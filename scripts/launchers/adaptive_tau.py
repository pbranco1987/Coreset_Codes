#!/usr/bin/env python
"""
Adaptive-tau NSGA-II launcher for geographic coreset selection.

This module implements the primary construction-only launcher for the clean
experimental pipeline.  It runs a multi-objective (NSGA-II) coreset selector
with an *adaptive penalty parameter* (tau) that controls the tightness of
geographic proportionality constraints.

Why adaptive tau?
-----------------
Geographic constraints (e.g. "each state gets a population-proportional share
of the coreset") are encoded as soft penalty terms with a temperature
parameter tau.  A tau that is too small makes the feasible region vanishingly
narrow and the search stalls; a tau that is too large under-penalizes
violations, producing infeasible Pareto fronts.  Rather than hand-tuning tau
per (k, space, constraint) triple, this launcher *automatically calibrates*
tau so that roughly 50 % of the NSGA-II population is feasible -- the sweet
spot where evolutionary pressure drives solutions toward feasibility without
starving the search of diversity.

Three-phase calibration protocol
---------------------------------
1. **Probe** (exponential search).  Starting from the greedy KL floor
   (Corollary 2 in the manuscript), tau is doubled every PROBE_PATIENCE
   generations until at least 50 % of the population becomes feasible.
   This establishes a bracket [tau_lo, tau_hi] where tau_lo is the last
   tau that failed the feasibility threshold and tau_hi is the first that
   succeeded.

2. **Bisect** (binary search with adaptive patience).  A standard bisection
   narrows [tau_lo, tau_hi] until the relative gap falls below 5 %.
   Patience per bisection step is proportional to the remaining gap (wider
   gaps get more generations to let the population adapt).  A *trend-aware
   extension* avoids premature bisection decisions: if feasibility is still
   climbing (positive slope over the last BISECT_TREND_WINDOW generations)
   but has not yet reached the threshold, the step gets BISECT_EXTEND_GENS
   extra generations before committing.

3. **Production** (fixed-tau exploitation).  Once bisection converges, the
   final tau_hi is locked and the optimizer runs for COMMITTED_GENS
   generations of pure Pareto-front refinement.

Architecture decisions
----------------------
- **Construction only**: this script outputs coreset masks and Pareto
  objective values (coreset.npz, representatives/).  All downstream quality
  metrics (Nystrom error, KRR RMSE, geographic KL, etc.) are computed by the
  separate ``scripts/analysis/evaluate_coresets.py`` script.  This separation
  keeps the launcher fast and allows re-evaluation without re-running the
  expensive optimization.

- **Generalized CLI**: the same launcher handles all scenario combinations
  (space in {vae, raw, pca}, constraint in {popsoft, munisoft, unconstrained},
  arbitrary objective sets) via command-line flags, eliminating per-scenario
  launcher scripts.

- **Deterministic seeding**: each replica index maps to a fixed seed via
  SEED_MAP, ensuring exact reproducibility across servers and reruns.

Usage examples
--------------
Single replica::

    python scripts/launchers/adaptive_tau.py \\
        --k 100 --space vae --constraint-mode popsoft --rep 0 \\
        --cache-dir replicate_cache_seed4200 \\
        --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

All 5 replicas::

    python scripts/launchers/adaptive_tau.py \\
        --k 300 --space raw --constraint-mode popsoft --all \\
        --cache-dir replicate_cache_seed4200 \\
        --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project imports -- add the repository root so that ``coreset_selection``
# is importable regardless of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from coreset_selection.geo.info import build_geo_info, GeoInfo
from coreset_selection.geo.projector import GeographicConstraintProjector
from coreset_selection.constraints.proportionality import (
    build_population_share_constraint,
    build_municipality_share_constraint,
    ProportionalityConstraintSet,
)
from coreset_selection.objectives.computer import build_space_objective_computer
from coreset_selection.config._dc_components import MMDConfig, SinkhornConfig
from coreset_selection.optimization.nsga2_internal import (
    fast_non_dominated_sort,
    _constraint_dominated_sort,
    _assign_crowding,
    _tournament_select,
    _uniform_crossover,
    _bitflip_mutation_exact_k,
    _swap_mutation_quota,
    _repair_mask,
    _evaluate_population,
)
from coreset_selection.data.cache import load_replicate_cache
from coreset_selection.optimization.selection import select_pareto_representatives

# ── Fixed NSGA-II hyperparameters ──────────────────────────────────────────
POP_SIZE: int = 300               # Number of individuals per generation.
CROSSOVER_PROB: float = 0.9       # Probability of applying uniform crossover.
MUTATION_PROB: float = 0.1        # Per-offspring probability of mutation.
MUTATION_SWAPS: int = 1           # Number of swap mutations per triggered event.
ALPHA_GEO: float = 1.0           # Geographic constraint strictness (1.0 = exact share).

# ── Probe phase constants ──────────────────────────────────────────────────
# The probe phase performs an exponential search for a tau that makes at
# least PROBE_FEAS_RATIO of the population feasible.
PROBE_PATIENCE: int = 15         # Generations to wait before doubling tau.
PROBE_GROWTH: float = 2.0        # Multiplicative factor for tau each bump.
PROBE_FEAS_RATIO: float = 0.50   # Target: >= 50 % of population feasible.

# ── Bisect phase constants ─────────────────────────────────────────────────
# Binary search refines the [tau_lo, tau_hi] bracket.  Patience is
# adaptive: wider gaps receive more generations (up to BISECT_PATIENCE_MAX)
# because the population needs more time to re-equilibrate after a large
# tau change.
BISECT_PATIENCE_BASE: int = 40   # Minimum generations per bisect step.
BISECT_PATIENCE_MAX: int = 60    # Maximum generations per bisect step.
BISECT_FEAS_RATIO: float = 0.50  # Same 50 % feasibility target.
BISECT_TOLERANCE: float = 0.05   # Convergence: (tau_hi - tau_lo)/tau_hi < 5%.

# ── Trend-aware extension ─────────────────────────────────────────────────
# If feasibility is still climbing (positive slope over the last
# BISECT_TREND_WINDOW gens) but has not reached the target, we extend
# patience by BISECT_EXTEND_GENS to avoid a premature "infeasible" verdict
# that would incorrectly lower tau_hi.
BISECT_TREND_WINDOW: int = 10    # Window size for slope estimation.
BISECT_TREND_SLOPE: float = 0.02 # Minimum positive slope to trigger extension.
BISECT_EXTEND_GENS: int = 15     # Extra generations granted by extension.

# ── Production phase ───────────────────────────────────────────────────────
COMMITTED_GENS: int = 1500       # Fixed-tau exploitation generations.

# ── Deterministic seed mapping ─────────────────────────────────────────────
# Each replica index (0--4) maps to a unique seed.  These seeds were chosen
# to be fresh (not reusing old 2026-2030 or 7001-7005 ranges) to avoid any
# accidental correlation with earlier pilot experiments.
SEED_MAP: Dict[int, int] = {0: 4200, 1: 4201, 2: 4202, 3: 4203, 4: 4204}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    """Return the first 12 characters of the current git commit hash.

    Used to stamp experiment manifests for provenance tracking.

    Returns
    -------
    str
        Abbreviated commit hash, or ``"unknown"`` if git is unavailable.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def build_experiment_name(
    space: str,
    constraint_mode: str,
    k: int,
    rep_id: int,
    objectives: Tuple[str, ...] = ("mmd", "sinkhorn"),
) -> str:
    """Build a descriptive experiment folder name from scenario parameters.

    The naming convention encodes the full experimental condition so that
    folder names are self-documenting and can be parsed programmatically
    by downstream analysis scripts.

    Parameters
    ----------
    space : str
        Representation space (``"vae"``, ``"pca"``, or ``"raw"``).
    constraint_mode : str
        Geographic constraint mode (``"popsoft"``, ``"munisoft"``, or
        ``"unconstrained"``).
    k : int
        Target coreset cardinality.
    rep_id : int
        Replica index (0--4).
    objectives : tuple of str
        Objective function names (e.g. ``("mmd", "sinkhorn")``).

    Returns
    -------
    str
        Experiment folder name, e.g. ``"nsga2-vae-popsoft-k100-rep0"``.
    """
    obj_set = set(objectives)
    # Standard bi-objective (MMD + Sinkhorn): the default pipeline
    if obj_set == {"mmd", "sinkhorn"}:
        return f"nsga2-{space}-{constraint_mode}-k{k}-rep{rep_id}"
    # Ablation: MMD-only (removes transport-based objective)
    elif obj_set == {"mmd"}:
        return f"ablation-mmdonly-k{k}-rep{rep_id}"
    # Ablation: Sinkhorn-only (removes kernel-based objective)
    elif obj_set == {"sinkhorn"}:
        return f"ablation-sinkonly-k{k}-rep{rep_id}"
    # Tri-objective extension: adds Nystrom log-determinant
    elif "nystrom_logdet" in obj_set or "logdet" in obj_set:
        return f"ablation-triobjective-k{k}-rep{rep_id}"
    # Catch-all for future objective combinations
    else:
        obj_tag = "-".join(sorted(objectives))
        return f"nsga2-{space}-{constraint_mode}-{obj_tag}-k{k}-rep{rep_id}"


def build_constraint_set(
    geo: GeoInfo,
    population: np.ndarray,
    constraint_mode: str,
    tau: float = 1e6,
) -> Tuple[Optional[ProportionalityConstraintSet], str]:
    """Build the geographic proportionality constraint set.

    Three constraint modes are supported:

    - **popsoft**: Population-share constraint.  Each geographic group
      (state/region) must receive a coreset share proportional to its
      population weight.  Violations are penalized via a soft KL-divergence
      penalty scaled by tau.

    - **munisoft**: Municipality-share constraint.  Each municipality must
      receive a share proportional to its count (unweighted by population).
      This is a stricter spatial coverage requirement.

    - **unconstrained**: No geographic constraint.  The optimizer freely
      selects any k points.  Used as an ablation baseline.

    Parameters
    ----------
    geo : GeoInfo
        Geographic metadata (group labels, population weights, etc.).
    population : np.ndarray
        Population weight array, shape ``(N,)``.
    constraint_mode : str
        One of ``"popsoft"``, ``"munisoft"``, ``"unconstrained"``.
    tau : float
        Initial penalty temperature.  Higher tau makes the constraint
        easier to satisfy (wider feasible region).

    Returns
    -------
    constraint_set : ProportionalityConstraintSet or None
        The constructed constraint set, or ``None`` for unconstrained mode.
    weight_type : str
        Weight type string (``"pop"`` or ``"muni"``) passed to the
        geographic projector to determine how target counts are computed.
    """
    constraints: List = []
    weight_type: str = "pop"  # default for the geographic projector

    if constraint_mode == "popsoft":
        # Population-share constraint: coreset share ~ population share
        c = build_population_share_constraint(
            geo=geo, population=population, alpha=ALPHA_GEO, tau=tau,
        )
        constraints.append(c)
        weight_type = "pop"
    elif constraint_mode == "munisoft":
        # Municipality-share constraint: coreset share ~ municipality count share
        c = build_municipality_share_constraint(
            geo=geo, alpha=ALPHA_GEO, tau=tau,
        )
        constraints.append(c)
        weight_type = "muni"
    elif constraint_mode == "unconstrained":
        # No geographic constraint -- optimizer has full freedom
        pass
    else:
        raise ValueError(f"Unsupported constraint mode: {constraint_mode}. "
                         f"Supported: popsoft, munisoft, unconstrained. "
                         f"Hard/joint modes use the standard runner, not adaptive-tau.")

    if not constraints:
        # Unconstrained: return None so the caller skips all constraint logic
        return None, weight_type

    # Wrap individual constraints into a set that:
    #   - enforces min_one_per_group (every state gets at least 1 point)
    #   - does NOT preserve historical group counts (allows redistribution)
    #   - uses up to 200 repair iterations for feasibility projection
    cs = ProportionalityConstraintSet(
        geo=geo, constraints=constraints,
        min_one_per_group=True, preserve_group_counts=False, max_iters=200,
    )
    return cs, weight_type


def get_representation(
    assets: object, space: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract the representation matrix for the requested optimization space.

    Parameters
    ----------
    assets : object
        Loaded replicate cache (has attributes ``Z_vae``, ``Z_pca``,
        ``X_scaled``, and optionally ``Z_logvar``).
    space : str
        One of ``"vae"`` (VAE latent), ``"pca"`` (PCA embedding), or
        ``"raw"`` (scaled original features).

    Returns
    -------
    X : np.ndarray
        Representation matrix, shape ``(N, d)``.
    logvar : np.ndarray or None
        VAE log-variance matrix (shape ``(N, d)``), only for ``space="vae"``.
        Used by the Sinkhorn objective to incorporate encoding uncertainty.
    """
    if space == "vae":
        X = np.asarray(assets.Z_vae, dtype=np.float64)
        logvar = getattr(assets, "Z_logvar", None)
        if logvar is not None:
            logvar = np.asarray(logvar, dtype=np.float64)
        return X, logvar
    elif space == "pca":
        return np.asarray(assets.Z_pca, dtype=np.float64), None
    elif space == "raw":
        return np.asarray(assets.X_scaled, dtype=np.float64), None
    else:
        raise ValueError(f"Unknown space: {space}")


def compute_greedy_kl_floor(
    geo: GeoInfo,
    constraint_set: ProportionalityConstraintSet,
    k: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """Compute the greedy KL floor for initial tau calibration (Corollary 2).

    The greedy KL floor is the minimum KL divergence achievable by a
    deterministic greedy algorithm that assigns points to states in order
    of decreasing KL reduction.  It provides a principled lower bound for
    the initial tau: starting below this value would make *every* solution
    infeasible, wasting probe generations.

    Parameters
    ----------
    geo : GeoInfo
        Geographic metadata.
    constraint_set : ProportionalityConstraintSet
        Constraint set (used to evaluate violation of the greedy solution).
    k : int
        Target coreset cardinality.
    rng : np.random.Generator
        Random generator (greedy init may break ties randomly).

    Returns
    -------
    greedy_kl : float
        Maximum KL violation of the greedy solution across all constraints.
    greedy_mask : np.ndarray
        Boolean selection mask of the greedy solution, shape ``(N,)``.
    """
    from coreset_selection.optimization.nsga2_internal import _greedy_kl_init

    greedy_mask = _greedy_kl_init(constraint_set, k, rng)
    # Evaluate worst-case KL across all constraints in the set
    greedy_kl = 0.0
    for c in constraint_set.constraints:
        greedy_kl = max(greedy_kl, c.value(greedy_mask, geo))
    return greedy_kl, greedy_mask


# ---------------------------------------------------------------------------
# Core: adaptive-tau NSGA-II (construction only, no downstream metrics)
# ---------------------------------------------------------------------------

def run_adaptive_tau_nsga2(
    *,
    rep_id: int,
    seed: int,
    k: int,
    space: str,
    constraint_mode: str,
    objectives: Tuple[str, ...],
    cache_path: str,
    verbose: bool = True,
) -> dict:
    """Run one replicate of adaptive-tau NSGA-II coreset construction.

    This is the main optimization loop.  It performs three phases (probe,
    bisect, production) to calibrate tau, then returns the final Pareto
    front of coreset masks and their objective values.

    Parameters
    ----------
    rep_id : int
        Replica index (0--4), used for logging.
    seed : int
        Random seed for full reproducibility.
    k : int
        Target coreset cardinality.
    space : str
        Representation space (``"vae"``, ``"pca"``, ``"raw"``).
    constraint_mode : str
        Geographic constraint mode (``"popsoft"``, ``"munisoft"``,
        ``"unconstrained"``).
    objectives : tuple of str
        Names of objective functions to optimize (e.g. ``("mmd", "sinkhorn")``).
    cache_path : str
        Path to the replicate cache file (``assets.npz``).
    verbose : bool
        If True, print per-generation progress.

    Returns
    -------
    dict
        Result dictionary containing:

        - ``X_pareto``: boolean mask array, shape ``(n_front, N)``
        - ``F_pareto``: objective values, shape ``(n_front, n_obj)``
        - ``greedy_kl_floor``: initial KL floor value
        - ``tau_init``, ``tau_final``: first and last tau values
        - ``total_gens``: total generations executed (all phases)
        - ``total_time``: wall-clock seconds
        - ``gen_log``: per-generation diagnostics list
    """
    t_start = time.time()

    # ── Load data assets ───────────────────────────────────────────────────
    assets = load_replicate_cache(cache_path)
    state_labels = assets.state_labels       # Geographic group label per point
    pop_array = np.asarray(assets.population, dtype=np.float64)  # Population weights
    N: int = len(state_labels)               # Total number of candidate points

    # ── Build geographic metadata ──────────────────────────────────────────
    geo: GeoInfo = build_geo_info(state_labels, population_weights=pop_array)

    # ── Build constraint set ───────────────────────────────────────────────
    # Use a large temporary tau (1e6) for the greedy KL floor computation;
    # the actual tau will be set by the probe/bisect phases.
    constraint_set, weight_type = build_constraint_set(
        geo, pop_array, constraint_mode, tau=1e6,
    )

    # ── Handle unconstrained mode ──────────────────────────────────────────
    is_unconstrained: bool = constraint_set is None
    if is_unconstrained:
        # No adaptive tau needed -- skip directly to production phase.
        # All constraint violation values will be zero throughout.
        greedy_kl_floor = 0.0
        greedy_mask = None
        tau_init = 0.0
    else:
        # Compute greedy KL floor (Corollary 2) to initialize tau.
        # This is the minimum achievable KL for cardinality k, providing
        # a principled starting point for the exponential probe.
        rng_greedy = np.random.default_rng(seed)
        greedy_kl_floor, greedy_mask = compute_greedy_kl_floor(
            geo, constraint_set, k, rng_greedy,
        )
        tau_init = greedy_kl_floor
        print(f"[Rep {rep_id}] Greedy KL floor(k={k}) = {greedy_kl_floor:.6f}")

        # Replace the temporary large tau with the greedy floor
        constraint_set.constraints = [
            dataclasses.replace(c, tau=tau_init)
            for c in constraint_set.constraints
        ]

    # ── Build geographic projector ─────────────────────────────────────────
    # The projector computes target counts per group and is used by the
    # repair operator to fix infeasible offspring.
    projector: GeographicConstraintProjector = GeographicConstraintProjector(
        geo=geo, alpha_geo=ALPHA_GEO, min_one_per_group=True,
        weight_type=weight_type,
    )

    # ── Build objective computer ───────────────────────────────────────────
    # Wraps MMD and Sinkhorn (and optionally other) objective computations
    # over the chosen representation space.
    X_repr, logvars = get_representation(assets, space)
    computer = build_space_objective_computer(
        X=X_repr, logvars=logvars,
        mmd_cfg=MMDConfig(), sinkhorn_cfg=SinkhornConfig(),
        seed=seed,
    )

    # ── Precompute group structure for mutation operators ───────────────────
    rng: np.random.Generator = np.random.default_rng(seed)
    gid: np.ndarray = geo.group_ids                          # Group label per point
    G: int = geo.G                                            # Number of groups
    group_to_indices: List[np.ndarray] = [                    # Point indices per group
        np.flatnonzero(gid == g) for g in range(G)
    ]

    # ── Initialize NSGA-II population ──────────────────────────────────────
    # Each individual is a boolean mask of length N with exactly k True values.
    pop_X: np.ndarray = np.zeros((POP_SIZE, N), dtype=bool)
    target_counts: np.ndarray = projector.target_counts(k)

    # Slot 0: seed the greedy-KL solution (if constrained) to ensure at
    # least one high-quality feasible individual is present from generation 0.
    if greedy_mask is not None:
        pop_X[0] = greedy_mask
        start_idx = 1
    else:
        start_idx = 0

    # Remaining slots: random stratified initialization.
    # Each individual is created by sampling target_counts[g] points from
    # each geographic group g, then repaired to enforce exact cardinality k.
    for i in range(start_idx, POP_SIZE):
        mask = np.zeros(N, dtype=bool)
        for g in range(G):
            c_g = int(target_counts[g])
            if c_g <= 0:
                continue
            idx_g = group_to_indices[g]
            if idx_g.size == 0:
                continue
            picks = rng.choice(idx_g, size=min(c_g, idx_g.size), replace=False)
            mask[picks] = True
        # Repair ensures exactly k selected points and geographic feasibility
        pop_X[i] = _repair_mask(
            projector, constraint_set, rng, mask, k,
            use_quota=not is_unconstrained, enforce_exact_k=True,
        )

    # ── Initial objective evaluation ───────────────────────────────────────
    # _obj_cache memoizes objective values by mask hash to avoid redundant
    # kernel/transport computations for identical masks.
    _obj_cache: dict = {}
    pop_F: np.ndarray = _evaluate_population(
        pop_X, computer, objectives, cache=_obj_cache
    )

    # ── Initial constraint violation ───────────────────────────────────────
    if is_unconstrained:
        pop_CV: np.ndarray = np.zeros(POP_SIZE, dtype=np.float64)
    else:
        pop_CV = np.array([
            constraint_set.total_violation(m) for m in pop_X
        ], dtype=np.float64)

    n_feas: int = int((pop_CV <= 0.0).sum())
    print(f"[Rep {rep_id}] Init: feasible={n_feas}/{POP_SIZE}, tau={tau_init:.6f}")

    # ══════════════════════════════════════════════════════════════════════
    # State machine: probe -> bisect -> production -> done
    #
    # The state machine governs tau calibration.  Transitions:
    #   probe  --[>=50% feasible]--> bisect  (or production if gap < 5%)
    #   probe  --[patience expired]-> probe  (double tau, reset counter)
    #   bisect --[converged]-------> production
    #   bisect --[step done]-------> bisect  (new midpoint)
    #   production --[COMMITTED_GENS done]--> STOP
    #
    # For unconstrained mode, the state starts directly at "production".
    # ══════════════════════════════════════════════════════════════════════
    current_tau: float = tau_init
    global_gen: int = 0
    state: str = "production" if is_unconstrained else "probe"
    committed_gens_remaining: int = COMMITTED_GENS if is_unconstrained else 0

    # -- Probe state variables --
    probe_counter: int = 0             # Generations since last tau bump

    # -- Bisect state variables --
    bisect_steps: int = 0              # Total bisection steps completed
    tau_lo: float = 0.0                # Lower bracket bound (last infeasible tau)
    tau_hi: Optional[float] = None     # Upper bracket bound (first feasible tau)
    bisect_counter: int = 0            # Generations in current bisect step
    bisect_patience_current: int = BISECT_PATIENCE_MAX  # Adaptive patience for this step
    bisect_extended: bool = False      # Whether trend extension was already applied
    bisect_feas_history: List[float] = []  # Feasibility ratios for trend detection

    # Edge case: if the initial population is already >= 50 % feasible at
    # tau_init, skip probe and bisect entirely -- go straight to production.
    if not is_unconstrained and n_feas >= int(PROBE_FEAS_RATIO * POP_SIZE):
        tau_hi = current_tau
        state = "production"
        committed_gens_remaining = COMMITTED_GENS
        print(f"[Rep {rep_id}] Skip search -> PRODUCTION tau={current_tau:.6f}")

    gen_log: List[dict] = []

    # ── Main evolutionary loop ─────────────────────────────────────────────
    while True:
        # ------------------------------------------------------------------
        # STEP 1: Non-dominated sorting with constraint domination.
        #
        # Constraint-dominated sorting (Deb 2002) gives priority to feasible
        # solutions: a feasible individual always dominates an infeasible one,
        # and among infeasible individuals, lower total violation wins.
        # ------------------------------------------------------------------
        fronts, rank = _constraint_dominated_sort(pop_F, pop_CV)

        # ------------------------------------------------------------------
        # STEP 2: Crowding distance assignment.
        #
        # Within each non-dominated front, crowding distance measures how
        # isolated an individual is in objective space.  Higher crowding
        # distance means more diversity, and is preferred in tournament
        # selection to maintain spread across the Pareto front.
        # ------------------------------------------------------------------
        crowd: np.ndarray = _assign_crowding(pop_F, fronts)

        # ------------------------------------------------------------------
        # STEP 3: Binary tournament selection.
        #
        # Pairs of individuals are compared by (rank, crowding distance).
        # Lower rank wins; ties are broken by higher crowding distance.
        # This selects POP_SIZE parents for offspring generation.
        # ------------------------------------------------------------------
        parents_idx: np.ndarray = _tournament_select(
            rng, rank, crowd, n_select=POP_SIZE
        )

        # ------------------------------------------------------------------
        # STEP 4: Offspring generation (crossover + mutation + repair).
        #
        # Parents are paired sequentially.  For each pair:
        #   (a) Uniform crossover: each gene (point) is independently
        #       inherited from one parent with probability CROSSOVER_PROB.
        #   (b) Swap mutation: with probability MUTATION_PROB, a selected
        #       point is swapped with an unselected point from the SAME
        #       geographic group (quota-preserving mutation).
        #   (c) Repair: the mask is projected back to exactly k selected
        #       points with valid geographic quotas.
        # ------------------------------------------------------------------
        offspring_X: np.ndarray = np.zeros((POP_SIZE, N), dtype=bool)
        for i in range(0, POP_SIZE, 2):
            p1 = pop_X[parents_idx[i]]
            p2 = pop_X[parents_idx[(i + 1) % POP_SIZE]]

            # (a) Uniform crossover: gene-wise mixing of two parent masks
            c1, c2 = _uniform_crossover(rng, p1, p2, CROSSOVER_PROB)

            # (b) Swap mutation: replace a selected point with an unselected
            #     point from the same geographic group.  This preserves
            #     per-group counts, avoiding unnecessary repair overhead.
            if rng.random() < MUTATION_PROB:
                for _ in range(MUTATION_SWAPS):
                    c1 = _swap_mutation_quota(rng, c1, gid, group_to_indices)
            if rng.random() < MUTATION_PROB:
                for _ in range(MUTATION_SWAPS):
                    c2 = _swap_mutation_quota(rng, c2, gid, group_to_indices)

            # (c) Repair: enforce exact cardinality k and geographic quotas.
            #     The projector redistributes excess/deficit points across
            #     groups, then randomly adds/removes within groups to hit k.
            c1 = _repair_mask(
                projector, constraint_set, rng, c1, k,
                use_quota=not is_unconstrained, enforce_exact_k=True,
            )
            c2 = _repair_mask(
                projector, constraint_set, rng, c2, k,
                use_quota=not is_unconstrained, enforce_exact_k=True,
            )

            offspring_X[i] = c1
            if i + 1 < POP_SIZE:
                offspring_X[i + 1] = c2

        # ------------------------------------------------------------------
        # STEP 5: Offspring evaluation.
        #
        # Compute objective values (MMD, Sinkhorn, etc.) and constraint
        # violations for all offspring.
        # ------------------------------------------------------------------
        offspring_F: np.ndarray = _evaluate_population(
            offspring_X, computer, objectives, cache=_obj_cache,
        )
        if is_unconstrained:
            offspring_CV: np.ndarray = np.zeros(POP_SIZE, dtype=np.float64)
        else:
            offspring_CV = np.array([
                constraint_set.total_violation(m) for m in offspring_X
            ], dtype=np.float64)

        # ------------------------------------------------------------------
        # STEP 6: Environmental selection (mu + lambda).
        #
        # Combine parents and offspring into a pool of 2 * POP_SIZE, then
        # select the best POP_SIZE by constraint-dominated sorting + crowding.
        # Fronts are added greedily; when the next front would overflow,
        # individuals within that front are ranked by crowding distance
        # (highest first) to fill the remaining slots.
        # ------------------------------------------------------------------
        combined_X: np.ndarray = np.vstack([pop_X, offspring_X])
        combined_F: np.ndarray = np.vstack([pop_F, offspring_F])
        combined_CV: np.ndarray = np.concatenate([pop_CV, offspring_CV])

        fronts, rank = _constraint_dominated_sort(combined_F, combined_CV)
        crowd = _assign_crowding(combined_F, fronts)

        # Select POP_SIZE survivors: fill front by front, break last front by crowding
        next_idx_list: List[int] = []
        for front in fronts:
            if len(next_idx_list) + front.size <= POP_SIZE:
                # Entire front fits -- add all members
                next_idx_list.extend(front.tolist())
            else:
                # Partial front: pick individuals with highest crowding distance
                remaining = POP_SIZE - len(next_idx_list)
                cd = crowd[front]
                order = np.argsort(-cd)  # Descending crowding distance
                next_idx_list.extend(front[order[:remaining]].tolist())
                break

        next_idx: np.ndarray = np.asarray(next_idx_list, dtype=int)
        pop_X = combined_X[next_idx]
        pop_F = combined_F[next_idx]
        pop_CV = combined_CV[next_idx]

        # Update feasibility count for state machine decisions
        n_feas = int((pop_CV <= 0.0).sum())
        global_gen += 1

        # ── Per-generation logging ─────────────────────────────────────────
        elapsed: float = time.time() - t_start
        f_min: np.ndarray = pop_F.min(axis=0)  # Best value per objective
        gen_log.append({
            "gen": global_gen, "tau": current_tau,
            "n_feasible": n_feas, "state": state,
            "f_min": [float(x) for x in f_min],
            "elapsed_s": elapsed,
        })

        # Print progress: every 50 gens in production, every gen during calibration
        if global_gen % 50 == 0 or state != "production":
            f_str = ", ".join(f"{x:.4f}" for x in f_min)
            print(
                f"[Rep {rep_id} gen {global_gen:5d}] "
                f"{state:10s} tau={current_tau:.6f} "
                f"feas={n_feas:3d}/{POP_SIZE} f_min=[{f_str}] "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Helper: atomically update tau across all constraints and
        # recompute constraint violations for the entire population.
        # This is called whenever the state machine changes tau.
        # ------------------------------------------------------------------
        def _set_tau(new_tau: float) -> None:
            nonlocal current_tau, n_feas
            current_tau = new_tau
            # Replace tau in every constraint via dataclass immutable update
            constraint_set.constraints = [
                dataclasses.replace(c, tau=current_tau)
                for c in constraint_set.constraints
            ]
            # Recompute violations under the new tau -- existing masks are
            # unchanged, but their violation values shift because tau scales
            # the feasibility boundary.
            pop_CV[:] = np.array(
                [constraint_set.total_violation(m) for m in pop_X],
                dtype=np.float64,
            )
            n_feas = int((pop_CV <= 0.0).sum())

        feas_ratio: float = n_feas / POP_SIZE

        # ==================================================================
        # STATE MACHINE TRANSITIONS
        # ==================================================================

        if state == "probe":
            # --------------------------------------------------------------
            # PROBE PHASE: exponential search for a feasible tau.
            #
            # Strategy: run PROBE_PATIENCE generations at the current tau.
            # If >= 50 % of the population becomes feasible, we have found
            # tau_hi and can transition.  If not, record current tau as
            # tau_lo (last known infeasible) and double tau to widen the
            # feasible region.
            #
            # Why double?  Exponential growth quickly spans orders of
            # magnitude, which is necessary because the relationship between
            # tau and feasibility is highly non-linear and problem-dependent.
            # A 2x factor balances speed (few probe steps) against overshoot
            # (not jumping too far past the optimal tau).
            # --------------------------------------------------------------
            probe_counter += 1

            if feas_ratio >= PROBE_FEAS_RATIO:
                # SUCCESS: this tau achieves sufficient feasibility.
                tau_hi = current_tau

                # Check if the bracket [tau_lo, tau_hi] is already tight enough
                gap: float = (tau_hi - tau_lo) / tau_hi if tau_lo > 0 else 0.0
                if gap > BISECT_TOLERANCE:
                    # Gap too wide -- need bisection to refine.
                    state = "bisect"
                    bisect_counter = 0
                    bisect_extended = False
                    bisect_feas_history = []
                    # Adaptive patience: wider gaps get more generations
                    # because larger tau changes need more evolutionary
                    # adaptation time.
                    bisect_patience_current = int(
                        BISECT_PATIENCE_BASE
                        + gap * (BISECT_PATIENCE_MAX - BISECT_PATIENCE_BASE)
                    )
                    # Start bisection at the midpoint of the bracket
                    _set_tau((tau_lo + tau_hi) / 2.0)
                    print(f"  PROBE OK -> BISECT: [{tau_lo:.6f}, {tau_hi:.6f}]")
                else:
                    # Bracket is already narrow -- go directly to production
                    state = "production"
                    committed_gens_remaining = COMMITTED_GENS
                    print(f"  PROBE -> PRODUCTION tau={current_tau:.6f}")

            elif probe_counter >= PROBE_PATIENCE:
                # FAILURE: tau is too tight.  Record it as the lower bound
                # and double tau to make constraints easier to satisfy.
                tau_lo = current_tau
                _set_tau(current_tau * PROBE_GROWTH)
                probe_counter = 0
                print(f"  PROBE BUMP -> tau={current_tau:.6f}")

        elif state == "bisect":
            # --------------------------------------------------------------
            # BISECT PHASE: binary search to refine the tau bracket.
            #
            # At each step, we test the midpoint tau_mid = (tau_lo+tau_hi)/2.
            # After bisect_patience_current generations:
            #   - If >= 50 % feasible: tau_mid becomes new tau_hi
            #   - Otherwise: tau_mid becomes new tau_lo
            # Then we compute the new midpoint and repeat.
            #
            # Adaptive patience: patience scales with the relative gap.
            # Rationale: a large tau change between bisection steps requires
            # more generations for the population to re-equilibrate; short
            # patience would read a transient feasibility ratio and make
            # incorrect bracket updates.
            # --------------------------------------------------------------
            bisect_counter += 1
            bisect_feas_history.append(feas_ratio)
            reached: bool = bisect_counter >= bisect_patience_current

            # -- Trend-aware extension --
            # If patience is exhausted but feasibility is still climbing
            # (positive slope over the last BISECT_TREND_WINDOW gens),
            # grant BISECT_EXTEND_GENS extra generations.  This prevents
            # a premature "infeasible" verdict when the population is
            # clearly converging toward feasibility but needs a few more
            # generations.  Extension is applied at most once per bisect
            # step to bound total calibration time.
            if (reached and not bisect_extended
                    and len(bisect_feas_history) >= BISECT_TREND_WINDOW):
                recent = bisect_feas_history[-BISECT_TREND_WINDOW:]
                slope = recent[-1] - recent[0]
                if slope > BISECT_TREND_SLOPE and feas_ratio < BISECT_FEAS_RATIO:
                    bisect_extended = True
                    bisect_patience_current += BISECT_EXTEND_GENS
                    reached = False  # Not done yet -- continue with extended patience

            if reached:
                # This bisection step is complete -- commit the verdict
                bisect_steps += 1
                if feas_ratio >= BISECT_FEAS_RATIO:
                    # Midpoint is feasible enough -- it becomes the new upper bound
                    tau_hi = current_tau
                else:
                    # Midpoint is too tight -- it becomes the new lower bound
                    tau_lo = current_tau

                # Check convergence: relative gap below tolerance
                gap = (tau_hi - tau_lo) / tau_hi
                if gap < BISECT_TOLERANCE:
                    # Converged: use the upper bound (guaranteed feasible)
                    # and enter production.
                    _set_tau(tau_hi)
                    state = "production"
                    committed_gens_remaining = COMMITTED_GENS
                    print(f"  BISECT CONVERGED ({bisect_steps} steps) -> PRODUCTION tau={current_tau:.6f}")
                else:
                    # Not yet converged: reset counters, compute new midpoint
                    bisect_counter = 0
                    bisect_extended = False
                    bisect_feas_history = []
                    # Recalculate adaptive patience for the narrower bracket
                    bisect_patience_current = int(
                        BISECT_PATIENCE_BASE
                        + gap * (BISECT_PATIENCE_MAX - BISECT_PATIENCE_BASE)
                    )
                    _set_tau((tau_lo + tau_hi) / 2.0)

        elif state == "production":
            # --------------------------------------------------------------
            # PRODUCTION PHASE: fixed-tau Pareto front refinement.
            #
            # Tau is locked at the bisected value.  The optimizer runs for
            # COMMITTED_GENS generations of pure multi-objective optimization
            # to converge the Pareto front.
            # --------------------------------------------------------------
            committed_gens_remaining -= 1
            if committed_gens_remaining <= 0:
                print(f"  PRODUCTION done ({COMMITTED_GENS} gens). Done.")
                break

    # ══════════════════════════════════════════════════════════════════════
    # Extract final Pareto front from the terminal population
    # ══════════════════════════════════════════════════════════════════════
    total_time: float = time.time() - t_start
    feas_mask: np.ndarray = pop_CV <= 0.0

    if feas_mask.any():
        # Standard case: extract the first non-dominated front among
        # feasible individuals.
        idx_f = np.flatnonzero(feas_mask)
        fronts_f, _ = fast_non_dominated_sort(pop_F[idx_f])
        f0 = idx_f[fronts_f[0]]
    else:
        # Fallback: no feasible solutions found (should be rare with
        # adaptive tau).  Return the 50 least-infeasible individuals
        # so downstream analysis can still proceed with a warning.
        order = np.argsort(pop_CV)
        f0 = order[:min(50, len(order))]
        print(f"  [WARNING] No feasible solutions — returning {len(f0)} least-infeasible")

    X_pareto: np.ndarray = pop_X[f0]
    F_pareto: np.ndarray = pop_F[f0]

    print(
        f"\n[Rep {rep_id}] DONE: {global_gen} gens, tau={current_tau:.6f}, "
        f"front={len(f0)} solutions, wall_clock={total_time:.0f}s"
    )

    return {
        "rep_id": rep_id, "seed": seed, "k": k,
        "space": space, "constraint_mode": constraint_mode,
        "objectives": list(objectives),
        "X_pareto": X_pareto, "F_pareto": F_pareto,
        "greedy_kl_floor": greedy_kl_floor,
        "tau_init": tau_init, "tau_final": current_tau,
        "total_gens": global_gen, "total_time": total_time,
        "gen_log": gen_log,
    }


# ---------------------------------------------------------------------------
# Save construction output (Pareto front only -- NO downstream metrics)
# ---------------------------------------------------------------------------

def save_construction_output(result: dict, output_dir: Path) -> None:
    """Save the Pareto front, representative coresets, and metadata.

    This function persists everything needed by the downstream evaluation
    script (``scripts/analysis/evaluate_coresets.py``) to compute quality
    metrics without re-running the optimization.

    Output structure::

        output_dir/
            coreset.npz             # Full Pareto front (masks + objectives)
            representatives/        # Named representative coresets
                best_mmd.npz
                best_sinkhorn.npz
                knee.npz
            adaptive-tau-log.json   # Per-generation calibration diagnostics
            wall-clock.json         # Timing summary
            manifest.json           # Full experiment metadata for provenance

    Parameters
    ----------
    result : dict
        Output from :func:`run_adaptive_tau_nsga2`.
    output_dir : Path
        Directory to write output files into (created if needed).
    """
    X_pareto: np.ndarray = result["X_pareto"]
    F_pareto: np.ndarray = result["F_pareto"]
    objectives: list = result["objectives"]

    os.makedirs(output_dir, exist_ok=True)

    # ── Save full Pareto front as compressed numpy archive ─────────────────
    np.savez_compressed(
        str(output_dir / "coreset.npz"),
        X=X_pareto,           # Boolean masks, shape (n_front, N)
        F=F_pareto,           # Objective values, shape (n_front, n_obj)
        objectives=np.array(objectives, dtype=object),
    )

    # ── Select and save representative coresets ────────────────────────────
    # Representatives are named solutions from the Pareto front:
    #   - best_mmd:      lowest MMD value (best kernel fidelity)
    #   - best_sinkhorn: lowest Sinkhorn value (best transport fidelity)
    #   - knee:          knee-point solution (best trade-off)
    reps: Dict[str, int] = select_pareto_representatives(
        F_pareto, objectives=tuple(objectives)
    )
    reps_dir: Path = output_dir / "representatives"
    os.makedirs(reps_dir, exist_ok=True)

    for rep_name, pareto_idx in reps.items():
        mask = X_pareto[pareto_idx]
        indices = np.flatnonzero(mask)  # Integer indices of selected points
        np.savez_compressed(
            str(reps_dir / f"{rep_name}.npz"),
            indices=indices,
            mask=mask,
            pareto_idx=np.array([pareto_idx]),
        )

    # ── Save adaptive-tau calibration log ──────────────────────────────────
    # JSON array of per-generation records; useful for plotting convergence
    # curves and diagnosing calibration behavior.
    with open(output_dir / "adaptive-tau-log.json", "w") as f:
        json.dump(result["gen_log"], f, indent=1)

    # ── Save wall-clock timing ─────────────────────────────────────────────
    with open(output_dir / "wall-clock.json", "w") as f:
        json.dump({
            "total_seconds": result["total_time"],
            "total_generations": result["total_gens"],
        }, f, indent=2)

    # ── Save experiment manifest for provenance ────────────────────────────
    # Contains all hyperparameters, calibration results, and git commit
    # so that any experiment folder is fully self-describing.
    manifest: dict = {
        "rep_id": result["rep_id"],
        "seed": result["seed"],
        "k": result["k"],
        "space": result["space"],
        "constraint_mode": result["constraint_mode"],
        "objectives": result["objectives"],
        "pop_size": POP_SIZE,
        "committed_gens": COMMITTED_GENS,
        "greedy_kl_floor": result["greedy_kl_floor"],
        "tau_init": result["tau_init"],
        "tau_final": result["tau_final"],
        "total_gens": result["total_gens"],
        "n_pareto_solutions": int(X_pareto.shape[0]),
        "n_representatives": len(reps),
        "representative_names": list(reps.keys()),
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "alpha_geo": ALPHA_GEO,
        "probe_patience": PROBE_PATIENCE,
        "bisect_tolerance": BISECT_TOLERANCE,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    coreset.npz ({X_pareto.shape[0]} front members)")
    print(f"    representatives/: {list(reps.keys())}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and run adaptive-tau NSGA-II.

    Supports running a single replica (``--rep N``) or all 5 replicas
    (``--all``).  Existing experiment folders with a ``coreset.npz`` are
    automatically skipped to support safe re-execution of partially
    completed batches.
    """
    parser = argparse.ArgumentParser(
        description="Adaptive-tau NSGA-II — construction only (no metrics).",
    )
    parser.add_argument("--k", type=int, required=True,
                        help="Coreset cardinality")
    parser.add_argument("--space", type=str, required=True,
                        choices=["vae", "raw", "pca"],
                        help="Representation space for optimization")
    parser.add_argument("--constraint-mode", type=str, required=True,
                        choices=["popsoft", "munisoft", "unconstrained"],
                        help="Constraint mode")
    parser.add_argument("--objectives", type=str, nargs="+",
                        default=["mmd", "sinkhorn"],
                        help="Objective functions (default: mmd sinkhorn)")
    parser.add_argument("--rep", type=int, default=None,
                        help="Replica ID (0-4)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 5 replicas")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Path to replicate cache directory "
                             "(contains rep00/, rep01/, ...)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base output directory for experiment folders")
    args = parser.parse_args()

    objectives: Tuple[str, ...] = tuple(args.objectives)

    # Determine which replicas to run
    if args.all:
        reps: List[int] = list(range(5))
    elif args.rep is not None:
        reps = [args.rep]
    else:
        parser.error("Specify --rep N or --all")

    cache_dir: Path = Path(args.cache_dir).resolve()
    output_base: Path = Path(args.output_dir).resolve()

    for rep_id in reps:
        seed: int = SEED_MAP[rep_id]
        cache_path: str = str(cache_dir / f"rep{rep_id:02d}" / "assets.npz")

        exp_name: str = build_experiment_name(
            args.space, args.constraint_mode, args.k, rep_id, objectives,
        )
        exp_dir: Path = output_base / exp_name

        # Skip already-completed experiments (idempotent re-execution)
        if (exp_dir / "coreset.npz").exists():
            print(f"\n[SKIP] {exp_name} — coreset.npz already exists")
            continue

        print(f"\n{'='*70}")
        print(f"  {exp_name} (seed={seed})")
        print(f"{'='*70}\n")

        result: dict = run_adaptive_tau_nsga2(
            rep_id=rep_id, seed=seed, k=args.k,
            space=args.space, constraint_mode=args.constraint_mode,
            objectives=objectives, cache_path=cache_path,
        )

        save_construction_output(result, exp_dir)


if __name__ == "__main__":
    main()
