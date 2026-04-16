"""
Pure-Numpy NSGA-II implementation for binary coreset selection.

This repository originally used ``pymoo`` for NSGA-II. To make the package
self-contained and fully reproducible, we implement the required NSGA-II
components directly.

Design goals:
- Works on binary masks in {0,1}^N with exact-k or quota constraints.
- Uses uniform crossover and swap-style mutations (quota-preserving when enabled).
- Applies projection repairs after variation so evolution proceeds on feasible masks.

This module is *not* intended as a general-purpose evolutionary library; it is
tailored to the manuscript's Algorithm 3 setup and the coreset-selection use case.

Verbosity Levels:
-----------------
The ``verbose`` parameter controls the amount of logging output:
- 0 or False: Silent mode (no output)
- 1 or True: Basic progress (every 10 generations)
- 2: Detailed progress (every generation with objective statistics)
- 3: Full diagnostics (includes population diversity, repair stats, timing)

Implementation Notes:
---------------------
The implementation is split across sub-modules for maintainability:

- ``_nsga2_logging``   -- statistics tracking, verbose output, formatting
- ``_nsga2_operators``  -- genetic operators, constraint handling, evaluation

All names are re-exported here so that external code can continue to import
from ``optimization.nsga2_internal`` without change.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..geo.projector import GeographicConstraintProjector
from ..constraints.proportionality import ProportionalityConstraintSet
from ..objectives.computer import SpaceObjectiveComputer
from ..utils.debug_timing import timer

# ── Re-exports from sub-modules ────────────────────────────────────────────
from ._nsga2_logging import (               # noqa: F401
    NSGA2Stats,
    NSGA2VerboseLogger,
    _compute_population_diversity,
    _compute_hypervolume_2d,
    _format_objective_stats,
    _format_progress_bar,
)

from ._nsga2_operators import (              # noqa: F401
    _dominates,
    _assign_crowding,
    _tournament_select,
    _uniform_crossover,
    _bitflip_mutation_exact_k,
    _swap_mutation_quota,
    _repair_mask,
    _constraint_dominated_sort,
    _evaluate_population,
)

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# CORE PUBLIC FUNCTIONS (kept in this module)
# =============================================================================

def fast_non_dominated_sort(F: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Fast non-dominated sorting (Deb et al., NSGA-II).

    Uses vectorized dominance testing: for each point *p* the entire
    population is compared at once via broadcasting, replacing the
    original O(n²·m) Python loop with O(n²·m) numpy ops — typically
    10-30× faster for pop_size <= 500.

    Parameters
    ----------
    F : np.ndarray
        Objective matrix (n, m), minimization.

    Returns
    -------
    fronts : List[np.ndarray]
        List of fronts; each front is an array of indices.
    rank : np.ndarray
        Rank of each point (0 = non-dominated).
    """
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]

    # Vectorised dominance: for every pair (p, q) determine if p dominates q.
    # dom[p, q] = True  iff  F[p] <= F[q] element-wise AND F[p] < F[q] on
    # at least one objective.
    # Shape of intermediate: (n, n, m) – acceptable for pop_size <= ~500.
    leq = F[:, None, :] <= F[None, :, :]   # (n, n, m)
    lt  = F[:, None, :] <  F[None, :, :]   # (n, n, m)
    dom = leq.all(axis=2) & lt.any(axis=2) # (n, n) — dom[p,q]=p dominates q

    # S[p]: set of indices that p dominates
    S: List[List[int]] = [[] for _ in range(n)]
    # n_dom[p]: number of points that dominate p
    n_dom = dom.sum(axis=0).astype(int)  # column-sum = how many dominate q

    rank = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        S[p] = np.flatnonzero(dom[p]).tolist()
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    # Convert to arrays, drop last empty
    fronts_arr = [np.asarray(front, dtype=int) for front in fronts if len(front) > 0]
    return fronts_arr, rank


def _save_checkpoint(
    path: str,
    *,
    pop_X: np.ndarray,
    pop_F: np.ndarray,
    pop_CV: np.ndarray,
    gen: int,
    seed: int,
    rng: np.random.Generator,
    repair_needed: list,
    repair_magnitude: list,
) -> None:
    """Atomically save an NSGA-II checkpoint to disk.

    Writes to a temporary file first, then renames — so a power loss
    mid-write never corrupts an existing checkpoint.
    """
    import os
    import pickle
    # np.savez_compressed auto-appends '.npz' if the filename doesn't
    # already end with it.  Make the temp file end with '.npz' so numpy
    # doesn't add a second extension, and os.replace works correctly.
    if path.endswith(".npz"):
        tmp = path[:-4] + ".tmp.npz"
    else:
        tmp = path + ".tmp"
    # Serialize RNG state (BitGenerator state dict) alongside arrays.
    # pickle → bytes → uint8 array so it fits inside npz.
    rng_state_bytes = pickle.dumps(rng.bit_generator.state)
    rng_state_arr = np.frombuffer(rng_state_bytes, dtype=np.uint8).copy()
    np.savez_compressed(
        tmp,
        pop_X=pop_X,
        pop_F=pop_F,
        pop_CV=pop_CV,
        gen=np.array([gen]),
        seed=np.array([seed]),
        rng_state=rng_state_arr,
        repair_needed=np.asarray(repair_needed, dtype=bool),
        repair_magnitude=np.asarray(repair_magnitude, dtype=np.int32),
    )
    # Atomic rename (same filesystem) — either the old or new file
    # exists, never a half-written state.
    os.replace(tmp, path)


def _load_checkpoint(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load an NSGA-II checkpoint if it exists and is valid."""
    import os
    if not os.path.isfile(path):
        return None
    try:
        data = dict(np.load(path, allow_pickle=False))
        # Validate required keys
        for key in ("pop_X", "pop_F", "pop_CV", "gen", "seed"):
            if key not in data:
                logger.warning("Checkpoint %s missing key %s — ignoring", path, key)
                return None
        return data
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s — starting fresh", path, e)
        return None


import os as _os

_PARETO_WARN_MIN_NONDOM = int(_os.environ.get("PARETO_WARN_MIN_NONDOM", "3"))
_PARETO_WARN_MAX_CORR = float(_os.environ.get("PARETO_WARN_MAX_CORR", "0.85"))


def _validate_pareto_front(
    F: np.ndarray,
    has_constraints: bool,
    cv: np.ndarray,
) -> None:
    """
    Validate Pareto front quality and emit warnings if degenerate.

    Checks:
    1. Number of non-dominated points (warn if < PARETO_WARN_MIN_NONDOM)
    2. Objective correlation (warn if > PARETO_WARN_MAX_CORR for bi-objective)
    3. Whether any solution is feasible (warn if none)

    Warnings are emitted via both ``logging.warning`` and ``print`` to ensure
    visibility in batch runs.
    """
    n_pts, n_obj = F.shape
    warnings_issued: List[str] = []

    # 1. Count non-dominated points
    fronts, _ = fast_non_dominated_sort(F)
    n_nondom = len(fronts[0])
    if n_nondom < _PARETO_WARN_MIN_NONDOM:
        msg = (
            f"[PARETO-WARNING] Degenerate front: only {n_nondom} non-dominated "
            f"point(s) out of {n_pts} returned. The Pareto front may not "
            f"represent a genuine trade-off surface."
        )
        warnings_issued.append(msg)

    # 2. Objective correlation (bi-objective case)
    if n_obj == 2 and n_pts >= 5:
        corr = np.corrcoef(F[:, 0], F[:, 1])[0, 1]
        if abs(corr) > _PARETO_WARN_MAX_CORR:
            msg = (
                f"[PARETO-WARNING] High objective correlation: "
                f"corr(obj0, obj1) = {corr:.4f}. Objectives may not be "
                f"providing distinct selection pressure."
            )
            warnings_issued.append(msg)

    # 3. Feasibility check
    if has_constraints:
        n_feasible = int(np.sum(cv <= 0.0))
        if n_feasible == 0:
            msg = (
                f"[PARETO-WARNING] No feasible solutions in returned front "
                f"({n_pts} points, min CV = {cv.min():.6f}). "
                f"Consider relaxing constraint tolerance (tau)."
            )
            warnings_issued.append(msg)
        elif n_feasible < _PARETO_WARN_MIN_NONDOM:
            msg = (
                f"[PARETO-WARNING] Only {n_feasible} feasible solution(s) "
                f"out of {n_pts}. Constraint may be too tight."
            )
            warnings_issued.append(msg)

    # Emit all warnings
    for w in warnings_issued:
        logger.warning(w)
        print(w, flush=True)

    if not warnings_issued:
        print(
            f"[PARETO-OK] Front quality: {n_nondom} non-dominated points "
            f"out of {n_pts} returned.",
            flush=True,
        )


def _greedy_kl_init(
    constraint_set: "ProportionalityConstraintSet",
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one boolean mask (N,) that greedily minimises population-share KL.

    Algorithm:
      1. Seed one municipality per state (the one whose population is closest
         to the ideal per-municipality contribution for that state).
      2. Greedily add municipalities that most reduce KL(π ‖ π̂) until |S|=k.

    This uses the *first* constraint in `constraint_set` whose weights are
    population-based (name == 'population_share').  Falls back to uniform
    weights if no population constraint is found.

    The resulting mask is a high-quality KL seed for the NSGA-II initial
    population.
    """
    geo = constraint_set.geo
    N, G = geo.N, geo.G

    # Find the population-share constraint
    pc = None
    for c in constraint_set.constraints:
        if c.name == "population_share":
            pc = c
            break
    if pc is None:
        # Fallback: use uniform weights
        pc = constraint_set.constraints[0] if constraint_set.constraints else None
    if pc is None:
        raise ValueError("No constraint available for greedy-KL init.")

    weights = pc.weights          # (N,) – population per municipality
    target_pi = pc.target_pi      # (G,) – target state shares
    alpha = pc.alpha

    # --- Step 1: seed one municipality per state ---
    selected = np.zeros(N, dtype=bool)
    for g in range(G):
        idx_g = geo.group_to_indices[g]
        if idx_g.size == 0:
            continue
        # Pick the municipality whose population is closest to the
        # "ideal" per-municipality contribution:  ideal_w = pi_g * W_total / 1
        # Since W_total is unknown at init, pick the median-population municipality.
        pops_g = weights[idx_g]
        median_pos = np.argsort(pops_g)[len(pops_g) // 2]
        selected[idx_g[median_pos]] = True

    k_init = int(selected.sum())
    if k_init > k:
        # More states than k (unlikely for k>=100, G=27); randomly drop
        on = np.flatnonzero(selected)
        drop = rng.choice(on, size=k_init - k, replace=False)
        selected[drop] = False
        return selected

    # --- Step 2: greedy KL minimisation ---
    # Pre-compute per-group weight accumulators for current selection
    Wg = np.zeros(G, dtype=np.float64)
    for g in range(G):
        idx_g = geo.group_to_indices[g]
        sel_in_g = idx_g[selected[idx_g]]
        if sel_in_g.size > 0:
            Wg[g] = weights[sel_in_g].sum()
    W_total = Wg.sum()

    unselected = np.flatnonzero(~selected)
    group_of = geo.group_ids  # (N,)

    for _ in range(k - k_init):
        best_kl = np.inf
        best_idx = -1
        for ui in range(len(unselected)):
            i = unselected[ui]
            g = group_of[i]
            wi = weights[i]
            # Tentative update
            Wg_new_g = Wg[g] + wi
            W_new = W_total + wi
            # Compute KL with only the changed group
            kl = 0.0
            for gg in range(G):
                if target_pi[gg] <= 0:
                    continue
                wg_gg = Wg_new_g if gg == g else Wg[gg]
                q_gg = (wg_gg + alpha) / (W_new + alpha * G)
                kl += target_pi[gg] * np.log(target_pi[gg] / q_gg)
            if kl < best_kl:
                best_kl = kl
                best_idx = i

        if best_idx < 0:
            break
        selected[best_idx] = True
        g_sel = group_of[best_idx]
        Wg[g_sel] += weights[best_idx]
        W_total += weights[best_idx]
        # Remove from unselected
        unselected = unselected[unselected != best_idx]

    return selected


def nsga2_optimize(
    *,
    computer: SpaceObjectiveComputer,
    projector: GeographicConstraintProjector,
    constraint_set: Optional[ProportionalityConstraintSet] = None,
    k: int,
    objectives: Sequence[str],
    pop_size: int,
    n_gen: int,
    crossover_prob: float,
    mutation_prob: float,
    mutation_swaps: int = 1,
    use_quota: bool,
    enforce_exact_k: bool = True,
    seed: int,
    verbose: Union[bool, int] = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 10,
    convergence_patience: int = 0,
    convergence_rtol: float = 1e-4,
    # Generation logging
    generation_log_dir: Optional[str] = None,
    snapshot_every: int = 25,
    # Adaptive tau
    adaptive_tau_check_every: int = 0,
    adaptive_tau_factor: float = 1.5,
    adaptive_tau_max: float = 0.20,
    adaptive_tau_min_nondom: int = 3,
    # Greedy-KL initialization
    greedy_kl_init: bool = False,
    auto_tau_multiplier: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Run NSGA-II and return the final non-dominated set (front 0).

    Parameters
    ----------
    computer : SpaceObjectiveComputer
        Objective function computer for the optimization space
    projector : GeographicConstraintProjector
        Projector for exact-k and municipality-share quota repair.
    constraint_set : Optional[ProportionalityConstraintSet]
        Optional weighted proportionality constraints (e.g., population-share).
    k : int
        Target coreset cardinality
    objectives : Sequence[str]
        Names of objectives to optimize (e.g., ['skl', 'mmd', 'sinkhorn'])
    pop_size : int
        Population size for NSGA-II
    n_gen : int
        Number of generations per epoch (resets on adaptive tau relaxation)
    crossover_prob : float
        Probability of crossover (typically 0.9)
    mutation_prob : float
        Probability of mutation per offspring (typically 0.1)
    use_quota : bool
        If True, enforce geographic quota constraints (state proportionality)
    enforce_exact_k : bool
        If True, enforce exact-k cardinality via repair
    seed : int
        Random seed for reproducibility
    verbose : Union[bool, int]
        Verbosity level (0=silent, 1=basic, 2=detailed, 3=full)
    checkpoint_path : str, optional
        Path for power-loss recovery checkpoints.
    checkpoint_every : int
        Checkpoint save frequency (default: 10).
    convergence_patience : int
        Early stopping patience (0=disabled). Stops epoch if no improvement.
    convergence_rtol : float
        Relative tolerance for improvement detection (default: 1e-4).
    generation_log_dir : str, optional
        Directory to save per-generation diagnostics. If None, no logging.
    snapshot_every : int
        Save full population snapshot every N global generations (default: 25).
    adaptive_tau_check_every : int
        Check Pareto front degeneracy every N local generations (0=disabled).
        If degenerate, relax tau and restart the generation counter.
    adaptive_tau_factor : float
        Multiply tau by this factor on each relaxation (default: 1.5).
    adaptive_tau_max : float
        Maximum allowed tau (default: 0.20).
    adaptive_tau_min_nondom : int
        Front is degenerate if < this many non-dominated points (default: 3).

    Returns
    -------
    X_pareto : np.ndarray
        Boolean masks for the final non-dominated set, shape (n_pareto, N)
    F_pareto : np.ndarray
        Objective values for the final non-dominated set, shape (n_pareto, M)
    stats : Dict[str, np.ndarray]
        Dictionary containing repair stats, history, and generation log info.
    """
    import sys as _sys

    rng = np.random.default_rng(int(seed))

    N = computer.X.shape[0]
    objectives = tuple([str(o).lower() for o in objectives])
    n_obj = len(objectives)

    # Initialize verbose logger
    vlogger = NSGA2VerboseLogger(
        verbose=verbose,
        objectives=objectives,
        n_gen=n_gen,
        pop_size=pop_size,
        k=k,
        use_quota=use_quota,
        enforce_exact_k=enforce_exact_k,
    )
    vlogger.log_header()

    # Precompute group index lists for quota swap mutation
    group_to_indices: List[np.ndarray] = []
    if use_quota:
        gid = projector.geo.group_ids
        G = projector.geo.G
        group_to_indices = [np.flatnonzero(gid == g) for g in range(G)]

    # Track repair activity (manuscript R6 diagnostics)
    repair_needed: List[bool] = []
    repair_magnitude: List[int] = []

    # ── Generation logging setup ──────────────────────────────────────
    _do_gen_log = generation_log_dir is not None
    if _do_gen_log:
        _os.makedirs(generation_log_dir, exist_ok=True)
    # Pre-allocate history arrays (worst case: 10 epochs × n_gen)
    _MAX_GENS = 10 * int(n_gen)
    _hist = {}
    if _do_gen_log:
        _hist["f_min"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["f_max"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["f_mean"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["f_std"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["cv_min"] = np.full(_MAX_GENS, np.nan)
        _hist["cv_mean"] = np.full(_MAX_GENS, np.nan)
        _hist["cv_max"] = np.full(_MAX_GENS, np.nan)
        _hist["n_feasible"] = np.full(_MAX_GENS, np.nan)
        _hist["front0_size"] = np.full(_MAX_GENS, np.nan)
        _hist["front0_f_min"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["front0_f_max"] = np.full((_MAX_GENS, n_obj), np.nan)
        _hist["front0_corr"] = np.full(_MAX_GENS, np.nan)
        _hist["n_unique"] = np.full(_MAX_GENS, np.nan)
        _hist["repair_rate"] = np.full(_MAX_GENS, np.nan)
        _hist["mean_repair_mag"] = np.full(_MAX_GENS, np.nan)
        _hist["elapsed_s"] = np.full(_MAX_GENS, np.nan)
        _hist["tau_value"] = np.full(_MAX_GENS, np.nan)
        _hist["epoch"] = np.full(_MAX_GENS, np.nan)

    # ── Checkpoint resume ────────────────────────────────────────────
    start_gen = 0
    _ckpt = _load_checkpoint(checkpoint_path) if checkpoint_path else None
    if _ckpt is not None:
        ckpt_gen = int(_ckpt["gen"][0])
        ckpt_seed = int(_ckpt["seed"][0])
        ckpt_pop_X = _ckpt["pop_X"].astype(bool)
        ckpt_pop_F = _ckpt["pop_F"].astype(np.float64)
        ckpt_pop_CV = _ckpt["pop_CV"].astype(np.float64)

        if (ckpt_seed == seed
                and ckpt_pop_X.shape == (pop_size, N)
                and ckpt_pop_F.shape[1] == n_obj):
            pop_X = ckpt_pop_X
            pop_F = ckpt_pop_F
            pop_CV = ckpt_pop_CV
            start_gen = ckpt_gen + 1

            if "rng_state" in _ckpt:
                import pickle
                rng_state = pickle.loads(_ckpt["rng_state"].tobytes())
                rng.bit_generator.state = rng_state
            if "repair_needed" in _ckpt:
                repair_needed = _ckpt["repair_needed"].tolist()
            if "repair_magnitude" in _ckpt:
                repair_magnitude = _ckpt["repair_magnitude"].tolist()

            print(
                f"[NSGA-II] Resumed from checkpoint at generation {ckpt_gen} "
                f"(skipping to gen {start_gen}/{n_gen})",
                flush=True,
            )
            _obj_cache: dict = {}
            for i in range(pop_X.shape[0]):
                _obj_cache[pop_X[i].tobytes()] = pop_F[i]
            timer.checkpoint("Resumed from checkpoint", gen=ckpt_gen)
        else:
            print("[NSGA-II] Checkpoint incompatible — starting fresh", flush=True)
            _ckpt = None

    if _ckpt is None:
        # ── Initialization (fresh start) ──────────────────────────────
        timer.checkpoint("NSGA-II starting initialization", pop_size=pop_size, N=N, k=k)

        with timer.section("NSGA-II_population_init"):
            pop_X = np.zeros((pop_size, N), dtype=bool)

            # Optionally seed initial individuals with greedy-KL-optimal
            # mask + random perturbations for diversity
            _greedy_start = 0
            if greedy_kl_init and constraint_set is not None:
                try:
                    gk_mask = _greedy_kl_init(constraint_set, k, rng)
                    # Number of greedy seeds: half the population (at least 1)
                    n_greedy = max(1, pop_size // 2)
                    gk_base = _repair_mask(projector, constraint_set, rng,
                                           gk_mask, k, use_quota, enforce_exact_k)
                    pop_X[0] = gk_base
                    # Create perturbed variants by random same-group swaps
                    for j in range(1, n_greedy):
                        variant = gk_base.copy()
                        # Apply a few random swaps (more swaps for later seeds)
                        n_swaps = min(1 + j, k // 5)
                        for _ in range(n_swaps):
                            variant = _swap_mutation_quota(
                                rng, variant,
                                projector.geo.group_ids, group_to_indices,
                            )
                        pop_X[j] = _repair_mask(projector, constraint_set, rng,
                                                variant, k, use_quota, enforce_exact_k)
                    _greedy_start = n_greedy
                    if verbose:
                        cv0 = constraint_set.total_violation(pop_X[0])
                        print(f"[NSGA-II] Greedy-KL seeds: {n_greedy}/{pop_size} "
                              f"individuals (base CV={cv0:.6f})", flush=True)
                except Exception as e:
                    logger.warning("Greedy-KL init failed, falling back: %s", e)
                    _greedy_start = 0

            for i in range(_greedy_start, pop_size):
                if use_quota:
                    mask = np.zeros(N, dtype=bool)
                    target = projector.target_counts(int(k))
                    for g in range(projector.geo.G):
                        c_g = int(target[g])
                        if c_g <= 0:
                            continue
                        idx_g = group_to_indices[g]
                        if idx_g.size == 0:
                            continue
                        picks = rng.choice(idx_g, size=c_g, replace=False)
                        mask[picks] = True
                else:
                    mask = np.zeros(N, dtype=bool)
                    sel = rng.choice(N, size=int(k), replace=False)
                    mask[sel] = True
                pop_X[i] = _repair_mask(projector, constraint_set, rng, mask, k, use_quota, enforce_exact_k)

        # ── Auto-tau: set tau from greedy-KL floor ──────────────────
        if greedy_kl_init and constraint_set is not None and _greedy_start > 0:
            # The greedy seed (pop_X[0]) has near-minimal KL.
            # Use its KL value as the floor and set tau = floor * multiplier.
            greedy_kl_floor = 0.0
            for c in constraint_set.constraints:
                greedy_kl_floor = max(greedy_kl_floor,
                                     c.value(pop_X[0], constraint_set.geo))
            auto_tau = greedy_kl_floor * auto_tau_multiplier
            old_tau = constraint_set.max_tau()
            # Only override if auto_tau is larger (more permissive) than current
            if auto_tau > old_tau:
                import dataclasses as _dc
                constraint_set.constraints = [
                    _dc.replace(c, tau=auto_tau)
                    for c in constraint_set.constraints
                ]
                print(
                    f"[NSGA-II] Auto-tau: greedy KL floor = {greedy_kl_floor:.6f}, "
                    f"setting tau = {auto_tau:.6f} "
                    f"({auto_tau_multiplier:.1f}× floor, was {old_tau:.6f})",
                    flush=True,
                )

        timer.checkpoint("Population initialized", pop_size=pop_size)
        _obj_cache = {}

        with timer.section("NSGA-II_initial_evaluation"):
            pop_F = _evaluate_population(pop_X, computer, objectives, cache=_obj_cache)
        pop_CV = np.zeros(pop_X.shape[0], dtype=np.float64)
        if constraint_set is not None:
            pop_CV = np.array([constraint_set.total_violation(m) for m in pop_X], dtype=np.float64)

    timer.checkpoint("Initial population evaluated",
                     pop_F_shape=pop_F.shape,
                     has_nan=bool(np.any(np.isnan(pop_F))),
                     has_inf=bool(np.any(np.isinf(pop_F))),
                     f_min=pop_F.min(axis=0).tolist(),
                     f_max=pop_F.max(axis=0).tolist())
    vlogger.log_initialization(pop_F)

    # =====================================================================
    # Evolution loop — outer epoch loop + inner generation loop
    # =====================================================================
    timer.checkpoint("Starting evolution loop", n_gen=n_gen)
    loop_start_time = time.perf_counter()
    global_gen = 0       # total generation counter across all epochs
    epoch = 0
    _adaptive_tau = adaptive_tau_check_every > 0 and constraint_set is not None
    _prev_n_feasible = int((pop_CV <= 0.0).sum())  # baseline from initial pop

    # Keep track of last offspring/combined for final dump
    _last_offspring_X = None
    _last_offspring_F = None
    _last_offspring_CV = None
    _last_combined_F = None
    _last_combined_CV = None
    _last_next_idx = None

    while True:
        tau_relaxed_this_epoch = False  # kept for generation-log compatibility

        # Reset early stopping state per epoch
        _early_stop = convergence_patience > 0
        _best_f = pop_F.min(axis=0).copy() if _early_stop else None
        _stale_gens = 0
        early_stopped = False

        current_tau = constraint_set.max_tau() if constraint_set is not None else 0.0

        print(f"\n[NSGA-II] === Epoch {epoch} | tau={current_tau:.4f} | "
              f"n_gen={n_gen} | global_gen={global_gen} ===", flush=True)

        for local_gen in range(int(n_gen)):
            vlogger.start_generation()

            # Compute fronts / rank / crowding
            fronts, rank = _constraint_dominated_sort(pop_F, pop_CV)
            crowd = _assign_crowding(pop_F, fronts)
            front0_size = len(fronts[0])

            # Print progress
            elapsed = time.perf_counter() - loop_start_time
            f_min_vals = pop_F.min(axis=0)
            f_min_str = ", ".join(f"{v:.4f}" for v in f_min_vals)
            n_feas = int(np.sum(pop_CV <= 0.0)) if constraint_set is not None else pop_size
            print(f"[GEN {global_gen:4d} (ep{epoch}:{local_gen:3d}/{n_gen})] "
                  f"elapsed={elapsed:6.1f}s | "
                  f"front0={front0_size:3d} | "
                  f"feasible={n_feas:3d} | "
                  f"tau={current_tau:.4f} | "
                  f"f_min=[{f_min_str}]",
                  file=_sys.stderr, flush=True)

            # ── Generation logging ────────────────────────────────────
            if _do_gen_log and global_gen < _MAX_GENS:
                g = global_gen
                _hist["f_min"][g] = pop_F.min(axis=0)
                _hist["f_max"][g] = pop_F.max(axis=0)
                _hist["f_mean"][g] = pop_F.mean(axis=0)
                _hist["f_std"][g] = pop_F.std(axis=0)
                _hist["cv_min"][g] = pop_CV.min()
                _hist["cv_mean"][g] = pop_CV.mean()
                _hist["cv_max"][g] = pop_CV.max()
                _hist["n_feasible"][g] = n_feas
                _hist["front0_size"][g] = front0_size
                f0_F = pop_F[fronts[0]]
                _hist["front0_f_min"][g] = f0_F.min(axis=0)
                _hist["front0_f_max"][g] = f0_F.max(axis=0)
                if n_obj == 2 and front0_size >= 3:
                    _hist["front0_corr"][g] = np.corrcoef(f0_F[:, 0], f0_F[:, 1])[0, 1]
                _hist["n_unique"][g] = len(set(pop_X[i].tobytes() for i in range(pop_size)))
                _hist["elapsed_s"][g] = elapsed
                _hist["tau_value"][g] = current_tau
                _hist["epoch"][g] = epoch

            # ── Snapshot save ─────────────────────────────────────────
            if _do_gen_log and global_gen % snapshot_every == 0:
                snap_path = _os.path.join(generation_log_dir, f"snapshot_gen{global_gen:04d}.npz")
                np.savez_compressed(snap_path,
                    pop_X=pop_X, pop_F=pop_F, pop_CV=pop_CV,
                    rank=rank, crowd=crowd)

            # Track repairs for this generation
            gen_repairs_needed: List[bool] = []
            gen_repair_magnitudes: List[int] = []

            # Parent selection
            parents_idx = _tournament_select(rng, rank, crowd, n_select=pop_size)

            # Variation
            offspring_X = np.zeros_like(pop_X)
            for i in range(0, pop_size, 2):
                p1 = pop_X[parents_idx[i]]
                p2 = pop_X[parents_idx[(i + 1) % pop_size]]
                c1, c2 = _uniform_crossover(rng, p1, p2, crossover_prob)

                if rng.random() < mutation_prob:
                    if use_quota:
                        for _ in range(mutation_swaps):
                            c1 = _swap_mutation_quota(rng, c1, projector.geo.group_ids, group_to_indices)
                    else:
                        c1 = _bitflip_mutation_exact_k(rng, c1, flip_prob=1.0 / max(1, N))
                if rng.random() < mutation_prob:
                    if use_quota:
                        for _ in range(mutation_swaps):
                            c2 = _swap_mutation_quota(rng, c2, projector.geo.group_ids, group_to_indices)
                    else:
                        c2 = _bitflip_mutation_exact_k(rng, c2, flip_prob=1.0 / max(1, N))

                c1_pre = c1
                c1 = _repair_mask(projector, constraint_set, rng, c1, k, use_quota, enforce_exact_k)
                needed = bool(np.any(c1_pre != c1))
                mag = int(np.sum(c1_pre != c1))
                repair_needed.append(needed)
                repair_magnitude.append(mag)
                gen_repairs_needed.append(needed)
                gen_repair_magnitudes.append(mag)

                c2_pre = c2
                c2 = _repair_mask(projector, constraint_set, rng, c2, k, use_quota, enforce_exact_k)
                needed = bool(np.any(c2_pre != c2))
                mag = int(np.sum(c2_pre != c2))
                repair_needed.append(needed)
                repair_magnitude.append(mag)
                gen_repairs_needed.append(needed)
                gen_repair_magnitudes.append(mag)

                offspring_X[i] = c1
                if i + 1 < pop_size:
                    offspring_X[i + 1] = c2

            offspring_F = _evaluate_population(offspring_X, computer, objectives, cache=_obj_cache)
            offspring_CV = np.zeros(offspring_X.shape[0], dtype=np.float64)
            if constraint_set is not None:
                offspring_CV = np.array([constraint_set.total_violation(m) for m in offspring_X], dtype=np.float64)

            # Combine and select next generation
            combined_X = np.vstack([pop_X, offspring_X])
            combined_F = np.vstack([pop_F, offspring_F])
            combined_CV = np.concatenate([pop_CV, offspring_CV], axis=0)

            fronts, rank = _constraint_dominated_sort(combined_F, combined_CV)
            crowd = _assign_crowding(combined_F, fronts)

            next_idx_list: List[int] = []
            for front in fronts:
                if len(next_idx_list) + front.size <= pop_size:
                    next_idx_list.extend(front.tolist())
                else:
                    remaining = pop_size - len(next_idx_list)
                    cd = crowd[front]
                    order = np.argsort(-cd)
                    next_idx_list.extend(front[order[:remaining]].tolist())
                    break

            next_idx = np.asarray(next_idx_list, dtype=int)
            pop_X = combined_X[next_idx]
            pop_F = combined_F[next_idx]
            pop_CV = combined_CV[next_idx]

            # Save references for final dump
            _last_offspring_X = offspring_X
            _last_offspring_F = offspring_F
            _last_offspring_CV = offspring_CV
            _last_combined_F = combined_F
            _last_combined_CV = combined_CV
            _last_next_idx = next_idx

            # Log generation repair stats
            if _do_gen_log and global_gen < _MAX_GENS:
                if gen_repairs_needed:
                    _hist["repair_rate"][global_gen] = sum(gen_repairs_needed) / len(gen_repairs_needed)
                    mags = [m for m in gen_repair_magnitudes if m > 0]
                    _hist["mean_repair_mag"][global_gen] = np.mean(mags) if mags else 0.0

            # Verbose logging
            fronts_log, _ = _constraint_dominated_sort(pop_F, pop_CV)
            crowd_log = _assign_crowding(pop_F, fronts_log)
            vlogger.log_generation(
                gen=global_gen,
                pop_F=pop_F,
                pop_X=pop_X,
                fronts=fronts_log,
                crowd=crowd_log,
                gen_repairs_needed=gen_repairs_needed,
                gen_repair_magnitudes=gen_repair_magnitudes,
            )

            # ── Early stopping check ──────────────────────────────────
            if _early_stop:
                curr_f = pop_F.min(axis=0)
                improved = np.any(curr_f < _best_f * (1.0 - convergence_rtol))
                if improved:
                    _best_f = np.minimum(_best_f, curr_f)
                    _stale_gens = 0
                else:
                    _stale_gens += 1
                if _stale_gens >= convergence_patience:
                    print(
                        f"[NSGA-II] Early stopping at epoch {epoch}, "
                        f"local_gen {local_gen} (global {global_gen}): "
                        f"no improvement for {convergence_patience} gens.",
                        flush=True,
                    )
                    early_stopped = True
                    global_gen += 1
                    break

            # ── Adaptive tau check ────────────────────────────────────
            # Every N gens, check if feasible count is increasing.
            # If stagnant or decreasing, bump tau by factor (e.g. 10%).
            # If increasing, leave tau alone.
            if (_adaptive_tau
                    and (local_gen + 1) % adaptive_tau_check_every == 0):
                n_feasible = int((pop_CV <= 0.0).sum())
                if n_feasible > _prev_n_feasible or n_feasible >= pop_size:
                    # Feasible count is growing (or all feasible) — leave tau
                    _prev_n_feasible = n_feasible
                else:
                    # Stagnant or decreasing — bump tau
                    old_tau = constraint_set.max_tau()
                    if old_tau < adaptive_tau_max:
                        constraint_set.relax_tau(adaptive_tau_factor)
                        new_tau = constraint_set.max_tau()
                        if new_tau > adaptive_tau_max:
                            constraint_set.relax_tau(adaptive_tau_max / new_tau)
                            new_tau = constraint_set.max_tau()
                        pop_CV = np.array(
                            [constraint_set.total_violation(m) for m in pop_X],
                            dtype=np.float64,
                        )
                        current_tau = new_tau
                        n_now_feasible = int((pop_CV <= 0.0).sum())
                        print(
                            f"[NSGA-II] Adaptive tau: {old_tau:.4f} → {new_tau:.4f} "
                            f"at gen {global_gen} (feasible stagnant: "
                            f"{_prev_n_feasible} → {n_feasible}). "
                            f"Now feasible: {n_now_feasible}/{pop_size}.",
                            flush=True,
                        )
                        _prev_n_feasible = n_now_feasible
                    else:
                        print(
                            f"[NSGA-II] Adaptive tau: max ({adaptive_tau_max:.4f}) "
                            f"reached, feasible={n_feasible}/{pop_size}.",
                            flush=True,
                        )
                        _prev_n_feasible = n_feasible

            # ── Periodic checkpoint ───────────────────────────────────
            if checkpoint_path and (global_gen + 1) % checkpoint_every == 0:
                _save_checkpoint(
                    checkpoint_path,
                    pop_X=pop_X,
                    pop_F=pop_F,
                    pop_CV=pop_CV,
                    gen=global_gen,
                    seed=seed,
                    rng=rng,
                    repair_needed=repair_needed,
                    repair_magnitude=repair_magnitude,
                )

            global_gen += 1
        # ── end inner loop ────────────────────────────────────────────

        if early_stopped or not tau_relaxed_this_epoch:
            break  # done: either converged or completed full epoch
        epoch += 1

    # ── Save generation history and final dump ────────────────────────
    if _do_gen_log:
        # Trim history arrays to actual length
        actual_gens = global_gen
        save_dict = {}
        for key, arr in _hist.items():
            save_dict[key] = arr[:actual_gens]
        save_dict["objectives"] = np.array(objectives)
        save_dict["n_epochs"] = np.array([epoch + 1])
        save_dict["total_gens"] = np.array([actual_gens])
        np.savez_compressed(
            _os.path.join(generation_log_dir, "history.npz"),
            **save_dict,
        )

        # Final snapshot (always)
        snap_path = _os.path.join(generation_log_dir, f"snapshot_gen{global_gen - 1:04d}.npz")
        fronts_final, rank_final = _constraint_dominated_sort(pop_F, pop_CV)
        crowd_final = _assign_crowding(pop_F, fronts_final)
        np.savez_compressed(snap_path,
            pop_X=pop_X, pop_F=pop_F, pop_CV=pop_CV,
            rank=rank_final, crowd=crowd_final)

        # Final dump — save EVERYTHING
        dump_data = {
            "pop_X": pop_X,
            "pop_F": pop_F,
            "pop_CV": pop_CV,
            "rank": rank_final,
            "crowd": crowd_final,
            "fronts_sizes": np.array([len(f) for f in fronts_final]),
            "front0_idx": fronts_final[0],
            "obj_cache_size": np.array([len(_obj_cache)]),
            "repair_needed_all": np.asarray(repair_needed, dtype=bool),
            "repair_magnitude_all": np.asarray(repair_magnitude, dtype=np.int32),
            "final_tau": np.array([current_tau]),
            "total_epochs": np.array([epoch + 1]),
            "total_gens": np.array([global_gen]),
        }
        if _last_offspring_X is not None:
            dump_data["offspring_X"] = _last_offspring_X
            dump_data["offspring_F"] = _last_offspring_F
            dump_data["offspring_CV"] = _last_offspring_CV
        if _last_combined_F is not None:
            dump_data["combined_F"] = _last_combined_F
            dump_data["combined_CV"] = _last_combined_CV
        if _last_next_idx is not None:
            dump_data["next_idx"] = _last_next_idx
        np.savez_compressed(
            _os.path.join(generation_log_dir, "final_dump.npz"),
            **dump_data,
        )
        print(f"[NSGA-II] Generation logs saved to {generation_log_dir}/", flush=True)

    # ── Clean up checkpoint on successful completion ──────────────────
    if checkpoint_path:
        if _os.path.isfile(checkpoint_path):
            _os.remove(checkpoint_path)
            print(
                f"[NSGA-II] Evolution complete — checkpoint removed "
                f"({checkpoint_path})",
                flush=True,
            )
        for _tmp in [
            checkpoint_path + ".tmp",
            checkpoint_path[:-4] + ".tmp.npz" if checkpoint_path.endswith(".npz") else checkpoint_path + ".tmp",
            checkpoint_path + ".tmp.npz",
        ]:
            if _os.path.isfile(_tmp):
                _os.remove(_tmp)

    # ── Return final non-dominated set ────────────────────────────────
    fronts, _ = _constraint_dominated_sort(pop_F, pop_CV)
    f0 = fronts[0]
    if constraint_set is not None:
        feas = pop_CV <= 0.0
        if feas.any():
            idx_f = np.flatnonzero(feas)
            fronts_f, _ = fast_non_dominated_sort(pop_F[idx_f])
            f0 = idx_f[fronts_f[0]]
        else:
            order = np.argsort(pop_CV)
            f0 = order[: min(50, len(order))]

    X_pareto = pop_X[f0]
    F_pareto = pop_F[f0]

    _validate_pareto_front(F_pareto, constraint_set is not None, pop_CV[f0])
    vlogger.log_completion(X_pareto, F_pareto)

    repair_stats = {
        "repair_needed": np.asarray(repair_needed, dtype=bool),
        "repair_magnitude": np.asarray(repair_magnitude, dtype=np.int32),
    }
    if vlogger.verbose >= 1:
        repair_stats["history"] = vlogger.get_history_summary()

    return X_pareto, F_pareto, repair_stats
