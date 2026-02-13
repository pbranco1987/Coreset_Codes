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
    use_quota: bool,
    enforce_exact_k: bool = True,
    seed: int,
    verbose: Union[bool, int] = False,
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
        Number of generations to evolve
    crossover_prob : float
        Probability of crossover (typically 0.9)
    mutation_prob : float
        Probability of mutation per offspring (typically 0.1)
    use_quota : bool
        If True, enforce geographic quota constraints (state proportionality)
    enforce_exact_k : bool
        If True, enforce exact-k cardinality via repair
        If False, allow subset size to drift (for ablations)
    seed : int
        Random seed for reproducibility
    verbose : Union[bool, int]
        Verbosity level:
        - 0 or False: Silent mode
        - 1 or True: Basic progress (every 10 generations)
        - 2: Detailed per-generation statistics
        - 3: Full diagnostics with diversity, timing, hypervolume

    Returns
    -------
    X_pareto : np.ndarray
        Boolean masks for the final non-dominated set, shape (n_pareto, N)
    F_pareto : np.ndarray
        Objective values for the final non-dominated set, shape (n_pareto, M)
    stats : Dict[str, np.ndarray]
        Dictionary containing:
        - 'repair_needed': Boolean array of repair flags
        - 'repair_magnitude': Integer array of bits changed per repair
        - 'history': Optimization history summary (if verbose >= 1)
    """
    rng = np.random.default_rng(int(seed))

    N = computer.X.shape[0]
    objectives = tuple([str(o).lower() for o in objectives])

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

    # -------------------------
    # Initialization
    # -------------------------
    timer.checkpoint("NSGA-II starting initialization", pop_size=pop_size, N=N, k=k)

    with timer.section("NSGA-II_population_init"):
        pop_X = np.zeros((pop_size, N), dtype=bool)
        for i in range(pop_size):
            if use_quota:
                # Quota-constrained initialization: sample within each group according
                # to the KL-optimal quota counts c*(k).
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

            # Repair as projection (safety)
            pop_X[i] = _repair_mask(projector, constraint_set, rng, mask, k, use_quota, enforce_exact_k)

    timer.checkpoint("Population initialized", pop_size=pop_size)

    # Objective cache: avoids re-evaluating masks that survive across
    # generations.  Keyed by the raw bytes of the boolean mask row.
    _obj_cache: dict = {}

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

    # -------------------------
    # Evolution loop
    # -------------------------
    timer.checkpoint("Starting evolution loop", n_gen=n_gen)
    gen_start_time = time.perf_counter()

    # Import for generation-level progress output
    import sys as _sys

    # Get number of objectives for flexible printing
    n_objectives = len(objectives)

    for gen in range(int(n_gen)):
        vlogger.start_generation()

        # Compute fronts / rank / crowding ONCE per generation (was 3×).
        fronts, rank = _constraint_dominated_sort(pop_F, pop_CV)
        crowd = _assign_crowding(pop_F, fronts)

        # Print generation progress for EVERY generation (always visible)
        elapsed = time.perf_counter() - gen_start_time
        avg_time_per_gen = elapsed / max(1, gen) if gen > 0 else 0
        eta = avg_time_per_gen * (n_gen - gen) if gen > 0 else 0
        front0_size = len(fronts[0])

        # Compute current best objectives (handle any number)
        f_min = pop_F.min(axis=0)
        f_min_str = ", ".join(f"{v:.4f}" for v in f_min)

        print(f"[GEN {gen:4d}/{n_gen}] "
              f"elapsed={elapsed:6.1f}s | "
              f"ETA={eta:6.1f}s | "
              f"front0={front0_size:3d} | "
              f"f_min=[{f_min_str}]",
              file=_sys.stderr, flush=True)

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

            # Mutation (applied per-offspring with probability mutation_prob)
            if rng.random() < mutation_prob:
                if use_quota:
                    c1 = _swap_mutation_quota(rng, c1, projector.geo.group_ids, group_to_indices)
                else:
                    c1 = _bitflip_mutation_exact_k(rng, c1, flip_prob=1.0 / max(1, N))
            if rng.random() < mutation_prob:
                if use_quota:
                    c2 = _swap_mutation_quota(rng, c2, projector.geo.group_ids, group_to_indices)
                else:
                    c2 = _bitflip_mutation_exact_k(rng, c2, flip_prob=1.0 / max(1, N))

            # Repair projection + activity tracking
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

        next_idx: List[int] = []
        for front in fronts:
            if len(next_idx) + front.size <= pop_size:
                next_idx.extend(front.tolist())
            else:
                # Fill remaining slots by crowding distance
                remaining = pop_size - len(next_idx)
                cd = crowd[front]
                order = np.argsort(-cd)  # descending
                next_idx.extend(front[order[:remaining]].tolist())
                break

        next_idx = np.asarray(next_idx, dtype=int)
        pop_X = combined_X[next_idx]
        pop_F = combined_F[next_idx]
        pop_CV = combined_CV[next_idx]

        # Log generation — reuse the fronts/crowd that will be computed
        # at the start of the next iteration; for the last generation we
        # compute them once here (cheap compared to objective evaluation).
        fronts_log, _ = _constraint_dominated_sort(pop_F, pop_CV)
        crowd_log = _assign_crowding(pop_F, fronts_log)
        vlogger.log_generation(
            gen=gen,
            pop_F=pop_F,
            pop_X=pop_X,
            fronts=fronts_log,
            crowd=crowd_log,
            gen_repairs_needed=gen_repairs_needed,
            gen_repair_magnitudes=gen_repair_magnitudes,
        )

    # Return final non-dominated set
    fronts, _ = _constraint_dominated_sort(pop_F, pop_CV)
    # Prefer the first feasible front if available
    f0 = fronts[0]
    if constraint_set is not None:
        feas = pop_CV <= 0.0
        if feas.any():
            idx_f = np.flatnonzero(feas)
            fronts_f, _ = fast_non_dominated_sort(pop_F[idx_f])
            f0 = idx_f[fronts_f[0]]
        else:
            # No feasible solutions; return the least-violating individuals
            order = np.argsort(pop_CV)
            f0 = order[: min(50, len(order))]

    X_pareto = pop_X[f0]
    F_pareto = pop_F[f0]

    vlogger.log_completion(X_pareto, F_pareto)

    repair_stats = {
        "repair_needed": np.asarray(repair_needed, dtype=bool),
        "repair_magnitude": np.asarray(repair_magnitude, dtype=np.int32),
    }

    # Add history if verbose logging was enabled
    if vlogger.verbose >= 1:
        repair_stats["history"] = vlogger.get_history_summary()

    return X_pareto, F_pareto, repair_stats
