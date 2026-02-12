"""
NSGA-II genetic operators, constraint handling, and population evaluation.

Extracted from ``nsga2_internal.py`` to reduce module size.
All names are re-exported from ``nsga2_internal`` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geo.projector import GeographicConstraintProjector
from ..constraints.proportionality import ProportionalityConstraintSet
from ..objectives.computer import SpaceObjectiveComputer
from .selection import crowding_distance


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance for minimization."""
    return np.all(a <= b) and np.any(a < b)


def _assign_crowding(F: np.ndarray, fronts: List[np.ndarray]) -> np.ndarray:
    """Compute crowding distance for each point, per-front."""
    n = F.shape[0]
    cd = np.zeros(n, dtype=np.float64)

    for front in fronts:
        if front.size == 0:
            continue
        cd_front = crowding_distance(F[front])
        cd[front] = cd_front
    return cd


def _tournament_select(
    rng: np.random.Generator,
    rank: np.ndarray,
    crowd: np.ndarray,
    n_select: int,
) -> np.ndarray:
    """Binary tournament selection by (rank asc, crowd desc)."""
    n = rank.shape[0]
    picks = np.empty(n_select, dtype=int)

    for i in range(n_select):
        a, b = rng.integers(0, n, size=2, endpoint=False)
        if rank[a] < rank[b]:
            picks[i] = a
        elif rank[b] < rank[a]:
            picks[i] = b
        else:
            # higher crowding wins; break ties randomly
            if crowd[a] > crowd[b]:
                picks[i] = a
            elif crowd[b] > crowd[a]:
                picks[i] = b
            else:
                picks[i] = a if rng.random() < 0.5 else b

    return picks


def _uniform_crossover(
    rng: np.random.Generator,
    p1: np.ndarray,
    p2: np.ndarray,
    prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover on boolean masks."""
    if rng.random() >= prob:
        return p1.copy(), p2.copy()

    # take_from_p1: when True, child1 takes gene from p1 (and child2 from p2)
    # when False, child1 takes gene from p2 (and child2 from p1)
    take_from_p1 = rng.random(p1.shape[0]) < 0.5
    c1 = np.where(take_from_p1, p1, p2)
    c2 = np.where(take_from_p1, p2, p1)
    return c1, c2


def _bitflip_mutation_exact_k(
    rng: np.random.Generator,
    mask: np.ndarray,
    *,
    flip_prob: float,
) -> np.ndarray:
    """Independent bit flips with probability 1/N per indicator (manuscript)."""
    mask = np.asarray(mask, dtype=bool).copy()
    if flip_prob <= 0:
        return mask
    flips = rng.random(mask.shape[0]) < flip_prob
    if not np.any(flips):
        return mask
    mask[flips] = ~mask[flips]
    return mask


def _swap_mutation_quota(
    rng: np.random.Generator,
    mask: np.ndarray,
    group_ids: np.ndarray,
    group_to_indices: List[np.ndarray],
    max_tries: int = 50,
) -> np.ndarray:
    """
    Within-group swap mutation: choose a group with both selected and unselected
    indices, then swap one in/out. Preserves quota counts within that group.
    """
    G = len(group_to_indices)
    mask = mask.copy()

    for _ in range(max_tries):
        g = int(rng.integers(0, G))
        idx_g = group_to_indices[g]
        if idx_g.size == 0:
            continue

        sel_g = idx_g[mask[idx_g]]
        unsel_g = idx_g[~mask[idx_g]]
        if sel_g.size == 0 or unsel_g.size == 0:
            continue

        i_del = int(rng.choice(sel_g))
        i_add = int(rng.choice(unsel_g))
        mask[i_del] = False
        mask[i_add] = True
        return mask

    return mask


def _repair_mask(
    projector: GeographicConstraintProjector,
    constraint_set: Optional[ProportionalityConstraintSet],
    rng: np.random.Generator,
    mask: np.ndarray,
    k: int,
    use_quota: bool,
    enforce_exact_k: bool,
) -> np.ndarray:
    """Repair a mask under the selected constraint-handling regime.

    Regimes (use_quota, enforce_exact_k):
      - (True,  True): project to quota counts for the configured k
      - (True,  False): project to quota counts for the *current* cardinality
                       (preserves |S| but enforces state proportionality)
      - (False, True): project to exact-k only (no geographic constraints)
      - (False, False): no repair
    """
    mask = np.asarray(mask, dtype=bool).copy()
    if use_quota:
        k_eff = int(k) if enforce_exact_k else int(mask.sum())
        mask = projector.project_to_quota_mask(mask, k_eff, rng=rng)
    elif enforce_exact_k:
        mask = projector.project_to_exact_k_mask(mask, int(k), rng=rng)
    # Weighted proportionality repair (population-share / joint)
    if constraint_set is not None:
        mask = constraint_set.repair(mask, rng=rng)
    return mask


def _constraint_dominated_sort(
    F: np.ndarray,
    CV: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Constraint-domination sorting.

    Feasible solutions (CV==0) always dominate infeasible. Among feasible,
    use Pareto non-dominated sorting on F. Among infeasible, rank by CV.
    """
    # Import here to avoid circular dependency: fast_non_dominated_sort
    # is defined in the facade module (nsga2_internal) and uses _dominates
    # from this module. We import lazily.
    from .nsga2_internal import fast_non_dominated_sort

    CV = np.asarray(CV, dtype=np.float64)
    feasible = CV <= 0.0
    fronts: List[np.ndarray] = []
    rank = np.full(F.shape[0], fill_value=10**9, dtype=int)

    if feasible.any():
        idx_f = np.flatnonzero(feasible)
        fronts_f, rank_f = fast_non_dominated_sort(F[idx_f])
        for fr in fronts_f:
            fronts.append(idx_f[fr])
        rank[idx_f] = rank_f

    idx_inf = np.flatnonzero(~feasible)
    if idx_inf.size:
        # Single front for infeasible, sorted by violation
        order = idx_inf[np.argsort(CV[idx_inf])]
        fronts.append(order)
        # Assign ranks after all feasible ranks
        base = int(rank[feasible].max() + 1) if feasible.any() else 0
        for i, j in enumerate(order):
            rank[j] = base + i

    return fronts, rank


def _evaluate_population(
    pop_X: np.ndarray,
    computer: SpaceObjectiveComputer,
    objectives: Sequence[str],
) -> np.ndarray:
    """Evaluate objective matrix F for a population of masks."""
    P = pop_X.shape[0]
    M = len(objectives)
    F = np.zeros((P, M), dtype=np.float64)

    for i in range(P):
        idx = np.flatnonzero(pop_X[i])
        for j, obj in enumerate(objectives):
            try:
                val = float(computer.compute_single(idx, obj))
                if not np.isfinite(val):
                    val = 1e18
                F[i, j] = val
            except Exception:
                F[i, j] = 1e18

    return F
