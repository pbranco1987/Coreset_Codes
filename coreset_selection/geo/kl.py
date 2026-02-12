"""
KL divergence utilities for geographic constraints.

Contains:
- kl_pi_hat_from_counts: Compute KL divergence from empirical counts
- kl_optimal_integer_counts_bounded: Find KL-optimal integer allocation (heap-based)
- min_achievable_geo_kl_bounded: Compute minimum achievable geographic KL
- compute_quota_path: Incremental quota path c*(k) and KL_min(k) over a k-grid
"""

from __future__ import annotations

import heapq
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def kl_pi_hat_from_counts(
    pi: np.ndarray,
    counts: np.ndarray,
    k: int,
    alpha: float = 1.0,
) -> float:
    """
    Compute KL divergence D_KL(π || π̂) with Dirichlet smoothing.

    Parameters
    ----------
    pi : np.ndarray
        Target distribution (population proportions), shape (G,)
    counts : np.ndarray
        Empirical counts per group, shape (G,)
    k : int
        Total count (sum of counts)
    alpha : float
        Dirichlet smoothing parameter

    Returns
    -------
    float
        KL divergence
    """
    pi = np.asarray(pi, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    G = len(pi)

    # Smoothed empirical distribution
    pi_hat = (counts + alpha) / (k + alpha * G)

    # KL divergence: D_KL(π || π̂) = Σ π_g * log(π_g / π̂_g)
    kl = 0.0
    for g in range(G):
        if pi[g] > 0:
            kl += pi[g] * np.log(pi[g] / (pi_hat[g] + 1e-30))

    return float(kl)


# ---------------------------------------------------------------------------
# Marginal-gain helpers (Theorem 1(ii))
# ---------------------------------------------------------------------------

def _marginal_gain(pi_g: float, t: int, alpha: float) -> float:
    r"""Marginal gain Δ_g(t) = π_g · [log(t + α + 1) − log(t + α)].

    This is the discrete derivative of f_g(c) = π_g · log(c + α) and is
    guaranteed non-increasing in *t* for fixed π_g (concavity), which is
    the key property that makes the greedy / lazy-heap approach optimal
    (Theorem 1(ii)).
    """
    if pi_g <= 0.0:
        return -np.inf
    return pi_g * (np.log(t + alpha + 1) - np.log(t + alpha))


def kl_optimal_integer_counts_bounded(
    pi: np.ndarray,
    k: int,
    alpha: float,
    *,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    r"""Find integer counts that minimize KL divergence within bounds.

    Implements the **lazy-heap** variant of Algorithm 1 (KL-Guided Quota
    Construction).  The heap stores ``(-gain, group)`` pairs and lazily
    re-evaluates stale entries, yielding :math:`O(k \log G)` amortised
    complexity instead of the naïve :math:`O(k G)` scan.

    Optimality follows from the fact that the objective
    :math:`F(c) = \sum_g \pi_g \log(c_g + \alpha)` is separable and
    each :math:`f_g` is concave, so the marginal gains
    :math:`\Delta_g(t)` are non-increasing (Theorem 1(ii)).

    Parameters
    ----------
    pi : np.ndarray
        Target distribution, shape (G,)
    k : int
        Total count to allocate
    alpha : float
        Dirichlet smoothing parameter
    lb : np.ndarray
        Lower bounds per group, shape (G,)
    ub : np.ndarray
        Upper bounds per group, shape (G,)

    Returns
    -------
    np.ndarray
        Optimal integer counts, shape (G,)
    """
    pi = np.asarray(pi, dtype=np.float64)
    lb = np.asarray(lb, dtype=int)
    ub = np.asarray(ub, dtype=int)
    G = len(pi)

    # Start with lower bounds (Algorithm 1, line 7)
    counts = lb.copy()
    remaining = k - int(counts.sum())  # R = k − Σ ℓ_g (Algorithm 1, line 8)

    if remaining < 0:
        raise ValueError(f"Cannot satisfy lower bounds: need {lb.sum()}, have k={k}")

    if remaining == 0:
        return counts

    # ---- Lazy-heap greedy allocation (Algorithm 1, lines 9-15) ----
    # Heap entries: (-gain, group_idx).  Python heapq is a min-heap so we
    # negate gains to extract the *maximum* gain first.
    heap: List[Tuple[float, int]] = []
    for g in range(G):
        if counts[g] < ub[g] and pi[g] > 0:
            gain = _marginal_gain(pi[g], counts[g], alpha)
            heapq.heappush(heap, (-gain, g))

    for _ in range(remaining):
        if not heap:
            raise ValueError("Cannot allocate all samples within bounds")

        # Lazy evaluation: pop, re-compute gain at current count; if the
        # gain decreased (because another allocation happened), push back
        # and retry.  Because gains are non-increasing this loop terminates
        # in O(log G) amortised time.
        while True:
            neg_gain, g = heapq.heappop(heap)
            if counts[g] >= ub[g]:
                # Group became saturated since this entry was pushed
                continue
            current_gain = _marginal_gain(pi[g], counts[g], alpha)
            # Check if this gain is still the top (up to float tolerance)
            if heap and (-heap[0][0]) > current_gain + 1e-15:
                # Stale — push with refreshed gain and try again
                heapq.heappush(heap, (-current_gain, g))
                continue
            # Accept this allocation
            counts[g] += 1
            # Push updated gain for next potential allocation (if not saturated)
            if counts[g] < ub[g]:
                next_gain = _marginal_gain(pi[g], counts[g], alpha)
                heapq.heappush(heap, (-next_gain, g))
            break

    return counts


def _kl_min_from_counts(
    pi: np.ndarray,
    counts: np.ndarray,
    k: int,
    alpha: float,
) -> float:
    r"""Theorem 1(iii): KL_min(k) = Σ π_g log π_g + log(k + αG) − Σ π_g log(c*_g + α)."""
    G = len(pi)
    entropy_term = 0.0
    allocation_term = 0.0
    for g in range(G):
        if pi[g] > 0:
            entropy_term += pi[g] * np.log(pi[g])
            allocation_term += pi[g] * np.log(counts[g] + alpha)
    log_normalizer = np.log(k + alpha * G)
    return float(entropy_term + log_normalizer - allocation_term)


def min_achievable_geo_kl_bounded(
    pi: np.ndarray,
    group_sizes: np.ndarray,
    k: int,
    alpha_geo: float,
    min_one_per_group: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Compute minimum achievable geographic KL divergence (KL_min(k)).

    Implements Algorithm 1 from the manuscript. Finds the optimal allocation
    of k samples across groups that minimises KL divergence from the
    population distribution, subject to availability constraints.

    Parameters
    ----------
    pi : np.ndarray
        Population proportions, shape (G,)
    group_sizes : np.ndarray
        Available samples per group (capacities n_g), shape (G,)
    k : int
        Total samples to select
    alpha_geo : float
        Dirichlet smoothing parameter (α_geo > 0)
    min_one_per_group : bool
        Whether to require at least one sample per supported group
        (only when k >= number of supported groups)

    Returns
    -------
    Tuple[float, np.ndarray]
        (KL_min(k), optimal_counts c*(k))
    """
    pi = np.asarray(pi, dtype=np.float64)
    group_sizes = np.asarray(group_sizes, dtype=int)
    G = len(pi)

    # Define supported groups G_π = {g : π_g > 0} (Algorithm 1, line 2)
    supported_groups = pi > 0
    G_pi = int(np.sum(supported_groups))  # |G_π|

    # Set lower bounds (Algorithm 1, line 3):
    # ℓ_g = 1 if (min_one_per_group and k >= G_π and g ∈ G_π) else 0
    lb = np.zeros(G, dtype=int)
    if min_one_per_group and k >= G_pi:
        lb[supported_groups] = 1

    # Upper bounds (capacities n_g)
    ub = group_sizes.copy()

    # Check feasibility (Algorithm 1, line 4-6)
    if np.any(lb > ub):
        raise ValueError("Infeasible: lower bound exceeds capacity for some group")
    if lb.sum() > k:
        raise ValueError(f"Infeasible: sum of lower bounds {lb.sum()} exceeds k={k}")
    if ub.sum() < k:
        raise ValueError(f"Infeasible: total capacity {ub.sum()} is less than k={k}")

    # Find optimal counts using lazy-heap greedy (Algorithm 1, lines 7-15)
    optimal_counts = kl_optimal_integer_counts_bounded(
        pi, k, alpha_geo, lb=lb, ub=ub
    )

    kl_min = _kl_min_from_counts(pi, optimal_counts, k, alpha_geo)

    return float(kl_min), optimal_counts


# ---------------------------------------------------------------------------
# Incremental quota path over k-grid (Phase 6, §6.1)
# ---------------------------------------------------------------------------

def compute_quota_path(
    pi: np.ndarray,
    group_sizes: np.ndarray,
    k_grid: Sequence[int],
    alpha_geo: float,
    min_one_per_group: bool = True,
) -> List[Dict]:
    r"""Compute c*(k) and KL_min(k) for every k in *k_grid* incrementally.

    Instead of calling :func:`min_achievable_geo_kl_bounded` independently
    for each k (which would repeat work), this function builds the quota
    allocation **incrementally** from the smallest k to the largest.  For
    each transition k → k+1 a single heap-pop suffices, giving amortised
    :math:`O(k_{\max} \log G)` total cost for the full path.

    Parameters
    ----------
    pi : np.ndarray
        Target distribution, shape (G,).
    group_sizes : np.ndarray
        Per-group capacities n_g, shape (G,).
    k_grid : sequence of int
        Coreset sizes to evaluate (need not be sorted).
    alpha_geo : float
        Dirichlet smoothing parameter.
    min_one_per_group : bool
        Require ℓ_g ≥ 1 for supported groups when k ≥ |G_π|.

    Returns
    -------
    list of dict
        One dict per k in *k_grid* (sorted ascending) with keys:
        ``k``, ``kl_min``, ``cstar`` (list[int]), ``geo_l1``, ``geo_maxdev``.
    """
    pi = np.asarray(pi, dtype=np.float64)
    group_sizes = np.asarray(group_sizes, dtype=int)
    G = len(pi)
    k_sorted = sorted(set(int(k) for k in k_grid))

    if not k_sorted:
        return []

    k_min = k_sorted[0]
    k_max = k_sorted[-1]

    # Bounds (same logic as min_achievable_geo_kl_bounded)
    supported = pi > 0
    G_pi = int(np.sum(supported))

    lb = np.zeros(G, dtype=int)
    if min_one_per_group and k_min >= G_pi:
        lb[supported] = 1

    ub = group_sizes.copy()

    # Feasibility for the smallest k
    if np.any(lb > ub):
        raise ValueError("Infeasible: lower bound exceeds capacity for some group")
    if int(lb.sum()) > k_min:
        raise ValueError(
            f"Infeasible: sum of lower bounds {lb.sum()} exceeds k_min={k_min}"
        )
    if int(ub.sum()) < k_max:
        raise ValueError(
            f"Infeasible: total capacity {ub.sum()} < k_max={k_max}"
        )

    # Bootstrap: solve for k_min with the heap allocator
    counts = kl_optimal_integer_counts_bounded(pi, k_min, alpha_geo, lb=lb, ub=ub)

    # Build a live heap from current state for incremental extension
    heap: List[Tuple[float, int]] = []
    for g in range(G):
        if counts[g] < ub[g] and pi[g] > 0:
            gain = _marginal_gain(pi[g], counts[g], alpha_geo)
            heapq.heappush(heap, (-gain, g))

    # Helper to record a snapshot for a given k
    def _snapshot(k_val: int) -> Dict:
        pi_hat = counts / float(max(1, counts.sum()))
        l1 = float(np.sum(np.abs(pi - pi_hat)))
        maxdev = float(np.max(np.abs(pi - pi_hat)))
        return {
            "k": k_val,
            "kl_min": _kl_min_from_counts(pi, counts, k_val, alpha_geo),
            "cstar": counts.tolist(),
            "geo_l1": l1,
            "geo_maxdev": maxdev,
        }

    # Collect results: start from k_min, incrementally grow to k_max
    results: List[Dict] = []
    current_k = k_min

    # Record k_min if it's in the grid
    if current_k in k_sorted:
        results.append(_snapshot(current_k))

    k_set = set(k_sorted)
    for target_k in range(k_min + 1, k_max + 1):
        # Allocate one more unit using the live heap
        allocated = False
        while heap:
            neg_gain, g = heapq.heappop(heap)
            if counts[g] >= ub[g]:
                continue  # saturated
            current_gain = _marginal_gain(pi[g], counts[g], alpha_geo)
            if heap and (-heap[0][0]) > current_gain + 1e-15:
                heapq.heappush(heap, (-current_gain, g))
                continue
            counts[g] += 1
            if counts[g] < ub[g]:
                next_gain = _marginal_gain(pi[g], counts[g], alpha_geo)
                heapq.heappush(heap, (-next_gain, g))
            allocated = True
            break

        if not allocated:
            raise ValueError(
                f"Cannot extend quota path to k={target_k}: no eligible groups"
            )
        current_k = target_k

        if current_k in k_set:
            results.append(_snapshot(current_k))

    return results


def save_quota_path(
    path_rows: List[Dict],
    output_dir: str,
    *,
    json_name: str = "quota_path.json",
    csv_name: str = "kl_floor.csv",
) -> Tuple[str, str]:
    """Persist the quota path to ``quota_path.json`` and ``kl_floor.csv``.

    Parameters
    ----------
    path_rows : list of dict
        Output of :func:`compute_quota_path`.
    output_dir : str
        Directory in which to write files.
    json_name, csv_name : str
        File names.

    Returns
    -------
    (json_path, csv_path) : tuple of str
    """
    import csv as csv_mod
    import os

    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, json_name)
    csv_path = os.path.join(output_dir, csv_name)

    # JSON: full detail (includes cstar vectors)
    with open(json_path, "w") as f:
        json.dump(path_rows, f, indent=2)

    # CSV: summary (one row per k, no cstar vector for readability)
    fieldnames = ["k", "kl_min", "geo_l1", "geo_maxdev"]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in path_rows:
            writer.writerow({fn: row[fn] for fn in fieldnames})

    return json_path, csv_path


def kl_weighted_from_subset(
    target_pi: np.ndarray,
    weights: np.ndarray,
    group_ids: np.ndarray,
    mask: np.ndarray,
    alpha: float = 1.0,
    G: int = None,
) -> float:
    r"""Compute D^{(w)}(S) = KL(π^{(w)} || π̂^{(w,α)}(S)) for general weights.

    Per manuscript Eq. (3)-(4):
      π̂_g^{(w,α)}(S) = (W_g(S) + α) / (W(S) + αG)
      D^{(w)}(S) = KL(π^{(w)} || π̂^{(w,α)}(S))

    Parameters
    ----------
    target_pi : np.ndarray
        Full-data weighted target distribution π^{(w)}, shape (G,).
    weights : np.ndarray
        Per-item nonneg. weights w_i, shape (N,).
    group_ids : np.ndarray
        Group assignment per item, shape (N,).
    mask : np.ndarray
        Boolean selection mask, shape (N,).
    alpha : float
        Laplace smoothing pseudo-count (α > 0).
    G : int, optional
        Number of groups (inferred from target_pi if None).

    Returns
    -------
    float
        Forward KL divergence D^{(w)}(S).
    """
    target_pi = np.asarray(target_pi, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if G is None:
        G = len(target_pi)

    # Weighted group totals in S
    Wg = np.zeros(G, dtype=np.float64)
    sel_idx = np.where(mask)[0]
    for i in sel_idx:
        Wg[int(group_ids[i])] += weights[i]
    W = float(Wg.sum())

    # Smoothed subset distribution
    q = (Wg + alpha) / (W + alpha * G)

    # KL(target || q)
    kl = 0.0
    for g in range(G):
        if target_pi[g] > 0:
            kl += target_pi[g] * np.log(target_pi[g] / (q[g] + 1e-30))
    return float(kl)


def compute_constraint_violations(
    constraints,
    mask: np.ndarray,
    geo,
) -> Tuple[float, np.ndarray]:
    r"""Total violation V(S) = Σ_h max{D^{(w^{(h)})}(S) − τ_h, 0}.

    Per manuscript Section V-D.

    Parameters
    ----------
    constraints : sequence of ProportionalityConstraint
        Active constraints (each has .value(mask, geo) and .tau).
    mask : np.ndarray
        Boolean selection mask (N,).
    geo : GeoInfo
        Geographic group information.

    Returns
    -------
    total_violation : float
        Sum of positive violations across all constraints.
    per_constraint : np.ndarray
        Violation per constraint, shape (H,).
    """
    violations = np.array(
        [max(c.value(mask, geo) - c.tau, 0.0) for c in constraints],
        dtype=np.float64,
    )
    return float(violations.sum()), violations


def proportional_allocation(
    pi: np.ndarray,
    k: int,
    group_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute proportional allocation (largest remainder method).
    
    Parameters
    ----------
    pi : np.ndarray
        Target proportions
    k : int
        Total to allocate
    group_sizes : np.ndarray
        Upper bounds (available)
        
    Returns
    -------
    np.ndarray
        Integer allocation
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()  # Normalize
    G = len(pi)
    
    # Fractional allocation
    frac = pi * k
    
    # Integer parts
    counts = np.floor(frac).astype(int)
    
    # Remainders
    remainders = frac - counts
    
    # Distribute remaining samples
    remaining = k - counts.sum()
    
    if remaining > 0:
        # Sort by remainder descending
        order = np.argsort(-remainders)
        for i in range(remaining):
            g = order[i % G]
            if counts[g] < group_sizes[g]:
                counts[g] += 1
    
    return counts
