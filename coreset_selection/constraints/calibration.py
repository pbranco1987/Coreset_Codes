r"""Constraint tolerance (τ) calibration helpers (Phase 5 — Milestone 5.3).

The manuscript treats τ as a user-controlled tolerance for the proportionality
constraint ``D(S) = KL(π || π̂(S)) ≤ τ``.  Reviewers will ask *why this τ?*

This module provides utilities to:

1. **Estimate the feasible τ range** for a given ``(k, weight_family)`` pair
   by evaluating the KL divergence of random subsets.
2. **Run a τ sensitivity sweep** varying τ and reporting whether feasible
   solutions exist (and at what fidelity cost).
3. **Produce a calibration summary** suitable for a supplementary table.

Usage
-----
>>> from coreset_selection.constraints.calibration import estimate_feasible_tau_range
>>> result = estimate_feasible_tau_range(geo, k=300, weight_type="pop", n_samples=500, seed=42)
>>> print(result)
{'tau_min_achievable': 0.0015, 'tau_q10': 0.012, 'tau_median': 0.035, ...}

The τ-sensitivity sweep (``tau_sensitivity_sweep``) is designed to be called
from the experiment runner or from a standalone analysis script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geo.info import GeoInfo
from ..geo.kl import kl_pi_hat_from_counts, min_achievable_geo_kl_bounded


# ---------------------------------------------------------------------------
# 1. Estimate feasible τ range by Monte-Carlo sampling
# ---------------------------------------------------------------------------

def estimate_feasible_tau_range(
    geo: GeoInfo,
    k: int,
    *,
    weight_type: str = "muni",
    alpha: float = 1.0,
    n_samples: int = 500,
    seed: int = 42,
    population: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    r"""Estimate the distribution of ``D(S)`` for random subsets of size k.

    Draws *n_samples* uniform random subsets of size ``k`` and computes the
    KL divergence ``D(S)`` for each.  Returns summary statistics that help
    the user choose a reasonable τ.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information.
    k : int
        Coreset cardinality.
    weight_type : str
        ``"muni"`` (w ≡ 1) or ``"pop"`` (w = population).
    alpha : float
        Laplace smoothing parameter.
    n_samples : int
        Number of random subsets to draw.
    seed : int
        RNG seed.
    population : np.ndarray, optional
        Per-point population weights (required when ``weight_type="pop"``).

    Returns
    -------
    dict
        Summary statistics of the sampled KL distribution:
        ``tau_min_achievable`` — theoretical KL_min(k) from the quota planner;
        ``tau_q05``, ``tau_q10``, ``tau_q25``, ``tau_median``, ``tau_q75``,
        ``tau_q90``, ``tau_q95`` — quantiles of the Monte-Carlo distribution;
        ``tau_mean``, ``tau_std``.
    """
    rng = np.random.default_rng(seed)
    N = geo.N
    G = geo.G

    # Theoretical floor
    pi = geo.get_target_distribution(weight_type)
    kl_floor, _ = min_achievable_geo_kl_bounded(
        pi, geo.group_sizes, k, alpha_geo=alpha, min_one_per_group=True,
    )

    # Monte-Carlo: draw random subsets and compute KL
    kl_values = np.empty(n_samples, dtype=np.float64)
    all_idx = np.arange(N, dtype=int)

    for s in range(n_samples):
        sel = rng.choice(all_idx, size=k, replace=False)
        kl_values[s] = _kl_for_subset(geo, sel, pi, alpha, weight_type, population)

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qs = np.quantile(kl_values, quantiles)

    return {
        "tau_min_achievable": float(kl_floor),
        "tau_q05": float(qs[0]),
        "tau_q10": float(qs[1]),
        "tau_q25": float(qs[2]),
        "tau_median": float(qs[3]),
        "tau_q75": float(qs[4]),
        "tau_q90": float(qs[5]),
        "tau_q95": float(qs[6]),
        "tau_mean": float(np.mean(kl_values)),
        "tau_std": float(np.std(kl_values)),
        "k": k,
        "weight_type": weight_type,
        "n_samples": n_samples,
    }


def _kl_for_subset(
    geo: GeoInfo,
    sel: np.ndarray,
    pi: np.ndarray,
    alpha: float,
    weight_type: str,
    population: Optional[np.ndarray],
) -> float:
    """Compute KL(π || π̂(S)) for a given subset."""
    G = geo.G
    if weight_type == "muni":
        # Count-based
        counts = np.zeros(G, dtype=np.float64)
        for i in sel:
            counts[int(geo.group_ids[i])] += 1.0
        W = float(counts.sum())
    else:
        # Population-weighted
        if population is None:
            pop = geo.population_weights
            if pop is None:
                raise ValueError("population required for weight_type='pop'")
        else:
            pop = np.asarray(population, dtype=np.float64)
        counts = np.zeros(G, dtype=np.float64)
        for i in sel:
            counts[int(geo.group_ids[i])] += pop[i]
        W = float(counts.sum())

    if W <= 0:
        return float("inf")

    q = (counts + alpha) / (W + alpha * G)
    kl = 0.0
    for g in range(G):
        if pi[g] > 0:
            kl += pi[g] * np.log(pi[g] / (q[g] + 1e-30))
    return float(kl)


# ---------------------------------------------------------------------------
# 2. τ sensitivity sweep
# ---------------------------------------------------------------------------

@dataclass
class TauSweepResult:
    """Result of one τ value in the sensitivity sweep."""
    tau: float
    feasible_frac: float          # fraction of random subsets with D(S) ≤ τ
    kl_mean_feasible: float       # mean KL among feasible subsets
    kl_min_achievable: float      # theoretical floor
    is_achievable: bool           # τ >= kl_min_achievable


def tau_sensitivity_sweep(
    geo: GeoInfo,
    k: int,
    *,
    tau_values: Sequence[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05),
    weight_type: str = "muni",
    alpha: float = 1.0,
    n_samples: int = 300,
    seed: int = 42,
    population: Optional[np.ndarray] = None,
) -> List[TauSweepResult]:
    r"""Run a τ sensitivity sweep: for each candidate τ, estimate feasibility.

    This directly answers the reviewer question:
        "Your constraint tolerance was arbitrarily chosen; what happens if it
        is tightened?"

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information.
    k : int
        Coreset cardinality.
    tau_values : sequence of float
        Candidate τ values to evaluate.
    weight_type : str
        ``"muni"`` or ``"pop"``.
    alpha : float
        Laplace smoothing parameter.
    n_samples : int
        Number of random subsets per τ evaluation.
    seed : int
        RNG seed.
    population : np.ndarray, optional
        Per-point population weights.

    Returns
    -------
    list of TauSweepResult
        One entry per τ value, sorted by τ ascending.
    """
    rng = np.random.default_rng(seed)
    N = geo.N

    pi = geo.get_target_distribution(weight_type)
    kl_floor, _ = min_achievable_geo_kl_bounded(
        pi, geo.group_sizes, k, alpha_geo=alpha, min_one_per_group=True,
    )

    # Pre-compute KL for all random subsets once
    all_idx = np.arange(N, dtype=int)
    kl_values = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        sel = rng.choice(all_idx, size=k, replace=False)
        kl_values[s] = _kl_for_subset(geo, sel, pi, alpha, weight_type, population)

    results = []
    for tau in sorted(tau_values):
        feas_mask = kl_values <= tau
        frac = float(feas_mask.mean())
        mean_kl = float(kl_values[feas_mask].mean()) if feas_mask.any() else float("nan")
        results.append(TauSweepResult(
            tau=float(tau),
            feasible_frac=frac,
            kl_mean_feasible=mean_kl,
            kl_min_achievable=float(kl_floor),
            is_achievable=float(tau) >= float(kl_floor),
        ))

    return results


def tau_sweep_to_csv_rows(results: List[TauSweepResult]) -> List[Dict[str, float]]:
    """Convert sweep results to a list of dicts for CSV / DataFrame export."""
    rows = []
    for r in results:
        rows.append({
            "tau": r.tau,
            "feasible_frac": r.feasible_frac,
            "kl_mean_feasible": r.kl_mean_feasible,
            "kl_min_achievable": r.kl_min_achievable,
            "is_achievable": float(r.is_achievable),
        })
    return rows
