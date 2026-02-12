"""
Geographic diagnostic metrics for coreset selection.

Contains:
- geo_diagnostics: Compute KL divergence, L1 distance, and max deviation
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..geo import GeoInfo


def geo_diagnostics(
    geo: GeoInfo,
    idx_sel: np.ndarray,
    k: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    r"""
    Compute geographic distribution diagnostics.

    Metrics per manuscript Section 5.11 (Geographic evaluation):
    - Smoothed KL proportionality gap: KL(pi || \hat{pi}^{(alpha)}(s)),
      where only the *subset* histogram is Dirichlet-smoothed.
    - L1 deviation between population and subset histograms (unsmoothed).
    - Maximum absolute deviation between population and subset histograms.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    idx_sel : np.ndarray
        Indices of selected points
    k : int
        Expected subset size (used only for reporting; KL uses |S| in smoothing denom)
    alpha : float
        Dirichlet smoothing parameter (alpha_geo)

    Returns
    -------
    Dict[str, float]
        Keys:
        - "geo_kl": KL(pi || \hat{pi}^{(alpha)}(s))
        - "geo_l1": ||pi - \hat{pi}(s)||_1 (unsmoothed)
        - "geo_maxdev": max_g |pi_g - \hat{pi}_g(s)| (unsmoothed)
        - "geo_counts": counts per group
        - "geo_k_actual": |S|
    """
    idx_sel = np.asarray(idx_sel, dtype=int)

    # Count selections per group
    counts = np.zeros(geo.G, dtype=int)
    for idx in idx_sel:
        g = int(geo.group_ids[idx])
        counts[g] += 1

    k_actual = int(counts.sum())
    if k_actual == 0:
        return {
            "geo_kl": float("inf"),
            "geo_l1": 2.0,
            "geo_maxdev": 1.0,
            "geo_counts": counts.tolist(),
            "geo_k_actual": 0,
        }

    # Smoothed subset histogram (Eq. 14 in manuscript, "smooth-hist")
    pi_hat_smooth = (counts + alpha) / (k_actual + alpha * geo.G)

    # Population distribution (UNSMOOTHED) pi
    pi = np.asarray(geo.pi, dtype=np.float64)

    # KL(pi || pi_hat_smooth) = sum_g pi_g log(pi_g / pi_hat_smooth_g)
    # Only terms with pi_g > 0 contribute.
    mask = pi > 0
    kl = float(np.sum(pi[mask] * np.log(pi[mask] / (pi_hat_smooth[mask] + 1e-30))))

    # Unsmooothed subset histogram for deviation metrics
    pi_hat = counts / k_actual
    l1 = float(np.sum(np.abs(pi - pi_hat)))
    maxdev = float(np.max(np.abs(pi - pi_hat)))

    return {
        "geo_kl": kl,
        "geo_l1": l1,
        "geo_maxdev": maxdev,
        "geo_counts": counts.tolist(),
        "geo_k_actual": k_actual,
    }



def compute_quota_satisfaction(
    geo: GeoInfo,
    idx_sel: np.ndarray,
    target_counts: np.ndarray,
) -> Dict[str, float]:
    """
    Check how well selection satisfies quota constraints.
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    idx_sel : np.ndarray
        Indices of selected points
    target_counts : np.ndarray
        Target count for each group
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - "quota_satisfied": 1.0 if all quotas met, 0.0 otherwise
        - "quota_total_deviation": Sum of |actual - target| over groups
        - "quota_violations": Number of groups with wrong count
    """
    idx_sel = np.asarray(idx_sel, dtype=int)
    target_counts = np.asarray(target_counts, dtype=int)
    
    # Count actual selections per group
    actual_counts = np.zeros(geo.G, dtype=int)
    for idx in idx_sel:
        g = geo.group_ids[idx]
        actual_counts[g] += 1
    
    deviations = np.abs(actual_counts - target_counts)
    
    return {
        "quota_satisfied": float(np.all(deviations == 0)),
        "quota_total_deviation": int(np.sum(deviations)),
        "quota_violations": int(np.sum(deviations > 0)),
    }


def state_coverage_report(
    geo: GeoInfo,
    idx_sel: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Generate detailed per-state coverage report.
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    idx_sel : np.ndarray
        Indices of selected points
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict: state_name -> {count, fraction, population_fraction, ratio}
    """
    idx_sel = np.asarray(idx_sel, dtype=int)
    k = len(idx_sel)
    
    if k == 0:
        return {}
    
    # Count per group
    counts = np.zeros(geo.G, dtype=int)
    for idx in idx_sel:
        g = geo.group_ids[idx]
        counts[g] += 1
    
    report = {}
    for g in range(geo.G):
        state_name = geo.groups[g]
        pop_frac = geo.pi[g]
        sel_frac = counts[g] / k
        
        report[state_name] = {
            "count": int(counts[g]),
            "fraction": float(sel_frac),
            "population_fraction": float(pop_frac),
            "ratio": float(sel_frac / (pop_frac + 1e-10)),
        }
    
    return report


def geo_diagnostics_weighted(
    geo: GeoInfo,
    idx_sel: np.ndarray,
    k: int,
    weight_type: str = "pop",
    alpha: float = 1.0,
) -> Dict[str, float]:
    r"""Weighted proportionality diagnostics.

    Computes KL, L1, maxdev for the specified weight type (``"pop"`` for
    population-share, ``"muni"`` for municipality/count-share).  Returns
    metrics with a ``_{weight_type}`` suffix.

    Per manuscript Section VII-G.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information (must have ``pi_pop`` for ``"pop"``).
    idx_sel : np.ndarray
        Selected indices.
    k : int
        Expected subset cardinality.
    weight_type : str
        ``"pop"`` or ``"muni"``.
    alpha : float
        Laplace smoothing parameter.

    Returns
    -------
    Dict[str, float]
        Keys like ``geo_kl_pop``, ``geo_l1_pop``, ``geo_maxdev_pop``.
    """
    suffix = f"_{weight_type}"
    idx_sel = np.asarray(idx_sel, dtype=int)
    nan_result = {
        f"geo_kl{suffix}": float("nan"),
        f"geo_l1{suffix}": float("nan"),
        f"geo_maxdev{suffix}": float("nan"),
    }

    if weight_type == "muni":
        # Count-based: delegate to standard geo_diagnostics
        d = geo_diagnostics(geo, idx_sel, k, alpha=alpha)
        return {
            f"geo_kl{suffix}": d["geo_kl"],
            f"geo_l1{suffix}": d["geo_l1"],
            f"geo_maxdev{suffix}": d["geo_maxdev"],
        }

    if weight_type == "pop":
        if geo.population_weights is None or geo.pi_pop is None:
            return nan_result

        pop = geo.population_weights
        pi = geo.pi_pop
        mask = np.zeros(geo.N, dtype=bool)
        mask[idx_sel] = True

        # Weighted group totals W_g(S) and W(S)
        Wg = np.zeros(geo.G, dtype=np.float64)
        for i in idx_sel:
            Wg[int(geo.group_ids[i])] += pop[i]
        W = float(Wg.sum())
        if W <= 0:
            return nan_result

        # Smoothed: (W_g(S) + α) / (W(S) + αG)
        q_smooth = (Wg + alpha) / (W + alpha * geo.G)
        kl = 0.0
        for g in range(geo.G):
            if pi[g] > 0:
                kl += pi[g] * np.log(pi[g] / (q_smooth[g] + 1e-30))

        # Unsmoothed weighted histogram
        q_raw = Wg / W
        l1 = float(np.sum(np.abs(pi - q_raw)))
        maxdev = float(np.max(np.abs(pi - q_raw)))

        return {
            f"geo_kl{suffix}": float(kl),
            f"geo_l1{suffix}": l1,
            f"geo_maxdev{suffix}": maxdev,
        }

    return nan_result


def dual_geo_diagnostics(
    geo: GeoInfo,
    idx_sel: np.ndarray,
    k: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    r"""Compute **both** municipality-share and population-share diagnostics.

    Per manuscript Sections VIII.2 and VIII.5, every evaluation row should
    contain proportionality metrics under *both* weight regimes so that
    downstream artifacts (figures & tables) can display either view without
    re-running experiments.

    This convenience wrapper calls :func:`geo_diagnostics` for the legacy
    un-suffixed keys, then appends ``_muni`` and ``_pop`` variants via
    :func:`geo_diagnostics_weighted`.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information (with optional ``population_weights``).
    idx_sel : np.ndarray
        Selected indices.
    k : int
        Expected subset cardinality.
    alpha : float
        Dirichlet smoothing parameter.

    Returns
    -------
    Dict[str, float]
        Combined dictionary with keys:
        - ``geo_kl``, ``geo_l1``, ``geo_maxdev`` (legacy / backward-compat)
        - ``geo_kl_muni``, ``geo_l1_muni``, ``geo_maxdev_muni``
        - ``geo_kl_pop``, ``geo_l1_pop``, ``geo_maxdev_pop`` (if pop weights available;
          ``NaN`` otherwise)
        - ``geo_k_actual``
    """
    row: Dict[str, float] = {}

    # 1. Legacy un-suffixed keys (count-based / municipality-share)
    base = geo_diagnostics(geo, idx_sel, k, alpha=alpha)
    for kk, vv in base.items():
        if kk == "geo_counts":
            continue
        row[kk] = vv

    # 2. Municipality-share with explicit _muni suffix
    muni = geo_diagnostics_weighted(geo, idx_sel, k, weight_type="muni", alpha=alpha)
    row.update(muni)

    # 3. Population-share with _pop suffix (NaN when unavailable)
    pop = geo_diagnostics_weighted(geo, idx_sel, k, weight_type="pop", alpha=alpha)
    row.update(pop)

    return row


def geographic_entropy(
    geo: GeoInfo,
    idx_sel: np.ndarray,
) -> float:
    """
    Compute entropy of geographic distribution of selection.
    
    Higher entropy indicates more uniform distribution across states.
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    idx_sel : np.ndarray
        Indices of selected points
        
    Returns
    -------
    float
        Shannon entropy in nats
    """
    idx_sel = np.asarray(idx_sel, dtype=int)
    k = len(idx_sel)
    
    if k == 0:
        return 0.0
    
    # Count per group
    counts = np.zeros(geo.G, dtype=int)
    for idx in idx_sel:
        g = geo.group_ids[idx]
        counts[g] += 1
    
    # Empirical distribution
    p = counts / k
    
    # Entropy
    entropy = 0.0
    for pg in p:
        if pg > 0:
            entropy -= pg * np.log(pg)
    
    return float(entropy)


def geographic_concentration_index(
    geo: GeoInfo,
    idx_sel: np.ndarray,
) -> float:
    """
    Compute geographic concentration (Herfindahl-Hirschman Index).
    
    HHI = Σ p_g² where p_g is fraction in group g.
    Ranges from 1/G (uniform) to 1 (all in one group).
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    idx_sel : np.ndarray
        Indices of selected points
        
    Returns
    -------
    float
        HHI value
    """
    idx_sel = np.asarray(idx_sel, dtype=int)
    k = len(idx_sel)
    
    if k == 0:
        return 1.0
    
    # Count per group
    counts = np.zeros(geo.G, dtype=int)
    for idx in idx_sel:
        g = geo.group_ids[idx]
        counts[g] += 1
    
    # HHI
    p = counts / k
    hhi = float(np.sum(p ** 2))
    
    return hhi
