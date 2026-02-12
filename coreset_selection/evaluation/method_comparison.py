r"""Cross-method comparison and effect isolation.

Post-hoc utilities for analysing saved experiment results, implementing the
comparison protocol from the kernel k-means vs MMD+Sinkhorn+NSGA-II analysis:

    1. Kernel k-means unconstrained (KKN)
    2. Kernel k-means quota-matched  (SKKN)
    3. NSGA-II Pareto selection       (any Pareto rep)

Plus all other baselines in both regimes (exactk / quota).

Functions
---------
effect_isolation_table
    Decomposes Pareto improvement over KKN into constraint vs objective.
rank_table
    Per-metric ranking of methods.
pairwise_dominance_matrix
    Pareto dominance across metric subsets.
stability_summary
    Mean ± std across replicates.
build_comparison_report
    One-call report combining all of the above.
load_result_rows
    Load result CSV rows saved by the experiment runner.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# IO helpers
# =====================================================================

def load_result_rows(path: str) -> List[Dict[str, Any]]:
    """Load a CSV of result rows (as written by ``saver.save_rows``)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def group_by_method(rows: List[Dict[str, Any]], key: str = "method") -> Dict[str, List[Dict]]:
    """Group result rows by method name."""
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        m = str(r.get(key, "unknown"))
        out.setdefault(m, []).append(r)
    return out


def mean_per_method(by_method: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    """Compute per-method mean of every numeric column."""
    out = {}
    for meth, reps in by_method.items():
        combined: Dict[str, List[float]] = {}
        for r in reps:
            for k, v in r.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    combined.setdefault(k, []).append(float(v))
        out[meth] = {k: float(np.mean(vals)) for k, vals in combined.items()}
    return out


# =====================================================================
# Effect isolation
# =====================================================================

def effect_isolation_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    nsga_key: str = "Pareto",
    kkn_key: str = "KKN",
    skkn_key: str = "SKKN",
) -> List[Dict[str, Any]]:
    r"""Decompose Pareto's improvement over unconstrained kernel k-means.

    For each metric *m* (lower-is-better convention):

        Δ_total      = m(KKN)  − m(Pareto)
        Δ_constraint  = m(KKN)  − m(SKKN)   ← pure proportionality effect
        Δ_objective   = m(SKKN) − m(Pareto)  ← MMD+Sinkhorn objective effect

    Returns a list of dicts (one per metric) ready for CSV.
    """
    if kkn_key not in results or skkn_key not in results:
        return []

    # Find the NSGA-II / Pareto key
    if nsga_key not in results:
        candidates = [k for k in results if "pareto" in k.lower() or "nsga" in k.lower()]
        if candidates:
            nsga_key = candidates[0]
        else:
            return []

    rows: List[Dict[str, Any]] = []
    for m in metrics:
        v_kkn = results[kkn_key].get(m, np.nan)
        v_skkn = results[skkn_key].get(m, np.nan)
        v_nsga = results[nsga_key].get(m, np.nan)

        dt = v_kkn - v_nsga
        dc = v_kkn - v_skkn
        do = v_skkn - v_nsga

        pct_c = dc / abs(dt) * 100.0 if np.isfinite(dt) and abs(dt) > 1e-15 else np.nan
        pct_o = do / abs(dt) * 100.0 if np.isfinite(dt) and abs(dt) > 1e-15 else np.nan

        rows.append({
            "metric": m,
            "kkn": v_kkn,
            "skkn": v_skkn,
            "nsga": v_nsga,
            "delta_total": dt,
            "delta_constraint": dc,
            "delta_objective": do,
            "pct_constraint": pct_c,
            "pct_objective": pct_o,
        })
    return rows


# =====================================================================
# Rank table
# =====================================================================

def rank_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    lower_is_better: bool = True,
) -> List[Dict[str, Any]]:
    """Per-metric ranking of methods plus average rank.

    Returns a list of dicts (one per method).
    """
    methods = sorted(results.keys())
    ranks: Dict[str, Dict[str, Any]] = {m: {"method": m} for m in methods}

    for metric in metrics:
        vals = [(meth, results[meth].get(metric, np.inf if lower_is_better else -np.inf))
                for meth in methods]
        vals.sort(key=lambda x: x[1], reverse=(not lower_is_better))
        for rank_idx, (meth, _) in enumerate(vals, start=1):
            ranks[meth][f"rank_{metric}"] = rank_idx

    for meth in methods:
        r_vals = [v for k, v in ranks[meth].items() if k.startswith("rank_")]
        ranks[meth]["avg_rank"] = float(np.mean(r_vals)) if r_vals else np.nan

    return list(ranks.values())


# =====================================================================
# Pairwise dominance
# =====================================================================

def pairwise_dominance_matrix(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    lower_is_better: bool = True,
) -> List[Dict[str, Any]]:
    """Return rows of (method_a, method_b, a_dominates_b)."""
    methods = sorted(results.keys())
    rows = []
    for a in methods:
        for b in methods:
            if a == b:
                continue
            # a dominates b if a ≤ b on all metrics and a < b on at least one
            better_any = False
            dominates = True
            for m in metrics:
                va = results[a].get(m, np.inf if lower_is_better else -np.inf)
                vb = results[b].get(m, np.inf if lower_is_better else -np.inf)
                if lower_is_better:
                    if va > vb:
                        dominates = False; break
                    if va < vb:
                        better_any = True
                else:
                    if va < vb:
                        dominates = False; break
                    if va > vb:
                        better_any = True
            rows.append({
                "method_a": a, "method_b": b,
                "a_dominates_b": dominates and better_any,
            })
    return rows


# =====================================================================
# Stability summary
# =====================================================================

def stability_summary(
    by_method: Dict[str, List[Dict]],
    metrics: List[str],
) -> List[Dict[str, Any]]:
    """Mean ± std ± CV of selected metrics across replicates."""
    rows = []
    for meth, reps in sorted(by_method.items()):
        row: Dict[str, Any] = {"method": meth}
        for m in metrics:
            vals = [r.get(m, np.nan) for r in reps]
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            if vals:
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                row[f"mean_{m}"] = mu
                row[f"std_{m}"] = sd
                row[f"cv_{m}"] = sd / max(abs(mu), 1e-15)
        rows.append(row)
    return rows


# =====================================================================
# Comprehensive report
# =====================================================================

# Canonical metrics from the analysis documents
DOWNSTREAM_LOWER = [
    "krr_rmse_4G", "krr_rmse_5G", "krr_rmse_mean",
    "nystrom_error", "kpca_distortion",
    "abs_err_p90_4G", "abs_err_p95_4G", "abs_err_p99_4G",
    "abs_err_p90_5G", "abs_err_p95_5G", "abs_err_p99_5G",
    "overall_mae_4G", "overall_mae_5G",
    "macro_rmse_4G", "macro_rmse_5G",
    "worst_group_rmse_4G", "worst_group_rmse_5G",
    "rmse_dispersion_4G", "rmse_dispersion_5G",
    "macro_mae_4G", "macro_mae_5G",
    "worst_group_mae_4G", "worst_group_mae_5G",
    "max_kpi_drift_4G", "max_kpi_drift_5G",
    "geo_kl", "geo_l1", "geo_maxdev",
    # QoS downstream: pooled models (lower is better)
    "qos_ols_pooled_rmse", "qos_ols_pooled_mae",
    "qos_ridge_pooled_rmse", "qos_ridge_pooled_mae",
    "qos_elastic_net_pooled_rmse", "qos_elastic_net_pooled_mae",
    "qos_pls_pooled_rmse", "qos_pls_pooled_mae",
    "qos_constrained_pooled_rmse", "qos_constrained_pooled_mae",
    "qos_heuristic_rmse", "qos_heuristic_mae",
    # QoS downstream: fixed-effects models (lower is better)
    "qos_ols_fe_rmse", "qos_ols_fe_mae",
    "qos_ridge_fe_rmse", "qos_ridge_fe_mae",
    "qos_elastic_net_fe_rmse", "qos_elastic_net_fe_mae",
    "qos_pls_fe_rmse", "qos_pls_fe_mae",
    "qos_constrained_fe_rmse", "qos_constrained_fe_mae",
]

DOWNSTREAM_HIGHER = [
    "overall_r2_4G", "overall_r2_5G",
    "macro_r2_4G", "macro_r2_5G",
    "worst_group_r2_4G", "worst_group_r2_5G",
    "kendall_tau_4G", "kendall_tau_5G",
    # QoS downstream: pooled models (higher is better)
    "qos_ols_pooled_r2",
    "qos_ridge_pooled_r2",
    "qos_elastic_net_pooled_r2",
    "qos_pls_pooled_r2",
    "qos_constrained_pooled_r2",
    "qos_heuristic_r2",
    # QoS downstream: fixed-effects models (higher is better)
    "qos_ols_fe_r2",
    "qos_ridge_fe_r2",
    "qos_elastic_net_fe_r2",
    "qos_pls_fe_r2",
    "qos_constrained_fe_r2",
]


def build_comparison_report(
    rows: List[Dict[str, Any]],
    output_dir: str,
    method_key: str = "method",
):
    r"""Build the full comparison report from saved result rows.

    Writes CSVs for:
      - effect_isolation.csv
      - rank_table_lower.csv / rank_table_higher.csv
      - pairwise_dominance.csv
      - stability_summary.csv
      - comparison_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)

    by_method = group_by_method(rows, method_key)
    means = mean_per_method(by_method)

    # Filter to metrics actually present
    available = set()
    for v in means.values():
        available.update(v.keys())
    lower_present = [m for m in DOWNSTREAM_LOWER if m in available]
    higher_present = [m for m in DOWNSTREAM_HIGHER if m in available]

    def _write_csv(path, data):
        if not data:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()), extrasaction="ignore")
            w.writeheader()
            for r in data:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"  ✓ {path}  ({len(data)} rows)")

    # Effect isolation (needs KKN + SKKN + a Pareto key)
    iso_rows = effect_isolation_table(means, lower_present + higher_present)
    if iso_rows:
        _write_csv(os.path.join(output_dir, "effect_isolation.csv"), iso_rows)

    # Rank tables
    if lower_present:
        _write_csv(os.path.join(output_dir, "rank_table_lower.csv"),
                   rank_table(means, lower_present, lower_is_better=True))
    if higher_present:
        _write_csv(os.path.join(output_dir, "rank_table_higher.csv"),
                   rank_table(means, higher_present, lower_is_better=False))

    # Pairwise dominance
    if lower_present:
        _write_csv(os.path.join(output_dir, "pairwise_dominance.csv"),
                   pairwise_dominance_matrix(means, lower_present))

    # Stability
    stab_metrics = [m for m in lower_present + higher_present
                    if any(len(reps) > 1 for reps in by_method.values())]
    if stab_metrics:
        _write_csv(os.path.join(output_dir, "stability_summary.csv"),
                   stability_summary(by_method, stab_metrics))

    # JSON summary
    summary = {
        "n_methods": len(means),
        "methods": sorted(means.keys()),
        "n_lower_metrics": len(lower_present),
        "n_higher_metrics": len(higher_present),
        "lower_metrics": lower_present,
        "higher_metrics": higher_present,
    }
    spath = os.path.join(output_dir, "comparison_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ {spath}")
