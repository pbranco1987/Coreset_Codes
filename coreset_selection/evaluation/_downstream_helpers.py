r"""Internal helper functions for downstream supervised evaluation metrics.

Split from evaluation/downstream_metrics.py for modularity.

Contains:
- _rmse, _mae, _r2: basic metric functions
- tail_absolute_errors: quantile thresholds of absolute error
- per_state_downstream_metrics: RMSE/MAE/R² per unique state
- aggregate_group_metrics: macro / worst-group / best-group summaries
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# =====================================================================
# Helpers
# =====================================================================

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


# =====================================================================
# 1. Tail error metrics
# =====================================================================

def tail_absolute_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> Dict[str, float]:
    r"""Compute quantile thresholds of |y_true − y_pred|.

    Parameters
    ----------
    y_true, y_pred : (n,) arrays
    quantiles : tuple of floats in (0, 1)

    Returns
    -------
    Dict  e.g. {"abs_err_p90": 0.12, "abs_err_p95": 0.18, ...}
    """
    ae = np.abs(np.asarray(y_true, dtype=np.float64)
                - np.asarray(y_pred, dtype=np.float64))
    out: Dict[str, float] = {}
    for q in quantiles:
        key = f"abs_err_p{int(q * 100)}"
        out[key] = float(np.quantile(ae, q))
    out["abs_err_max"] = float(ae.max())
    return out


# =====================================================================
# 2–3. Per-state RMSE/MAE/R² and aggregates
# =====================================================================

def per_state_downstream_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    r"""Compute RMSE, MAE, R² per unique state.

    Returns
    -------
    Dict[state_label_str, Dict[str, float]]
        e.g. {"SP": {"rmse": 0.03, "mae": 0.02, "r2": 0.91, "n": 312}, ...}
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    state_labels = np.asarray(state_labels)

    results: Dict[str, Dict[str, float]] = {}
    for g in np.unique(state_labels):
        mask = state_labels == g
        n_g = int(mask.sum())
        if n_g < 2:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        results[str(g)] = {
            "rmse": _rmse(yt, yp),
            "mae": _mae(yt, yp),
            "r2": _r2(yt, yp),
            "n": n_g,
        }
    return results


def aggregate_group_metrics(
    per_state: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    r"""Macro / worst-group / best-group summaries from per-state results.

    Returns keys like:
      macro_rmse, worst_group_rmse, best_group_rmse,
      macro_mae, worst_group_mae, macro_r2, worst_group_r2,
      rmse_dispersion (std of per-state RMSE), n_groups_evaluated
    """
    if not per_state:
        return {}

    rmses = [v["rmse"] for v in per_state.values()]
    maes = [v["mae"] for v in per_state.values()]
    r2s = [v["r2"] for v in per_state.values()]

    return {
        # RMSE
        "macro_rmse": float(np.mean(rmses)),
        "worst_group_rmse": float(np.max(rmses)),
        "best_group_rmse": float(np.min(rmses)),
        "rmse_dispersion": float(np.std(rmses)),
        "rmse_iqr": float(np.quantile(rmses, 0.75) - np.quantile(rmses, 0.25)),
        # MAE
        "macro_mae": float(np.mean(maes)),
        "worst_group_mae": float(np.max(maes)),
        # R²
        "macro_r2": float(np.mean(r2s)),
        "worst_group_r2": float(np.min(r2s)),
        "best_group_r2": float(np.max(r2s)),
        # Counts
        "n_groups_evaluated": len(per_state),
    }
