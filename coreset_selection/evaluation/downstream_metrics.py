r"""Downstream supervised evaluation metrics (incremental).

Implements the additional metrics recommended by the kernel k-means vs
MMD+Sinkhorn+NSGA-II comparative analysis:

1. **Tail error metrics**: P90 / P95 of |error| per target — captures
   whether a method leaves "coverage holes" that produce large outlier
   errors.  Sinkhorn-driven selection is expected to reduce these.

2. **Per-state KRR RMSE** (full breakdown): returns a dict mapping each
   unique state label to its local RMSE so downstream scripts can produce
   heatmaps, rank tables, etc.

3. **Worst-group / macro-averaged RMSE**: first-class aggregate metrics
   that weight each state equally (or report the maximum), isolating the
   effect of proportionality constraints.

4. **R² per group**: coefficient of determination per state.

5. **MAE overall and per group**: sometimes preferred for robustness
   comparisons because it is not dominated by a few large residuals.

All functions accept raw numpy arrays and are agnostic to the upstream
selection method (NSGA-II, kernel k-means, etc.), so they can be called
from any evaluation harness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._downstream_helpers import (
    _rmse,
    _mae,
    _r2,
    tail_absolute_errors,
    per_state_downstream_metrics,
    aggregate_group_metrics,
)


# =====================================================================
# 4. Full downstream evaluation (combines tail + per-state + aggregates)
# =====================================================================

def full_downstream_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_suffix: str = "",
) -> Dict[str, Any]:
    r"""One-call evaluation producing all downstream metrics.

    Parameters
    ----------
    y_true, y_pred : (n,) arrays
    state_labels : (n,) optional — if given, per-state and macro metrics
        are computed.
    target_suffix : str
        Appended to every metric key, e.g. "_4G".

    Returns
    -------
    Dict with keys like:
      overall_rmse_{suf}, overall_mae_{suf}, overall_r2_{suf},
      abs_err_p90_{suf}, abs_err_p95_{suf}, abs_err_p99_{suf},
      macro_rmse_{suf}, worst_group_rmse_{suf}, ...
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    sfx = target_suffix

    out: Dict[str, Any] = {}

    # Global
    out[f"overall_rmse{sfx}"] = _rmse(y_true, y_pred)
    out[f"overall_mae{sfx}"] = _mae(y_true, y_pred)
    out[f"overall_r2{sfx}"] = _r2(y_true, y_pred)

    # Tail quantiles
    for k, v in tail_absolute_errors(y_true, y_pred).items():
        out[f"{k}{sfx}"] = v

    # Per-state (if labels provided)
    if state_labels is not None:
        state_labels = np.asarray(state_labels)
        per_state = per_state_downstream_metrics(y_true, y_pred, state_labels)
        agg = aggregate_group_metrics(per_state)
        for k, v in agg.items():
            out[f"{k}{sfx}"] = v
        # Stash the raw per-state dict for later CSV export
        out[f"_per_state_detail{sfx}"] = per_state

    return out


# =====================================================================
# 5. Multi-target wrapper (loops over columns)
# =====================================================================

def multitarget_downstream_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    r"""Evaluate all targets and return combined dict.

    Parameters
    ----------
    y_true, y_pred : (n, T) arrays
    state_labels : (n,) optional
    target_names : list of T strings

    Returns
    -------
    Dict combining all per-target metric dicts, plus cross-target
    summaries (mean_macro_rmse, mean_worst_group_rmse).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    T = y_true.shape[1]
    if target_names is None:
        if T == 2:
            target_names = ["_4G", "_5G"]
        elif T == 1:
            target_names = [""]
        else:
            target_names = [f"_{i}" for i in range(T)]

    combined: Dict[str, Any] = {}
    macro_rmses = []
    worst_rmses = []

    for t in range(T):
        sfx = target_names[t]
        res = full_downstream_evaluation(
            y_true[:, t], y_pred[:, t],
            state_labels=state_labels,
            target_suffix=sfx,
        )
        combined.update(res)
        if f"macro_rmse{sfx}" in res:
            macro_rmses.append(res[f"macro_rmse{sfx}"])
        if f"worst_group_rmse{sfx}" in res:
            worst_rmses.append(res[f"worst_group_rmse{sfx}"])

    # Cross-target summaries
    if macro_rmses:
        combined["mean_macro_rmse"] = float(np.mean(macro_rmses))
    if worst_rmses:
        combined["mean_worst_group_rmse"] = float(np.mean(worst_rmses))

    return combined


# =====================================================================
# 6. Convenience: evaluate a Nystrom landmark set end-to-end
# =====================================================================

def evaluate_nystrom_landmarks_downstream(
    *,
    X_raw: np.ndarray,
    S_idx: np.ndarray,
    y: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    eval_idx: np.ndarray,
    sigma_sq: float,
    state_labels: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    ridge_lambdas: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    r"""Build Nystrom features from landmarks, fit KRR, evaluate downstream.

    This is a self-contained function that:
      1. Builds Nystrom features from landmarks S on eval set E.
      2. Fits ridge regression (KRR) on E_train.
      3. Predicts on E_test.
      4. Computes all downstream metrics (overall, tail, per-state, macro).

    Parameters
    ----------
    X_raw : (N, D) full feature matrix
    S_idx : selected landmark indices
    y : (N, T) targets
    eval_train_idx, eval_test_idx, eval_idx : standard split indices
    sigma_sq : RBF kernel bandwidth squared
    state_labels : (N,) group labels
    target_names : list of target column names
    ridge_lambdas : grid for CV; default logspace(-6, 6, 13)

    Returns
    -------
    Dict with all downstream metrics
    """
    from .raw_space import (
        _rbf_kernel, _nystrom_components, _nystrom_features,
        _safe_cholesky_solve, _select_lambda_ridge,
    )

    X_raw = np.asarray(X_raw, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    eval_idx = np.asarray(eval_idx, dtype=int)
    eval_train_idx = np.asarray(eval_train_idx, dtype=int)
    eval_test_idx = np.asarray(eval_test_idx, dtype=int)

    if ridge_lambdas is None:
        ridge_lambdas = np.logspace(-6, 6, 13)

    # Map absolute indices -> positions within eval_idx
    N = X_raw.shape[0]
    pos = np.full(N, -1, dtype=int)
    pos[eval_idx] = np.arange(eval_idx.size, dtype=int)
    tr_pos = pos[eval_train_idx]; tr_pos = tr_pos[tr_pos >= 0]
    te_pos = pos[eval_test_idx];  te_pos = te_pos[te_pos >= 0]

    # Nystrom features
    X_E = X_raw[eval_idx]
    X_S = X_raw[S_idx]
    C, W, lambda_nys = _nystrom_components(X_E, X_S, sigma_sq)
    Phi = _nystrom_features(C, W, lambda_nys)

    Phi_tr = Phi[tr_pos]
    Phi_te = Phi[te_pos]
    y_tr = y[eval_train_idx]
    y_te = y[eval_test_idx]

    T = y.shape[1]
    if target_names is None:
        if T == 2:
            target_names = ["_4G", "_5G"]
        elif T == 1:
            target_names = [""]
        else:
            target_names = [f"_{i}" for i in range(T)]

    # Fit KRR per target and collect predictions
    y_pred_all = np.zeros_like(y_te)
    for t in range(T):
        yt_tr = y_tr[:, t]
        best_lam = _select_lambda_ridge(
            Phi_tr, yt_tr, lambdas=ridge_lambdas, n_folds=5, seed=12345 + t,
        )
        A = Phi_tr.T @ Phi_tr + float(best_lam) * np.eye(Phi_tr.shape[1])
        b = Phi_tr.T @ yt_tr
        w, _ = _safe_cholesky_solve(A, b)
        y_pred_all[:, t] = Phi_te @ w

    # State labels for test set
    te_states = state_labels[eval_test_idx] if state_labels is not None else None

    return multitarget_downstream_evaluation(
        y_true=y_te,
        y_pred=y_pred_all,
        state_labels=te_states,
        target_names=target_names,
    )


# =====================================================================
# 7. Unified evaluation dispatch (Phase 2 -- target-type-aware)
# =====================================================================

def evaluate_target_auto(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_suffix: str = "",
    target_type: str = "auto",
    cardinality_threshold: int = 50,
) -> Dict[str, Any]:
    r"""Unified evaluation that dispatches to regression or classification
    metrics based on target type.

    Parameters
    ----------
    y_true, y_pred : (n,) arrays
    state_labels : (n,) optional
    target_suffix : str
    target_type : str
        ``"auto"`` to infer, ``"regression"`` for RMSE/MAE/R²,
        ``"classification"`` for accuracy/kappa/F1.
    cardinality_threshold : int
        Used when ``target_type="auto"`` — max unique values to
        consider a target as classification.

    Returns
    -------
    Dict of metric name -> value.
    """
    from .classification_metrics import (
        infer_target_type,
        full_classification_evaluation,
    )

    if target_type == "auto":
        target_type = infer_target_type(
            y_true, cardinality_threshold=cardinality_threshold,
        )

    if target_type == "classification":
        return full_classification_evaluation(
            y_true, y_pred,
            state_labels=state_labels,
            target_suffix=target_suffix,
        )
    else:
        return full_downstream_evaluation(
            y_true, y_pred,
            state_labels=state_labels,
            target_suffix=target_suffix,
        )


def multitarget_evaluate_auto(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    cardinality_threshold: int = 50,
) -> Dict[str, Any]:
    r"""Multi-target evaluation with per-target type dispatch.

    Parameters
    ----------
    y_true, y_pred : (n, T) arrays
    state_labels : (n,) optional
    target_names : list of T strings (suffixes)
    target_types : list of T strings, each ``"auto"``, ``"regression"``,
        or ``"classification"``.  If None, all targets use auto-detection.
    cardinality_threshold : int

    Returns
    -------
    Dict combining all per-target metric dicts.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    T = y_true.shape[1]
    if target_names is None:
        target_names = [f"_{i}" for i in range(T)] if T > 1 else [""]
    if target_types is None:
        target_types = ["auto"] * T

    combined: Dict[str, Any] = {}
    for t in range(T):
        sfx = target_names[t]
        tt = target_types[t] if t < len(target_types) else "auto"
        res = evaluate_target_auto(
            y_true[:, t], y_pred[:, t],
            state_labels=state_labels,
            target_suffix=sfx,
            target_type=tt,
            cardinality_threshold=cardinality_threshold,
        )
        combined.update(res)

    return combined
