r"""State-conditioned KPI stability metrics (manuscript Section VII).

The manuscript defines stability via *target means* per state, not RMSE.

For a scalar KPI u_i (e.g. y^{4G}_i or y^{5G}_i), define:
  μ_g^{full}(u)  = (1/n_g) Σ_{i: g_i=g} u_i
  μ_g^{S}(u)     = (1/c_g(S)) Σ_{i∈S: g_i=g} u_i   (c_g(S)>0 guaranteed by ℓ_g=1)

Reported metrics:
  - max_kpi_drift_{target}: max_g |μ_g^S - μ_g^full|
  - avg_kpi_drift_{target}: mean_g |μ_g^S - μ_g^full|
  - kendall_tau_{target}: Kendall's τ between {μ_g^S}_g and {μ_g^full}_g
  - worst_state_rmse_{target}: max-state KRR RMSE (predictive)
  - state_rmse_dispersion_{target}: std of per-state KRR RMSE
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def state_kpi_stability(
    *,
    y: np.ndarray,
    state_labels: np.ndarray,
    S_idx: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict[str, float]:
    r"""Compute state-conditioned KPI drift and ranking stability.

    Parameters
    ----------
    y : np.ndarray
        Target matrix (N, T) or (N,).
    state_labels : np.ndarray
        State labels per municipality (N,).
    S_idx : np.ndarray
        Selected landmark indices.
    target_names : list, optional
        Names for each target column. Defaults to ["4G", "5G", ...].

    Returns
    -------
    Dict[str, float]
        max_kpi_drift_{t}, avg_kpi_drift_{t}, kendall_tau_{t} for each target.
    """
    from scipy.stats import kendalltau

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    state_labels = np.asarray(state_labels)
    S_idx = np.asarray(S_idx, dtype=int)

    N, T = y.shape
    if target_names is None:
        if T == 1:
            target_names = [""]
        elif T == 2:
            target_names = ["_4G", "_5G"]
        else:
            target_names = [f"_{i}" for i in range(T)]

    S_set = set(S_idx.tolist())
    S_mask = np.zeros(N, dtype=bool)
    S_mask[S_idx] = True

    unique_g = np.unique(state_labels)
    G = len(unique_g)

    out: Dict[str, float] = {}

    for t in range(T):
        suffix = target_names[t]
        full_means = np.zeros(G, dtype=np.float64)
        sub_means = np.zeros(G, dtype=np.float64)
        valid = np.ones(G, dtype=bool)

        for gi, g in enumerate(unique_g):
            mask_g = state_labels == g
            full_means[gi] = y[mask_g, t].mean()
            in_S_g = mask_g & S_mask
            c_g = in_S_g.sum()
            if c_g == 0:
                valid[gi] = False
                sub_means[gi] = np.nan
            else:
                sub_means[gi] = y[in_S_g, t].mean()

        v = valid
        if v.sum() < 2:
            continue

        drifts = np.abs(sub_means[v] - full_means[v])
        out[f"max_kpi_drift{suffix}"] = float(drifts.max())
        out[f"avg_kpi_drift{suffix}"] = float(drifts.mean())

        tau, _ = kendalltau(full_means[v], sub_means[v])
        out[f"kendall_tau{suffix}"] = float(tau) if np.isfinite(tau) else 0.0
        out[f"n_states_valid{suffix}"] = int(v.sum())

    return out


def per_state_kpi_drift_matrix(
    *,
    y: np.ndarray,
    state_labels: np.ndarray,
    S_idx: np.ndarray,
    target_names: Optional[list] = None,
    state_names: Optional[list] = None,
) -> Dict[str, Any]:
    r"""Compute the full per-state × per-target KPI drift matrix.

    For each state *g* and target *t*, computes the absolute deviation
    of the subset mean from the full-dataset mean:

        drift_{g,t} = |μ_g^{S}(u_t) − μ_g^{full}(u_t)|

    Parameters
    ----------
    y : np.ndarray
        Target matrix ``(N, T)`` or ``(N,)``.
    state_labels : np.ndarray
        State labels per municipality ``(N,)``.
    S_idx : np.ndarray
        Selected landmark indices.
    target_names : list, optional
        Human-readable names for each target column.
    state_names : list, optional
        Override names for each unique state (sorted).

    Returns
    -------
    Dict[str, Any]
        ``"states"`` – list of state names (rows),
        ``"targets"`` – list of target names (columns),
        ``"drift"`` – ``(G, T)`` float array of absolute drifts,
        ``"full_means"`` – ``(G, T)`` array of full-dataset means,
        ``"sub_means"`` – ``(G, T)`` array of subset means (NaN if c_g=0).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    state_labels = np.asarray(state_labels)
    S_idx = np.asarray(S_idx, dtype=int)

    N, T = y.shape
    if target_names is None:
        if T == 2:
            target_names = ["4G", "5G"]
        else:
            target_names = [f"target_{i}" for i in range(T)]

    unique_g = np.unique(state_labels)
    G = len(unique_g)
    if state_names is None:
        state_names = [str(g) for g in unique_g]

    S_mask = np.zeros(N, dtype=bool)
    S_mask[S_idx] = True

    full_means = np.zeros((G, T), dtype=np.float64)
    sub_means = np.full((G, T), np.nan, dtype=np.float64)

    for gi, g in enumerate(unique_g):
        mask_g = state_labels == g
        full_means[gi] = y[mask_g].mean(axis=0)
        in_S_g = mask_g & S_mask
        c_g = in_S_g.sum()
        if c_g > 0:
            sub_means[gi] = y[in_S_g].mean(axis=0)

    drift = np.where(np.isfinite(sub_means),
                     np.abs(sub_means - full_means),
                     np.nan)

    return {
        "states": state_names,
        "targets": target_names,
        "drift": drift,
        "full_means": full_means,
        "sub_means": sub_means,
    }


def export_state_kpi_drift_csv(
    *,
    y: np.ndarray,
    state_labels: np.ndarray,
    S_idx: np.ndarray,
    output_path: str,
    target_names: Optional[list] = None,
    state_names: Optional[list] = None,
    run_id: str = "",
    k: int = 0,
) -> str:
    r"""Export per-state KPI drift to a long-format CSV for heatmap visualization.

    Produces a CSV with columns ``state, target, drift, full_mean, sub_mean``
    suitable for direct consumption by ``ManuscriptArtifacts.fig_state_kpi_heatmap``.

    Parameters
    ----------
    y : np.ndarray
        Target matrix ``(N, T)`` or ``(N,)``.
    state_labels : np.ndarray
        State labels per municipality ``(N,)``.
    S_idx : np.ndarray
        Selected landmark indices.
    output_path : str
        Destination CSV file path.
    target_names : list, optional
        Human-readable target names.
    state_names : list, optional
        Override state names.
    run_id : str, optional
        Run identifier (written into the CSV for traceability).
    k : int, optional
        Coreset size (written into the CSV for traceability).

    Returns
    -------
    str
        The path written (same as *output_path*).
    """
    import os
    mat = per_state_kpi_drift_matrix(
        y=y,
        state_labels=state_labels,
        S_idx=S_idx,
        target_names=target_names,
        state_names=state_names,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    states = mat["states"]
    targets = mat["targets"]
    drift = mat["drift"]
    full_m = mat["full_means"]
    sub_m = mat["sub_means"]

    with open(output_path, "w") as f:
        f.write("state,target,drift,full_mean,sub_mean,run_id,k\n")
        for gi, st in enumerate(states):
            for ti, tgt in enumerate(targets):
                d = f"{drift[gi, ti]:.6f}" if np.isfinite(drift[gi, ti]) else ""
                fm = f"{full_m[gi, ti]:.6f}"
                sm = f"{sub_m[gi, ti]:.6f}" if np.isfinite(sub_m[gi, ti]) else ""
                f.write(f"{st},{tgt},{d},{fm},{sm},{run_id},{k}\n")

    return output_path


def state_krr_stability(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: np.ndarray,
    target_names: Optional[list] = None,
) -> Dict[str, float]:
    """Per-state KRR RMSE and dispersion (worst-state, std) per target.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True/predicted targets on E_test, shape (n_test, T).
    state_labels : np.ndarray
        State labels for E_test, shape (n_test,).
    target_names : list, optional
        Names for targets.

    Returns
    -------
    Dict[str, float]
        worst_state_rmse_{t}, state_rmse_std_{t}.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    state_labels = np.asarray(state_labels)

    T = y_true.shape[1]
    if target_names is None:
        target_names = ["_4G", "_5G"] if T == 2 else [f"_{i}" for i in range(T)]

    unique_g = np.unique(state_labels)
    out: Dict[str, float] = {}

    for t in range(T):
        suffix = target_names[t]
        rmses = []
        for g in unique_g:
            mask = state_labels == g
            if mask.sum() < 2:
                continue
            err = np.sqrt(np.mean((y_true[mask, t] - y_pred[mask, t]) ** 2))
            rmses.append(err)
        if rmses:
            out[f"worst_state_rmse{suffix}"] = float(max(rmses))
            out[f"state_rmse_std{suffix}"] = float(np.std(rmses))
            out[f"mean_state_rmse{suffix}"] = float(np.mean(rmses))

    return out
