"""
R11 Post-hoc Diagnostics -- Analysis functions.

Split from ``r7_diagnostics.py`` for maintainability.
Public API is re-exported via ``r7_diagnostics.py`` (facade).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ._r7_objectives import (
    SurrogateSensitivityConfig,
    compute_anchored_sinkhorn,
    compute_rff_mmd,
)


def surrogate_sensitivity_analysis(
    X: np.ndarray,
    pareto_indices: List[np.ndarray],
    config: SurrogateSensitivityConfig,
    bandwidth: float,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run surrogate sensitivity analysis (manuscript Section VIII.K, Table IV).

    For a fixed pool of candidate subsets, computes Spearman \u03c1 between
    rankings induced by the reference and alternative parameter settings.

    Parameters
    ----------
    X : np.ndarray
        Full dataset, shape (N, d).
    pareto_indices : List[np.ndarray]
        List of index arrays for each candidate solution.
    config : SurrogateSensitivityConfig
        Configuration for sensitivity analysis.
    bandwidth : float
        RBF kernel bandwidth for MMD.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Results including correlations and objective values.
    """
    n_solutions = len(pareto_indices)

    # Compute reference objectives
    ref_mmd = np.array([
        compute_rff_mmd(X, idx, config.reference_rff, bandwidth, seed)
        for idx in pareto_indices
    ])
    ref_sink = np.array([
        compute_anchored_sinkhorn(X, idx, config.reference_anchors, seed=seed)
        for idx in pareto_indices
    ])

    results: Dict[str, Any] = {
        "reference": {
            "rff_dim": config.reference_rff,
            "n_anchors": config.reference_anchors,
            "mmd": ref_mmd.tolist(),
            "sinkhorn": ref_sink.tolist(),
        },
        "rff_sensitivity": {},
        "anchor_sensitivity": {},
    }

    # RFF dimension sensitivity (Table IV, section 1)
    for m in config.rff_dims:
        if m == config.reference_rff:
            # Self-correlation is 1.0 by definition
            results["rff_sensitivity"][m] = {
                "mmd": ref_mmd.tolist(),
                "spearman_rho": 1.0,
                "spearman_pval": 0.0,
            }
            continue
        alt_mmd = np.array([
            compute_rff_mmd(X, idx, m, bandwidth, seed)
            for idx in pareto_indices
        ])

        if n_solutions > 2:
            rho, pval = stats.spearmanr(ref_mmd, alt_mmd)
        else:
            rho, pval = np.nan, np.nan

        results["rff_sensitivity"][m] = {
            "mmd": alt_mmd.tolist(),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_pval": float(pval) if not np.isnan(pval) else None,
        }

    # Anchor count sensitivity (Table IV, section 2)
    for A in config.anchor_counts:
        if A == config.reference_anchors:
            results["anchor_sensitivity"][A] = {
                "sinkhorn": ref_sink.tolist(),
                "spearman_rho": 1.0,
                "spearman_pval": 0.0,
            }
            continue
        alt_sink = np.array([
            compute_anchored_sinkhorn(X, idx, A, seed=seed)
            for idx in pareto_indices
        ])

        if n_solutions > 2:
            rho, pval = stats.spearmanr(ref_sink, alt_sink)
        else:
            rho, pval = np.nan, np.nan

        results["anchor_sensitivity"][A] = {
            "sinkhorn": alt_sink.tolist(),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_pval": float(pval) if not np.isnan(pval) else None,
        }

    return results


# =============================================================================
# Cross-Space Objective Re-evaluation (Section VIII.K -- Table IV, section 3)
# =============================================================================


def cross_space_evaluation(
    X_vae: np.ndarray,
    X_pca: np.ndarray,
    X_raw: np.ndarray,
    pareto_indices: List[np.ndarray],
    bandwidth_vae: float,
    bandwidth_pca: float,
    bandwidth_raw: float,
    rff_dim: int = 2000,
    n_anchors: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Re-evaluate objectives in all three representation spaces.

    Per manuscript Section VIII.K (Table IV, cross-representation section):
    Compute MMD and Sinkhorn in VAE, PCA, and raw spaces for the same
    subset indices.  Report Spearman \u03c1 between every pair of spaces.

    Parameters
    ----------
    X_vae : np.ndarray
        VAE latent embeddings, shape (N, d_vae).
    X_pca : np.ndarray
        PCA embeddings, shape (N, d_pca).
    X_raw : np.ndarray
        Raw features, shape (N, d_raw).
    pareto_indices : List[np.ndarray]
        List of index arrays for each candidate solution.
    bandwidth_vae, bandwidth_pca, bandwidth_raw : float
        Kernel bandwidths for each space.
    rff_dim : int
        RFF dimension for MMD.
    n_anchors : int
        Anchor count for Sinkhorn.
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Cross-space objective values and pairwise Spearman correlations.
    """
    spaces = {
        "vae": (X_vae, bandwidth_vae),
        "pca": (X_pca, bandwidth_pca),
        "raw": (X_raw, bandwidth_raw),
    }

    objectives: Dict[str, Dict[str, list]] = {}
    for space_name, (X, bw) in spaces.items():
        mmd_vals = [
            compute_rff_mmd(X, idx, rff_dim, bw, seed)
            for idx in pareto_indices
        ]
        sink_vals = [
            compute_anchored_sinkhorn(X, idx, n_anchors, seed=seed)
            for idx in pareto_indices
        ]
        objectives[space_name] = {
            "mmd": mmd_vals,
            "sinkhorn": sink_vals,
        }

    # Compute cross-space correlations for both MMD and Sinkhorn
    correlations: Dict[str, Dict[str, Dict[str, float]]] = {
        "mmd": {},
        "sinkhorn": {},
    }
    space_names = list(spaces.keys())

    for i, s1 in enumerate(space_names):
        for s2 in space_names[i + 1:]:
            pair_key = f"{s1}_vs_{s2}"

            # MMD correlation
            v1 = np.array(objectives[s1]["mmd"])
            v2 = np.array(objectives[s2]["mmd"])
            if len(v1) > 2:
                rho, pval = stats.spearmanr(v1, v2)
                correlations["mmd"][pair_key] = {
                    "spearman_rho": float(rho),
                    "spearman_pval": float(pval),
                }

            # Sinkhorn correlation
            v1 = np.array(objectives[s1]["sinkhorn"])
            v2 = np.array(objectives[s2]["sinkhorn"])
            if len(v1) > 2:
                rho, pval = stats.spearmanr(v1, v2)
                correlations["sinkhorn"][pair_key] = {
                    "spearman_rho": float(rho),
                    "spearman_pval": float(pval),
                }

    return {
        "objectives": objectives,
        "correlations": correlations,
    }


# =============================================================================
# Objective-Metric Alignment (Section VIII.K -- Fig 4)
# =============================================================================


def objective_metric_alignment(
    objective_values: Dict[str, np.ndarray],
    metric_values: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """
    Compute Spearman correlations between objectives and downstream metrics.

    Per manuscript Section VIII.K and Figure 4:
    For each (objective, metric) pair, compute Spearman rank correlation
    to assess alignment between optimization objectives and evaluation
    metrics.

    Objectives: {f_MMD, f_SD, optionally f_SKL}
    Metrics: {e_Nys, e_kPCA, RMSE_4G, RMSE_5G, geo_kl, geo_l1}

    Parameters
    ----------
    objective_values : Dict[str, np.ndarray]
        Objective values keyed by objective name.
        Each array has shape (n_solutions,).
    metric_values : Dict[str, np.ndarray]
        Metric values keyed by metric name.
        Each array has shape (n_solutions,).

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Optional[float]]]]
        Nested dict: objective -> metric -> {spearman_rho, spearman_pval}.
    """
    alignment: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    for obj_name, obj_vals in objective_values.items():
        alignment[obj_name] = {}
        obj_arr = np.asarray(obj_vals, dtype=np.float64)

        for met_name, met_vals in metric_values.items():
            met_arr = np.asarray(met_vals, dtype=np.float64)

            if len(obj_arr) != len(met_arr):
                continue
            if len(obj_arr) < 3:
                alignment[obj_name][met_name] = {
                    "spearman_rho": None,
                    "spearman_pval": None,
                }
                continue

            # Remove any NaN values
            valid = ~(np.isnan(obj_arr) | np.isnan(met_arr))
            if valid.sum() < 3:
                alignment[obj_name][met_name] = {
                    "spearman_rho": None,
                    "spearman_pval": None,
                }
                continue

            rho, pval = stats.spearmanr(obj_arr[valid], met_arr[valid])
            alignment[obj_name][met_name] = {
                "spearman_rho": float(rho),
                "spearman_pval": float(pval),
            }

    return alignment


def compute_alignment_heatmap_data(
    pareto_F: np.ndarray,
    objective_names: Tuple[str, ...],
    metrics_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Prepare data for objective\u2013metric alignment heatmap (Fig 4).

    Per manuscript Figure 4:
    Returns a correlation matrix suitable for annotated heatmap
    visualization with a diverging colormap centered at 0.

    Parameters
    ----------
    pareto_F : np.ndarray
        Objective values, shape (n_solutions, n_objectives).
    objective_names : Tuple[str, ...]
        Names of objectives (e.g., ("mmd", "sinkhorn", "skl")).
    metrics_dict : Dict[str, np.ndarray]
        Metric values keyed by metric name.

    Returns
    -------
    Tuple[np.ndarray, List[str], List[str]]
        (correlation_matrix, row_labels, col_labels)
    """
    n_obj = pareto_F.shape[1]
    metric_names = list(metrics_dict.keys())
    n_met = len(metric_names)

    corr_matrix = np.zeros((n_obj, n_met))

    for i, obj_name in enumerate(objective_names):
        obj_vals = pareto_F[:, i]
        for j, met_name in enumerate(metric_names):
            met_vals = metrics_dict[met_name]

            valid = ~(np.isnan(obj_vals) | np.isnan(met_vals))
            if valid.sum() >= 3:
                rho, _ = stats.spearmanr(obj_vals[valid], met_vals[valid])
                corr_matrix[i, j] = rho
            else:
                corr_matrix[i, j] = np.nan

    return corr_matrix, list(objective_names), metric_names
