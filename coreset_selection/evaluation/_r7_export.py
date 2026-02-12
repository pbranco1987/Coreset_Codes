"""
R11 Post-hoc Diagnostics -- Export, IO, and orchestration.

Split from ``r7_diagnostics.py`` for maintainability.
Public API is re-exported via ``r7_diagnostics.py`` (facade).
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._r7_objectives import (
    SurrogateSensitivityConfig,
    compute_anchored_sinkhorn,
    compute_rff_mmd,
)
from ._r7_analysis import (
    cross_space_evaluation,
    objective_metric_alignment,
    surrogate_sensitivity_analysis,
)


# =============================================================================
# Repair Activity Diagnostics (Section VIII)
# =============================================================================


@dataclass
class RepairDiagnostics:
    """
    Aggregated repair activity statistics for R11 reporting.

    Per manuscript Section VIII:
    Reports fraction of offspring requiring repair and
    distribution of Hamming distances.

    Attributes
    ----------
    total_generations : int
        Total number of generations.
    total_offspring : int
        Total offspring processed across all generations.
    repaired_count : int
        Number of offspring requiring repair.
    repaired_fraction : float
        Fraction of offspring requiring repair.
    hamming_mean : float
        Mean Hamming distance for repaired offspring.
    hamming_std : float
        Standard deviation of Hamming distances.
    hamming_median : float
        Median Hamming distance.
    hamming_q25 : float
        25th percentile of Hamming distances.
    hamming_q75 : float
        75th percentile of Hamming distances.
    hamming_max : int
        Maximum Hamming distance observed.
    hamming_histogram : Dict[str, int]
        Histogram of Hamming distances by bin.
    """
    total_generations: int
    total_offspring: int
    repaired_count: int
    repaired_fraction: float
    hamming_mean: float
    hamming_std: float
    hamming_median: float
    hamming_q25: float
    hamming_q75: float
    hamming_max: int
    hamming_histogram: Dict[str, int] = field(default_factory=dict)


def aggregate_repair_diagnostics(
    tracker_summaries: List[Dict[str, Any]],
) -> RepairDiagnostics:
    """
    Aggregate repair activity from multiple runs/generations.

    Parameters
    ----------
    tracker_summaries : List[Dict[str, Any]]
        List of RepairActivityTracker.summary() outputs.

    Returns
    -------
    RepairDiagnostics
        Aggregated statistics.
    """
    total_offspring = sum(s.get("total_offspring", 0) for s in tracker_summaries)
    repaired_count = sum(s.get("repaired_count", 0) for s in tracker_summaries)

    # Collect all Hamming distances
    all_hd: List[int] = []
    for s in tracker_summaries:
        hd_list = s.get("hamming_distances", [])
        if isinstance(hd_list, list):
            all_hd.extend(hd_list)

    if not all_hd:
        return RepairDiagnostics(
            total_generations=len(tracker_summaries),
            total_offspring=total_offspring,
            repaired_count=repaired_count,
            repaired_fraction=0.0,
            hamming_mean=0.0,
            hamming_std=0.0,
            hamming_median=0.0,
            hamming_q25=0.0,
            hamming_q75=0.0,
            hamming_max=0,
            hamming_histogram={},
        )

    hd_arr = np.array(all_hd)

    # Compute histogram bins
    bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
    bin_labels = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    hist_counts: Dict[str, int] = {}
    for i in range(len(bins) - 1):
        mask = (hd_arr >= bins[i]) & (hd_arr < bins[i + 1])
        hist_counts[bin_labels[i]] = int(mask.sum())

    return RepairDiagnostics(
        total_generations=len(tracker_summaries),
        total_offspring=total_offspring,
        repaired_count=repaired_count,
        repaired_fraction=repaired_count / max(1, total_offspring),
        hamming_mean=float(np.mean(hd_arr)),
        hamming_std=float(np.std(hd_arr)),
        hamming_median=float(np.median(hd_arr)),
        hamming_q25=float(np.percentile(hd_arr, 25)),
        hamming_q75=float(np.percentile(hd_arr, 75)),
        hamming_max=int(np.max(hd_arr)),
        hamming_histogram=hist_counts,
    )


# =============================================================================
# CSV Export Utilities -- Phase 7 Deliverables
# =============================================================================


def export_proxy_stability_csv(
    surrogate_results: Dict[str, Any],
    cross_space_results: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Export proxy stability diagnostics as a structured CSV (Table IV data).

    The CSV has columns::

        section, parameter, comparison, objective, spearman_rho, spearman_pval

    Three sections match manuscript Table IV:
    1. ``rff_sweep``    -- MMD RFF dimension sweep (m vs reference m=2000)
    2. ``anchor_sweep`` -- Sinkhorn anchor count sweep (A vs reference A=200)
    3. ``cross_repr``   -- Cross-representation correlations (VAE-vs-raw, ...)

    Parameters
    ----------
    surrogate_results : Dict[str, Any]
        Output of :func:`surrogate_sensitivity_analysis`.
    cross_space_results : Dict[str, Any]
        Output of :func:`cross_space_evaluation`.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        The *output_path* written.
    """
    rows: List[Dict[str, Any]] = []

    ref_rff = surrogate_results.get("reference", {}).get("rff_dim", 2000)
    ref_anc = surrogate_results.get("reference", {}).get("n_anchors", 200)

    # --- Section 1: RFF dimension sweep ---
    for m_str, data in surrogate_results.get("rff_sensitivity", {}).items():
        m = int(m_str)
        rows.append({
            "section": "rff_sweep",
            "parameter": f"m={m}",
            "comparison": f"m={m} vs m={ref_rff}",
            "objective": "MMD",
            "spearman_rho": data.get("spearman_rho"),
            "spearman_pval": data.get("spearman_pval"),
        })

    # --- Section 2: Anchor count sweep ---
    for a_str, data in surrogate_results.get("anchor_sensitivity", {}).items():
        A = int(a_str)
        rows.append({
            "section": "anchor_sweep",
            "parameter": f"A={A}",
            "comparison": f"A={A} vs A={ref_anc}",
            "objective": "Sinkhorn",
            "spearman_rho": data.get("spearman_rho"),
            "spearman_pval": data.get("spearman_pval"),
        })

    # --- Section 3: Cross-representation ---
    corr = cross_space_results.get("correlations", {})
    for obj_key in ("mmd", "sinkhorn"):
        obj_label = "MMD" if obj_key == "mmd" else "Sinkhorn"
        for pair_name, pair_data in corr.get(obj_key, {}).items():
            rows.append({
                "section": "cross_repr",
                "parameter": pair_name,
                "comparison": pair_name.replace("_", " "),
                "objective": obj_label,
                "spearman_rho": pair_data.get("spearman_rho"),
                "spearman_pval": pair_data.get("spearman_pval"),
            })

    # Write CSV
    fieldnames = [
        "section", "parameter", "comparison", "objective",
        "spearman_rho", "spearman_pval",
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_path


def export_objective_metric_alignment_csv(
    alignment_results: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    output_path: str,
) -> str:
    """
    Export objective\u2013metric alignment as a structured CSV (Fig 4 data).

    The CSV has columns::

        objective, metric, spearman_rho, spearman_pval

    Each row is one (objective, metric) Spearman correlation suitable
    for rendering as an annotated heatmap (Fig 4).

    Parameters
    ----------
    alignment_results : Dict[str, Dict[str, Dict[str, Optional[float]]]]
        Output of :func:`objective_metric_alignment`.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        The *output_path* written.
    """
    rows: List[Dict[str, Any]] = []

    for obj_name, metrics in alignment_results.items():
        for met_name, corr_data in metrics.items():
            rows.append({
                "objective": obj_name,
                "metric": met_name,
                "spearman_rho": corr_data.get("spearman_rho"),
                "spearman_pval": corr_data.get("spearman_pval"),
            })

    fieldnames = ["objective", "metric", "spearman_rho", "spearman_pval"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output_path


def export_alignment_heatmap_csv(
    corr_matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    output_path: str,
) -> str:
    """
    Export the correlation matrix as a pivot-table CSV (Fig 4 alternative).

    Rows = objectives, columns = metrics, cells = Spearman \u03c1.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Shape (n_objectives, n_metrics).
    row_labels : List[str]
        Objective names.
    col_labels : List[str]
        Metric names.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        The *output_path* written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["objective"] + col_labels)
        for i, obj_name in enumerate(row_labels):
            row_vals = [
                f"{corr_matrix[i, j]:.4f}" if not np.isnan(corr_matrix[i, j]) else ""
                for j in range(len(col_labels))
            ]
            writer.writerow([obj_name] + row_vals)

    return output_path


# =============================================================================
# Candidate Pool Construction -- Load R10 Baseline Indices
# =============================================================================


def load_baseline_indices_from_dir(
    results_dir: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    Scan *results_dir* for saved coreset index files and return them.

    The runner saves baseline coresets as ``coreset_<method>_<space>_<regime>.npz``
    with an ``indices`` array inside.  This function globs for those files.

    Parameters
    ----------
    results_dir : str
        Path to a rep's ``results/`` directory.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of (label, indices_array) pairs.
    """
    found: Dict[str, np.ndarray] = {}

    # Glob patterns covering the formats produced by the runner's saver
    patterns = [
        os.path.join(results_dir, "coreset_*.npz"),
    ]
    # Also look for direct method_space_regime patterns
    for space in ("raw", "vae", "pca"):
        patterns.append(os.path.join(results_dir, f"*_{space}_*.npz"))

    for pat in patterns:
        for fpath in glob.glob(pat):
            bname = os.path.basename(fpath)
            if bname in found:
                continue
            try:
                data = np.load(fpath, allow_pickle=True)
                # Standard saver format: "indices" key
                if "indices" in data:
                    found[bname] = np.asarray(data["indices"], dtype=int)
                # Fallback: first integer array in the archive
                elif len(data.files) >= 1:
                    key0 = data.files[0]
                    arr = np.asarray(data[key0])
                    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                        found[bname] = arr
            except Exception:
                continue

    return [(label, idx) for label, idx in found.items()]


def find_r10_results_dir(
    output_dir: str,
    rep_id: int = 0,
) -> Optional[str]:
    """
    Locate the R10 results directory for a given replicate.

    Searches for directories matching ``R10*/rep{rep_id:02d}/results``
    under *output_dir*.

    Parameters
    ----------
    output_dir : str
        Top-level experiment output directory.
    rep_id : int
        Replicate index.

    Returns
    -------
    Optional[str]
        Path to the R10 results directory, or None if not found.
    """
    # Try exact "R10" first, then "R10_k300", then any R10*
    candidates = [
        os.path.join(output_dir, "R10", f"rep{rep_id:02d}", "results"),
        os.path.join(output_dir, "R10_k300", f"rep{rep_id:02d}", "results"),
    ]
    # Glob fallback
    for g in glob.glob(os.path.join(output_dir, "R10*", f"rep{rep_id:02d}", "results")):
        if g not in candidates:
            candidates.append(g)

    for cand in candidates:
        if os.path.isdir(cand):
            return cand

    return None


# =============================================================================
# R11 Results Dataclass
# =============================================================================


@dataclass
class R11Results:
    """
    Complete R11 diagnostic results bundle.

    Attributes
    ----------
    surrogate_sensitivity : Optional[Dict]
        Results from :func:`surrogate_sensitivity_analysis`.
    cross_space : Optional[Dict]
        Results from :func:`cross_space_evaluation`.
    objective_metric_alignment : Optional[Dict]
        Results from :func:`objective_metric_alignment`.
    repair_diagnostics : Optional[RepairDiagnostics]
        Aggregated repair statistics.
    proxy_stability_csv : Optional[str]
        Path to the exported ``proxy_stability.csv``.
    objective_metric_alignment_csv : Optional[str]
        Path to the exported ``objective_metric_alignment.csv``.
    alignment_heatmap_csv : Optional[str]
        Path to the exported ``objective_metric_alignment_heatmap.csv``.
    """
    surrogate_sensitivity: Optional[Dict] = None
    cross_space: Optional[Dict] = None
    objective_metric_alignment: Optional[Dict] = None
    repair_diagnostics: Optional[RepairDiagnostics] = None
    proxy_stability_csv: Optional[str] = None
    objective_metric_alignment_csv: Optional[str] = None
    alignment_heatmap_csv: Optional[str] = None


# Backward-compatible alias (was R6Results before Phase 7 rename)
R6Results = R11Results


# =============================================================================
# R11 Runner: Orchestrate All Diagnostics
# =============================================================================


def run_r7_diagnostics(
    X_vae: np.ndarray,
    X_pca: np.ndarray,
    X_raw: np.ndarray,
    pareto_F: np.ndarray,
    pareto_X: np.ndarray,
    objective_names: Tuple[str, ...],
    metrics_dict: Dict[str, np.ndarray],
    tracker_summaries: Optional[List[Dict]] = None,
    bandwidth_vae: Optional[float] = None,
    bandwidth_pca: Optional[float] = None,
    bandwidth_raw: Optional[float] = None,
    seed: int = 0,
    output_dir: Optional[str] = None,
    baseline_indices: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> R11Results:
    """
    Run all R11 post-hoc diagnostics.

    This is the main orchestrator called by the experiment runner for R11.
    It performs:

    1. Surrogate sensitivity analysis (RFF sweep + anchor sweep) -> Table IV S1-2
    2. Cross-space objective re-evaluation (VAE vs PCA vs raw) -> Table IV S3
    3. Objective-metric alignment (Spearman correlations) -> Fig 4
    4. Repair diagnostics (if tracker summaries provided)

    When *output_dir* is given, structured CSV deliverables are written:
      - ``proxy_stability.csv``                    (Table IV data)
      - ``objective_metric_alignment.csv``          (Fig 4 long-form)
      - ``objective_metric_alignment_heatmap.csv``  (Fig 4 pivot-table)

    Parameters
    ----------
    X_vae : np.ndarray
        VAE embeddings (N, d_vae).
    X_pca : np.ndarray
        PCA embeddings (N, d_pca).
    X_raw : np.ndarray
        Raw features (N, d_raw).
    pareto_F : np.ndarray
        Pareto front objectives (n_pareto, n_objectives).
    pareto_X : np.ndarray
        Pareto front decision variables (n_pareto, N) boolean.
    objective_names : Tuple[str, ...]
        Names of objectives (e.g., ("mmd", "sinkhorn")).
    metrics_dict : Dict[str, np.ndarray]
        Downstream metric values per Pareto solution.
    tracker_summaries : Optional[List[Dict]]
        Repair tracker summaries for repair diagnostics.
    bandwidth_vae, bandwidth_pca, bandwidth_raw : Optional[float]
        Kernel bandwidths (computed via median heuristic if None).
    seed : int
        Random seed.
    output_dir : Optional[str]
        Directory for CSV deliverables. If None, no CSVs are written.
    baseline_indices : Optional[List[Tuple[str, np.ndarray]]]
        Additional baseline subset indices from R10 to include in the
        candidate pool for the objective-metric alignment analysis.

    Returns
    -------
    R11Results
        Complete diagnostic results including CSV file paths.
    """
    from ..utils.math import median_sq_dist

    # Convert Pareto decision variables to index lists
    pareto_indices = [
        np.where(pareto_X[i])[0] for i in range(pareto_X.shape[0])
    ]

    # Merge baseline indices into the candidate pool if provided
    all_indices = list(pareto_indices)
    baseline_labels: List[str] = []
    if baseline_indices:
        for label, idx in baseline_indices:
            all_indices.append(np.asarray(idx, dtype=int))
            baseline_labels.append(label)

    # Compute bandwidths if not provided (median heuristic, sigma^2 = median(d^2)/2)
    if bandwidth_vae is None:
        bandwidth_vae = median_sq_dist(X_vae, sample_size=2048, seed=seed) / 2.0
    if bandwidth_pca is None:
        bandwidth_pca = median_sq_dist(X_pca, sample_size=2048, seed=seed) / 2.0
    if bandwidth_raw is None:
        bandwidth_raw = median_sq_dist(X_raw, sample_size=2048, seed=seed) / 2.0

    results = R11Results()

    # ------------------------------------------------------------------
    # 1. Surrogate sensitivity analysis (Table IV, sections 1-2)
    # ------------------------------------------------------------------
    config = SurrogateSensitivityConfig()
    results.surrogate_sensitivity = surrogate_sensitivity_analysis(
        X_vae, all_indices, config, bandwidth_vae, seed
    )

    # ------------------------------------------------------------------
    # 2. Cross-space evaluation (Table IV, section 3)
    # ------------------------------------------------------------------
    results.cross_space = cross_space_evaluation(
        X_vae, X_pca, X_raw, all_indices,
        bandwidth_vae, bandwidth_pca, bandwidth_raw,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 3. Objective-metric alignment (Fig 4)
    # ------------------------------------------------------------------
    n_pareto = pareto_F.shape[0]
    n_total = len(all_indices)

    # Extend objective arrays: re-evaluate objectives for baseline solutions
    # in VAE space to make them comparable with Pareto-front objectives.
    extended_objectives: Dict[str, np.ndarray] = {}
    for i, obj_name in enumerate(objective_names):
        obj_vals = np.full(n_total, np.nan, dtype=np.float64)
        obj_vals[:n_pareto] = pareto_F[:, i]

        # Re-evaluate objectives for baseline indices
        if n_total > n_pareto:
            for j in range(n_pareto, n_total):
                idx = all_indices[j]
                if obj_name in ("mmd",):
                    try:
                        obj_vals[j] = compute_rff_mmd(
                            X_vae, idx, config.reference_rff,
                            bandwidth_vae, seed,
                        )
                    except Exception:
                        pass
                elif obj_name in ("sinkhorn", "sd"):
                    try:
                        obj_vals[j] = compute_anchored_sinkhorn(
                            X_vae, idx, config.reference_anchors, seed=seed,
                        )
                    except Exception:
                        pass
                # SKL is not re-evaluated for baselines (optional ablation only)
        extended_objectives[obj_name] = obj_vals

    # Extend metrics_dict for baseline solutions.  If the runner has already
    # appended baseline metrics the lengths match; otherwise pad with NaN.
    extended_metrics: Dict[str, np.ndarray] = {}
    for met_name, met_vals in metrics_dict.items():
        met_arr = np.asarray(met_vals, dtype=np.float64)
        if len(met_arr) < n_total:
            padded = np.full(n_total, np.nan, dtype=np.float64)
            padded[:len(met_arr)] = met_arr
            extended_metrics[met_name] = padded
        else:
            extended_metrics[met_name] = met_arr[:n_total]

    results.objective_metric_alignment = objective_metric_alignment(
        extended_objectives, extended_metrics
    )

    # ------------------------------------------------------------------
    # 4. Repair diagnostics
    # ------------------------------------------------------------------
    if tracker_summaries:
        results.repair_diagnostics = aggregate_repair_diagnostics(
            tracker_summaries
        )

    # ------------------------------------------------------------------
    # 5. Export CSV deliverables (Phase 7 requirement)
    # ------------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # proxy_stability.csv (Table IV)
        proxy_csv_path = os.path.join(output_dir, "proxy_stability.csv")
        results.proxy_stability_csv = export_proxy_stability_csv(
            surrogate_results=results.surrogate_sensitivity,
            cross_space_results=results.cross_space,
            output_path=proxy_csv_path,
        )

        # objective_metric_alignment.csv (Fig 4 -- long-form)
        alignment_csv_path = os.path.join(
            output_dir, "objective_metric_alignment.csv"
        )
        results.objective_metric_alignment_csv = (
            export_objective_metric_alignment_csv(
                alignment_results=results.objective_metric_alignment,
                output_path=alignment_csv_path,
            )
        )

        # Heatmap pivot-table CSV (Fig 4 -- wide-form)
        heatmap_csv_path = os.path.join(
            output_dir, "objective_metric_alignment_heatmap.csv"
        )
        obj_names_list = list(extended_objectives.keys())
        met_names_list = list(extended_metrics.keys())
        n_obj = len(obj_names_list)
        n_met = len(met_names_list)
        corr_matrix = np.full((n_obj, n_met), np.nan)
        for ii, oname in enumerate(obj_names_list):
            for jj, mname in enumerate(met_names_list):
                entry = results.objective_metric_alignment.get(oname, {}).get(mname, {})
                rho = entry.get("spearman_rho")
                if rho is not None:
                    corr_matrix[ii, jj] = rho
        results.alignment_heatmap_csv = export_alignment_heatmap_csv(
            corr_matrix=corr_matrix,
            row_labels=obj_names_list,
            col_labels=met_names_list,
            output_path=heatmap_csv_path,
        )

    return results
