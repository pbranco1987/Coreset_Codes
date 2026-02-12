"""Manuscript-specific table generators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._tables_core import format_mean_std, generate_latex_table


def klmin_summary_table(
    k_values: List[int],
    klmin_values: List[float],
    quota_counts: List[np.ndarray],
    group_names: List[str],
) -> str:
    """
    Generate KL_min summary table (tab:klmin-summary).

    Per manuscript Section 6.3, summarizes Algorithm 1 outputs.

    Parameters
    ----------
    k_values : List[int]
        Cardinality values
    klmin_values : List[float]
        KL_min(k) for each k
    quota_counts : List[np.ndarray]
        c*(k) quota vectors for each k
    group_names : List[str]
        Names of geographic groups

    Returns
    -------
    str
        LaTeX table code
    """
    # Summary table with KL_min and basic stats
    headers = ["$k$", "$\\mathrm{KL}_{\\min}(k)$", "Min quota", "Max quota", "Groups at 1"]
    data = []

    for i, k in enumerate(k_values):
        counts = quota_counts[i]
        data.append([
            str(k),
            f"{klmin_values[i]:.4f}",
            str(int(counts.min())),
            str(int(counts.max())),
            str(int((counts == 1).sum())),
        ])

    return generate_latex_table(
        data, headers,
        caption="KL-guided quota summary across cardinalities",
        label="tab:klmin-summary",
        column_format="r|r|r|r|r",
    )


def front_stats_table(
    run_results: Dict[str, Dict],
    objective_names: Tuple[str, ...] = ("SKL", "MMD", "Sinkhorn"),
) -> str:
    """
    Generate Pareto front statistics table (tab:front-stats).

    Per manuscript Section 6.4, reports front size, min/max per objective,
    and hypervolume for each configuration.

    Parameters
    ----------
    run_results : Dict[str, Dict]
        Results keyed by run_id, each containing:
        - 'pareto_F': Pareto front objectives (n_solutions, n_obj)
        - 'hypervolume': Hypervolume indicator value
    objective_names : Tuple[str, ...]
        Names of objectives

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Run", "|P*|"] + \
              [f"${o}^{{\\min}}$" for o in objective_names] + \
              [f"${o}^{{\\max}}$" for o in objective_names] + \
              ["HV"]

    data = []
    for run_id, res in run_results.items():
        F = res.get('pareto_F')
        if F is None:
            continue

        row = [run_id, str(F.shape[0])]

        # Min values
        for j in range(F.shape[1]):
            row.append(f"{F[:, j].min():.4f}")

        # Max values
        for j in range(F.shape[1]):
            row.append(f"{F[:, j].max():.4f}")

        # Hypervolume
        hv = res.get('hypervolume', np.nan)
        row.append(f"{hv:.4f}" if not np.isnan(hv) else "--")

        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Pareto front statistics",
        label="tab:front-stats",
        column_format="l|r|" + "r" * len(objective_names) + "|" + "r" * len(objective_names) + "|r",
    )


def cardinality_metrics_table(
    k_values: List[int],
    metrics_by_k: Dict[int, Dict[str, Tuple[float, float]]],
    metric_names: List[str] = None,
) -> str:
    """
    Generate cardinality vs metrics table (tab:cardinality-metrics).

    Per manuscript Section 6.4, shows how metrics vary with k.

    Parameters
    ----------
    k_values : List[int]
        Cardinality values
    metrics_by_k : Dict[int, Dict[str, Tuple[float, float]]]
        For each k, dict of metric_name -> (mean, std)
    metric_names : List[str]
        Metrics to include (if None, use all)

    Returns
    -------
    str
        LaTeX table code
    """
    if metric_names is None:
        # Get all metrics from first k
        first_k = list(metrics_by_k.keys())[0]
        metric_names = list(metrics_by_k[first_k].keys())

    headers = ["$k$"] + [m.replace("_", " ") for m in metric_names]
    data = []

    for k in k_values:
        if k not in metrics_by_k:
            continue
        row = [str(k)]
        for m in metric_names:
            if m in metrics_by_k[k]:
                mean, std = metrics_by_k[k][m]
                row.append(format_mean_std(mean, std, precision=4))
            else:
                row.append("--")
        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Downstream metrics across coreset sizes",
        label="tab:cardinality-metrics",
        column_format="r|" + "r" * len(metric_names),
    )


def objective_ablations_table(
    ablation_results: Dict[str, Dict[str, Tuple[float, float]]],
    metric_names: List[str],
) -> str:
    """
    Generate objective ablation table (tab:objective-ablations).

    Per manuscript Section 6.5, compares R1 vs R3-R5 (removing one objective).

    Parameters
    ----------
    ablation_results : Dict[str, Dict[str, Tuple[float, float]]]
        Results keyed by configuration name (e.g., "R1 (full)", "R3 (no SKL)")
        Each contains metric_name -> (mean, std)
    metric_names : List[str]
        Metrics to report

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Configuration", "Objectives"] + [m.replace("_", " ") for m in metric_names]

    config_objectives = {
        "R1": "SKL, MMD, Sink",
        "R3": "MMD, Sink",
        "R4": "SKL, MMD",
        "R5": "SKL, Sink",
    }

    data = []
    for config_name, metrics in ablation_results.items():
        # Extract run id (R1, R3, etc.)
        run_id = config_name.split()[0] if " " in config_name else config_name
        objectives = config_objectives.get(run_id, "--")

        row = [config_name, objectives]
        for m in metric_names:
            if m in metrics:
                mean, std = metrics[m]
                row.append(format_mean_std(mean, std, precision=4))
            else:
                row.append("--")
        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Objective ablation study results",
        label="tab:objective-ablations",
        column_format="l|l|" + "r" * len(metric_names),
    )


def representation_transfer_table(
    transfer_results: Dict[str, Dict[str, Tuple[float, float]]],
    metric_names: List[str],
) -> str:
    """
    Generate representation transfer table (tab:repr-transfer).

    Per manuscript Section 6.6, compares VAE vs PCA vs raw space selection.

    Parameters
    ----------
    transfer_results : Dict[str, Dict[str, Tuple[float, float]]]
        Results keyed by space name ("VAE (R1)", "PCA (R7)", "Raw (R8)")
        Each contains metric_name -> (mean, std)
    metric_names : List[str]
        Metrics to report

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Space", "Objectives"] + [m.replace("_", " ") for m in metric_names]

    space_info = {
        "VAE": ("R1", "SKL, MMD, Sink"),
        "PCA": ("R7", "MMD, Sink"),
        "Raw": ("R8", "MMD, Sink"),
    }

    data = []
    for space_name, metrics in transfer_results.items():
        base_space = space_name.split()[0] if " " in space_name else space_name
        run_id, objectives = space_info.get(base_space, ("--", "--"))

        row = [f"{base_space} ({run_id})", objectives]
        for m in metric_names:
            if m in metrics:
                mean, std = metrics[m]
                row.append(format_mean_std(mean, std, precision=4))
            else:
                row.append("--")
        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Representation transfer comparison",
        label="tab:repr-transfer",
        column_format="l|l|" + "r" * len(metric_names),
    )


def surrogate_sensitivity_table(
    sensitivity_results: Dict[str, Dict],
) -> str:
    """
    Generate surrogate sensitivity table (tab:surrogate-sensitivity).

    Per manuscript Section 6.7.1, reports rank correlations vs reference.

    Parameters
    ----------
    sensitivity_results : Dict[str, Dict]
        Results from surrogate_sensitivity_analysis()

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Setting", "Parameter", "Spearman $\\rho$", "$p$-value"]
    data = []

    # RFF sensitivity
    rff_sens = sensitivity_results.get("rff_sensitivity", {})
    for m, vals in sorted(rff_sens.items()):
        rho = vals.get("spearman_rho")
        pval = vals.get("spearman_pval")
        data.append([
            "RFF dim",
            f"$m = {m}$",
            f"{rho:.3f}" if rho is not None else "--",
            f"{pval:.3e}" if pval is not None else "--",
        ])

    # Anchor sensitivity
    anchor_sens = sensitivity_results.get("anchor_sensitivity", {})
    for A, vals in sorted(anchor_sens.items()):
        rho = vals.get("spearman_rho")
        pval = vals.get("spearman_pval")
        data.append([
            "Anchors",
            f"$A = {A}$",
            f"{rho:.3f}" if rho is not None else "--",
            f"{pval:.3e}" if pval is not None else "--",
        ])

    return generate_latex_table(
        data, headers,
        caption="Surrogate sensitivity analysis (vs. reference $m=2000$, $A=200$)",
        label="tab:surrogate-sensitivity",
        column_format="l|l|r|r",
    )


def repair_activity_table(
    repair_diagnostics,
) -> str:
    """
    Generate repair activity table (tab:repair-activity).

    Per manuscript Section 6.8, reports repair statistics.

    Parameters
    ----------
    repair_diagnostics : RepairDiagnostics
        Aggregated repair statistics

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Statistic", "Value"]
    data = [
        ["Total offspring", str(repair_diagnostics.total_offspring)],
        ["Repaired count", str(repair_diagnostics.repaired_count)],
        ["Repair fraction", f"{repair_diagnostics.repaired_fraction:.2%}"],
        ["Hamming mean", f"{repair_diagnostics.hamming_mean:.2f}"],
        ["Hamming std", f"{repair_diagnostics.hamming_std:.2f}"],
        ["Hamming median", f"{repair_diagnostics.hamming_median:.2f}"],
        ["Hamming Q25", f"{repair_diagnostics.hamming_q25:.2f}"],
        ["Hamming Q75", f"{repair_diagnostics.hamming_q75:.2f}"],
        ["Hamming max", str(repair_diagnostics.hamming_max)],
    ]

    return generate_latex_table(
        data, headers,
        caption="Repair operator activity statistics",
        label="tab:repair-activity",
        column_format="l|r",
    )


def crossspace_objectives_table(
    cross_space_results: Dict[str, Any],
) -> str:
    """
    Generate cross-space objectives table (tab:crossspace-objectives).

    Per manuscript Section 6.7.2, reports correlations between spaces.

    Parameters
    ----------
    cross_space_results : Dict[str, Any]
        Results from cross_space_evaluation()

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Objective", "Space Pair", "Spearman $\\rho$", "$p$-value"]
    data = []

    correlations = cross_space_results.get("correlations", {})

    for obj_name in ["mmd", "sinkhorn"]:
        obj_corrs = correlations.get(obj_name, {})
        for pair, vals in obj_corrs.items():
            rho = vals.get("spearman_rho")
            pval = vals.get("spearman_pval")
            data.append([
                obj_name.upper(),
                pair.replace("_", " "),
                f"{rho:.3f}" if rho is not None else "--",
                f"{pval:.3e}" if pval is not None else "--",
            ])

    return generate_latex_table(
        data, headers,
        caption="Cross-space objective correlations",
        label="tab:crossspace-objectives",
        column_format="l|l|r|r",
    )


def baseline_quota_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    baseline_names: List[str],
    metric_names: List[str],
    k: int,
) -> str:
    """
    Generate baseline comparison table with quota constraints (tab:baseline-quota).

    Parameters
    ----------
    results : Dict[str, Dict[str, Tuple[float, float]]]
        Results keyed by method name
    baseline_names : List[str]
        Names of baseline methods
    metric_names : List[str]
        Metrics to report
    k : int
        Coreset size

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Method"] + [m.replace("_", " ") for m in metric_names]
    data = []

    # Add Pareto methods first
    pareto_methods = ["Pareto knee", "Pareto best MMD", "Pareto best Sink"]
    for method in pareto_methods:
        if method in results:
            row = [method]
            for m in metric_names:
                if m in results[method]:
                    mean, std = results[method][m]
                    row.append(format_mean_std(mean, std, precision=4))
                else:
                    row.append("--")
            data.append(row)

    # Add baselines
    for method in baseline_names:
        if method in results:
            row = [method.replace("_", " ").title()]
            for m in metric_names:
                if m in results[method]:
                    mean, std = results[method][m]
                    row.append(format_mean_std(mean, std, precision=4))
                else:
                    row.append("--")
            data.append(row)

    return generate_latex_table(
        data, headers,
        caption=f"Baseline comparison with quota constraints ($k={k}$)",
        label="tab:baseline-quota",
        column_format="l|" + "r" * len(metric_names),
    )


def baseline_unconstrained_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    baseline_names: List[str],
    metric_names: List[str],
    k: int,
) -> str:
    """
    Generate baseline comparison table without quota constraints (tab:baseline-unconstrained).

    Parameters
    ----------
    results : Dict[str, Dict[str, Tuple[float, float]]]
        Results keyed by method name
    baseline_names : List[str]
        Names of baseline methods
    metric_names : List[str]
        Metrics to report
    k : int
        Coreset size

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Method"] + [m.replace("_", " ") for m in metric_names]
    data = []

    # Add Pareto methods first (R2 ablation)
    pareto_methods = ["Pareto knee (R2)", "Pareto best MMD (R2)"]
    for method in pareto_methods:
        if method in results:
            row = [method]
            for m in metric_names:
                if m in results[method]:
                    mean, std = results[method][m]
                    row.append(format_mean_std(mean, std, precision=4))
                else:
                    row.append("--")
            data.append(row)

    # Add baselines
    for method in baseline_names:
        if method in results:
            row = [method.replace("_", " ").title()]
            for m in metric_names:
                if m in results[method]:
                    mean, std = results[method][m]
                    row.append(format_mean_std(mean, std, precision=4))
                else:
                    row.append("--")
            data.append(row)

    return generate_latex_table(
        data, headers,
        caption=f"Baseline comparison without quota constraints ($k={k}$)",
        label="tab:baseline-unconstrained",
        column_format="l|" + "r" * len(metric_names),
    )


def objective_metric_alignment_table(
    alignment_results: Dict[str, Dict[str, Dict[str, float]]],
) -> str:
    """
    Generate objective-metric alignment table.

    Per manuscript Section 6.7.3, shows Spearman correlations between
    objectives and downstream metrics.

    Parameters
    ----------
    alignment_results : Dict[str, Dict[str, Dict[str, float]]]
        From objective_metric_alignment()

    Returns
    -------
    str
        LaTeX table code
    """
    # Get all objectives and metrics
    objectives = list(alignment_results.keys())
    metrics = list(alignment_results[objectives[0]].keys()) if objectives else []

    headers = ["Objective"] + [m.replace("_", " ") for m in metrics]
    data = []

    for obj in objectives:
        row = [obj.upper()]
        for met in metrics:
            rho = alignment_results[obj].get(met, {}).get("spearman_rho")
            if rho is not None:
                row.append(f"{rho:.3f}")
            else:
                row.append("--")
        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Objective-metric alignment (Spearman $\\rho$)",
        label="tab:obj-metric-align",
        column_format="l|" + "r" * len(metrics),
    )
