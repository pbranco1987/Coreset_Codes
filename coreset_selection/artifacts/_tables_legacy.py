"""Legacy / basic table generators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._tables_core import format_mean_std, generate_latex_table


def results_summary_table(
    results_df,
    methods: List[str],
    metrics: List[str],
    k_values: Optional[List[int]] = None,
    precision: int = 3,
) -> Tuple[List[List[str]], List[str]]:
    """
    Generate summary table data from results dataframe.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results with columns: method, k, rep_id, {metrics}
    methods : List[str]
        Methods to include
    metrics : List[str]
        Metrics to include
    k_values : Optional[List[int]]
        k values to include. If None, uses all.
    precision : int
        Decimal precision

    Returns
    -------
    Tuple[List[List[str]], List[str]]
        (data_rows, headers)
    """
    import pandas as pd

    if k_values is None:
        k_values = sorted(results_df['k'].unique())

    headers = ["Method"] + [f"{m}" for m in metrics]
    data = []

    for method in methods:
        method_data = results_df[results_df['method'] == method]

        if k_values:
            method_data = method_data[method_data['k'].isin(k_values)]

        row = [method]

        for metric in metrics:
            if metric in method_data.columns:
                mean = method_data[metric].mean()
                std = method_data[metric].std()
                row.append(format_mean_std(mean, std, precision))
            else:
                row.append("--")

        data.append(row)

    return data, headers


def baseline_comparison_table(
    results_df,
    baseline_methods: List[str],
    pareto_methods: List[str],
    metrics: List[str],
    k: int,
    precision: int = 3,
) -> Tuple[str, str]:
    """
    Generate baseline comparison table in LaTeX and CSV.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results dataframe
    baseline_methods : List[str]
        Baseline method names
    pareto_methods : List[str]
        Pareto method names
    metrics : List[str]
        Metrics to compare
    k : int
        Coreset size
    precision : int
        Decimal precision

    Returns
    -------
    Tuple[str, str]
        (latex_table, csv_content)
    """
    # Filter to specific k
    df_k = results_df[results_df['k'] == k]

    all_methods = pareto_methods + baseline_methods
    headers = ["Method"] + [m.replace("_", " ").title() for m in metrics]

    data = []
    metric_values = {m: [] for m in metrics}

    for method in all_methods:
        method_data = df_k[df_k['method'] == method]
        row = [method.replace("_", " ").replace("baseline ", "").replace("pareto ", "Pareto ")]

        for metric in metrics:
            if metric in method_data.columns and len(method_data) > 0:
                mean = method_data[metric].mean()
                std = method_data[metric].std()
                metric_values[metric].append((method, mean))
                row.append(format_mean_std(mean, std, precision, show_pm=False))
            else:
                row.append("--")

        data.append(row)

    # Generate LaTeX
    latex = generate_latex_table(
        data, headers,
        caption=f"Method comparison at $k={k}$",
        label=f"tab:comparison_k{k}",
    )

    # Generate CSV content
    csv_lines = [",".join(headers)]
    for row in data:
        csv_lines.append(",".join(row))
    csv_content = "\n".join(csv_lines)

    return latex, csv_content


def run_specs_table(
    run_specs: Dict[str, Any],
) -> str:
    """
    Generate LaTeX table for run specifications (Table 4 in paper).

    Parameters
    ----------
    run_specs : Dict[str, Any]
        Dictionary of run specifications

    Returns
    -------
    str
        LaTeX table code
    """
    headers = ["Run", "k", "Geo", "Space", "Objectives", "Description"]
    data = []

    for run_id, spec in run_specs.items():
        row = [
            run_id,
            str(spec.k) if hasattr(spec, 'k') else "--",
            "yes" if getattr(spec, 'use_geo', True) else "no",
            getattr(spec, 'space', 'VAE'),
            ", ".join(getattr(spec, 'objectives', ['SKL', 'MMD', 'Sink'])),
            getattr(spec, 'description', ''),
        ]
        data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Experimental run configurations",
        label="tab:run_specs",
        column_format="l|c|c|c|l|l",
    )


def geographic_distribution_table(
    group_names: List[str],
    population_fractions: np.ndarray,
    selected_counts: np.ndarray,
    target_counts: Optional[np.ndarray] = None,
) -> str:
    """
    Generate table showing geographic distribution.

    Parameters
    ----------
    group_names : List[str]
        State/region names
    population_fractions : np.ndarray
        Population proportion per group
    selected_counts : np.ndarray
        Selected count per group
    target_counts : Optional[np.ndarray]
        Target quota counts

    Returns
    -------
    str
        LaTeX table code
    """
    k = selected_counts.sum()
    selected_fracs = selected_counts / k

    if target_counts is not None:
        headers = ["State", "Pop. \\%", "Target", "Selected", "Sel. \\%"]
        data = []
        for i, name in enumerate(group_names):
            row = [
                name,
                f"{population_fractions[i]*100:.1f}",
                str(int(target_counts[i])),
                str(int(selected_counts[i])),
                f"{selected_fracs[i]*100:.1f}",
            ]
            data.append(row)
    else:
        headers = ["State", "Pop. \\%", "Selected", "Sel. \\%"]
        data = []
        for i, name in enumerate(group_names):
            row = [
                name,
                f"{population_fractions[i]*100:.1f}",
                str(int(selected_counts[i])),
                f"{selected_fracs[i]*100:.1f}",
            ]
            data.append(row)

    return generate_latex_table(
        data, headers,
        caption="Geographic distribution of selected coreset",
        label="tab:geo_dist",
    )
