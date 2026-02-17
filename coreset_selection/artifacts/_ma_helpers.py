"""Shared helpers for the ManuscriptArtifacts mixin modules.

Provides common utilities (style setup, figure saving) used across the
various manuscript-artifact generation sub-modules.

R/ggplot2 integration:
    ``use_r()`` returns True when Rscript is available and not disabled.
    ``_save_r()`` renders a figure via R/ggplot2 with automatic fallback to
    matplotlib if R is unavailable or the script fails.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# IEEE publication-quality style (matplotlib)
# ---------------------------------------------------------------------------

def _set_style():
    """Set matplotlib style for IEEE-compliant manuscript figures.

    Enforces:
    - Base font size >= 10 pt (axes labels), >= 8 pt (annotations/legend).
    - Top/right spines removed for clean appearance.
    - Transparent grid at 25 % alpha.
    - 300 dpi saved output.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # Font sizes (IEEE requires >= 8 pt at final column width)
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        # Figure quality
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # Grid & spines
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })


def _save(fig, path: str) -> str:
    """Save figure to *path* at publication quality (300 dpi, tight bbox)."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# R / ggplot2 integration
# ---------------------------------------------------------------------------

def use_r() -> bool:
    """Return True if R/ggplot2 rendering should be used.

    Conditions for returning True:
    1. ``CORESET_FORCE_MATPLOTLIB`` env var is **not** set to ``1``.
    2. ``Rscript`` is found on PATH.

    The result is cached after the first call.
    """
    from ._ma_r_bridge import r_is_available
    return r_is_available()


def _save_r(
    script_name: str,
    data: pd.DataFrame,
    output_path: str,
    fallback_fn: Callable[[], str],
    extra_args: Optional[Dict[str, str]] = None,
) -> str:
    """Try rendering a figure via R/ggplot2; fall back to matplotlib.

    Parameters
    ----------
    script_name : str
        R script filename inside ``r_scripts/``, e.g. ``"fig_kl_floor_vs_k.R"``.
    data : pd.DataFrame
        Tidy-format data to pass to the R script as CSV.
    output_path : str
        Absolute path for the output PDF.
    fallback_fn : callable
        Zero-argument callable that renders the matplotlib version and returns
        the output path.  Called when R is unavailable or fails.
    extra_args : dict, optional
        Additional ``--key=value`` arguments forwarded to the R script.

    Returns
    -------
    str
        Path to the generated PDF.
    """
    if not use_r():
        return fallback_fn()

    from ._ma_r_bridge import run_r_figure, RScriptError

    try:
        return run_r_figure(
            script_name=script_name,
            data=data,
            output_pdf=output_path,
            extra_args=extra_args,
        )
    except (RScriptError, FileNotFoundError, TimeoutError, OSError) as exc:
        print(f"[ManuscriptArtifacts] R fallback for {script_name}: {exc}")
        return fallback_fn()


# ---------------------------------------------------------------------------
# Pareto front analysis helpers
# ---------------------------------------------------------------------------

import numpy as np
from typing import Any, List, Tuple


def get_knee_values(results_df: pd.DataFrame, run_pattern: str,
                    k: int = 300) -> Dict[str, float]:
    """Extract knee-point metric values from all_results.csv.

    Parameters
    ----------
    results_df : pd.DataFrame
        Concatenated ``all_results.csv`` (with ``rep_name`` column).
    run_pattern : str
        Substring match on ``run_id`` (e.g. ``"R1"``).
    k : int
        Cardinality filter.

    Returns
    -------
    dict
        Mapping ``{metric_name: float}`` for the knee point.
    """
    d = results_df[results_df["run_id"].astype(str).str.contains(run_pattern)].copy()
    d = d[d["k"].fillna(k).astype(int) == k]
    knee = d[d.get("rep_name", pd.Series(dtype=str)).astype(str) == "knee"]
    if knee.empty:
        knee = d  # fallback: use all rows
    if knee.empty:
        return {}
    # Return mean across seeds for each numeric column
    return {c: float(knee[c].mean()) for c in knee.columns
            if knee[c].dtype.kind in ("f", "i") and knee[c].notna().any()}


def get_best_per_metric(front_df: pd.DataFrame,
                        metrics: List[str]) -> Dict[str, pd.Series]:
    """For each metric, find the Pareto solution minimizing it.

    Parameters
    ----------
    front_df : pd.DataFrame
        Full front evaluation (from ``front_metrics_{space}.csv``).
    metrics : list of str
        Column names to minimize (e.g. ``["nystrom_error", "krr_rmse_4G"]``).

    Returns
    -------
    dict
        Mapping ``{metric_name: pd.Series}`` -- the full row of the
        minimizing solution.
    """
    result: Dict[str, pd.Series] = {}
    for m in metrics:
        if m not in front_df.columns or front_df[m].isna().all():
            continue
        idx = front_df[m].idxmin()
        result[m] = front_df.loc[idx]
    return result


# Metric-column alias resolution (shared across figures)
_METRIC_ALIASES: Dict[str, List[str]] = {
    "nystrom_error": ["nystrom_error", "nys_error", "e_nys"],
    "kpca_distortion": ["kpca_distortion", "e_kpca"],
    "krr_rmse_4G": ["krr_rmse_4G", "krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
    "krr_rmse_5G": ["krr_rmse_5G", "krr_rmse_cov_area_5G", "krr_rmse_area_5G"],
    "geo_kl": ["geo_kl", "geo_kl_pop"],
}


def resolve_metric(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Find the actual column name for a canonical metric."""
    if canonical in df.columns:
        return canonical
    for alias in _METRIC_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


def plot_pareto_scatter(
    ax,
    front_dfs: Dict[str, pd.DataFrame],
    colors: Dict[str, str],
    x_metric: str = "nystrom_error",
    y_metric: str = "krr_rmse_4G",
    knee_values: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_points: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_markers: Optional[Dict[str, str]] = None,
    x_label: str = r"$e_{\mathrm{Nys}}$",
    y_label: str = r"RMSE$_{4\mathrm{G}}$",
    title: str = "",
) -> None:
    """Plot overlaid Pareto fronts with knee and best-per-metric markers.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    front_dfs : dict
        ``{label: DataFrame}`` -- each DataFrame is the full Pareto front
        for one run/configuration.
    colors : dict
        ``{label: color}`` -- one color per configuration.
    x_metric, y_metric : str
        Canonical metric names for x and y axes.
    knee_values : dict, optional
        ``{label: {metric: value}}`` -- knee-point metric values per config.
    baseline_points : dict, optional
        ``{label: {metric: value}}`` -- single-point baselines to overlay.
    baseline_markers : dict, optional
        ``{label: marker_shape}`` -- marker style per baseline.
    x_label, y_label : str
        Axis labels (LaTeX-safe).
    title : str
        Axes title.
    """
    import matplotlib.pyplot as plt

    # --- Plot each Pareto front as a point cloud ---
    for label, fdf in front_dfs.items():
        x_col = resolve_metric(fdf, x_metric)
        y_col = resolve_metric(fdf, y_metric)
        if x_col is None or y_col is None:
            continue
        valid = fdf[[x_col, y_col]].dropna()
        if valid.empty:
            continue
        color = colors.get(label, "#999999")
        ax.scatter(valid[x_col], valid[y_col], s=18, alpha=0.45,
                   color=color, edgecolors="none", zorder=3,
                   label=f"{label} front")

        # --- Knee point (star) ---
        if knee_values and label in knee_values:
            kv = knee_values[label]
            kx = kv.get(x_col, kv.get(x_metric))
            ky = kv.get(y_col, kv.get(y_metric))
            if kx is not None and ky is not None:
                ax.scatter(kx, ky, s=140, marker="*", color=color,
                           edgecolors="black", linewidths=0.7, zorder=10,
                           label=f"{label} knee")

        # --- Best-per-x-metric (diamond pointing left) ---
        bpm = get_best_per_metric(fdf, [x_col, y_col] if x_col != y_col else [x_col])
        if x_col in bpm:
            bx = float(bpm[x_col][x_col])
            by = float(bpm[x_col][y_col]) if y_col in bpm[x_col].index else np.nan
            if np.isfinite(bx) and np.isfinite(by):
                ax.scatter(bx, by, s=80, marker="d", color=color,
                           edgecolors="black", linewidths=0.7, zorder=9)

        # --- Best-per-y-metric (triangle pointing down) ---
        if y_col in bpm and y_col != x_col:
            bx2 = float(bpm[y_col][x_col]) if x_col in bpm[y_col].index else np.nan
            by2 = float(bpm[y_col][y_col])
            if np.isfinite(bx2) and np.isfinite(by2):
                ax.scatter(bx2, by2, s=80, marker="v", color=color,
                           edgecolors="black", linewidths=0.7, zorder=9)

    # --- Baseline single points ---
    if baseline_points:
        default_markers = ["s", "^", "P", "X", "h", "D", "p", "8"]
        for i, (blabel, bvals) in enumerate(baseline_points.items()):
            bx = bvals.get(x_metric, bvals.get(resolve_metric(
                pd.DataFrame([bvals]), x_metric) or x_metric))
            by = bvals.get(y_metric, bvals.get(resolve_metric(
                pd.DataFrame([bvals]), y_metric) or y_metric))
            if bx is None or by is None:
                continue
            marker = (baseline_markers or {}).get(
                blabel, default_markers[i % len(default_markers)])
            ax.scatter(bx, by, s=70, marker=marker, color="#777777",
                       edgecolors="black", linewidths=0.6, zorder=7,
                       label=blabel)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
