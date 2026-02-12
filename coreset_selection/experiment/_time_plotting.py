"""Time complexity plotting and analysis helpers.

Extracted from ``time_complexity.py`` to keep the main module focused on the
experiment runner.  All public names are re-exported by ``time_complexity``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Complexity curve fitting
# ---------------------------------------------------------------------------

def fit_power_law(ks: np.ndarray, ts: np.ndarray) -> Tuple[float, float]:
    """Fit t = a * k^b via log-linear regression.

    Returns (a, b) where t ~ a * k^b.  Returns (nan, nan) on failure.
    """
    mask = (ks > 0) & (ts > 0) & np.isfinite(ks) & np.isfinite(ts)
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    logk = np.log(ks[mask].astype(float))
    logt = np.log(ts[mask].astype(float))
    try:
        b, log_a = np.polyfit(logk, logt, 1)
        return (float(np.exp(log_a)), float(b))
    except Exception:
        return (float("nan"), float("nan"))


def annotate_complexity_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Append empirical power-law exponents to the timing DataFrame.

    For each unique ``(method, phase)`` group with >=3 k-values, fits
    ``t = a * k^b`` and appends columns ``fit_a`` and ``fit_b``.
    """
    results = []
    for (method, phase), g in df.groupby(["method", "phase"]):
        g = g.sort_values("k")
        a, b = fit_power_law(g["k"].values, g["time_mean_s"].values)
        for _, row in g.iterrows():
            r = row.to_dict()
            r["fit_a"] = a
            r["fit_b"] = b
            results.append(r)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Publication-quality plotting
# ---------------------------------------------------------------------------

def plot_time_complexity(
    df: pd.DataFrame,
    out_dir: str,
) -> List[str]:
    """Generate time complexity figures.

    Produces:
      - time_vs_k_selection.pdf: Selection time vs k by method
      - time_vs_k_evaluation.pdf: Evaluation time vs k by phase
      - time_vs_k_combined.pdf: Combined 2x2 panel
      - time_vs_k_nsga2_loglog.pdf: Log-log with power-law fit
      - time_phase_breakdown_stacked.pdf: Stacked bar per k showing phase breakdown

    Returns list of generated file paths.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Colour palettes
    sel_colors = {
        "nsga2": "#1f77b4", "uniform": "#aec7e8", "kmeans": "#ff7f0e",
        "herding": "#2ca02c", "farthest_first": "#d62728",
        "kernel_thinning": "#9467bd", "rls": "#8c564b", "dpp": "#e377c2",
    }
    eval_colors = {
        "all_eval": "#1f77b4", "nystrom_eval": "#ff7f0e",
        "kpca_eval": "#2ca02c", "krr_eval": "#d62728",
        "geo_eval": "#9467bd",
    }

    df_sel = df[df["phase"] == "selection"]
    df_eval = df[df["phase"] == "eval"]

    # --- 1. Selection time vs k ---
    if not df_sel.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method in df_sel["method"].unique():
            d = df_sel[df_sel["method"] == method].sort_values("k")
            color = sel_colors.get(method, None)
            ax.errorbar(
                d["k"], d["time_mean_s"], yerr=d["time_std_s"],
                marker="o", linewidth=1.3, capsize=3, label=method,
                color=color, markersize=5,
            )
        ax.set_xlabel("Coreset size $k$")
        ax.set_ylabel("Selection time (s)")
        ax.set_title("Selection time vs. coreset size")
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        p = os.path.join(out_dir, "time_vs_k_selection.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # --- 2. Evaluation time vs k ---
    if not df_eval.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method in df_eval["method"].unique():
            d = df_eval[df_eval["method"] == method].sort_values("k")
            color = eval_colors.get(method, None)
            ax.errorbar(
                d["k"], d["time_mean_s"], yerr=d["time_std_s"],
                marker="s", linewidth=1.3, capsize=3, label=method.replace("_", " "),
                color=color, markersize=5,
            )
        ax.set_xlabel("Coreset size $k$")
        ax.set_ylabel("Evaluation time (s)")
        ax.set_title("Evaluation time vs. coreset size")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = os.path.join(out_dir, "time_vs_k_evaluation.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # --- 3. Combined 2x2 panel ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: selection
    ax = axes[0, 0]
    for method in df_sel["method"].unique():
        d = df_sel[df_sel["method"] == method].sort_values("k")
        ax.plot(d["k"], d["time_mean_s"], marker="o", linewidth=1.3,
                label=method, color=sel_colors.get(method), markersize=4)
    ax.set_xlabel("$k$")
    ax.set_ylabel("Time (s)")
    ax.set_title("(a) Selection time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Top-right: evaluation sub-phases
    ax = axes[0, 1]
    for method in df_eval["method"].unique():
        d = df_eval[df_eval["method"] == method].sort_values("k")
        ax.plot(d["k"], d["time_mean_s"], marker="s", linewidth=1.3,
                label=method.replace("_", " "), color=eval_colors.get(method), markersize=4)
    ax.set_xlabel("$k$")
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Evaluation sub-phases")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-left: phase breakdown stacked bars
    ax = axes[1, 0]
    _plot_phase_stacked(df, ax)
    ax.set_title("(c) Phase breakdown (NSGA-II pipeline)")

    # Bottom-right: log-log NSGA-II
    ax = axes[1, 1]
    df_nsga = df_sel[df_sel["method"] == "nsga2"].sort_values("k")
    if not df_nsga.empty and len(df_nsga) >= 3:
        d = df_nsga
        ax.loglog(d["k"], d["time_mean_s"], "o-", linewidth=1.5, color="#1f77b4")
        # Power-law fit
        a, b = fit_power_law(d["k"].values, d["time_mean_s"].values)
        if np.isfinite(b):
            kfit = np.linspace(d["k"].min(), d["k"].max(), 50)
            ax.plot(kfit, a * kfit**b, "--", color="gray", alpha=0.7,
                    label=f"fit: $t \\propto k^{{{b:.2f}}}$")
            ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("$k$ (log)")
    ax.set_ylabel("Time (s, log)")
    ax.set_title("(d) NSGA-II scaling (log-log)")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    p = os.path.join(out_dir, "time_vs_k_combined.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- 4. Standalone log-log NSGA-II ---
    if not df_nsga.empty and len(df_nsga) >= 3:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        d = df_nsga.sort_values("k")
        ax.loglog(d["k"], d["time_mean_s"], "o-", linewidth=1.5)
        a, b = fit_power_law(d["k"].values, d["time_mean_s"].values)
        if np.isfinite(b):
            kfit = np.linspace(d["k"].min(), d["k"].max(), 50)
            ax.plot(kfit, a * kfit**b, "--", color="gray", alpha=0.7,
                    label=f"slope \u2248 {b:.2f}")
            ax.legend()
        ax.set_xlabel("$k$")
        ax.set_ylabel("Time (s)")
        ax.set_title("NSGA-II selection time (log-log)")
        ax.grid(True, alpha=0.3, which="both")
        p = os.path.join(out_dir, "time_vs_k_nsga2_loglog.pdf")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # --- 5. Phase breakdown stacked bar (standalone) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_phase_stacked(df, ax)
    ax.set_title("Pipeline phase breakdown vs. coreset size $k$")
    fig.tight_layout()
    p = os.path.join(out_dir, "time_phase_breakdown_stacked.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def _plot_phase_stacked(df: pd.DataFrame, ax) -> None:
    """Plot a stacked bar chart of pipeline phases for NSGA-II."""
    phase_order = [
        "quota_computation", "objective_setup", "nsga2",
        "nystrom_eval", "kpca_eval", "krr_eval", "geo_eval",
    ]
    phase_labels = [
        "Quota comp.", "Obj. setup", "NSGA-II",
        "Nystr\u00f6m eval", "kPCA eval", "KRR eval", "Geo diag.",
    ]
    phase_colors = [
        "#aec7e8", "#ffbb78", "#1f77b4",
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    ]

    k_vals = sorted(df["k"].unique())
    bottom = np.zeros(len(k_vals))
    bar_width = 0.6 * min(np.diff(k_vals)) if len(k_vals) > 1 else 30

    for phase, label, color in zip(phase_order, phase_labels, phase_colors):
        sub = df[df["method"] == phase]
        if sub.empty:
            continue
        vals = []
        for kv in k_vals:
            row = sub[sub["k"] == kv]
            vals.append(float(row["time_mean_s"].iloc[0]) if not row.empty else 0.0)
        vals = np.array(vals)
        ax.bar(k_vals, vals, bottom=bottom, width=bar_width,
               label=label, color=color, edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel("Coreset size $k$")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")


def save_time_complexity_summary(df: pd.DataFrame, out_dir: str) -> str:
    """Save a publication-ready summary CSV with complexity annotations.

    Returns the path to the written CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Augment with power-law fits
    df_aug = annotate_complexity_fits(df)

    path = os.path.join(out_dir, "time_complexity_results.csv")
    df_aug.to_csv(path, index=False, float_format="%.6f")
    return path
