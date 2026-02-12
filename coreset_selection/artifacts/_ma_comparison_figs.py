"""Comparison figure methods for ManuscriptArtifacts (mixin)."""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ._ma_helpers import _save


class ComparisonFigsMixin:
    """Mixin providing comparison/diagnostic figure methods."""

    def fig_geo_ablation_scatter(self, df: pd.DataFrame) -> str:
        r"""Fig 1: Composition drift vs Nyström error (R6, Section VIII.B).

        Scatter plot of geographic :math:`\ell_1` drift (x-axis) vs Nyström
        approximation error (y-axis, **log-scaled**).  Data source: all
        candidates from the R6 (unconstrained) Pareto front at k = 300.

        Manuscript compliance fixes (Phase 8):
        * y-axis is log-scaled.
        * Constraint regimes distinguished by marker shape.
        * Quota-feasible reference point added.
        * R1 constrained knee-point overlaid for comparison.
        * IEEE single-column width (~3.5 in).
        * Font sizes ≥ 8 pt (annotations), ≥ 10 pt (labels).
        """
        import matplotlib.pyplot as plt

        # ---- data: R6 unconstrained candidates at k=300 ----
        if df.empty or "run_id" not in df.columns:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "geo_ablation_tradeoff_scatter.pdf"))

        d_r6 = df[df["run_id"].astype(str).str.contains("R6")].copy()
        if d_r6.empty:
            d_r6 = df[df["k"].fillna(300).astype(int) == 300].copy()
        has_cols = (not d_r6.empty
                    and "geo_l1" in d_r6.columns
                    and "nystrom_error" in d_r6.columns)
        if not has_cols:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No R6 data available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "geo_ablation_tradeoff_scatter.pdf"))

        # ---- plot ----
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        # Marker shapes per constraint regime
        regime_styles = {
            "exactk_only": ("o", "#1f77b4", "Exact-$k$ only"),
            "exactk":      ("o", "#1f77b4", "Exact-$k$ only"),
            "quota":       ("s", "#ff7f0e", "Quota-matched"),
            "quota+exactk":("^", "#2ca02c", "Quota + exact-$k$"),
            "unconstrained":("D","#9467bd", "Unconstrained"),
            "none":        ("D", "#9467bd", "Unconstrained"),
        }
        cr = d_r6.get("constraint_regime",
                       pd.Series(["unconstrained"] * len(d_r6), index=d_r6.index))
        plotted_labels = set()
        for regime in cr.unique():
            mk, col, lab = regime_styles.get(
                str(regime), ("o", "#7f7f7f", str(regime)))
            sub = d_r6[cr == regime]
            show_label = lab not in plotted_labels
            ax.scatter(sub["nystrom_error"], sub["geo_l1"],
                       s=22, alpha=0.65, marker=mk, color=col,
                       label=lab if show_label else None,
                       edgecolors="k", linewidths=0.3)
            plotted_labels.add(lab)

        # ---- overlay R1 constrained knee-point ----
        d_r1 = df[df["run_id"].astype(str).str.contains("R1")].copy()
        d_r1 = d_r1[d_r1["k"].fillna(300).astype(int) == 300]
        if not d_r1.empty:
            # Use knee-point row if available, else best Nyström
            knee = d_r1[d_r1.get("rep_name", pd.Series()).astype(str) == "knee"]
            if knee.empty:
                knee = d_r1
            if "geo_l1" in knee.columns and "nystrom_error" in knee.columns:
                knee_row = knee.loc[knee["nystrom_error"].idxmin()]
                ax.scatter(knee_row["nystrom_error"], knee_row["geo_l1"],
                           s=90, marker="*", color="#d62728", edgecolors="k",
                           linewidths=0.5, zorder=10,
                           label="R1 knee (constrained)")

        # ---- quota-feasible reference line ----
        if "geo_kl" in d_r6.columns:
            try:
                from ..config.constants import ALPHA_GEO
                tau_ref = getattr(
                    __import__("coreset_selection.config.constants", fromlist=["TAU_MUNICIPALITY"]),
                    "TAU_MUNICIPALITY", 0.02)
            except Exception:
                tau_ref = 0.02
            # Draw horizontal line at approximate feasibility boundary
            feas = d_r6[d_r6["geo_kl"] <= tau_ref]
            if not feas.empty:
                boundary_l1 = feas["geo_l1"].max()
                ax.axhline(boundary_l1, color="gray", linestyle="--",
                           linewidth=0.8, alpha=0.6)
                ax.text(ax.get_xlim()[0], boundary_l1, " feasible",
                        fontsize=7, va="bottom", ha="left", color="gray")

        ax.set_xlabel(r"Nyström error $e_{\mathrm{Nys}}$", fontsize=10)
        ax.set_ylabel(r"Geographic $\ell_1$ drift", fontsize=10)
        ax.set_xscale("log")
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title("Composition drift vs.\\ Nyström error "
                      "(R6, $k{=}300$)", fontsize=10, pad=6)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "geo_ablation_tradeoff_scatter.pdf"))

    def fig_regional_validity_k300(self, df: pd.DataFrame) -> str:
        r"""Fig 3: State-conditioned KPI stability at k = 300 (Section VIII.D).

        Compares R1 (population-share only) and R5 (joint constraints):
        max drift, avg drift, and Kendall's :math:`\tau` for both
        :math:`y^{(4\mathrm{G})}` and :math:`y^{(5\mathrm{G})}`.

        Manuscript compliance fixes (Phase 8):
        * R1 **and** R5 shown side-by-side (grouped bars).
        * Both 4G and 5G targets are shown explicitly.
        * Three panels: (a) max drift, (b) avg drift, (c) Kendall's τ.
        * IEEE double-column width (~7 in).
        * Font sizes ≥ 8 pt annotations, ≥ 10 pt labels.
        """
        import matplotlib.pyplot as plt

        if df.empty or "k" not in df.columns:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "regional_validity_k300.pdf"))

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        r1 = d[d["run_id"].astype(str).str.contains("R1")]
        r5 = d[d["run_id"].astype(str).str.contains("R5")]

        # ---- Discover drift & tau columns ----
        all_cols = set(d.columns)
        # Canonical column names expected from Phase 4/7
        max_drift_4g = next((c for c in all_cols if "max" in c and "drift" in c and "4G" in c), None)
        avg_drift_4g = next((c for c in all_cols if "avg" in c and "drift" in c and "4G" in c), None)
        tau_4g       = next((c for c in all_cols if "kendall" in c.lower() and "4G" in c), None)
        max_drift_5g = next((c for c in all_cols if "max" in c and "drift" in c and "5G" in c), None)
        avg_drift_5g = next((c for c in all_cols if "avg" in c and "drift" in c and "5G" in c), None)
        tau_5g       = next((c for c in all_cols if "kendall" in c.lower() and "5G" in c), None)

        # Fallback: try generic names without technology suffix
        if max_drift_4g is None:
            max_drift_4g = next((c for c in all_cols if "max" in c and ("kpi" in c or "drift" in c)), None)
        if avg_drift_4g is None:
            avg_drift_4g = next((c for c in all_cols if "avg" in c and ("kpi" in c or "drift" in c)), None)
        if tau_4g is None:
            tau_4g = next((c for c in all_cols if "kendall" in c.lower() or "tau" in c.lower()), None)

        found_any = any(c is not None for c in [max_drift_4g, avg_drift_4g, tau_4g])
        if not found_any or (r1.empty and r5.empty):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5,
                    "No KPI stability data found for R1/R5 at $k=300$.\n"
                    "Columns searched: max_drift, avg_drift, kendall_tau",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "regional_validity_k300.pdf"))

        # ---- Build grouped data ----
        def _safe_mean(sub, col):
            if col is not None and col in sub.columns:
                v = sub[col].dropna()
                return float(v.mean()) if len(v) > 0 else np.nan
            return np.nan

        # Three metric groups: max_drift, avg_drift, kendall_tau
        # Each has up to 2 targets (4G, 5G) × 2 runs (R1, R5)
        metric_groups = [
            ("Max drift",       [(max_drift_4g, "4G"), (max_drift_5g, "5G")]),
            ("Avg drift",       [(avg_drift_4g, "4G"), (avg_drift_5g, "5G")]),
            (r"Kendall's $\tau$", [(tau_4g, "4G"),     (tau_5g,       "5G")]),
        ]

        n_panels = len(metric_groups)
        fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 3.5))
        if n_panels == 1:
            axes = [axes]
        panel_labels = ["(a)", "(b)", "(c)"]

        bar_width = 0.18
        colors_r1 = ["#1f77b4", "#aec7e8"]   # dark blue / light blue for 4G/5G
        colors_r5 = ["#ff7f0e", "#ffbb78"]   # dark orange / light orange for 4G/5G

        for p_idx, (ax, (group_name, targets)) in enumerate(zip(axes, metric_groups)):
            x_positions = []
            tick_labels = []
            bars_drawn = []

            x = 0
            for col, tech in targets:
                if col is None:
                    continue
                r1_val = _safe_mean(r1, col)
                r5_val = _safe_mean(r5, col)

                c_r1 = colors_r1[0] if tech == "4G" else colors_r1[1]
                c_r5 = colors_r5[0] if tech == "4G" else colors_r5[1]

                if np.isfinite(r1_val):
                    b = ax.bar(x, r1_val, width=bar_width, color=c_r1,
                               edgecolor="k", linewidth=0.3)
                    bars_drawn.append((b, f"R1 {tech}"))
                if np.isfinite(r5_val):
                    b = ax.bar(x + bar_width, r5_val, width=bar_width,
                               color=c_r5, edgecolor="k", linewidth=0.3)
                    bars_drawn.append((b, f"R5 {tech}"))

                x_positions.append(x + bar_width / 2)
                tick_labels.append(tech)
                x += 2.5 * bar_width

            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_ylabel(group_name, fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)
            ax.annotate(panel_labels[p_idx], xy=(0.03, 0.95),
                        xycoords="axes fraction", fontsize=10,
                        fontweight="bold", va="top")

        # Shared legend (de-duplicate)
        seen = set()
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    handles.append(h)
                    labels.append(l)
                    seen.add(l)
        # Build manual legend entries
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors_r1[0], edgecolor="k", linewidth=0.3, label="R1 (4G)"),
            Patch(facecolor=colors_r1[1], edgecolor="k", linewidth=0.3, label="R1 (5G)"),
            Patch(facecolor=colors_r5[0], edgecolor="k", linewidth=0.3, label="R5 (4G)"),
            Patch(facecolor=colors_r5[1], edgecolor="k", linewidth=0.3, label="R5 (5G)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=4, fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle("State-conditioned KPI stability at $k{=}300$: "
                      "R1 (pop-share) vs R5 (joint)",
                      fontsize=11, y=1.02)
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])
        return _save(fig, os.path.join(self.fig_dir,
                     "regional_validity_k300.pdf"))

    def fig_objective_metric_alignment(self, df: pd.DataFrame) -> str:
        r"""Fig 4: Objective–metric Spearman ρ heatmap (R11, Section VIII.K).

        Annotated heatmap showing Spearman rank correlations between
        optimisation objectives (:math:`f_{\mathrm{MMD}}`,
        :math:`f_{\mathrm{SD}}`, optionally :math:`f_{\mathrm{SKL}}`)
        and raw-space evaluation metrics (:math:`e_{\mathrm{Nys}}`,
        :math:`e_{\mathrm{kPCA}}`, RMSE :sub:`4G`, RMSE :sub:`5G`,
        :math:`\mathrm{KL}_{\mathrm{geo}}`, :math:`\ell_1`).

        Manuscript compliance fixes (Phase 8):
        * Diverging colormap (``RdBu_r``) centred at 0.
        * Correlation values annotated inside each cell (≥ 8 pt).
        * Rows = objectives, columns = metrics, clearly labelled.
        * Data preferentially loaded from R11 ``front_metrics*.csv``
          or ``objective_metric_alignment*.csv``; falls back to k = 300
          Pareto data.
        * IEEE single-column width (~3.5 in) or slightly wider for
          readability.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        try:
            from scipy.stats import spearmanr
        except ImportError:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "scipy not available", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
            return _save(fig, os.path.join(self.fig_dir,
                         "objective_metric_alignment_heatmap.pdf"))

        if df.empty or "k" not in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "objective_metric_alignment_heatmap.pdf"))

        # ---- Load data: prefer R11 artefacts ----
        fdf = pd.DataFrame()
        for pattern in [
            os.path.join(self.runs_root, "**/objective_metric_alignment*.csv"),
            os.path.join(self.runs_root, "**/front_metrics*.csv"),
        ]:
            found = glob.glob(pattern, recursive=True)
            if found:
                fdf = pd.read_csv(found[0])
                break
        if fdf.empty:
            # Fallback: all k=300 data (R1 + baselines)
            fdf = df[df["k"].fillna(300).astype(int) == 300].copy()

        # ---- Identify objective & metric columns ----
        obj_cols = [c for c in fdf.columns if c.startswith("f_")]
        met_candidates = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_4G", "krr_rmse_5G",
            "geo_kl", "geo_l1", "geo_kl_pop", "geo_l1_pop",
        ]
        met_cols = [c for c in met_candidates if c in fdf.columns]

        if not obj_cols or not met_cols:
            fig, ax = plt.subplots(figsize=(4, 3))
            msg = ("Missing objective (f_*) or metric columns.\n"
                   f"Found objectives: {obj_cols}\n"
                   f"Found metrics: {met_cols}")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, family="monospace")
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "objective_metric_alignment_heatmap.pdf"))

        # ---- Compute Spearman matrix ----
        corr = np.full((len(obj_cols), len(met_cols)), np.nan)
        for i, o in enumerate(obj_cols):
            for j, m in enumerate(met_cols):
                valid = fdf[[o, m]].dropna()
                if len(valid) > 3:
                    corr[i, j] = spearmanr(valid[o], valid[m]).correlation

        # ---- Pretty labels ----
        obj_labels = [c.replace("f_", "").replace("mmd", "MMD")
                       .replace("sinkhorn", "SD").replace("skl", "SKL")
                      for c in obj_cols]
        met_labels = []
        _label_map = {
            "nystrom_error": r"$e_{\rm Nys}$",
            "kpca_distortion": r"$e_{\rm kPCA}$",
            "krr_rmse_4G": r"RMSE$_{\rm 4G}$",
            "krr_rmse_5G": r"RMSE$_{\rm 5G}$",
            "geo_kl": r"KL$_{\rm geo}$",
            "geo_l1": r"$\ell_1$",
            "geo_kl_pop": r"KL$_{\rm pop}$",
            "geo_l1_pop": r"$\ell_1^{\rm pop}$",
        }
        for c in met_cols:
            met_labels.append(_label_map.get(c, c.replace("_", " ")))

        # ---- Plot ----
        n_obj, n_met = len(obj_cols), len(met_cols)
        fig_w = max(3.5, n_met * 0.9 + 1.2)
        fig_h = max(2.0, n_obj * 0.65 + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Diverging norm centred at 0
        vmin = np.nanmin(corr) if np.any(np.isfinite(corr)) else -1
        vmax = np.nanmax(corr) if np.any(np.isfinite(corr)) else 1
        abs_lim = max(abs(vmin), abs(vmax), 0.05)
        norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0, vmax=abs_lim)

        im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", norm=norm)

        ax.set_yticks(range(n_obj))
        ax.set_yticklabels(obj_labels, fontsize=10)
        ax.set_xticks(range(n_met))
        ax.set_xticklabels(met_labels, rotation=40, ha="right", fontsize=9)

        # Cell annotations
        for i in range(n_obj):
            for j in range(n_met):
                val = corr[i, j]
                if np.isfinite(val):
                    # Use white text on dark cells for contrast
                    color = "white" if abs(val) > 0.55 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color, fontweight="medium")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.06)
        cbar.set_label(r"Spearman $\rho$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(r"Spearman $\rho$: optimisation objectives vs.\  "
                      "raw-space metrics", fontsize=10, pad=8)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "objective_metric_alignment_heatmap.pdf"))

    def fig_objective_ablation_bars(self, df: pd.DataFrame) -> str:
        r"""Fig N3: Objective ablation comparison — R1 vs R2 vs R3 at k=300.

        Grouped bar chart showing that neither MMD-only (R2) nor SD-only (R3)
        matches the bi-objective approach (R1) across all raw-space evaluation
        metrics.

        Phase 10a enhancements (manuscript Section VIII.C):
        * Metrics on x-axis with methods (R1-knee, R2, R3) as grouped bars.
        * Extended metric set: e_Nys, e_kPCA, RMSE_4G, RMSE_5G, geo_kl.
        * Value annotations on each bar.
        * Consistent colour scheme across all complementary figures.
        * IEEE double-column width for multi-metric clarity.
        * Defense value: quantifies the benefit of bi-objective optimisation;
          neither single-objective approach dominates on all metrics.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()

        # Extended metric set per Phase 10 spec
        metric_candidates = [
            ("nystrom_error",    r"$e_{\rm Nys}$"),
            ("kpca_distortion",  r"$e_{\rm kPCA}$"),
            ("krr_rmse_4G",      r"RMSE$_{\rm 4G}$"),
            ("krr_rmse_5G",      r"RMSE$_{\rm 5G}$"),
            ("geo_kl",           r"KL$_{\rm geo}$"),
        ]
        # Also try canonical column names
        _aliases = {
            "krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
            "krr_rmse_5G": ["krr_rmse_cov_area_5G", "krr_rmse_area_5G"],
        }
        for canon, aliases in _aliases.items():
            if canon not in d.columns:
                for alias in aliases:
                    if alias in d.columns:
                        d[canon] = d[alias]
                        break

        metrics = [(m, lab) for m, lab in metric_candidates if m in d.columns]
        if not metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No evaluation metrics available for ablation.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

        # Gather data for each run configuration
        runs = {
            "R1 (MMD+SD)":  ("R1", "#1f77b4"),
            "R2 (MMD only)": ("R2", "#ff7f0e"),
            "R3 (SD only)":  ("R3", "#2ca02c"),
        }
        bar_data = {}
        for label, (rid, _) in runs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if sub.empty:
                continue
            bar_data[label] = {m: float(sub[m].min()) for m, _ in metrics
                                if m in sub.columns and sub[m].notna().any()}

        if not bar_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No R1/R2/R3 data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

        # ---- Grouped bar chart ----
        metric_keys = [m for m, _ in metrics]
        metric_labels = [lab for _, lab in metrics]
        method_labels = list(bar_data.keys())
        method_colors = [runs[m][1] for m in method_labels]

        n_metrics = len(metric_keys)
        n_methods = len(method_labels)
        x = np.arange(n_metrics)
        total_width = 0.75
        width = total_width / max(n_methods, 1)

        fig, ax = plt.subplots(figsize=(max(7, n_metrics * 1.5), 4.2))

        for i, (method, color) in enumerate(zip(method_labels, method_colors)):
            vals = [bar_data[method].get(m, 0) for m in metric_keys]
            offset = (i - (n_methods - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9, color=color,
                          alpha=0.85, label=method, edgecolor="white",
                          linewidth=0.5)
            # Value annotations
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=6.5,
                            rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylabel("Metric value (lower is better)", fontsize=10)
        ax.set_title("Objective ablation at $k{=}300$: bi-objective vs.\\ "
                     "single-objective", fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
                  edgecolor="0.8")
        ax.grid(True, alpha=0.2, axis="y")
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

    def fig_constraint_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig N4: Constraint regime comparison — R1, R4, R5, R6 at k=300.

        Multi-panel grouped bar chart comparing the four constraint
        configurations across both composition diagnostics and fidelity
        metrics at k=300.

        Phase 10a enhancements (manuscript Section VIII.F):
        * Four metrics: geo_kl, geo_l1, e_Nys, RMSE_4G.
        * 2×2 panel layout (one panel per metric) for clear comparison.
        * Runs: R1 (pop-share), R4 (muni-quota), R5 (joint), R6 (none).
        * Consistent colour scheme; value annotations on bars.
        * IEEE double-column width.
        * Defense value: directly supports Section VIII.F (constraint
          swapping / joint constraints) by showing how different constraint
          configurations trade off composition quality against fidelity.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()

        # Resolve column aliases
        _aliases = {
            "krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
        }
        for canon, aliases in _aliases.items():
            if canon not in d.columns:
                for alias in aliases:
                    if alias in d.columns:
                        d[canon] = d[alias]
                        break

        # Metric panels — per manuscript Phase 10 spec
        metric_defs = [
            ("geo_kl",          r"KL$_{\rm geo}$ (composition)", "lower"),
            ("geo_l1",          r"$\ell_1$ drift (composition)", "lower"),
            ("nystrom_error",   r"$e_{\rm Nys}$ (fidelity)", "lower"),
            ("krr_rmse_4G",    r"RMSE$_{\rm 4G}$ (prediction)", "lower"),
        ]
        available_metrics = [(m, lab, dirn) for m, lab, dirn in metric_defs
                             if m in d.columns]

        runs = {
            "R1\n(pop-share)":     ("R1", "#1f77b4"),
            "R4\n(muni-quota)":    ("R4", "#ff7f0e"),
            "R5\n(joint)":         ("R5", "#2ca02c"),
            "R6\n(none)":          ("R6", "#d62728"),
        }

        # Collect data
        bar_data = {}
        for label, (rid, _) in runs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if not sub.empty:
                bar_data[label] = {m: float(sub[m].mean())
                                   for m, _, _ in available_metrics
                                   if m in sub.columns and sub[m].notna().any()}

        if not bar_data or not available_metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No constraint comparison data (R1/R4/R5/R6 at k=300).",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "constraint_comparison_bars_k300.pdf"))

        # ---- 2×2 panel layout ----
        n_panels = min(len(available_metrics), 4)
        nrows = 2 if n_panels > 2 else 1
        ncols = 2 if n_panels > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 5.5), squeeze=False)

        method_labels = list(bar_data.keys())
        method_colors = [runs[m][1] for m in method_labels]
        x = np.arange(len(method_labels))

        for idx, (metric, label, direction) in enumerate(available_metrics[:n_panels]):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]

            vals = [bar_data[m].get(metric, 0) for m in method_labels]
            bars = ax.bar(x, vals, color=method_colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5, width=0.65)

            # Value annotations
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, fontsize=8)
            ax.set_title(label, fontsize=9)
            ax.set_ylabel("Value", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(labelsize=8)

        # Hide unused panels
        for idx in range(n_panels, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle("Constraint regime comparison at $k{=}300$",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "constraint_comparison_bars_k300.pdf"))

    def fig_representation_transfer(self, df: pd.DataFrame) -> str:
        """R8/R9 representation transfer bars."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        runs = {"R1 (raw)": "R1", "R8 (PCA)": "R8", "R9 (VAE)": "R9"}
        metrics = ["nystrom_error", "kpca_distortion", "krr_rmse_4G"]
        metrics = [m for m in metrics if m in d.columns]

        bar_data = {}
        for label, rid in runs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if not sub.empty:
                bar_data[label] = {m: sub[m].min() for m in metrics}

        if not bar_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No R8/R9 data", ha="center", va="center",
                    transform=ax.transAxes)
            return _save(fig, os.path.join(self.fig_dir, "representation_transfer_bars.pdf"))

        bdf = pd.DataFrame(bar_data).T
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, metrics):
            if m in bdf.columns:
                bdf[m].plot(kind="bar", ax=ax, rot=15)
                ax.set_title(m.replace("_", " "))
        fig.suptitle("Representation transfer at $k=300$", y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "representation_transfer_bars.pdf"))

    def fig_baseline_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig N6: Comprehensive baseline comparison at k=300 (R10).

        Grouped bar chart comparing all 7 baselines against the R1
        knee-point across multiple evaluation metrics.

        Phase 10a enhancements (manuscript Section VIII.E / R10):
        * X-axis: methods (uniform, k-means, herding, farthest-first,
          kernel thinning, RLS, DPP, R1-knee).
        * Metrics as separate panels: e_Nys, RMSE_4G, geo_kl.
        * When available, show both unconstrained and quota-matched
          variants side-by-side (differentiated by hatching/alpha).
        * R1 knee-point shown as a horizontal reference line for each
          metric to ease comparison.
        * Value annotations on bars.
        * IEEE double-column width.
        * Defense value: demonstrates competitive performance of the
          constrained NSGA-II selection vs standard baselines; shows
          the cost of imposing proportionality on baselines.
        """
        import matplotlib.pyplot as plt

        d_all = df[df["k"].fillna(300).astype(int) == 300].copy()

        # Resolve column aliases
        _aliases = {
            "krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
        }
        for canon, aliases in _aliases.items():
            if canon not in d_all.columns:
                for alias in aliases:
                    if alias in d_all.columns:
                        d_all[canon] = d_all[alias]
                        break

        # Identify baseline rows
        d_base = d_all[d_all["run_id"].astype(str).str.contains("R10")].copy()
        if d_base.empty and "method" in d_all.columns:
            d_base = d_all[d_all["method"].astype(str) != "nsga2"].copy()

        # Metric panels
        metric_defs = [
            ("nystrom_error",  r"$e_{\rm Nys}$"),
            ("krr_rmse_4G",   r"RMSE$_{\rm 4G}$"),
            ("geo_kl",         r"KL$_{\rm geo}$"),
        ]
        metrics = [(m, lab) for m, lab in metric_defs if m in d_all.columns]

        if d_base.empty or not metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No baseline comparison data (R10) at k=300.\n"
                    "Run R10 first to generate baseline results.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "baseline_comparison_grouped.pdf"))

        # ---- Determine method and variant ----
        has_method = "method" in d_base.columns
        has_variant = "variant" in d_base.columns

        if has_method:
            methods_available = sorted(d_base["method"].unique())
        else:
            methods_available = ["baseline"]

        # Standard method display order
        method_order = ["uniform", "kmeans", "herding", "farthest_first",
                        "kernel_thinning", "leverage", "dpp"]
        # Keep only those present
        methods = [m for m in method_order if m in methods_available]
        # Add any extras not in the standard order
        methods.extend([m for m in methods_available if m not in methods])

        # Human-readable labels
        _method_labels = {
            "uniform": "Uniform",
            "kmeans": "k-Means",
            "herding": "Herding",
            "farthest_first": "Farthest-First",
            "kernel_thinning": "Kern. Thin.",
            "leverage": "RLS",
            "dpp": "DPP",
        }

        # ---- Gather R1 knee reference values ----
        d_r1 = d_all[d_all["run_id"].astype(str).str.contains("R1")]
        r1_ref = {}
        if not d_r1.empty:
            knee = d_r1[d_r1.get("rep_name", pd.Series(dtype=str)).astype(str) == "knee"]
            if knee.empty:
                knee = d_r1
            for m, _ in metrics:
                if m in knee.columns and knee[m].notna().any():
                    r1_ref[m] = float(knee[m].min())

        # ---- Build data structure ----
        # Two variants: unconstrained and quota-matched
        variants = ["unconstrained", "quota_matched"]
        variant_labels = {"unconstrained": "Uncons.", "quota_matched": "Quota-m."}

        # Gather per-method per-variant metric values
        method_data = {}  # {method: {variant: {metric: value}}}
        for method in methods:
            method_data[method] = {}
            if has_method:
                sub_m = d_base[d_base["method"] == method]
            else:
                sub_m = d_base

            if has_variant and sub_m["variant"].nunique() > 1:
                for var in variants:
                    sub_v = sub_m[sub_m["variant"].astype(str).str.contains(var)]
                    if not sub_v.empty:
                        method_data[method][var] = {
                            m: float(sub_v[m].mean())
                            for m, _ in metrics
                            if m in sub_v.columns and sub_v[m].notna().any()
                        }
            else:
                # Single variant or no variant column — treat as unconstrained
                if not sub_m.empty:
                    method_data[method]["unconstrained"] = {
                        m: float(sub_m[m].mean())
                        for m, _ in metrics
                        if m in sub_m.columns and sub_m[m].notna().any()
                    }

        # Check if we have any quota-matched data
        has_quota = any(
            "quota_matched" in method_data.get(m, {})
            for m in methods
        )

        # ---- Plot: one panel per metric ----
        n_panels = len(metrics)
        fig, axes = plt.subplots(1, n_panels,
                                  figsize=(max(7, n_panels * 4), 4.5))
        if n_panels == 1:
            axes = [axes]

        for ax, (metric, label) in zip(axes, metrics):
            n_methods = len(methods) + 1  # +1 for R1-knee
            x = np.arange(n_methods)

            if has_quota:
                # Side-by-side bars for unconstrained and quota-matched
                width = 0.35
                # Unconstrained bars
                uncons_vals = []
                for method in methods:
                    v = method_data.get(method, {}).get("unconstrained", {}).get(metric, 0)
                    uncons_vals.append(v)
                uncons_vals.append(r1_ref.get(metric, 0))  # R1 knee

                quota_vals = []
                for method in methods:
                    v = method_data.get(method, {}).get("quota_matched", {}).get(metric, 0)
                    quota_vals.append(v)
                quota_vals.append(r1_ref.get(metric, 0))  # R1 knee (same)

                bars1 = ax.bar(x - width / 2, uncons_vals, width=width * 0.9,
                               color="#1f77b4", alpha=0.8, label="Unconstrained",
                               edgecolor="white", linewidth=0.5)
                bars2 = ax.bar(x + width / 2, quota_vals, width=width * 0.9,
                               color="#ff7f0e", alpha=0.8, label="Quota-matched",
                               edgecolor="white", linewidth=0.5)
            else:
                # Single bars
                width = 0.55
                vals = []
                for method in methods:
                    v = method_data.get(method, {}).get("unconstrained", {}).get(metric, 0)
                    vals.append(v)
                vals.append(r1_ref.get(metric, 0))

                colors = ["#1f77b4"] * len(methods) + ["#d62728"]
                bars1 = ax.bar(x, vals, width=width, color=colors, alpha=0.85,
                               edgecolor="white", linewidth=0.5)

            # R1 knee reference line
            if metric in r1_ref:
                ax.axhline(r1_ref[metric], color="#d62728", linestyle="--",
                           linewidth=1.0, alpha=0.6)

            # X-axis labels
            xlabels = [_method_labels.get(m, m) for m in methods] + ["R1-knee"]
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7.5)
            ax.set_title(label, fontsize=9)
            ax.set_ylabel("Value", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(labelsize=8)
            if has_quota:
                ax.legend(fontsize=7, loc="upper left")

        fig.suptitle("Baseline comparison at $k{=}300$ (R10)",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "baseline_comparison_grouped.pdf"))

    def fig_state_kpi_heatmap(self, df: pd.DataFrame) -> str:
        r"""Fig N8: Per-state KPI drift heatmap (R1, k=300).

        Heatmap with states on the y-axis and coverage targets (4G, 5G)
        on the x-axis.  Colour intensity encodes
        :math:`|\mu_g^S - \mu_g^{\mathrm{full}}|` (absolute drift between
        full-data state mean and subset state mean).

        Phase 10b enhancements (manuscript Section VIII.D):
        * States sorted by drift magnitude (worst states at top).
        * Small-state annotations: states with :math:`n_g < 50` are
          highlighted with a marker to signal limited sample size.
        * Cell annotations showing drift values.
        * Colour bar labelled with drift interpretation.
        * If per-state drift CSV is unavailable, attempts to reconstruct
          from per-state columns in the main results DataFrame.
        * IEEE double-column width.
        * Defense value: makes small-state failures visible; addresses
          concern that aggregate metrics hide per-state problems.
        """
        import matplotlib.pyplot as plt

        # ---- Strategy 1: Load dedicated per-state drift CSV ----
        state_files = glob.glob(os.path.join(self.runs_root,
                                              "**/state_kpi_drift*.csv"),
                                recursive=True)
        sdf = None
        if state_files:
            try:
                sdf = pd.read_csv(state_files[0])
            except Exception:
                pass

        # ---- Strategy 2: Reconstruct from per-state columns ----
        if sdf is None:
            d = df[df["run_id"].astype(str).str.contains("R1")].copy()
            d = d[d["k"].fillna(300).astype(int) == 300]
            state_drift_cols = [c for c in d.columns
                                if c.startswith("state_drift_") or
                                c.startswith("kpi_drift_")]
            if state_drift_cols and not d.empty:
                # Pivot available state drift columns
                rows = []
                for col in state_drift_cols:
                    parts = col.replace("state_drift_", "").replace("kpi_drift_", "").split("_")
                    if len(parts) >= 2:
                        state = parts[0]
                        target = "_".join(parts[1:])
                        rows.append({
                            "state": state,
                            "target": target,
                            "drift": float(d[col].mean()),
                        })
                if rows:
                    sdf = pd.DataFrame(rows)

        if sdf is None or sdf.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5,
                    "Per-state KPI drift data not yet generated.\n"
                    "Run R1 with state-level diagnostics enabled,\n"
                    "or ensure state_kpi_drift.csv exists in runs_out/.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "state_kpi_heatmap.pdf"))

        # ---- Build pivot table ----
        pivot = sdf.pivot_table(index="state", columns="target",
                                 values="drift", aggfunc="mean")

        # Sort states by mean drift (worst at top)
        pivot["_mean_drift"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean_drift", ascending=False)
        pivot = pivot.drop(columns=["_mean_drift"])

        # ---- Load group sizes for small-state annotation ----
        group_sizes = None
        try:
            cache_files = glob.glob(os.path.join(self.cache_root,
                                                  "rep*/assets.npz"))
            if cache_files:
                from ..data.cache import load_replicate_cache
                assets = load_replicate_cache(cache_files[0])
                if hasattr(assets, "state_labels"):
                    labels = assets.state_labels
                    from collections import Counter
                    group_sizes = Counter(labels)
        except Exception:
            pass

        # ---- Plot ----
        n_states = len(pivot.index)
        n_targets = len(pivot.columns)
        fig_h = max(5, n_states * 0.22 + 1.5)
        fig_w = max(4, n_targets * 1.5 + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        data = pivot.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest")

        ax.set_yticks(range(n_states))
        ylabels = []
        for state in pivot.index:
            lbl = str(state)
            # Annotate small states
            if group_sizes and group_sizes.get(state, 999) < 50:
                lbl += f" (n={group_sizes[state]})"
                lbl += " ◆"
            ylabels.append(lbl)
        ax.set_yticklabels(ylabels, fontsize=7)

        ax.set_xticks(range(n_targets))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)

        # Cell annotations
        for i in range(n_states):
            for j in range(n_targets):
                val = data[i, j]
                if np.isfinite(val):
                    color = "white" if val > np.nanpercentile(data, 75) else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(r"$|\mu_g^S - \mu_g^{\rm full}|$ (KPI drift)",
                       fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title("Per-state KPI drift (R1, $k{=}300$)\n"
                     "States sorted by mean drift (worst at top)",
                     fontsize=10, pad=8)
        if group_sizes:
            ax.text(0.01, -0.08,
                    r"◆ = state with $n_g < 50$ municipalities",
                    transform=ax.transAxes, fontsize=7, color="#555555")

        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "state_kpi_heatmap.pdf"))

    def fig_composition_shift(self, df: pd.DataFrame) -> str:
        r"""Fig N9: State composition shift — R6 (unconstrained) vs R1 (constrained).

        Side-by-side horizontal bar chart comparing :math:`\pi_g` (full-data
        state proportions) against :math:`\hat\pi_g(S)` (subset state
        proportions) for both the R6 unconstrained run and the R1
        constrained run at k=300.

        This makes the abstract KL / :math:`\ell_1` drift metrics visually
        concrete and compelling.

        Phase 10b — NEW figure (manuscript Section VIII.B):
        * Left panel: R6 (unconstrained) — shows how selection distorts
          state composition without proportionality constraints.
        * Right panel: R1 (constrained) — shows composition preservation.
        * Target proportions :math:`\pi_g` drawn as reference.
        * States sorted by size for consistent ordering.
        * Defense value: makes the benefit of proportionality constraints
          visually obvious to reviewers.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()

        # Look for per-state proportion columns
        pi_cols = [c for c in d.columns if c.startswith("pi_g_")]
        pihat_cols = [c for c in d.columns if c.startswith("pihat_g_")]

        # Also try loading from geo diagnostics CSVs
        geo_files = glob.glob(os.path.join(self.runs_root,
                                            "**/geo_state_proportions*.csv"),
                              recursive=True)
        geo_df = None
        if geo_files:
            try:
                geo_df = pd.read_csv(geo_files[0])
            except Exception:
                pass

        if geo_df is not None and "state" in geo_df.columns:
            states = geo_df["state"].values
            pi_target = (geo_df["pi_target"].values if "pi_target" in geo_df.columns
                         else np.ones(len(states)) / len(states))

            # Build per-run pihat
            def _get_pihat(run_pattern):
                sub = geo_df[geo_df.get("run_id", pd.Series(dtype=str)).astype(str)
                             .str.contains(run_pattern)]
                if not sub.empty and "pihat" in sub.columns:
                    return sub["pihat"].values
                return None

            pihat_r6 = _get_pihat("R6")
            pihat_r1 = _get_pihat("R1")
        elif pi_cols and pihat_cols:
            # Reconstruct from column-level proportions
            states = sorted(set(c.replace("pi_g_", "") for c in pi_cols))
            pi_target = np.array([d[f"pi_g_{s}"].mean() for s in states
                                  if f"pi_g_{s}" in d.columns])
            def _get_pihat_from_cols(rid):
                sub = d[d["run_id"].astype(str).str.contains(rid)]
                if sub.empty:
                    return None
                return np.array([sub[f"pihat_g_{s}"].mean()
                                 for s in states if f"pihat_g_{s}" in sub.columns])

            pihat_r6 = _get_pihat_from_cols("R6")
            pihat_r1 = _get_pihat_from_cols("R1")
        else:
            # Fallback: placeholder
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5,
                    "No state-level proportion data available.\n"
                    "Run R1 and R6 with geo_diagnostics enabled\n"
                    "to produce per-state proportions.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "composition_shift_sankey.pdf"))
            pi_target = states = pihat_r6 = pihat_r1 = None

        # ---- Plot: side-by-side horizontal bar chart ----
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(states) * 0.2 + 1)))

        # Sort states by pi_target descending
        order = np.argsort(-pi_target)
        states_sorted = np.array(states)[order]
        pi_sorted = pi_target[order]
        y = np.arange(len(states_sorted))

        panels = [
            (axes[0], pihat_r6, "R6 (unconstrained)", "#d62728"),
            (axes[1], pihat_r1, "R1 (constrained)",    "#1f77b4"),
        ]
        for ax, pihat, title, color in panels:
            ax.barh(y, pi_sorted, height=0.35, align="center",
                    color="#cccccc", alpha=0.8, label=r"$\pi_g$ (target)")
            if pihat is not None and len(pihat) == len(order):
                pihat_sorted = pihat[order]
                ax.barh(y + 0.35, pihat_sorted, height=0.35, align="center",
                        color=color, alpha=0.7,
                        label=r"$\hat\pi_g(S)$ (subset)")

            ax.set_yticks(y + 0.175)
            ax.set_yticklabels(states_sorted, fontsize=6)
            ax.set_xlabel("Proportion", fontsize=9)
            ax.set_title(title, fontsize=10, pad=6)
            ax.legend(fontsize=7, loc="lower right")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis="x")
            ax.tick_params(labelsize=8)

        fig.suptitle("State composition shift at $k{=}300$: "
                     r"$\pi_g$ vs $\hat\pi_g(S)$",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "composition_shift_sankey.pdf"))

    def fig_constraint_tightness_vs_fidelity(self, df: pd.DataFrame) -> str:
        """Constraint tightness vs. operator fidelity (Nystrom error).

        Compares runs with different constraint regimes at k=300:
          R1 (population-share), R4 (municipality-share), R5 (joint),
          R6 (no proportionality / exact-k only).

        X-axis: geographic KL divergence (tighter -> lower KL -> more
        constrained).  Y-axis: Nystrom error (lower -> better fidelity).
        Ideally one would like both to be low -- the plot reveals the
        constraint-fidelity trade-off front.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No k=300 data", ha="center", va="center",
                    transform=ax.transAxes)
            return _save(fig, os.path.join(self.fig_dir, "constraint_tightness_vs_fidelity.pdf"))

        # Map run IDs to constraint regime labels
        regimes = {
            "R1": ("Pop-share", "#1f77b4", "o"),
            "R4": ("Muni-share", "#ff7f0e", "s"),
            "R5": ("Joint", "#2ca02c", "D"),
            "R6": ("None (exact-k)", "#d62728", "^"),
        }

        # Determine which geo_kl variant is available
        kl_col = None
        for candidate in ["geo_kl_pop", "geo_kl", "geo_kl_muni"]:
            if candidate in d.columns and d[candidate].notna().any():
                kl_col = candidate
                break

        fidelity_col = None
        for candidate in ["nystrom_error", "kpca_distortion", "krr_rmse_4G"]:
            if candidate in d.columns and d[candidate].notna().any():
                fidelity_col = candidate
                break

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Panel (a): scatter KL vs Nystrom
        ax = axes[0]
        has_data = False
        for rid, (label, color, marker) in regimes.items():
            sub = d[d["run_id"].astype(str).str.startswith(rid)]
            if sub.empty or kl_col is None or fidelity_col is None:
                continue
            vals = sub[[kl_col, fidelity_col]].dropna()
            if vals.empty:
                continue
            has_data = True
            ax.scatter(vals[fidelity_col], vals[kl_col],
                       label=label, color=color, marker=marker, s=50, alpha=0.8)

        if not has_data:
            ax.text(0.5, 0.5, "Insufficient data for scatter",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(fidelity_col.replace("_", " ") if fidelity_col else "Fidelity metric")
        ax.set_ylabel(f"Geographic divergence ({kl_col})" if kl_col else "Geographic KL")
        ax.set_title("(a) Constraint tightness vs. operator fidelity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel (b): grouped bar chart of geo and fidelity metrics side by side
        ax = axes[1]
        bar_metrics = []
        for c in ["geo_kl", "geo_kl_pop", "geo_l1", "nystrom_error", "kpca_distortion"]:
            if c in d.columns and d[c].notna().any():
                bar_metrics.append(c)
        bar_metrics = bar_metrics[:4]  # Limit to 4 for readability

        if bar_metrics:
            bar_data = {}
            for rid, (label, _, _) in regimes.items():
                sub = d[d["run_id"].astype(str).str.startswith(rid)]
                if not sub.empty:
                    bar_data[label] = {m: float(sub[m].mean()) for m in bar_metrics if m in sub.columns}
            if bar_data:
                bdf = pd.DataFrame(bar_data).T
                x = np.arange(len(bdf.index))
                width = 0.8 / max(len(bdf.columns), 1)
                for i, col in enumerate(bdf.columns):
                    ax.bar(x + i * width, bdf[col], width=width,
                           label=col.replace("_", " "), alpha=0.85)
                ax.set_xticks(x + width * (len(bdf.columns) - 1) / 2)
                ax.set_xticklabels(bdf.index, rotation=15, fontsize=8)
                ax.legend(fontsize=7)
        ax.set_title("(b) Metrics by constraint regime ($k=300$)")
        ax.grid(True, alpha=0.2, axis="y")

        fig.suptitle("Constraint tightness vs. fidelity trade-off", y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "constraint_tightness_vs_fidelity.pdf"))

    def fig_skl_ablation_comparison(self, df: pd.DataFrame) -> str:
        """R7 SKL ablation: bi-objective (R9 or R1) vs tri-objective (R7).

        The manuscript introduces SKL (symmetrised KL) as a third objective
        that exploits VAE latent-space uncertainty.  This figure compares
        key metrics between the bi-objective baseline (R9: VAE, MMD+Sinkhorn)
        and the tri-objective variant (R7: VAE, MMD+Sinkhorn+SKL) at k=300.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No k=300 data", ha="center", va="center",
                    transform=ax.transAxes)
            return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))

        # Identify rows for R7 (tri-obj) and R9 (bi-obj, same VAE space)
        r7 = d[d["run_id"].astype(str).str.startswith("R7")]
        r9 = d[d["run_id"].astype(str).str.startswith("R9")]
        r1 = d[d["run_id"].astype(str).str.startswith("R1")]

        metrics = ["nystrom_error", "kpca_distortion", "krr_rmse_4G",
                    "geo_kl", "geo_kl_pop", "geo_l1"]
        metrics = [m for m in metrics if m in d.columns and d[m].notna().any()]

        configs = {}
        if not r7.empty:
            configs["R7 (tri-obj)"] = r7
        if not r9.empty:
            configs["R9 (bi-obj VAE)"] = r9
        if not r1.empty:
            configs["R1 (bi-obj raw)"] = r1

        if not configs or not metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient R7/R9 data for comparison",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))

        n_metrics = min(len(metrics), 6)
        ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)

        colors = {"R7 (tri-obj)": "#2ca02c", "R9 (bi-obj VAE)": "#ff7f0e",
                  "R1 (bi-obj raw)": "#1f77b4"}

        for idx, m in enumerate(metrics[:n_metrics]):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            vals = []
            labels = []
            bar_colors = []
            for label, sub in configs.items():
                if m in sub.columns:
                    v = sub[m].dropna()
                    if not v.empty:
                        vals.append(float(v.mean()))
                        labels.append(label)
                        bar_colors.append(colors.get(label, "#999999"))
            if vals:
                bars = ax.bar(range(len(vals)), vals, color=bar_colors, alpha=0.85)
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(labels, rotation=20, fontsize=7)
                # Add value annotations
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(m.replace("_", " "), fontsize=9)
            ax.grid(True, alpha=0.2, axis="y")

        # Hide unused axes
        for idx in range(n_metrics, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle("SKL ablation: bi-objective vs. tri-objective ($k=300$)", y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))
