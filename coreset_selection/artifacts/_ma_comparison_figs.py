"""Comparison figure methods for ManuscriptArtifacts (mixin).

Manuscript figures rendered here (R/ggplot2 with matplotlib fallback):
  Fig 2: fig_geo_ablation_scatter       → geo_ablation_tradeoff_scatter.pdf
  Fig 5: fig_regional_validity_k300     → regional_validity_k300.pdf
  Fig 5b: fig_objective_ablation_bars   → objective_ablation_bars_k300.pdf  (Pareto scatter)
  Fig 6a: fig_constraint_comparison     → constraint_comparison_bars_k300.pdf (Pareto scatter)
  Fig 6b: fig_baseline_comparison       → baseline_comparison_grouped.pdf   (Pareto scatter + baselines)
  Fig 7: fig_representation_transfer    → representation_transfer_bars.pdf  (Pareto scatter)
  Fig 8: fig_objective_metric_alignment → objective_metric_alignment_heatmap.pdf
  Fig 9: fig_downstream_model_heatmap   → downstream_model_heatmap.pdf
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ._ma_helpers import _save, _save_r, use_r


class ComparisonFigsMixin:
    """Mixin providing comparison/diagnostic figure methods."""

    # ------------------------------------------------------------------
    # Fig 2: Geo ablation scatter (Section VIII.B)
    # ------------------------------------------------------------------
    def fig_geo_ablation_scatter(self, df: pd.DataFrame) -> str:
        r"""Fig 2: Composition drift vs Nystrom error (R6, Section VIII.B)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "geo_ablation_tradeoff_scatter.pdf")

        if df.empty or "run_id" not in df.columns:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

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
            return _save(fig, out_path)

        # ---- Build tidy data for R ----
        tidy_rows = []
        cr = d_r6.get("constraint_regime",
                       pd.Series(["unconstrained"] * len(d_r6), index=d_r6.index))
        for i, row in d_r6.iterrows():
            tidy_rows.append({
                "nystrom_error": float(row["nystrom_error"]),
                "geo_l1": float(row["geo_l1"]),
                "constraint_regime": str(cr.loc[i]) if i in cr.index else "unconstrained",
                "is_r1_knee": False,
            })

        # Add R1 knee point
        d_r1 = df[df["run_id"].astype(str).str.contains("R1")].copy()
        d_r1 = d_r1[d_r1["k"].fillna(300).astype(int) == 300]
        if not d_r1.empty and "geo_l1" in d_r1.columns and "nystrom_error" in d_r1.columns:
            knee = d_r1.loc[d_r1["nystrom_error"].idxmin()]
            tidy_rows.append({
                "nystrom_error": float(knee["nystrom_error"]),
                "geo_l1": float(knee["geo_l1"]),
                "constraint_regime": "r1_knee",
                "is_r1_knee": True,
            })

        # Feasible boundary
        extra_args = {}
        if "geo_kl" in d_r6.columns:
            try:
                tau_ref = 0.02
                feas = d_r6[d_r6["geo_kl"] <= tau_ref]
                if not feas.empty:
                    extra_args["feasible_boundary"] = str(float(feas["geo_l1"].max()))
            except Exception:
                pass

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            regime_styles = {
                "exactk_only": ("o", "#1f77b4", "Exact-$k$ only"),
                "exactk":      ("o", "#1f77b4", "Exact-$k$ only"),
                "quota":       ("s", "#ff7f0e", "Quota-matched"),
                "quota+exactk":("^", "#2ca02c", "Quota + exact-$k$"),
                "unconstrained":("D","#9467bd", "Unconstrained"),
                "none":        ("D", "#9467bd", "Unconstrained"),
            }
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
            if not d_r1.empty and "geo_l1" in d_r1.columns and "nystrom_error" in d_r1.columns:
                knee_row = d_r1.loc[d_r1["nystrom_error"].idxmin()]
                ax.scatter(knee_row["nystrom_error"], knee_row["geo_l1"],
                           s=90, marker="*", color="#d62728", edgecolors="k",
                           linewidths=0.5, zorder=10, label="R1 knee (constrained)")
            ax.set_xlabel(r"Nyström error $e_{\mathrm{Nys}}$", fontsize=10)
            ax.set_ylabel(r"Geographic $\ell_1$ drift", fontsize=10)
            ax.set_xscale("log")
            ax.tick_params(labelsize=9)
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9,
                      edgecolor="0.8", handletextpad=0.4)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_title("Composition drift vs.\\ Nyström error (R6, $k{=}300$)",
                          fontsize=10, pad=6)
            fig.tight_layout()
            return _save(fig, out_path)

        return _save_r("fig_geo_ablation_scatter.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback, extra_args=extra_args or None)

    # ------------------------------------------------------------------
    # Fig 5: Regional validity at k=300 (Section VIII.D)
    # ------------------------------------------------------------------
    def fig_regional_validity_k300(self, df: pd.DataFrame) -> str:
        r"""Fig 5: State-conditioned KPI stability at k = 300 (Section VIII.D)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "regional_validity_k300.pdf")

        if df.empty or "k" not in df.columns:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        r1 = d[d["run_id"].astype(str).str.contains("R1")]
        r5 = d[d["run_id"].astype(str).str.contains("R5")]

        all_cols = set(d.columns)
        max_drift_4g = next((c for c in all_cols if "max" in c and "drift" in c and "4G" in c), None)
        avg_drift_4g = next((c for c in all_cols if "avg" in c and "drift" in c and "4G" in c), None)
        tau_4g       = next((c for c in all_cols if "kendall" in c.lower() and "4G" in c), None)
        max_drift_5g = next((c for c in all_cols if "max" in c and "drift" in c and "5G" in c), None)
        avg_drift_5g = next((c for c in all_cols if "avg" in c and "drift" in c and "5G" in c), None)
        tau_5g       = next((c for c in all_cols if "kendall" in c.lower() and "5G" in c), None)

        if max_drift_4g is None:
            max_drift_4g = next((c for c in all_cols if "max" in c and ("kpi" in c or "drift" in c)), None)
        if avg_drift_4g is None:
            avg_drift_4g = next((c for c in all_cols if "avg" in c and ("kpi" in c or "drift" in c)), None)
        if tau_4g is None:
            tau_4g = next((c for c in all_cols if "kendall" in c.lower() or "tau" in c.lower()), None)

        found_any = any(c is not None for c in [max_drift_4g, avg_drift_4g, tau_4g])
        if not found_any or (r1.empty and r5.empty):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No KPI stability data found for R1/R5 at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        def _safe_mean(sub, col):
            if col is not None and col in sub.columns:
                v = sub[col].dropna()
                return float(v.mean()) if len(v) > 0 else np.nan
            return np.nan

        # ---- Build tidy data for R ----
        metric_groups = [
            ("Max drift",       [(max_drift_4g, "4G"), (max_drift_5g, "5G")]),
            ("Avg drift",       [(avg_drift_4g, "4G"), (avg_drift_5g, "5G")]),
            ("Kendall's tau",   [(tau_4g, "4G"),       (tau_5g,       "5G")]),
        ]

        tidy_rows = []
        for metric_name, targets in metric_groups:
            for col, tech in targets:
                if col is None:
                    continue
                for run_label, run_df in [("R1", r1), ("R5", r5)]:
                    val = _safe_mean(run_df, col)
                    if np.isfinite(val):
                        tidy_rows.append({
                            "run": run_label, "target": tech,
                            "metric_name": metric_name, "value": val,
                        })

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            from matplotlib.patches import Patch
            n_panels = len(metric_groups)
            fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 3.5))
            if n_panels == 1:
                axes = [axes]
            panel_labels = ["(a)", "(b)", "(c)"]
            bar_width = 0.18
            colors_r1 = ["#1f77b4", "#aec7e8"]
            colors_r5 = ["#ff7f0e", "#ffbb78"]
            for p_idx, (ax, (group_name, targets)) in enumerate(zip(axes, metric_groups)):
                x = 0
                x_positions, tick_labels = [], []
                for col, tech in targets:
                    if col is None:
                        continue
                    r1_val = _safe_mean(r1, col)
                    r5_val = _safe_mean(r5, col)
                    c_r1 = colors_r1[0] if tech == "4G" else colors_r1[1]
                    c_r5 = colors_r5[0] if tech == "4G" else colors_r5[1]
                    if np.isfinite(r1_val):
                        ax.bar(x, r1_val, width=bar_width, color=c_r1, edgecolor="k", linewidth=0.3)
                    if np.isfinite(r5_val):
                        ax.bar(x + bar_width, r5_val, width=bar_width, color=c_r5, edgecolor="k", linewidth=0.3)
                    x_positions.append(x + bar_width / 2)
                    tick_labels.append(tech)
                    x += 2.5 * bar_width
                ax.set_xticks(x_positions)
                ax.set_xticklabels(tick_labels, fontsize=9)
                ax.set_ylabel(group_name, fontsize=10)
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)
                ax.annotate(panel_labels[p_idx], xy=(0.03, 0.95),
                            xycoords="axes fraction", fontsize=10, fontweight="bold", va="top")
            legend_elements = [
                Patch(facecolor=colors_r1[0], edgecolor="k", linewidth=0.3, label="R1 (4G)"),
                Patch(facecolor=colors_r1[1], edgecolor="k", linewidth=0.3, label="R1 (5G)"),
                Patch(facecolor=colors_r5[0], edgecolor="k", linewidth=0.3, label="R5 (4G)"),
                Patch(facecolor=colors_r5[1], edgecolor="k", linewidth=0.3, label="R5 (5G)"),
            ]
            fig.legend(handles=legend_elements, loc="lower center",
                       ncol=4, fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
            fig.suptitle("State-conditioned KPI stability at $k{=}300$: R1 vs R5",
                          fontsize=11, y=1.02)
            fig.tight_layout(rect=[0, 0.06, 1, 0.98])
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_regional_validity_k300.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

    # ------------------------------------------------------------------
    # Fig 8: Objective-metric alignment heatmap (Section VIII.K)
    # ------------------------------------------------------------------
    def fig_objective_metric_alignment(self, df: pd.DataFrame) -> str:
        r"""Fig 8: Objective-metric Spearman rho heatmap (Section VIII.K)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "objective_metric_alignment_heatmap.pdf")

        try:
            from scipy.stats import spearmanr
        except ImportError:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "scipy not available", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9)
            return _save(fig, out_path)

        if df.empty or "k" not in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        # Load data: prefer R11 artefacts
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
            fdf = df[df["k"].fillna(300).astype(int) == 300].copy()

        obj_cols = [c for c in fdf.columns if c.startswith("f_")]
        met_candidates = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_4G", "krr_rmse_5G",
            "geo_kl", "geo_l1", "geo_kl_pop", "geo_l1_pop",
        ]
        met_cols = [c for c in met_candidates if c in fdf.columns]

        if not obj_cols or not met_cols:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "Missing objective (f_*) or metric columns.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_axis_off()
            return _save(fig, out_path)

        # Compute Spearman matrix
        corr = np.full((len(obj_cols), len(met_cols)), np.nan)
        for i, o in enumerate(obj_cols):
            for j, m in enumerate(met_cols):
                valid = fdf[[o, m]].dropna()
                if len(valid) > 3:
                    corr[i, j] = spearmanr(valid[o], valid[m]).correlation

        # Pretty labels
        obj_labels = [c.replace("f_", "").replace("mmd", "MMD")
                       .replace("sinkhorn", "SD").replace("skl", "SKL")
                      for c in obj_cols]
        _label_map = {
            "nystrom_error": "e_Nys", "kpca_distortion": "e_kPCA",
            "krr_rmse_4G": "RMSE_4G", "krr_rmse_5G": "RMSE_5G",
            "geo_kl": "KL_geo", "geo_l1": "l1",
            "geo_kl_pop": "KL_pop", "geo_l1_pop": "l1_pop",
        }
        met_labels = [_label_map.get(c, c.replace("_", " ")) for c in met_cols]

        # ---- Build tidy data for R ----
        tidy_rows = []
        for i, ol in enumerate(obj_labels):
            for j, ml in enumerate(met_labels):
                val = corr[i, j]
                if np.isfinite(val):
                    tidy_rows.append({
                        "objective": ol, "metric": ml, "spearman_rho": float(val),
                    })

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            from matplotlib.colors import TwoSlopeNorm
            n_obj, n_met = len(obj_cols), len(met_cols)
            fig_w = max(3.5, n_met * 0.9 + 1.2)
            fig_h = max(2.0, n_obj * 0.65 + 1.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            vmin = np.nanmin(corr) if np.any(np.isfinite(corr)) else -1
            vmax = np.nanmax(corr) if np.any(np.isfinite(corr)) else 1
            abs_lim = max(abs(vmin), abs(vmax), 0.05)
            norm = TwoSlopeNorm(vmin=-abs_lim, vcenter=0, vmax=abs_lim)
            im = ax.imshow(corr, aspect="auto", cmap="RdBu_r", norm=norm)
            ax.set_yticks(range(n_obj))
            ax.set_yticklabels(obj_labels, fontsize=10)
            ax.set_xticks(range(n_met))
            ax.set_xticklabels(met_labels, rotation=40, ha="right", fontsize=9)
            for i in range(n_obj):
                for j in range(n_met):
                    val = corr[i, j]
                    if np.isfinite(val):
                        color = "white" if abs(val) > 0.55 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color=color, fontweight="medium")
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.06)
            cbar.set_label(r"Spearman $\rho$", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(r"Spearman $\rho$: optimisation objectives vs.\ raw-space metrics",
                         fontsize=10, pad=8)
            fig.tight_layout()
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_objective_metric_heatmap.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

    # ------------------------------------------------------------------
    # Fig N3: Objective ablation bars (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_objective_ablation_bars(self, df: pd.DataFrame) -> str:
        r"""Fig 5: Objective ablation — Pareto fronts for R1/R2/R3 at k=300.

        Replaces the former grouped bar chart with overlaid Pareto front
        scatters in downstream metric space (e_Nys vs RMSE_4G).
        """
        import matplotlib.pyplot as plt
        from ._ma_helpers import plot_pareto_scatter, get_knee_values

        out_path = os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf")

        run_configs = [
            ("R1 (MMD+SD)", "R1", "#1f77b4"),
            ("R2 (MMD only)", "R2", "#ff7f0e"),
            ("R3 (SD only)", "R3", "#2ca02c"),
        ]

        # Load front metrics for each run
        front_dfs = {}
        colors = {}
        knee_vals = {}
        has_data = False
        for label, rid, color in run_configs:
            fdf = self._load_front_metrics(run_pattern=rid, space="raw")
            if fdf.empty:
                # Fallback: use all_results.csv rows
                fdf = df[(df["run_id"].astype(str).str.contains(rid))
                         & (df["k"].fillna(300).astype(int) == 300)].copy()
            else:
                fdf = fdf[fdf["k"].fillna(300).astype(int) == 300].copy()
            if not fdf.empty:
                front_dfs[label] = fdf
                colors[label] = color
                knee_vals[label] = get_knee_values(df, rid, k=300)
                has_data = True

        if not has_data:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.text(0.5, 0.5, "No R1/R2/R3 Pareto front data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_pareto_scatter(
            ax, front_dfs, colors,
            x_metric="nystrom_error", y_metric="krr_rmse_4G",
            knee_values=knee_vals,
            title="Objective ablation at $k{=}300$",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4, ncol=1)
        fig.tight_layout()
        return _save(fig, out_path)

    # ------------------------------------------------------------------
    # Fig N4: Constraint comparison bars (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_constraint_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig 6: Constraint regime comparison — Pareto fronts for R1/R4/R5/R6 at k=300.

        Replaces the former grouped bar chart with overlaid Pareto front
        scatters in downstream metric space (e_Nys vs RMSE_4G).
        """
        import matplotlib.pyplot as plt
        from ._ma_helpers import plot_pareto_scatter, get_knee_values

        out_path = os.path.join(self.fig_dir, "constraint_comparison_bars_k300.pdf")

        run_configs = [
            ("R1 (pop-share)", "R1", "#1f77b4"),
            ("R4 (muni-quota)", "R4", "#ff7f0e"),
            ("R5 (joint)", "R5", "#2ca02c"),
            ("R6 (none)", "R6", "#d62728"),
        ]

        # Load front metrics for each run
        front_dfs = {}
        colors = {}
        knee_vals = {}
        has_data = False
        for label, rid, color in run_configs:
            fdf = self._load_front_metrics(run_pattern=rid, space="raw")
            if fdf.empty:
                # Fallback: use all_results.csv rows
                fdf = df[(df["run_id"].astype(str).str.contains(rid))
                         & (df["k"].fillna(300).astype(int) == 300)].copy()
            else:
                fdf = fdf[fdf["k"].fillna(300).astype(int) == 300].copy()
            if not fdf.empty:
                front_dfs[label] = fdf
                colors[label] = color
                knee_vals[label] = get_knee_values(df, rid, k=300)
                has_data = True

        if not has_data:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.text(0.5, 0.5, "No R1/R4/R5/R6 Pareto front data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_pareto_scatter(
            ax, front_dfs, colors,
            x_metric="nystrom_error", y_metric="krr_rmse_4G",
            knee_values=knee_vals,
            title="Constraint regime comparison at $k{=}300$",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4, ncol=1)
        fig.tight_layout()
        return _save(fig, out_path)

    # ------------------------------------------------------------------
    # Fig 7: Representation transfer bars (Section VIII.G)
    # ------------------------------------------------------------------
    def fig_representation_transfer(self, df: pd.DataFrame) -> str:
        r"""Fig 7: Representation transfer — Pareto fronts for R1/R8/R9 at k=300.

        Replaces the former grouped bar chart with overlaid Pareto front
        scatters in downstream metric space (e_Nys vs RMSE_4G).
        """
        import matplotlib.pyplot as plt
        from ._ma_helpers import plot_pareto_scatter, get_knee_values

        out_path = os.path.join(self.fig_dir, "representation_transfer_bars.pdf")

        run_configs = [
            ("R1 (VAE mean)", "R1", "#1f77b4"),
            ("R8 (raw)", "R8", "#ff7f0e"),
            ("R9 (PCA)", "R9", "#2ca02c"),
        ]

        # Load front metrics for each run
        front_dfs = {}
        colors = {}
        knee_vals = {}
        has_data = False
        for label, rid, color in run_configs:
            fdf = self._load_front_metrics(run_pattern=rid, space="raw")
            if fdf.empty:
                # Fallback: use all_results.csv rows
                fdf = df[(df["run_id"].astype(str).str.contains(rid))
                         & (df["k"].fillna(300).astype(int) == 300)].copy()
            else:
                fdf = fdf[fdf["k"].fillna(300).astype(int) == 300].copy()
            if not fdf.empty:
                front_dfs[label] = fdf
                colors[label] = color
                knee_vals[label] = get_knee_values(df, rid, k=300)
                has_data = True

        if not has_data:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.text(0.5, 0.5, "No R1/R8/R9 Pareto front data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_pareto_scatter(
            ax, front_dfs, colors,
            x_metric="nystrom_error", y_metric="krr_rmse_4G",
            knee_values=knee_vals,
            title="Representation transfer at $k{=}300$",
        )
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4, ncol=1)
        fig.tight_layout()
        return _save(fig, out_path)

    # ------------------------------------------------------------------
    # Fig 6: Baseline comparison (Section VIII.E)
    # ------------------------------------------------------------------
    def fig_baseline_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig 6: Baseline comparison — R1 Pareto front + baselines at k=300.

        Replaces the former grouped bar chart.  The R1 Pareto front is shown
        as a point cloud in downstream metric space (e_Nys vs RMSE_4G), with
        knee (star) and best-per-metric (diamond/triangle) marked.  Each
        baseline method appears as a single distinctly-shaped point.
        """
        import matplotlib.pyplot as plt
        from ._ma_helpers import (plot_pareto_scatter, get_knee_values,
                                  resolve_metric)

        out_path = os.path.join(self.fig_dir, "baseline_comparison_grouped.pdf")

        # --- R1 Pareto front ---
        front_dfs = {}
        colors = {}
        knee_vals = {}

        fdf_r1 = self._load_front_metrics(run_pattern="R1", space="raw")
        if fdf_r1.empty:
            fdf_r1 = df[(df["run_id"].astype(str).str.contains("R1"))
                        & (df["k"].fillna(300).astype(int) == 300)].copy()
        else:
            fdf_r1 = fdf_r1[fdf_r1["k"].fillna(300).astype(int) == 300].copy()

        if not fdf_r1.empty:
            front_dfs["R1 (NSGA-II)"] = fdf_r1
            colors["R1 (NSGA-II)"] = "#1f77b4"
            knee_vals["R1 (NSGA-II)"] = get_knee_values(df, "R1", k=300)

        # --- Baseline single points from R10 ---
        d_all = df[df["k"].fillna(300).astype(int) == 300].copy()

        # Resolve metric aliases in d_all
        _aliases = {"krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"]}
        for canon, aliases in _aliases.items():
            if canon not in d_all.columns:
                for alias in aliases:
                    if alias in d_all.columns:
                        d_all[canon] = d_all[alias]
                        break

        d_base = d_all[d_all["run_id"].astype(str).str.contains("R10")].copy()
        if d_base.empty and "method" in d_all.columns:
            d_base = d_all[d_all["method"].astype(str) != "nsga2"].copy()

        _method_labels = {
            "uniform": "Uniform", "kmeans": "k-Means", "herding": "Herding",
            "farthest_first": "FF", "kernel_thinning": "K.Thin.",
            "leverage": "RLS", "dpp": "DPP",
        }

        baseline_points = {}
        baseline_markers = {}
        _markers = ["s", "^", "P", "X", "h", "D", "p", "8"]
        has_method = "method" in d_base.columns

        if has_method and not d_base.empty:
            for i, method in enumerate(sorted(d_base["method"].unique())):
                sub = d_base[d_base["method"] == method]
                blabel = _method_labels.get(method, method)
                bvals = {}
                for metric in ["nystrom_error", "krr_rmse_4G"]:
                    if metric in sub.columns and sub[metric].notna().any():
                        bvals[metric] = float(sub[metric].mean())
                if bvals:
                    baseline_points[blabel] = bvals
                    baseline_markers[blabel] = _markers[i % len(_markers)]
        elif not d_base.empty:
            # Single unnamed baseline
            bvals = {}
            for metric in ["nystrom_error", "krr_rmse_4G"]:
                if metric in d_base.columns and d_base[metric].notna().any():
                    bvals[metric] = float(d_base[metric].mean())
            if bvals:
                baseline_points["Baseline"] = bvals
                baseline_markers["Baseline"] = "s"

        if not front_dfs and not baseline_points:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.text(0.5, 0.5, "No baseline comparison data (R1/R10) at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_pareto_scatter(
            ax, front_dfs, colors,
            x_metric="nystrom_error", y_metric="krr_rmse_4G",
            knee_values=knee_vals,
            baseline_points=baseline_points if baseline_points else None,
            baseline_markers=baseline_markers if baseline_markers else None,
            title="Baseline comparison at $k{=}300$",
        )
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4, ncol=2)
        fig.tight_layout()
        return _save(fig, out_path)

    # ------------------------------------------------------------------
    # Fig N8: State KPI drift heatmap (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_state_kpi_heatmap(self, df: pd.DataFrame) -> str:
        r"""Fig N8: Per-state KPI drift heatmap (R1, k=300)."""
        import matplotlib.pyplot as plt

        state_files = glob.glob(os.path.join(self.runs_root, "**/state_kpi_drift*.csv"), recursive=True)
        sdf = None
        if state_files:
            try: sdf = pd.read_csv(state_files[0])
            except Exception: pass
        if sdf is None:
            d = df[df["run_id"].astype(str).str.contains("R1")].copy()
            d = d[d["k"].fillna(300).astype(int) == 300]
            state_drift_cols = [c for c in d.columns if c.startswith("state_drift_") or c.startswith("kpi_drift_")]
            if state_drift_cols and not d.empty:
                rows = []
                for col in state_drift_cols:
                    parts = col.replace("state_drift_", "").replace("kpi_drift_", "").split("_")
                    if len(parts) >= 2:
                        rows.append({"state": parts[0], "target": "_".join(parts[1:]),
                                     "drift": float(d[col].mean())})
                if rows: sdf = pd.DataFrame(rows)
        if sdf is None or sdf.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "Per-state KPI drift data not yet generated.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "state_kpi_heatmap.pdf"))

        pivot = sdf.pivot_table(index="state", columns="target", values="drift", aggfunc="mean")
        pivot["_mean_drift"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean_drift", ascending=False).drop(columns=["_mean_drift"])
        n_states = len(pivot.index); n_targets = len(pivot.columns)
        fig_h = max(5, n_states * 0.22 + 1.5); fig_w = max(4, n_targets * 1.5 + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        data = pivot.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(n_states)); ax.set_yticklabels(list(pivot.index), fontsize=7)
        ax.set_xticks(range(n_targets))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        for i in range(n_states):
            for j in range(n_targets):
                val = data[i, j]
                if np.isfinite(val):
                    color = "white" if val > np.nanpercentile(data, 75) else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(r"$|\mu_g^S - \mu_g^{\rm full}|$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title("Per-state KPI drift (R1, $k{=}300$)", fontsize=10, pad=8)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "state_kpi_heatmap.pdf"))

    # ------------------------------------------------------------------
    # Fig N9: Composition shift (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_composition_shift(self, df: pd.DataFrame) -> str:
        r"""Fig N9: State composition shift — R6 vs R1."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        pi_cols = [c for c in d.columns if c.startswith("pi_g_")]
        pihat_cols = [c for c in d.columns if c.startswith("pihat_g_")]
        geo_files = glob.glob(os.path.join(self.runs_root, "**/geo_state_proportions*.csv"), recursive=True)
        geo_df = None
        if geo_files:
            try: geo_df = pd.read_csv(geo_files[0])
            except Exception: pass

        if geo_df is not None and "state" in geo_df.columns:
            states = geo_df["state"].values
            pi_target = geo_df["pi_target"].values if "pi_target" in geo_df.columns else np.ones(len(states)) / len(states)
            def _get_pihat(run_pattern):
                sub = geo_df[geo_df.get("run_id", pd.Series(dtype=str)).astype(str).str.contains(run_pattern)]
                return sub["pihat"].values if not sub.empty and "pihat" in sub.columns else None
            pihat_r6 = _get_pihat("R6"); pihat_r1 = _get_pihat("R1")
        elif pi_cols and pihat_cols:
            states = sorted(set(c.replace("pi_g_", "") for c in pi_cols))
            pi_target = np.array([d[f"pi_g_{s}"].mean() for s in states if f"pi_g_{s}" in d.columns])
            def _get_pihat_from_cols(rid):
                sub = d[d["run_id"].astype(str).str.contains(rid)]
                return np.array([sub[f"pihat_g_{s}"].mean() for s in states if f"pihat_g_{s}" in sub.columns]) if not sub.empty else None
            pihat_r6 = _get_pihat_from_cols("R6"); pihat_r1 = _get_pihat_from_cols("R1")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No state-level proportion data available.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "composition_shift_sankey.pdf"))

        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(states) * 0.2 + 1)))
        order = np.argsort(-pi_target)
        states_sorted = np.array(states)[order]; pi_sorted = pi_target[order]
        y = np.arange(len(states_sorted))
        panels = [(axes[0], pihat_r6, "R6 (unconstrained)", "#d62728"),
                   (axes[1], pihat_r1, "R1 (constrained)", "#1f77b4")]
        for ax, pihat, title, color in panels:
            ax.barh(y, pi_sorted, height=0.35, align="center", color="#cccccc", alpha=0.8, label=r"$\pi_g$ (target)")
            if pihat is not None and len(pihat) == len(order):
                ax.barh(y + 0.35, pihat[order], height=0.35, align="center", color=color, alpha=0.7,
                        label=r"$\hat\pi_g(S)$ (subset)")
            ax.set_yticks(y + 0.175); ax.set_yticklabels(states_sorted, fontsize=6)
            ax.set_xlabel("Proportion", fontsize=9); ax.set_title(title, fontsize=10, pad=6)
            ax.legend(fontsize=7, loc="lower right"); ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis="x"); ax.tick_params(labelsize=8)
        fig.suptitle(r"State composition shift at $k{=}300$", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "composition_shift_sankey.pdf"))

    # ------------------------------------------------------------------
    # Constraint tightness vs fidelity (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_constraint_tightness_vs_fidelity(self, df: pd.DataFrame) -> str:
        """Constraint tightness vs. operator fidelity."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No k=300 data", ha="center", va="center", transform=ax.transAxes)
            return _save(fig, os.path.join(self.fig_dir, "constraint_tightness_vs_fidelity.pdf"))

        regimes = {"R1": ("Pop-share", "#1f77b4", "o"), "R4": ("Muni-share", "#ff7f0e", "s"),
                    "R5": ("Joint", "#2ca02c", "D"), "R6": ("None (exact-k)", "#d62728", "^")}
        kl_col = None
        for cand in ["geo_kl_pop", "geo_kl", "geo_kl_muni"]:
            if cand in d.columns and d[cand].notna().any():
                kl_col = cand; break
        fidelity_col = None
        for cand in ["nystrom_error", "kpca_distortion", "krr_rmse_4G"]:
            if cand in d.columns and d[cand].notna().any():
                fidelity_col = cand; break

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        has_data = False
        for rid, (label, color, marker) in regimes.items():
            sub = d[d["run_id"].astype(str).str.startswith(rid)]
            if sub.empty or kl_col is None or fidelity_col is None: continue
            vals = sub[[kl_col, fidelity_col]].dropna()
            if vals.empty: continue
            has_data = True
            ax.scatter(vals[fidelity_col], vals[kl_col], label=label, color=color, marker=marker, s=50, alpha=0.8)
        if not has_data:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel(fidelity_col.replace("_", " ") if fidelity_col else "Fidelity")
        ax.set_ylabel(f"Geographic divergence ({kl_col})" if kl_col else "Geographic KL")
        ax.set_title("(a) Constraint tightness vs. fidelity"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1]
        bar_metrics = [c for c in ["geo_kl", "geo_kl_pop", "geo_l1", "nystrom_error", "kpca_distortion"]
                       if c in d.columns and d[c].notna().any()][:4]
        if bar_metrics:
            bar_data = {}
            for rid, (label, _, _) in regimes.items():
                sub = d[d["run_id"].astype(str).str.startswith(rid)]
                if not sub.empty:
                    bar_data[label] = {m: float(sub[m].mean()) for m in bar_metrics if m in sub.columns}
            if bar_data:
                bdf = pd.DataFrame(bar_data).T
                x = np.arange(len(bdf.index)); width = 0.8 / max(len(bdf.columns), 1)
                for i, col in enumerate(bdf.columns):
                    ax.bar(x + i * width, bdf[col], width=width, label=col.replace("_", " "), alpha=0.85)
                ax.set_xticks(x + width * (len(bdf.columns) - 1) / 2)
                ax.set_xticklabels(bdf.index, rotation=15, fontsize=8); ax.legend(fontsize=7)
        ax.set_title("(b) Metrics by constraint regime ($k=300$)"); ax.grid(True, alpha=0.2, axis="y")
        fig.suptitle("Constraint tightness vs. fidelity trade-off", y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "constraint_tightness_vs_fidelity.pdf"))

    # ------------------------------------------------------------------
    # SKL ablation comparison (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_skl_ablation_comparison(self, df: pd.DataFrame) -> str:
        """R7 SKL ablation: bi-objective vs tri-objective."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No k=300 data", ha="center", va="center", transform=ax.transAxes)
            return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))

        r7 = d[d["run_id"].astype(str).str.startswith("R7")]
        r9 = d[d["run_id"].astype(str).str.startswith("R9")]
        r1 = d[d["run_id"].astype(str).str.startswith("R1")]
        metrics = [m for m in ["nystrom_error", "kpca_distortion", "krr_rmse_4G", "geo_kl", "geo_kl_pop", "geo_l1"]
                   if m in d.columns and d[m].notna().any()]
        configs = {}
        if not r7.empty: configs["R7 (tri-obj)"] = r7
        if not r9.empty: configs["R9 (bi-obj VAE)"] = r9
        if not r1.empty: configs["R1 (bi-obj raw)"] = r1
        if not configs or not metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient R7/R9 data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))

        n_metrics = min(len(metrics), 6); ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
        colors = {"R7 (tri-obj)": "#2ca02c", "R9 (bi-obj VAE)": "#ff7f0e", "R1 (bi-obj raw)": "#1f77b4"}
        for idx, m in enumerate(metrics[:n_metrics]):
            r, c = divmod(idx, ncols); ax = axes[r][c]
            vals, labels, bar_colors = [], [], []
            for label, sub in configs.items():
                if m in sub.columns:
                    v = sub[m].dropna()
                    if not v.empty:
                        vals.append(float(v.mean())); labels.append(label)
                        bar_colors.append(colors.get(label, "#999999"))
            if vals:
                bars = ax.bar(range(len(vals)), vals, color=bar_colors, alpha=0.85)
                ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=20, fontsize=7)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(m.replace("_", " "), fontsize=9); ax.grid(True, alpha=0.2, axis="y")
        for idx in range(n_metrics, nrows * ncols):
            r, c = divmod(idx, ncols); axes[r][c].set_visible(False)
        fig.suptitle("SKL ablation: bi-objective vs. tri-objective ($k=300$)", y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "skl_ablation_comparison.pdf"))

    # ------------------------------------------------------------------
    # Pareto front evolution (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_pareto_front_evolution(self, df: pd.DataFrame) -> str:
        """Pareto front evolution across k values."""
        import matplotlib.pyplot as plt

        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty or "k" not in d.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No R1 data for Pareto evolution",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "pareto_front_evolution.pdf"))

        obj_x = next((c for c in d.columns if c.startswith("f_") and "mmd" in c.lower()), None)
        obj_y = next((c for c in d.columns if c.startswith("f_") and ("sink" in c.lower() or "sd" in c.lower())), None)
        if obj_x is None or obj_y is None:
            for cx, cy in [("nystrom_error", "geo_kl"), ("nystrom_error", "kpca_distortion")]:
                if cx in d.columns and cy in d.columns:
                    obj_x, obj_y = cx, cy; break

        if obj_x is None or obj_y is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No suitable objective/metric pair found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "pareto_front_evolution.pdf"))

        fig, ax = plt.subplots(figsize=(6, 5))
        ks = sorted(d["k"].unique())
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(ks)))
        for ki, (k_val, color) in enumerate(zip(ks, cmap)):
            sub = d[d["k"] == k_val]
            ax.scatter(sub[obj_x], sub[obj_y], s=20, color=color, alpha=0.7,
                       label=f"k={int(k_val)}", edgecolors="k", linewidths=0.2)
        ax.set_xlabel(obj_x.replace("_", " "), fontsize=10)
        ax.set_ylabel(obj_y.replace("_", " "), fontsize=10)
        ax.set_title("Pareto front evolution across $k$ values (R1)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=2); ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "pareto_front_evolution.pdf"))

    # ------------------------------------------------------------------
    # Fig 9: Downstream model heatmap (Section VIII.C)
    # ------------------------------------------------------------------
    def fig_downstream_model_heatmap(self, df: pd.DataFrame) -> str:
        r"""Fig 9: Dual-colormap heatmap of downstream model performance.

        Left half  = regression targets (RMSE, lower is better).
        Right half = classification targets (balanced accuracy, higher is better).
        Models as rows; targets as columns.  A vertical divider separates the
        two metric families, and each half has its own colorbar.
        """
        import re as _re
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        out_path = os.path.join(self.fig_dir, "downstream_model_heatmap.pdf")

        # ---- filter to R1, k=300 ----
        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "Downstream model data not yet generated (R1 results needed).",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)
        d["k"] = d["k"].astype(int)
        d300 = d[d["k"] == 300]
        if d300.empty:
            d300 = d  # fall back to any k

        # ---- discover targets ----
        reg_models = {"krr": "KRR", "knn": "KNN", "rf": "RF", "gbt": "GBT"}
        cls_models = {"knn": "KNN", "rf": "RF", "lr": "LR", "gbt": "GBT"}

        reg_targets = set()
        for col in d300.columns:
            m = _re.match(r"^(krr|knn|rf|gbt)_rmse_(.+)$", col)
            if m:
                reg_targets.add(m.group(2))
        reg_targets = sorted(reg_targets)

        cls_targets = set()
        for col in d300.columns:
            m = _re.match(r"^(knn|rf|lr|gbt)_bal_accuracy_(.+)$", col)
            if m:
                cls_targets.add(m.group(2))
        if not cls_targets:
            for col in d300.columns:
                m = _re.match(r"^(knn|rf|lr|gbt)_accuracy_(.+)$", col)
                if m:
                    cls_targets.add(m.group(2))
        cls_targets = sorted(cls_targets)

        if not reg_targets and not cls_targets:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No downstream regression/classification columns found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        # ---- unified model list ----
        all_models = ["KRR", "KNN", "RF", "GBT", "LR"]
        reg_model_keys = {"KRR": "krr", "KNN": "knn", "RF": "rf", "GBT": "gbt"}
        cls_model_keys = {"KNN": "knn", "RF": "rf", "LR": "lr", "GBT": "gbt"}

        n_reg = len(reg_targets)
        n_cls = len(cls_targets)
        n_models = len(all_models)

        # ---- build data matrices ----
        reg_matrix = np.full((n_models, n_reg), np.nan)
        for i, model_label in enumerate(all_models):
            key = reg_model_keys.get(model_label)
            if key is None:
                continue
            for j, tgt in enumerate(reg_targets):
                col_name = f"{key}_rmse_{tgt}"
                if col_name in d300.columns and d300[col_name].notna().any():
                    reg_matrix[i, j] = float(d300[col_name].mean())

        cls_matrix = np.full((n_models, n_cls), np.nan)
        for i, model_label in enumerate(all_models):
            key = cls_model_keys.get(model_label)
            if key is None:
                continue
            for j, tgt in enumerate(cls_targets):
                bal_col = f"{key}_bal_accuracy_{tgt}"
                acc_col = f"{key}_accuracy_{tgt}"
                if bal_col in d300.columns and d300[bal_col].notna().any():
                    cls_matrix[i, j] = float(d300[bal_col].mean())
                elif acc_col in d300.columns and d300[acc_col].notna().any():
                    cls_matrix[i, j] = float(d300[acc_col].mean())

        # ---- shorten target names ----
        def _short(name: str) -> str:
            return (name.replace("cov_area_", "")
                        .replace("cov_hh_", "hh_")
                        .replace("cov_res_", "res_")
                        .replace("concentrated_", "conc_")
                        .replace("_", " "))

        reg_labels = [_short(t) for t in reg_targets]
        cls_labels = [_short(t) for t in cls_targets]
        all_labels = reg_labels + cls_labels

        # ---- plot ----
        n_total = n_reg + n_cls
        fig_w = max(7, n_total * 0.55 + 3.5)
        fig_h = max(3, n_models * 0.55 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # normalizations
        reg_finite = reg_matrix[np.isfinite(reg_matrix)]
        cls_finite = cls_matrix[np.isfinite(cls_matrix)]
        reg_norm = Normalize(vmin=reg_finite.min() if len(reg_finite) else 0,
                             vmax=reg_finite.max() if len(reg_finite) else 1)
        cls_norm = Normalize(vmin=cls_finite.min() if len(cls_finite) else 0,
                             vmax=cls_finite.max() if len(cls_finite) else 1)

        cmap_reg = plt.cm.RdYlGn_r   # lower RMSE = green
        cmap_cls = plt.cm.RdYlGn      # higher accuracy = green

        # draw cells manually for dual colormaps
        for i in range(n_models):
            # regression half
            for j in range(n_reg):
                val = reg_matrix[i, j]
                if np.isfinite(val):
                    color = cmap_reg(reg_norm(val))
                    ax.add_patch(plt.Rectangle((j, i - 0.5), 1, 1,
                                               facecolor=color, edgecolor="white",
                                               linewidth=0.5))
                    text_color = "white" if reg_norm(val) > 0.65 else "black"
                    ax.text(j + 0.5, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=5.5, color=text_color, fontweight="medium")
                else:
                    ax.add_patch(plt.Rectangle((j, i - 0.5), 1, 1,
                                               facecolor="#e0e0e0", edgecolor="white",
                                               linewidth=0.5))
            # classification half
            for j in range(n_cls):
                col_idx = n_reg + j
                val = cls_matrix[i, j]
                if np.isfinite(val):
                    color = cmap_cls(cls_norm(val))
                    ax.add_patch(plt.Rectangle((col_idx, i - 0.5), 1, 1,
                                               facecolor=color, edgecolor="white",
                                               linewidth=0.5))
                    text_color = "white" if cls_norm(val) > 0.65 else "black"
                    ax.text(col_idx + 0.5, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=5.5, color=text_color, fontweight="medium")
                else:
                    ax.add_patch(plt.Rectangle((col_idx, i - 0.5), 1, 1,
                                               facecolor="#e0e0e0", edgecolor="white",
                                               linewidth=0.5))

        # divider line
        ax.axvline(x=n_reg, color="black", linewidth=2.5, linestyle="-")

        # axes
        ax.set_xlim(0, n_total)
        ax.set_ylim(-0.5, n_models - 0.5)
        ax.invert_yaxis()
        ax.set_xticks([j + 0.5 for j in range(n_total)])
        ax.set_xticklabels(all_labels, rotation=50, ha="right", fontsize=7)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(all_models, fontsize=8, fontweight="bold")

        # half-labels above
        if n_reg > 0:
            ax.text(n_reg / 2, -1.0, "Regression (RMSE $\\downarrow$)",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#8B0000")
        if n_cls > 0:
            ax.text(n_reg + n_cls / 2, -1.0, "Classification (Bal. Acc. $\\uparrow$)",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#006400")

        ax.tick_params(axis="both", which="both", length=0)

        # colorbars
        sm_reg = plt.cm.ScalarMappable(cmap=cmap_reg, norm=reg_norm)
        sm_reg.set_array([])
        sm_cls = plt.cm.ScalarMappable(cmap=cmap_cls, norm=cls_norm)
        sm_cls.set_array([])

        cbar_reg = fig.colorbar(sm_reg, ax=ax, fraction=0.02, pad=0.01,
                                location="left", shrink=0.8)
        cbar_reg.set_label("RMSE", fontsize=7, labelpad=3)
        cbar_reg.ax.tick_params(labelsize=6)

        cbar_cls = fig.colorbar(sm_cls, ax=ax, fraction=0.02, pad=0.01,
                                location="right", shrink=0.8)
        cbar_cls.set_label("Bal. Accuracy", fontsize=7, labelpad=3)
        cbar_cls.ax.tick_params(labelsize=6)

        ax.set_title("Downstream model comparison (R1, $k{=}300$)", fontsize=10, pad=18)
        fig.tight_layout()
        return _save(fig, out_path)
