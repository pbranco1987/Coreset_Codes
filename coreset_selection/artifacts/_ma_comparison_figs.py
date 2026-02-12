"""Comparison figure methods for ManuscriptArtifacts (mixin).

Manuscript figures rendered here (R/ggplot2 with matplotlib fallback):
  Fig 2: fig_geo_ablation_scatter       → geo_ablation_tradeoff_scatter.pdf
  Fig 5: fig_regional_validity_k300     → regional_validity_k300.pdf
  Fig 6: fig_baseline_comparison        → baseline_comparison_grouped.pdf
  Fig 7: fig_representation_transfer    → representation_transfer_bars.pdf
  Fig 8: fig_objective_metric_alignment → objective_metric_alignment_heatmap.pdf
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
        r"""Fig N3: Objective ablation comparison — R1 vs R2 vs R3 at k=300."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        _aliases = {
            "krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
            "krr_rmse_5G": ["krr_rmse_cov_area_5G", "krr_rmse_area_5G"],
        }
        for canon, aliases in _aliases.items():
            if canon not in d.columns:
                for alias in aliases:
                    if alias in d.columns:
                        d[canon] = d[alias]; break

        metric_candidates = [
            ("nystrom_error",    r"$e_{\rm Nys}$"),
            ("kpca_distortion",  r"$e_{\rm kPCA}$"),
            ("krr_rmse_4G",      r"RMSE$_{\rm 4G}$"),
            ("krr_rmse_5G",      r"RMSE$_{\rm 5G}$"),
            ("geo_kl",           r"KL$_{\rm geo}$"),
        ]
        metrics = [(m, lab) for m, lab in metric_candidates if m in d.columns]
        if not metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No evaluation metrics available for ablation.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

        runs = {"R1 (MMD+SD)": ("R1", "#1f77b4"), "R2 (MMD only)": ("R2", "#ff7f0e"),
                "R3 (SD only)": ("R3", "#2ca02c")}
        bar_data = {}
        for label, (rid, _) in runs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if sub.empty: continue
            bar_data[label] = {m: float(sub[m].min()) for m, _ in metrics
                                if m in sub.columns and sub[m].notna().any()}
        if not bar_data:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No R1/R2/R3 data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

        metric_keys = [m for m, _ in metrics]
        metric_labels = [lab for _, lab in metrics]
        method_labels = list(bar_data.keys())
        method_colors = [runs[m][1] for m in method_labels]
        n_metrics = len(metric_keys); n_methods = len(method_labels)
        x = np.arange(n_metrics); total_width = 0.75; width = total_width / max(n_methods, 1)
        fig, ax = plt.subplots(figsize=(max(7, n_metrics * 1.5), 4.2))
        for i, (method, color) in enumerate(zip(method_labels, method_colors)):
            vals = [bar_data[method].get(m, 0) for m in metric_keys]
            offset = (i - (n_methods - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9, color=color,
                          alpha=0.85, label=method, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
        ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylabel("Metric value (lower is better)", fontsize=10)
        ax.set_title("Objective ablation at $k{=}300$", fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="0.8")
        ax.grid(True, alpha=0.2, axis="y"); ax.tick_params(labelsize=9)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "objective_ablation_bars_k300.pdf"))

    # ------------------------------------------------------------------
    # Fig N4: Constraint comparison bars (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_constraint_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig N4: Constraint regime comparison — R1, R4, R5, R6 at k=300."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        _aliases = {"krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"]}
        for canon, aliases in _aliases.items():
            if canon not in d.columns:
                for alias in aliases:
                    if alias in d.columns:
                        d[canon] = d[alias]; break

        metric_defs = [
            ("geo_kl", r"KL$_{\rm geo}$", "lower"), ("geo_l1", r"$\ell_1$ drift", "lower"),
            ("nystrom_error", r"$e_{\rm Nys}$", "lower"), ("krr_rmse_4G", r"RMSE$_{\rm 4G}$", "lower"),
        ]
        available_metrics = [(m, lab, d2) for m, lab, d2 in metric_defs if m in d.columns]
        runs = {"R1\n(pop-share)": ("R1", "#1f77b4"), "R4\n(muni-quota)": ("R4", "#ff7f0e"),
                "R5\n(joint)": ("R5", "#2ca02c"), "R6\n(none)": ("R6", "#d62728")}
        bar_data = {}
        for label, (rid, _) in runs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if not sub.empty:
                bar_data[label] = {m: float(sub[m].mean()) for m, _, _ in available_metrics
                                   if m in sub.columns and sub[m].notna().any()}
        if not bar_data or not available_metrics:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No constraint comparison data.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "constraint_comparison_bars_k300.pdf"))

        n_panels = min(len(available_metrics), 4)
        nrows = 2 if n_panels > 2 else 1; ncols = 2 if n_panels > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 5.5), squeeze=False)
        method_labels = list(bar_data.keys())
        method_colors = [runs[m][1] for m in method_labels]
        x = np.arange(len(method_labels))
        for idx, (metric, label, _) in enumerate(available_metrics[:n_panels]):
            r, c = divmod(idx, ncols); ax = axes[r][c]
            vals = [bar_data[m].get(metric, 0) for m in method_labels]
            bars = ax.bar(x, vals, color=method_colors, alpha=0.85, edgecolor="white",
                          linewidth=0.5, width=0.65)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{v:.4f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(method_labels, fontsize=8)
            ax.set_title(label, fontsize=9); ax.set_ylabel("Value", fontsize=8)
            ax.grid(True, alpha=0.2, axis="y"); ax.tick_params(labelsize=8)
        for idx in range(n_panels, nrows * ncols):
            r, c = divmod(idx, ncols); axes[r][c].set_visible(False)
        fig.suptitle("Constraint regime comparison at $k{=}300$", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "constraint_comparison_bars_k300.pdf"))

    # ------------------------------------------------------------------
    # Fig 7: Representation transfer bars (Section VIII.G)
    # ------------------------------------------------------------------
    def fig_representation_transfer(self, df: pd.DataFrame) -> str:
        r"""Fig 7: Representation transfer bars at k=300 (Section VIII.G)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "representation_transfer_bars.pdf")

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        _aliases = {
            "krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"],
            "krr_rmse_5G": ["krr_rmse_cov_area_5G", "krr_rmse_area_5G"],
        }
        for canon, aliases in _aliases.items():
            if canon not in d.columns:
                for alias in aliases:
                    if alias in d.columns:
                        d[canon] = d[alias]; break

        run_configs = [
            ("R1", "Raw (p=D)", "#1f77b4"),
            ("R8", "PCA (p=20)", "#ff7f0e"),
            ("R9", "VAE (p=32)", "#2ca02c"),
        ]
        metric_defs = [
            ("nystrom_error",   "e_Nys",   "(a)"),
            ("kpca_distortion", "e_kPCA",   "(b)"),
            ("krr_rmse_4G",     "RMSE_4G",  "(c)"),
        ]
        metrics = [(m, lab, pl) for m, lab, pl in metric_defs if m in d.columns]

        if not metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No evaluation metrics available for R1/R8/R9.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        # Gather data
        bar_vals = {}
        for rid, label, color in run_configs:
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            if sub.empty: continue
            vals = {}
            for mcol, _, _ in metrics:
                if mcol in sub.columns:
                    col_vals = sub[mcol].dropna()
                    if len(col_vals) > 0:
                        vals[mcol] = (float(col_vals.min()), float(col_vals.mean()),
                                      float(col_vals.std()))
            if vals:
                bar_vals[label] = vals

        if not bar_vals:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No R1/R8/R9 data at k=300",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        # ---- Build tidy data for R ----
        tidy_rows = []
        for repr_label, metric_dict in bar_vals.items():
            for mcol, mlabel, _ in metrics:
                if mcol in metric_dict:
                    best, mean, std = metric_dict[mcol]
                    tidy_rows.append({
                        "representation": repr_label, "metric": mlabel,
                        "best": best, "mean": mean, "std": std,
                    })

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            color_map = {label: color for _, label, color in run_configs}
            run_labels = [label for _, label, _ in run_configs if label in bar_vals]
            n_m = len(metrics)
            fig, axes_arr = plt.subplots(1, n_m, figsize=(max(7, 3.5 * n_m), 4.2))
            if n_m == 1:
                axes_arr = [axes_arr]
            for ax, (mcol, ylabel, panel_label) in zip(axes_arr, metrics):
                mpl_ylabel = ylabel.replace("e_Nys", r"$e_{\mathrm{Nys}}$") \
                                    .replace("e_kPCA", r"$e_{\mathrm{kPCA}}$") \
                                    .replace("RMSE_4G", r"RMSE$_{\mathrm{4G}}$")
                for i, rlabel in enumerate(run_labels):
                    if mcol not in bar_vals.get(rlabel, {}):
                        continue
                    best, mean, std = bar_vals[rlabel][mcol]
                    color = color_map.get(rlabel, "#999999")
                    ax.bar(i, best, width=0.55, color=color, alpha=0.75,
                           edgecolor="k", linewidth=0.5, zorder=4)
                    if std > 0:
                        ax.errorbar(i, mean, yerr=std, fmt="none", color="k",
                                    capsize=4, capthick=1.0, linewidth=1.0, zorder=5)
                    ax.annotate(f"{best:.4f}", xy=(i, best), xytext=(0, 5),
                                textcoords="offset points", fontsize=7.5,
                                ha="center", va="bottom", color="#333333")
                ax.set_xticks(range(len(run_labels)))
                ax.set_xticklabels(run_labels, rotation=15, ha="right", fontsize=8)
                ax.set_ylabel(mpl_ylabel, fontsize=10)
                ax.grid(True, alpha=0.2, axis="y"); ax.tick_params(labelsize=9)
                ax.annotate(panel_label, xy=(0.03, 0.95), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", va="top")
            fig.suptitle("Representation transfer: raw-space evaluation at $k{=}300$",
                         fontsize=10, y=1.03)
            fig.tight_layout()
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_representation_transfer.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

    # ------------------------------------------------------------------
    # Fig 6: Baseline comparison (Section VIII.E)
    # ------------------------------------------------------------------
    def fig_baseline_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig 6: Comprehensive baseline comparison at k=300 (Section VIII.E)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "baseline_comparison_grouped.pdf")

        d_all = df[df["k"].fillna(300).astype(int) == 300].copy()
        _aliases = {"krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"]}
        for canon, aliases in _aliases.items():
            if canon not in d_all.columns:
                for alias in aliases:
                    if alias in d_all.columns:
                        d_all[canon] = d_all[alias]; break

        d_base = d_all[d_all["run_id"].astype(str).str.contains("R10")].copy()
        if d_base.empty and "method" in d_all.columns:
            d_base = d_all[d_all["method"].astype(str) != "nsga2"].copy()

        metric_defs = [
            ("nystrom_error",  "e_Nys"), ("krr_rmse_4G", "RMSE_4G"), ("geo_kl", "KL_geo"),
        ]
        metrics = [(m, lab) for m, lab in metric_defs if m in d_all.columns]

        if d_base.empty or not metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No baseline comparison data (R10) at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        has_method = "method" in d_base.columns
        has_variant = "variant" in d_base.columns

        method_order = ["uniform", "kmeans", "herding", "farthest_first",
                        "kernel_thinning", "leverage", "dpp"]
        if has_method:
            methods_available = sorted(d_base["method"].unique())
        else:
            methods_available = ["baseline"]
        methods = [m for m in method_order if m in methods_available]
        methods.extend([m for m in methods_available if m not in methods])

        _method_labels = {
            "uniform": "Uniform", "kmeans": "k-Means", "herding": "Herding",
            "farthest_first": "Farthest-First", "kernel_thinning": "Kern. Thin.",
            "leverage": "RLS", "dpp": "DPP",
        }

        # R1 knee reference
        d_r1 = d_all[d_all["run_id"].astype(str).str.contains("R1")]
        r1_ref = {}
        if not d_r1.empty:
            knee = d_r1[d_r1.get("rep_name", pd.Series(dtype=str)).astype(str) == "knee"]
            if knee.empty: knee = d_r1
            for m, _ in metrics:
                if m in knee.columns and knee[m].notna().any():
                    r1_ref[m] = float(knee[m].min())

        # Determine variants
        variants = ["unconstrained", "quota_matched"]
        method_data = {}
        for method in methods:
            method_data[method] = {}
            sub_m = d_base[d_base["method"] == method] if has_method else d_base
            if has_variant and sub_m["variant"].nunique() > 1:
                for var in variants:
                    sub_v = sub_m[sub_m["variant"].astype(str).str.contains(var)]
                    if not sub_v.empty:
                        method_data[method][var] = {
                            m: float(sub_v[m].mean()) for m, _ in metrics
                            if m in sub_v.columns and sub_v[m].notna().any()}
            else:
                if not sub_m.empty:
                    method_data[method]["unconstrained"] = {
                        m: float(sub_m[m].mean()) for m, _ in metrics
                        if m in sub_m.columns and sub_m[m].notna().any()}

        has_quota = any("quota_matched" in method_data.get(m, {}) for m in methods)

        # ---- Build tidy data for R ----
        tidy_rows = []
        for method in methods:
            for var, mvals in method_data.get(method, {}).items():
                for m, mlabel in metrics:
                    if m in mvals:
                        tidy_rows.append({
                            "method": _method_labels.get(method, method),
                            "variant": var, "metric": mlabel,
                            "value": mvals[m], "is_r1_knee": False,
                        })
        # Add R1-knee
        for m, mlabel in metrics:
            if m in r1_ref:
                tidy_rows.append({
                    "method": "R1-knee", "variant": "unconstrained",
                    "metric": mlabel, "value": r1_ref[m], "is_r1_knee": True,
                })

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            n_panels = len(metrics)
            fig, axes_arr = plt.subplots(1, n_panels, figsize=(max(7, n_panels * 4), 4.5))
            if n_panels == 1:
                axes_arr = [axes_arr]
            mpl_metric_labels = {
                "e_Nys": r"$e_{\rm Nys}$", "RMSE_4G": r"RMSE$_{\rm 4G}$",
                "KL_geo": r"KL$_{\rm geo}$",
            }
            for ax, (metric, mlabel) in zip(axes_arr, metrics):
                n_methods = len(methods) + 1
                x = np.arange(n_methods)
                if has_quota:
                    width = 0.35
                    uncons_vals = [method_data.get(m, {}).get("unconstrained", {}).get(metric, 0)
                                   for m in methods] + [r1_ref.get(metric, 0)]
                    quota_vals = [method_data.get(m, {}).get("quota_matched", {}).get(metric, 0)
                                  for m in methods] + [r1_ref.get(metric, 0)]
                    ax.bar(x - width / 2, uncons_vals, width=width * 0.9,
                           color="#1f77b4", alpha=0.8, label="Unconstrained",
                           edgecolor="white", linewidth=0.5)
                    ax.bar(x + width / 2, quota_vals, width=width * 0.9,
                           color="#ff7f0e", alpha=0.8, label="Quota-matched",
                           edgecolor="white", linewidth=0.5)
                else:
                    vals = [method_data.get(m, {}).get("unconstrained", {}).get(metric, 0)
                            for m in methods] + [r1_ref.get(metric, 0)]
                    colors = ["#1f77b4"] * len(methods) + ["#d62728"]
                    ax.bar(x, vals, width=0.55, color=colors, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
                if metric in r1_ref:
                    ax.axhline(r1_ref[metric], color="#d62728", linestyle="--",
                               linewidth=1.0, alpha=0.6)
                xlabels = [_method_labels.get(m, m) for m in methods] + ["R1-knee"]
                ax.set_xticks(x)
                ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7.5)
                ax.set_title(mpl_metric_labels.get(mlabel, mlabel), fontsize=9)
                ax.set_ylabel("Value", fontsize=8)
                ax.grid(True, alpha=0.2, axis="y"); ax.tick_params(labelsize=8)
                if has_quota:
                    ax.legend(fontsize=7, loc="upper left")
            fig.suptitle("Baseline comparison at $k{=}300$ (R10)", fontsize=11, y=1.01)
            fig.tight_layout()
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_baseline_comparison.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

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
