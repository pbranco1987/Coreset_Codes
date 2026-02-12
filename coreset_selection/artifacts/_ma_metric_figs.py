"""Metric and distortion figure methods for ManuscriptArtifacts (mixin)."""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ._ma_helpers import _save


class MetricFigsMixin:
    """Mixin providing metric/distortion visualization methods."""

    def fig_distortion_cardinality_r1(self, df: pd.DataFrame) -> str:
        r"""Fig 2: Raw-space metrics vs *k* for R1 (Section VIII.C).

        2×2 panel showing metric-wise envelopes over the feasible
        non-dominated set for each :math:`k \in \mathcal{K}`:

        (a) :math:`e_{\mathrm{Nys}}` vs *k*,
        (b) :math:`e_{\mathrm{kPCA}}` vs *k*,
        (c) RMSE :math:`y^{(4\mathrm{G})}` vs *k*,
        (d) RMSE :math:`y^{(5\mathrm{G})}` vs *k*.

        Manuscript compliance fixes (Phase 8):
        * Exact 2×2 layout.
        * Panel labels ``(a)``–``(d)`` in upper-left corners.
        * Error bands (mean ± std) when multi-seed (5 seeds).
        * Diminishing-returns annotation for large *k*.
        * IEEE double-column width (~7 in).
        * Font sizes ≥ 8 pt, label sizes ≥ 10 pt.
        """
        import matplotlib.pyplot as plt

        if df.empty or "run_id" not in df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "distortion_cardinality_R1.pdf"))

        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No R1 data available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "distortion_cardinality_R1.pdf"))

        d["k"] = d["k"].astype(int)

        # Resolve KRR column names
        def _resolve(preferred, fallback_prefix):
            if preferred in d.columns:
                return preferred
            cands = [c for c in d.columns if c.startswith(fallback_prefix)]
            return cands[0] if cands else preferred

        krr_4g = _resolve("krr_rmse_4G", "krr_rmse_cov_area_4G")
        krr_5g = _resolve("krr_rmse_5G", "krr_rmse_cov_area_5G")

        metrics = [
            ("nystrom_error",   r"$e_{\mathrm{Nys}}$"),
            ("kpca_distortion", r"$e_{\mathrm{kPCA}}$"),
            (krr_4g,            r"RMSE$_{4\mathrm{G}}$"),
            (krr_5g,            r"RMSE$_{5\mathrm{G}}$"),
        ]
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
        axes_flat = axes.flatten()

        for idx, (ax, (mcol, ylabel)) in enumerate(zip(axes_flat, metrics)):
            if mcol not in d.columns:
                ax.text(0.5, 0.5, f"{mcol}\nnot available",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray")
                ax.set_visible(True)
                ax.set_axis_off()
                # Panel label even for missing panels
                ax.annotate(panel_labels[idx], xy=(0.02, 0.95),
                            xycoords="axes fraction", fontsize=10,
                            fontweight="bold", va="top")
                continue

            # Envelope: best (min) per k
            env = d.groupby("k")[mcol].min().reset_index().sort_values("k")

            # Multi-seed aggregation
            agg = d.groupby("k")[mcol].agg(["mean", "std", "count"]).reset_index()
            agg = agg.sort_values("k")
            multi_seed = agg["count"].max() > 1

            # Plot envelope (best)
            ax.plot(env["k"], env[mcol], "o-", linewidth=1.6,
                    markersize=4, color="#1f77b4", label="Envelope (best)",
                    zorder=5)

            # Error band if multi-seed
            if multi_seed and agg["std"].max() > 0:
                ax.fill_between(agg["k"],
                                agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"],
                                alpha=0.15, color="#1f77b4",
                                label=r"Mean $\pm$ 1 std")

            # Diminishing-returns annotation
            if len(env) >= 3:
                vals = env[mcol].values
                ks = env["k"].values
                # Relative improvement from second-to-last to last point
                if vals[-2] > 0:
                    rel_improv = (vals[-2] - vals[-1]) / vals[-2]
                    if rel_improv < 0.05:
                        ax.annotate("diminishing\nreturns",
                                    xy=(ks[-1], vals[-1]),
                                    xytext=(-35, 15),
                                    textcoords="offset points",
                                    fontsize=7, color="#888888",
                                    arrowprops=dict(arrowstyle="->",
                                                    color="#888888",
                                                    lw=0.7))

            ax.set_xlabel("$k$", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            if multi_seed:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

            # Panel label
            ax.annotate(panel_labels[idx], xy=(0.03, 0.95),
                        xycoords="axes fraction", fontsize=10,
                        fontweight="bold", va="top")

        fig.suptitle(
            r"R1: Raw-space metrics vs.\ coreset size $k$ "
            "(metric-wise envelope)",
            fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "distortion_cardinality_R1.pdf"))

    def fig_kl_floor_vs_k(self) -> str:
        r"""Fig N1: KL feasibility planning curve from R0 / quota computation.

        Illustrates the integrality-induced KL floor :math:`\mathrm{KL}_{\min}(k)`
        as a function of coreset size *k*.  For small *k*, perfect proportionality
        is impossible — this curve makes the minimum achievable geographic
        divergence transparent.

        Phase 10a enhancements (manuscript Section VIII.A):
        * Horizontal lines at common :math:`\tau` thresholds (0.01, 0.02, 0.05)
          to show feasibility thresholds — the intersection of each line with the
          curve indicates the minimum *k* for which that tolerance is achievable.
        * Manuscript cardinality grid points (:math:`\mathcal{K}`) are marked and
          annotated with their :math:`\mathrm{KL}_{\min}` values.
        * Shaded feasible region below each :math:`\tau` line.
        * IEEE single-column width (~5.5 in for clarity).
        * Defense value: directly supports the claim that quota constraints make
          proportionality loss transparent; answers "why not exact proportionality?"
        """
        import matplotlib.pyplot as plt

        # Try loading from R0 results
        df = self._load_df()
        d = df[df["run_id"].astype(str).str.contains("R0")] if not df.empty else pd.DataFrame()
        if not d.empty and "kl_min" in d.columns:
            d = d.sort_values("k")
            ks, vals = d["k"].values, d["kl_min"].values
        else:
            # Recompute from cache
            try:
                from ..data.cache import load_replicate_cache
                from ..geo.info import build_geo_info
                from ..geo.kl import min_achievable_geo_kl_bounded
            except ImportError:
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                ax.text(0.5, 0.5, "Cannot import KL modules — run R0 first.",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                return _save(fig, os.path.join(self.fig_dir, "kl_floor_vs_k.pdf"))
            import glob as _glob
            cfs = _glob.glob(os.path.join(self.cache_root, "rep*/assets.npz"))
            if not cfs:
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                ax.text(0.5, 0.5, "No caches available for KL floor computation.",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                return _save(fig, os.path.join(self.fig_dir, "kl_floor_vs_k.pdf"))
            assets = load_replicate_cache(cfs[0])
            geo = build_geo_info(assets.state_labels)
            ks = np.arange(27, 501)
            vals = []
            for k in ks:
                kl, _ = min_achievable_geo_kl_bounded(
                    pi=geo.pi, group_sizes=geo.group_sizes, k=int(k),
                    alpha_geo=1.0, min_one_per_group=True,
                )
                vals.append(float(kl))
            vals = np.array(vals)

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        # Main KL floor curve
        ax.plot(ks, vals, linewidth=2.0, color="#1f77b4",
                label=r"$\mathrm{KL}_{\min}(k)$", zorder=5)

        # Horizontal τ threshold lines with shaded feasible regions
        tau_thresholds = [
            (0.05, "#2ca02c", r"$\tau = 0.05$", 0.08),
            (0.02, "#ff7f0e", r"$\tau = 0.02$", 0.08),
            (0.01, "#d62728", r"$\tau = 0.01$", 0.08),
        ]
        for tau, color, label, alpha_fill in tau_thresholds:
            ax.axhline(tau, color=color, linestyle="--", linewidth=1.0,
                       alpha=0.75, label=label, zorder=3)
            # Find minimum k where KL_min ≤ τ
            feasible_idx = np.where(np.asarray(vals, dtype=float) <= tau)[0]
            if len(feasible_idx) > 0:
                k_min_feasible = float(ks[feasible_idx[0]])
                # Annotate the intersection point
                ax.plot(k_min_feasible, tau, "v", color=color, markersize=7,
                        zorder=8)
                ax.annotate(f"$k \\geq {int(k_min_feasible)}$",
                            xy=(k_min_feasible, tau),
                            xytext=(k_min_feasible + 25, tau + 0.008),
                            fontsize=7.5, color=color,
                            arrowprops=dict(arrowstyle="->", color=color,
                                            linewidth=0.8),
                            zorder=9)

        # Mark manuscript cardinality grid points K = {50, 100, 200, 300, 400, 500}
        grid_k = [50, 100, 200, 300, 400, 500]
        for gk in grid_k:
            idx = np.argmin(np.abs(np.asarray(ks, dtype=float) - gk))
            if idx < len(vals):
                ax.plot(ks[idx], vals[idx], "o", color="#d62728", markersize=6,
                        zorder=7, markeredgecolor="k", markeredgewidth=0.4)
                # Annotate with KL value for key grid points
                if gk in (50, 100, 300, 500):
                    offset_y = 0.006 if vals[idx] > 0.01 else 0.003
                    ax.annotate(f"{vals[idx]:.4f}",
                                xy=(ks[idx], vals[idx]),
                                xytext=(ks[idx], vals[idx] + offset_y),
                                fontsize=7, ha="center", color="#555555",
                                zorder=9)

        ax.set_xlabel("Coreset size $k$", fontsize=10)
        ax.set_ylabel(r"$\mathrm{KL}_{\min}(k)$", fontsize=10)
        ax.set_title(r"Feasibility floor $\mathrm{KL}_{\min}(k)$ vs.\ coreset budget",
                     fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
                  edgecolor="0.8", handletextpad=0.4)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.set_xlim(left=25)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "kl_floor_vs_k.pdf"))

    def fig_effort_quality(self, df: pd.DataFrame) -> str:
        r"""Fig N5: Effort sweep — selection time vs downstream quality (R12).

        Scatter / line plot showing that increasing NSGA-II effort improves
        solution quality with diminishing returns.  Each point represents
        one ``(pop_size, n_gen)`` configuration from the R12 effort grid.

        Phase 10a enhancements (manuscript Section VII.G, VIII.L):
        * Left panel: scatter of wall-clock time vs downstream metric
          (e_Nys or RMSE_4G), with iso-effort curves connecting same
          population sizes.
        * Right panel: metric vs total effort P×T with diminishing-returns
          annotation (elbow or plateau region highlighted).
        * Point size/colour encode pop_size and n_gen for easy identification.
        * Annotation of the default configuration (P=200, T=1000).
        * IEEE double-column width.
        * Defense value: makes computational cost transparent and helps
          practitioners choose effort knobs; addresses "Is 1000 generations
          enough?" concerns.
        """
        import matplotlib.pyplot as plt

        d = df[df["run_id"].astype(str).str.contains("R12")].copy()
        if d.empty or "wall_clock_s" not in d.columns:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.text(0.5, 0.5, "No R12 effort sweep data available.\n"
                    "Run R12 first to generate effort grid results.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "effort_quality_tradeoff.pdf"))

        # Select best available quality metric
        quality_col = None
        quality_label = ""
        for cand, lab in [("nystrom_error", r"$e_{\rm Nys}$"),
                          ("krr_rmse_4G", r"RMSE$_{\rm 4G}$"),
                          ("krr_rmse_cov_area_4G", r"RMSE$_{\rm 4G}$"),
                          ("kpca_distortion", r"$e_{\rm kPCA}$")]:
            if cand in d.columns and d[cand].notna().any():
                quality_col = cand
                quality_label = lab
                break

        if quality_col is None:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.text(0.5, 0.5, "No quality metric found in R12 data.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "effort_quality_tradeoff.pdf"))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Determine effort column
        has_effort = "effort_P_x_T" in d.columns
        has_pop = "pop_size" in d.columns
        has_ngen = "n_gen" in d.columns

        # ---- Left panel: wall-clock time vs quality ----
        ax = axes[0]
        if has_pop:
            # Colour by pop_size for visual grouping
            pop_sizes = sorted(d["pop_size"].unique())
            cmap_vals = plt.cm.viridis(np.linspace(0.2, 0.9, len(pop_sizes)))
            pop_color = {p: cmap_vals[i] for i, p in enumerate(pop_sizes)}

            for ps in pop_sizes:
                sub = d[d["pop_size"] == ps].sort_values("wall_clock_s")
                c = pop_color[ps]
                ax.plot(sub["wall_clock_s"], sub[quality_col], "o-",
                        color=c, markersize=6, linewidth=1.0, alpha=0.8,
                        label=f"$P={int(ps)}$")
        else:
            ax.scatter(d["wall_clock_s"], d[quality_col], s=30, alpha=0.7,
                       color="#1f77b4", edgecolors="k", linewidth=0.3)

        ax.set_xlabel("Wall-clock time (s)", fontsize=10)
        ax.set_ylabel(quality_label, fontsize=10)
        ax.set_title("(a) Quality vs.\\ selection time", fontsize=10, pad=6)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)

        # ---- Right panel: quality vs total effort P×T ----
        ax = axes[1]
        if has_effort:
            d_sorted = d.sort_values("effort_P_x_T").copy()
            ax.scatter(d_sorted["effort_P_x_T"], d_sorted[quality_col],
                       s=35, alpha=0.7, color="#d62728", edgecolors="k",
                       linewidth=0.3, zorder=4)
            # Best-so-far envelope (diminishing returns)
            d_sorted["best_so_far"] = d_sorted[quality_col].cummin()
            ax.plot(d_sorted["effort_P_x_T"], d_sorted["best_so_far"],
                    "-", linewidth=2.0, color="#1f77b4", zorder=5,
                    label="Best-so-far")

            # Annotate diminishing returns: find elbow
            bsf = d_sorted["best_so_far"].values
            if len(bsf) > 3:
                # Normalise and find max curvature (elbow)
                efforts = d_sorted["effort_P_x_T"].values.astype(float)
                e_norm = (efforts - efforts.min()) / max(efforts.max() - efforts.min(), 1e-12)
                b_norm = (bsf - bsf.min()) / max(bsf.max() - bsf.min(), 1e-12)
                # Simple elbow: largest drop in improvement
                improvements = np.diff(b_norm)
                if len(improvements) > 1:
                    elbow_idx = int(np.argmax(np.abs(np.diff(improvements)))) + 1
                    elbow_effort = efforts[elbow_idx]
                    ax.axvline(elbow_effort, color="gray", linestyle=":",
                               linewidth=1.0, alpha=0.6)
                    ax.annotate("diminishing\nreturns",
                                xy=(elbow_effort, bsf[elbow_idx]),
                                xytext=(elbow_effort * 1.2, bsf[elbow_idx] * 1.1),
                                fontsize=7.5, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray",
                                                linewidth=0.8))

            # Mark default configuration (P=200, T=1000)
            if has_pop and has_ngen:
                default_mask = (d["pop_size"] == 200) & (d["n_gen"] == 1000)
                default_rows = d[default_mask]
                if not default_rows.empty:
                    for _, dr in default_rows.iterrows():
                        ax.scatter(dr["effort_P_x_T"], dr[quality_col],
                                   s=100, marker="*", color="#ff7f0e",
                                   edgecolors="k", linewidths=0.6, zorder=10)
                        ax.annotate("default\n($P{=}200, T{=}1000$)",
                                    xy=(dr["effort_P_x_T"], dr[quality_col]),
                                    xytext=(dr["effort_P_x_T"] * 0.7,
                                            dr[quality_col] * 1.15),
                                    fontsize=7, color="#ff7f0e",
                                    arrowprops=dict(arrowstyle="->",
                                                    color="#ff7f0e",
                                                    linewidth=0.8))

            ax.set_xlabel("Total effort ($P \\times T$)", fontsize=10)
        else:
            ax.scatter(range(len(d)), d[quality_col], s=30, alpha=0.7,
                       color="#d62728")
            ax.set_xlabel("Run index", fontsize=10)

        ax.set_ylabel(quality_label, fontsize=10)
        ax.set_title("(b) Quality vs.\\ total effort", fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)

        fig.suptitle("Effort–quality trade-off (R12 sweep)", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "effort_quality_tradeoff.pdf"))

    def fig_multi_seed_boxplot(self, df: pd.DataFrame) -> str:
        r"""Fig N7: Multi-seed robustness boxplots — R1, R5 at k=300.

        Box-and-whisker plots of key metrics across 5 random seeds for
        both R1 (population-share) and R5 (joint constraints) to show
        that the method produces stable results.

        Phase 10b enhancements:
        * Extended metric set: e_Nys, RMSE_4G, geo_kl (composition + fidelity).
        * Mean annotations displayed on each box.
        * Paired R1/R5 comparison with consistent colour scheme.
        * Individual seed points overlaid as jittered scatter.
        * IEEE double-column width.
        * Defense value: addresses "how stable are these results across seeds?"
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

        runs = {"R1": ("R1", "#1f77b4"), "R5": ("R5", "#2ca02c")}
        metric_defs = [
            ("nystrom_error",  r"$e_{\rm Nys}$"),
            ("krr_rmse_4G",   r"RMSE$_{\rm 4G}$"),
            ("geo_kl",         r"KL$_{\rm geo}$"),
        ]
        metrics = [(m, lab) for m, lab in metric_defs if m in d.columns]

        if not metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "No multi-seed metric data available at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "multi_seed_stability_boxplot.pdf"))

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics,
                                  figsize=(max(7, 3.5 * n_metrics), 4.2))
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric, label) in zip(axes, metrics):
            box_data = []
            box_labels = []
            box_colors = []

            for run_label, (rid, color) in runs.items():
                sub = d[d["run_id"].astype(str).str.contains(rid)]
                if not sub.empty and metric in sub.columns:
                    vals = sub[metric].dropna().values
                    if len(vals) > 0:
                        box_data.append(vals)
                        box_labels.append(run_label)
                        box_colors.append(color)

            if not box_data:
                ax.text(0.5, 0.5, f"No data for {label}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8)
                continue

            # Draw box plots with custom colours
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5))

            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.2)
            for whisker, color in zip(bp["whiskers"],
                                      [c for c in box_colors for _ in range(2)]):
                whisker.set_color(color)
            for cap, color in zip(bp["caps"],
                                  [c for c in box_colors for _ in range(2)]):
                cap.set_color(color)

            # Overlay individual seed points (jittered)
            for i, (vals, color) in enumerate(zip(box_data, box_colors)):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full_like(vals, i + 1) + jitter, vals,
                           s=18, alpha=0.7, color=color, edgecolors="k",
                           linewidths=0.3, zorder=5)

            # Annotate mean ± std
            for i, vals in enumerate(box_data):
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                ax.annotate(f"$\\mu$={mean:.4f}\n$\\sigma$={std:.4f}",
                            xy=(i + 1, mean), xytext=(i + 1.35, mean),
                            fontsize=6.5, color="#555555",
                            arrowprops=dict(arrowstyle="->", color="#999999",
                                            linewidth=0.6),
                            ha="left", va="center")

            ax.set_title(label, fontsize=10)
            ax.set_ylabel("Value", fontsize=9)
            ax.grid(True, alpha=0.2, axis="y")
            ax.tick_params(labelsize=9)

        fig.suptitle("Multi-seed stability at $k{=}300$ (5 seeds per run)",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "multi_seed_stability_boxplot.pdf"))

    def fig_nystrom_error_distribution(self, df: pd.DataFrame) -> str:
        r"""Fig N11: Distribution of Nyström errors across Pareto solutions at k=300.

        Histogram or violin plot of :math:`e_{\mathrm{Nys}}` values across
        all Pareto-front solutions to show the practical range of outcomes
        available at k=300.

        Phase 10b — NEW figure (manuscript Section VIII.C):
        * Combined histogram + kernel density estimate.
        * Vertical lines marking the knee-point and envelope-best values.
        * Comparison across R1, R5, R9 if available.
        * Annotation showing median, IQR, and envelope range.
        * Defense value: justifies reporting envelope values by showing the
          Pareto front offers a range of quality levels, not just one point.
        """
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty or "nystrom_error" not in d.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5,
                    "No Nyström error data at k=300.\n"
                    "Run R1 and ensure nystrom_error is computed.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "nystrom_error_distribution.pdf"))

        # Gather e_Nys distributions per run
        run_configs = {
            "R1 (pop-share)": ("R1", "#1f77b4"),
            "R5 (joint)":     ("R5", "#2ca02c"),
            "R9 (VAE)":       ("R9", "#ff7f0e"),
        }

        available_runs = {}
        for label, (rid, color) in run_configs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            vals = sub["nystrom_error"].dropna().values
            if len(vals) > 0:
                available_runs[label] = (vals, color)

        if not available_runs:
            # Fallback: use all k=300 data
            vals = d["nystrom_error"].dropna().values
            if len(vals) == 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, "No Nyström error values.",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                return _save(fig, os.path.join(self.fig_dir,
                             "nystrom_error_distribution.pdf"))
            available_runs["All runs"] = (vals, "#1f77b4")

        n_runs = len(available_runs)
        fig, axes = plt.subplots(1, n_runs,
                                  figsize=(max(5, 4 * n_runs), 4.2),
                                  sharey=True)
        if n_runs == 1:
            axes = [axes]

        for ax, (label, (vals, color)) in zip(axes, available_runs.items()):
            # Histogram with KDE
            n_bins = min(30, max(5, len(vals) // 3))
            ax.hist(vals, bins=n_bins, density=True, alpha=0.5,
                    color=color, edgecolor="white", linewidth=0.5)

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method="silverman")
                x_range = np.linspace(vals.min() * 0.95, vals.max() * 1.05, 200)
                ax.plot(x_range, kde(x_range), "-", color=color,
                        linewidth=2.0, label="KDE")
            except ImportError:
                pass

            # Mark key statistics
            median = float(np.median(vals))
            best = float(np.min(vals))
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))

            ax.axvline(median, color="black", linestyle="-", linewidth=1.2,
                       alpha=0.7, label=f"Median: {median:.4f}")
            ax.axvline(best, color="#d62728", linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"Best: {best:.4f}")
            ax.axvspan(q25, q75, alpha=0.1, color=color)

            ax.set_xlabel(r"$e_{\mathrm{Nys}}$", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{label} ($n={len(vals)}$)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2, axis="x")
            ax.tick_params(labelsize=9)

            # Annotation: IQR range
            ax.annotate(f"IQR: [{q25:.4f}, {q75:.4f}]",
                        xy=(0.5, 0.95), xycoords="axes fraction",
                        fontsize=7, ha="center", va="top", color="#555555")

        fig.suptitle(r"Distribution of $e_{\mathrm{Nys}}$ across Pareto "
                     "solutions ($k{=}300$)", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "nystrom_error_distribution.pdf"))

    def fig_krr_worst_state_rmse_vs_k(self, df: pd.DataFrame) -> str:
        r"""Fig N12: Worst-state KRR RMSE vs coreset budget k.

        Line plot showing how the most-disadvantaged state's RMSE improves
        with increasing k, compared against the average RMSE across all
        states.  Plots both 4G and 5G targets.

        Phase 10b — NEW figure (manuscript Section VIII.D, equity):
        * Two-panel layout: (a) 4G target, (b) 5G target.
        * Lines: worst-state RMSE, average RMSE, best-state RMSE.
        * Shaded band between worst and best states.
        * If per-state data is unavailable, uses global worst_state_rmse
          and mean RMSE columns from the results DataFrame.
        * Defense value: directly addresses equity/fairness concerns —
          does the framework disproportionately serve some states?
        """
        import matplotlib.pyplot as plt

        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty or "k" not in d.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5,
                    "No R1 data with multiple k values available.\n"
                    "Run R1 with k sweep to generate.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "krr_worst_state_rmse_vs_k.pdf"))

        d["k"] = d["k"].astype(int)

        # ---- Identify available RMSE columns ----
        target_defs = [
            ("4G", r"RMSE $y^{(4G)}$", "#1f77b4"),
            ("5G", r"RMSE $y^{(5G)}$", "#ff7f0e"),
        ]

        # Resolve column names with various naming conventions
        def _find_col(d, patterns):
            for p in patterns:
                if p in d.columns and d[p].notna().any():
                    return p
            return None

        target_cols = {}
        for tag, label, color in target_defs:
            mean_col = _find_col(d, [
                f"krr_rmse_{tag}", f"krr_rmse_cov_area_{tag}",
                f"krr_rmse_area_{tag}",
            ])
            worst_col = _find_col(d, [
                f"worst_state_rmse_{tag}", f"krr_worst_state_rmse_{tag}",
                f"state_rmse_max_{tag}",
            ])
            best_col = _find_col(d, [
                f"best_state_rmse_{tag}", f"krr_best_state_rmse_{tag}",
                f"state_rmse_min_{tag}",
            ])
            dispersion_col = _find_col(d, [
                f"state_rmse_std_{tag}", f"krr_state_rmse_std_{tag}",
            ])
            if mean_col:
                target_cols[tag] = {
                    "mean": mean_col, "worst": worst_col,
                    "best": best_col, "std": dispersion_col,
                    "label": label, "color": color,
                }

        if not target_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5,
                    "No KRR RMSE columns found for R1.\n"
                    "Ensure evaluation pipeline produces krr_rmse_4G/5G.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                         "krr_worst_state_rmse_vs_k.pdf"))

        # ---- Plot ----
        n_panels = len(target_cols)
        fig, axes = plt.subplots(1, n_panels,
                                  figsize=(max(5.5, 5 * n_panels), 4.2))
        if n_panels == 1:
            axes = [axes]

        for ax, (tag, info) in zip(axes, target_cols.items()):
            color = info["color"]

            # Group by k and compute envelope/mean
            grp = d.groupby("k")

            # Average RMSE line
            mean_by_k = grp[info["mean"]].mean()
            ks = mean_by_k.index.values
            ax.plot(ks, mean_by_k.values, "o-", color=color,
                    linewidth=2.0, markersize=6, label="Average RMSE",
                    zorder=5)

            # Worst-state RMSE line
            if info["worst"]:
                worst_by_k = grp[info["worst"]].mean()
                ax.plot(ks, worst_by_k.values, "s--", color="#d62728",
                        linewidth=1.5, markersize=5,
                        label="Worst-state RMSE", zorder=6)

                # Best-state RMSE line + shaded band
                if info["best"]:
                    best_by_k = grp[info["best"]].mean()
                    ax.plot(ks, best_by_k.values, "^:", color="#2ca02c",
                            linewidth=1.2, markersize=4,
                            label="Best-state RMSE", zorder=4)
                    ax.fill_between(ks, best_by_k.values, worst_by_k.values,
                                    alpha=0.1, color="#d62728")
            elif info["std"]:
                # Fallback: use mean ± std as proxy for state dispersion
                std_by_k = grp[info["std"]].mean()
                upper = mean_by_k.values + std_by_k.values
                lower = np.maximum(mean_by_k.values - std_by_k.values, 0)
                ax.fill_between(ks, lower, upper, alpha=0.15, color=color,
                                label="Mean ± state-std")

            ax.set_xlabel("Coreset size $k$", fontsize=10)
            ax.set_ylabel("RMSE", fontsize=10)
            _panel = "a" if tag == "4G" else "b"
            ax.set_title(f"({_panel}) {info['label']}",
                         fontsize=10, pad=6)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(labelsize=9)

        fig.suptitle("Worst-state vs.\\ average RMSE across coreset budget "
                     "(R1, equity analysis)", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                     "krr_worst_state_rmse_vs_k.pdf"))
