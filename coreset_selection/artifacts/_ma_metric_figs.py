"""Metric and distortion figure methods for ManuscriptArtifacts (mixin).

Manuscript figures rendered here (R/ggplot2 with matplotlib fallback):
  Fig 1: fig_kl_floor_vs_k        → kl_floor_vs_k.pdf
  Fig 3: fig_distortion_cardinality_r1 → distortion_cardinality_R1.pdf
  Fig 4: fig_krr_worst_state_rmse_vs_k → krr_worst_state_rmse_vs_k.pdf
"""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ._ma_helpers import _save, _save_r, use_r


class MetricFigsMixin:
    """Mixin providing metric/distortion visualization methods."""

    # ------------------------------------------------------------------
    # Fig 3: Distortion vs cardinality R1 (2×2 faceted)
    # ------------------------------------------------------------------
    def fig_distortion_cardinality_r1(self, df: pd.DataFrame) -> str:
        r"""Fig 3: Raw-space metrics vs *k* for R1 (Section VIII.C).

        2×2 panel showing metric-wise envelopes over the feasible
        non-dominated set for each :math:`k \in \mathcal{K}`.
        """
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "distortion_cardinality_R1.pdf")

        if df.empty or "run_id" not in df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No experiment data loaded",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No R1 data available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

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
            ("nystrom_error",   "e_Nys"),
            ("kpca_distortion", "e_kPCA"),
            (krr_4g,            "RMSE_4G"),
            (krr_5g,            "RMSE_5G"),
        ]

        # ---- Build tidy data for R ----
        tidy_rows = []
        for mcol, mlabel in metrics:
            if mcol not in d.columns:
                continue
            # Envelope (best per k)
            env = d.groupby("k")[mcol].min().reset_index().sort_values("k")
            for _, row in env.iterrows():
                tidy_rows.append({"k": int(row["k"]), "metric": mlabel,
                                  "value": float(row[mcol]), "stat": "envelope_best"})
            # Mean and std bounds per k
            agg = d.groupby("k")[mcol].agg(["mean", "std"]).reset_index().sort_values("k")
            for _, row in agg.iterrows():
                tidy_rows.append({"k": int(row["k"]), "metric": mlabel,
                                  "value": float(row["mean"]), "stat": "mean"})
                std_val = float(row["std"]) if pd.notna(row["std"]) else 0.0
                tidy_rows.append({"k": int(row["k"]), "metric": mlabel,
                                  "value": float(row["mean"]) + std_val,
                                  "stat": "mean_plus_std"})
                tidy_rows.append({"k": int(row["k"]), "metric": mlabel,
                                  "value": float(row["mean"]) - std_val,
                                  "stat": "mean_minus_std"})

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            mpl_metrics = [
                ("nystrom_error",   r"$e_{\mathrm{Nys}}$"),
                ("kpca_distortion", r"$e_{\mathrm{kPCA}}$"),
                (krr_4g,            r"RMSE$_{4\mathrm{G}}$"),
                (krr_5g,            r"RMSE$_{5\mathrm{G}}$"),
            ]
            panel_labels = ["(a)", "(b)", "(c)", "(d)"]
            fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
            axes_flat = axes.flatten()
            for idx, (ax, (mcol, ylabel)) in enumerate(zip(axes_flat, mpl_metrics)):
                if mcol not in d.columns:
                    ax.text(0.5, 0.5, f"{mcol}\nnot available",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=8, color="gray")
                    ax.set_axis_off()
                    ax.annotate(panel_labels[idx], xy=(0.02, 0.95),
                                xycoords="axes fraction", fontsize=10,
                                fontweight="bold", va="top")
                    continue
                env = d.groupby("k")[mcol].min().reset_index().sort_values("k")
                agg = d.groupby("k")[mcol].agg(["mean", "std", "count"]).reset_index().sort_values("k")
                multi_seed = agg["count"].max() > 1
                ax.plot(env["k"], env[mcol], "o-", linewidth=1.6,
                        markersize=4, color="#1f77b4", label="Envelope (best)", zorder=5)
                if multi_seed and agg["std"].max() > 0:
                    ax.fill_between(agg["k"], agg["mean"] - agg["std"],
                                    agg["mean"] + agg["std"],
                                    alpha=0.15, color="#1f77b4", label=r"Mean $\pm$ 1 std")
                if len(env) >= 3:
                    vals_arr = env[mcol].values
                    ks_arr = env["k"].values
                    if vals_arr[-2] > 0:
                        rel_improv = (vals_arr[-2] - vals_arr[-1]) / vals_arr[-2]
                        if rel_improv < 0.05:
                            ax.annotate("diminishing\nreturns", xy=(ks_arr[-1], vals_arr[-1]),
                                        xytext=(-35, 15), textcoords="offset points",
                                        fontsize=7, color="#888888",
                                        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7))
                ax.set_xlabel("$k$", fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                if multi_seed:
                    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
                ax.annotate(panel_labels[idx], xy=(0.03, 0.95),
                            xycoords="axes fraction", fontsize=10,
                            fontweight="bold", va="top")
            fig.suptitle(r"R1: Raw-space metrics vs.\ coreset size $k$ "
                         "(metric-wise envelope)", fontsize=11, y=1.01)
            fig.tight_layout()
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_distortion_cardinality_R1.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

    # ------------------------------------------------------------------
    # Fig 1: KL feasibility floor vs k
    # ------------------------------------------------------------------
    def fig_kl_floor_vs_k(self) -> str:
        r"""Fig 1: KL feasibility floor vs coreset size k (Section VIII.A)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "kl_floor_vs_k.pdf")

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
                return _save(fig, out_path)
            import glob as _glob
            cfs = _glob.glob(os.path.join(self.cache_root, "rep*/assets.npz"))
            if not cfs:
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                ax.text(0.5, 0.5, "No caches available for KL floor computation.",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                return _save(fig, out_path)
            assets = load_replicate_cache(cfs[0])
            geo = build_geo_info(assets.state_labels)
            ks = np.arange(27, 501)
            kl_vals = []
            for k in ks:
                kl, _ = min_achievable_geo_kl_bounded(
                    pi=geo.pi, group_sizes=geo.group_sizes, k=int(k),
                    alpha_geo=1.0, min_one_per_group=True,
                )
                kl_vals.append(float(kl))
            vals = np.array(kl_vals)

        # ---- Build tidy data for R ----
        tidy_df = pd.DataFrame({"k": ks.astype(int), "kl_min": vals.astype(float)})

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot(ks, vals, linewidth=2.0, color="#1f77b4",
                    label=r"$\mathrm{KL}_{\min}(k)$", zorder=5)
            tau_thresholds = [
                (0.05, "#2ca02c", r"$\tau = 0.05$"),
                (0.02, "#ff7f0e", r"$\tau = 0.02$"),
                (0.01, "#d62728", r"$\tau = 0.01$"),
            ]
            for tau, color, label in tau_thresholds:
                ax.axhline(tau, color=color, linestyle="--", linewidth=1.0,
                           alpha=0.75, label=label, zorder=3)
                feasible_idx = np.where(np.asarray(vals, dtype=float) <= tau)[0]
                if len(feasible_idx) > 0:
                    k_min_feasible = float(ks[feasible_idx[0]])
                    ax.plot(k_min_feasible, tau, "v", color=color, markersize=7, zorder=8)
                    ax.annotate(f"$k \\geq {int(k_min_feasible)}$",
                                xy=(k_min_feasible, tau),
                                xytext=(k_min_feasible + 25, tau + 0.008),
                                fontsize=7.5, color=color,
                                arrowprops=dict(arrowstyle="->", color=color, linewidth=0.8),
                                zorder=9)
            grid_k = [50, 100, 200, 300, 400, 500]
            for gk in grid_k:
                idx = np.argmin(np.abs(np.asarray(ks, dtype=float) - gk))
                if idx < len(vals):
                    ax.plot(ks[idx], vals[idx], "o", color="#d62728", markersize=6,
                            zorder=7, markeredgecolor="k", markeredgewidth=0.4)
                    if gk in (50, 100, 300, 500):
                        offset_y = 0.006 if vals[idx] > 0.01 else 0.003
                        ax.annotate(f"{vals[idx]:.4f}",
                                    xy=(ks[idx], vals[idx]),
                                    xytext=(ks[idx], vals[idx] + offset_y),
                                    fontsize=7, ha="center", color="#555555", zorder=9)
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
            return _save(fig, out_path)

        return _save_r("fig_kl_floor_vs_k.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)

    # ------------------------------------------------------------------
    # Fig N5: Effort–quality trade-off (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_effort_quality(self, df: pd.DataFrame) -> str:
        r"""Fig N5: Effort sweep — selection time vs downstream quality (R12)."""
        import matplotlib.pyplot as plt

        d = df[df["run_id"].astype(str).str.contains("R12")].copy()
        if d.empty or "wall_clock_s" not in d.columns:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.text(0.5, 0.5, "No R12 effort sweep data available.\n"
                    "Run R12 first to generate effort grid results.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "effort_quality_tradeoff.pdf"))

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
        has_effort = "effort_P_x_T" in d.columns
        has_pop = "pop_size" in d.columns
        has_ngen = "n_gen" in d.columns

        ax = axes[0]
        if has_pop:
            pop_sizes = sorted(d["pop_size"].unique())
            cmap_vals = plt.cm.viridis(np.linspace(0.2, 0.9, len(pop_sizes)))
            pop_color = {p: cmap_vals[i] for i, p in enumerate(pop_sizes)}
            for ps in pop_sizes:
                sub = d[d["pop_size"] == ps].sort_values("wall_clock_s")
                ax.plot(sub["wall_clock_s"], sub[quality_col], "o-",
                        color=pop_color[ps], markersize=6, linewidth=1.0, alpha=0.8,
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

        ax = axes[1]
        if has_effort:
            d_sorted = d.sort_values("effort_P_x_T").copy()
            ax.scatter(d_sorted["effort_P_x_T"], d_sorted[quality_col],
                       s=35, alpha=0.7, color="#d62728", edgecolors="k",
                       linewidth=0.3, zorder=4)
            d_sorted["best_so_far"] = d_sorted[quality_col].cummin()
            ax.plot(d_sorted["effort_P_x_T"], d_sorted["best_so_far"],
                    "-", linewidth=2.0, color="#1f77b4", zorder=5, label="Best-so-far")
            bsf = d_sorted["best_so_far"].values
            if len(bsf) > 3:
                efforts = d_sorted["effort_P_x_T"].values.astype(float)
                b_norm = (bsf - bsf.min()) / max(bsf.max() - bsf.min(), 1e-12)
                improvements = np.diff(b_norm)
                if len(improvements) > 1:
                    elbow_idx = int(np.argmax(np.abs(np.diff(improvements)))) + 1
                    elbow_effort = efforts[elbow_idx]
                    ax.axvline(elbow_effort, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
                    ax.annotate("diminishing\nreturns",
                                xy=(elbow_effort, bsf[elbow_idx]),
                                xytext=(elbow_effort * 1.2, bsf[elbow_idx] * 1.1),
                                fontsize=7.5, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray", linewidth=0.8))
            if has_pop and has_ngen:
                default_mask = (d["pop_size"] == 200) & (d["n_gen"] == 1000)
                default_rows = d[default_mask]
                if not default_rows.empty:
                    for _, dr in default_rows.iterrows():
                        ax.scatter(dr["effort_P_x_T"], dr[quality_col],
                                   s=100, marker="*", color="#ff7f0e",
                                   edgecolors="k", linewidths=0.6, zorder=10)
            ax.set_xlabel("Total effort ($P \\times T$)", fontsize=10)
        else:
            ax.scatter(range(len(d)), d[quality_col], s=30, alpha=0.7, color="#d62728")
            ax.set_xlabel("Run index", fontsize=10)
        ax.set_ylabel(quality_label, fontsize=10)
        ax.set_title("(b) Quality vs.\\ total effort", fontsize=10, pad=6)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)
        fig.suptitle("Effort\u2013quality trade-off (R12 sweep)", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "effort_quality_tradeoff.pdf"))

    # ------------------------------------------------------------------
    # Fig N7: Multi-seed boxplot (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_multi_seed_boxplot(self, df: pd.DataFrame) -> str:
        r"""Fig N7: Multi-seed robustness boxplots — R1, R5 at k=300."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        _aliases = {"krr_rmse_4G": ["krr_rmse_cov_area_4G", "krr_rmse_area_4G"]}
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
            return _save(fig, os.path.join(self.fig_dir, "multi_seed_stability_boxplot.pdf"))

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(max(7, 3.5 * n_metrics), 4.2))
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric, label) in zip(axes, metrics):
            box_data, box_labels, box_colors = [], [], []
            for run_label, (rid, color) in runs.items():
                sub = d[d["run_id"].astype(str).str.contains(rid)]
                if not sub.empty and metric in sub.columns:
                    v = sub[metric].dropna().values
                    if len(v) > 0:
                        box_data.append(v)
                        box_labels.append(run_label)
                        box_colors.append(color)
            if not box_data:
                ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                continue
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color); patch.set_alpha(0.35)
                patch.set_edgecolor(color); patch.set_linewidth(1.2)
            for whisker, color in zip(bp["whiskers"], [c for c in box_colors for _ in range(2)]):
                whisker.set_color(color)
            for cap, color in zip(bp["caps"], [c for c in box_colors for _ in range(2)]):
                cap.set_color(color)
            for i, (v, color) in enumerate(zip(box_data, box_colors)):
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(v))
                ax.scatter(np.full_like(v, i + 1) + jitter, v, s=18, alpha=0.7,
                           color=color, edgecolors="k", linewidths=0.3, zorder=5)
            for i, v in enumerate(box_data):
                mean = float(np.mean(v)); std = float(np.std(v))
                ax.annotate(f"$\\mu$={mean:.4f}\n$\\sigma$={std:.4f}",
                            xy=(i + 1, mean), xytext=(i + 1.35, mean),
                            fontsize=6.5, color="#555555",
                            arrowprops=dict(arrowstyle="->", color="#999999", linewidth=0.6),
                            ha="left", va="center")
            ax.set_title(label, fontsize=10); ax.set_ylabel("Value", fontsize=9)
            ax.grid(True, alpha=0.2, axis="y"); ax.tick_params(labelsize=9)

        fig.suptitle("Multi-seed stability at $k{=}300$ (5 seeds per run)", fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "multi_seed_stability_boxplot.pdf"))

    # ------------------------------------------------------------------
    # Fig N11: Nystrom error distribution (complementary, matplotlib only)
    # ------------------------------------------------------------------
    def fig_nystrom_error_distribution(self, df: pd.DataFrame) -> str:
        r"""Fig N11: Distribution of Nystrom errors across Pareto solutions at k=300."""
        import matplotlib.pyplot as plt

        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if d.empty or "nystrom_error" not in d.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No Nystrom error data at k=300.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir, "nystrom_error_distribution.pdf"))

        run_configs = {
            "R1 (pop-share)": ("R1", "#1f77b4"),
            "R5 (joint)":     ("R5", "#2ca02c"),
            "R9 (VAE)":       ("R9", "#ff7f0e"),
        }
        available_runs = {}
        for label, (rid, color) in run_configs.items():
            sub = d[d["run_id"].astype(str).str.contains(rid)]
            v = sub["nystrom_error"].dropna().values
            if len(v) > 0:
                available_runs[label] = (v, color)
        if not available_runs:
            v = d["nystrom_error"].dropna().values
            if len(v) == 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(0.5, 0.5, "No Nystrom error values.", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_axis_off()
                return _save(fig, os.path.join(self.fig_dir, "nystrom_error_distribution.pdf"))
            available_runs["All runs"] = (v, "#1f77b4")

        n_runs = len(available_runs)
        fig, axes = plt.subplots(1, n_runs, figsize=(max(5, 4 * n_runs), 4.2), sharey=True)
        if n_runs == 1:
            axes = [axes]

        for ax, (label, (v, color)) in zip(axes, available_runs.items()):
            n_bins = min(30, max(5, len(v) // 3))
            ax.hist(v, bins=n_bins, density=True, alpha=0.5, color=color,
                    edgecolor="white", linewidth=0.5)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(v, bw_method="silverman")
                x_range = np.linspace(v.min() * 0.95, v.max() * 1.05, 200)
                ax.plot(x_range, kde(x_range), "-", color=color, linewidth=2.0, label="KDE")
            except ImportError:
                pass
            median = float(np.median(v)); best = float(np.min(v))
            q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
            ax.axvline(median, color="black", linestyle="-", linewidth=1.2, alpha=0.7,
                       label=f"Median: {median:.4f}")
            ax.axvline(best, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"Best: {best:.4f}")
            ax.axvspan(q25, q75, alpha=0.1, color=color)
            ax.set_xlabel(r"$e_{\mathrm{Nys}}$", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{label} ($n={len(v)}$)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2, axis="x"); ax.tick_params(labelsize=9)
            ax.annotate(f"IQR: [{q25:.4f}, {q75:.4f}]", xy=(0.5, 0.95),
                        xycoords="axes fraction", fontsize=7, ha="center", va="top", color="#555555")

        fig.suptitle(r"Distribution of $e_{\mathrm{Nys}}$ across Pareto solutions ($k{=}300$)",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir, "nystrom_error_distribution.pdf"))

    # ------------------------------------------------------------------
    # Fig 4: Worst-state KRR RMSE vs k
    # ------------------------------------------------------------------
    def fig_krr_worst_state_rmse_vs_k(self, df: pd.DataFrame) -> str:
        r"""Fig 4: Worst-state KRR RMSE vs coreset budget k (Section VIII.D)."""
        import matplotlib.pyplot as plt

        out_path = os.path.join(self.fig_dir, "krr_worst_state_rmse_vs_k.pdf")

        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty or "k" not in d.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No R1 data with multiple k values available.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        d["k"] = d["k"].astype(int)

        # Identify available RMSE columns
        target_defs = [("4G", "#1f77b4"), ("5G", "#ff7f0e")]

        def _find_col(frame, patterns):
            for p in patterns:
                if p in frame.columns and frame[p].notna().any():
                    return p
            return None

        target_cols = {}
        for tag, color in target_defs:
            mean_col = _find_col(d, [f"krr_rmse_{tag}", f"krr_rmse_cov_area_{tag}", f"krr_rmse_area_{tag}"])
            worst_col = _find_col(d, [f"worst_state_rmse_{tag}", f"krr_worst_state_rmse_{tag}", f"state_rmse_max_{tag}"])
            best_col = _find_col(d, [f"best_state_rmse_{tag}", f"krr_best_state_rmse_{tag}", f"state_rmse_min_{tag}"])
            if mean_col:
                target_cols[tag] = {"mean": mean_col, "worst": worst_col, "best": best_col, "color": color}

        if not target_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No KRR RMSE columns found for R1.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            return _save(fig, out_path)

        # ---- Build tidy data for R ----
        tidy_rows = []
        grp = d.groupby("k")
        for tag, info in target_cols.items():
            mean_by_k = grp[info["mean"]].mean()
            for k_val, rmse_val in mean_by_k.items():
                row = {"k": int(k_val), "target": tag, "mean_rmse": float(rmse_val),
                       "worst_rmse": np.nan, "best_rmse": np.nan}
                if info["worst"]:
                    worst_by_k = grp[info["worst"]].mean()
                    if k_val in worst_by_k.index:
                        row["worst_rmse"] = float(worst_by_k[k_val])
                if info["best"]:
                    best_by_k = grp[info["best"]].mean()
                    if k_val in best_by_k.index:
                        row["best_rmse"] = float(best_by_k[k_val])
                tidy_rows.append(row)

        tidy_df = pd.DataFrame(tidy_rows)

        # ---- Matplotlib fallback ----
        def _mpl_fallback():
            n_panels = len(target_cols)
            fig, axes = plt.subplots(1, n_panels, figsize=(max(5.5, 5 * n_panels), 4.2))
            if n_panels == 1:
                axes = [axes]
            for ax, (tag, info) in zip(axes, target_cols.items()):
                color = info["color"]
                mean_by_k = grp[info["mean"]].mean()
                ks_arr = mean_by_k.index.values
                ax.plot(ks_arr, mean_by_k.values, "o-", color=color,
                        linewidth=2.0, markersize=6, label="Average RMSE", zorder=5)
                if info["worst"]:
                    worst_by_k = grp[info["worst"]].mean()
                    ax.plot(ks_arr, worst_by_k.values, "s--", color="#d62728",
                            linewidth=1.5, markersize=5, label="Worst-state RMSE", zorder=6)
                    if info["best"]:
                        best_by_k = grp[info["best"]].mean()
                        ax.plot(ks_arr, best_by_k.values, "^:", color="#2ca02c",
                                linewidth=1.2, markersize=4, label="Best-state RMSE", zorder=4)
                        ax.fill_between(ks_arr, best_by_k.values, worst_by_k.values,
                                        alpha=0.1, color="#d62728")
                ax.set_xlabel("Coreset size $k$", fontsize=10)
                ax.set_ylabel("RMSE", fontsize=10)
                _panel = "a" if tag == "4G" else "b"
                ax.set_title(f"({_panel}) RMSE $y^{{({tag})}}$", fontsize=10, pad=6)
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.tick_params(labelsize=9)
            fig.suptitle("Worst-state vs.\\ average RMSE across coreset budget "
                         "(R1, equity analysis)", fontsize=11, y=1.01)
            fig.tight_layout()
            return _save(fig, out_path)

        if tidy_df.empty:
            return _mpl_fallback()

        return _save_r("fig_krr_worst_state_rmse.R", tidy_df, out_path,
                        fallback_fn=_mpl_fallback)
