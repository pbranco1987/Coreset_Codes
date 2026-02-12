"""
Figure generation mixin for ManuscriptArtifactGenerator.

Contains all ``figure_*`` methods that produce manuscript-facing PDF figures.
"""

from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import pandas as pd

from ..data.cache import load_replicate_cache
from ..geo import build_geo_info
from ..geo.kl import min_achievable_geo_kl_bounded
from ..experiment.saver import load_pareto_front


class GenFiguresMixin:
    """Mixin providing all ``figure_*`` methods for the artifact generator."""

    def figure_klmin_vs_k(self) -> str:
        """figures/klmin_vs_k.pdf"""
        import matplotlib.pyplot as plt

        # Prefer stored R0 rows if present; otherwise recompute from a cache.
        df_all = self._load_all_results()
        df_r0 = df_all[df_all.get("run_id", "").astype(str).str.startswith("R0")]

        if not df_r0.empty and "k" in df_r0.columns and ("kl_min" in df_r0.columns or "geo_kl" in df_r0.columns):
            df_plot = df_r0.copy()
            if "kl_min" not in df_plot.columns:
                df_plot["kl_min"] = df_plot["geo_kl"]
            df_plot = df_plot.sort_values("k")
            ks = df_plot["k"].astype(int).values
            vals = df_plot["kl_min"].astype(float).values
        else:
            cache_files = glob.glob(os.path.join(self.cache_root, "rep*/assets.npz"))
            if not cache_files:
                # Placeholder
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.text(0.5, 0.5, "No caches found", ha="center", va="center")
                ax.set_axis_off()
                return self._save_pdf(fig, "klmin_vs_k.pdf")
            assets = load_replicate_cache(cache_files[0])
            geo = build_geo_info(assets.state_labels)
            ks = np.array([50, 100, 200, 300, 400, 500], dtype=int)
            vals = []
            for k in ks:
                kl_min, _ = min_achievable_geo_kl_bounded(
                    pi=geo.pi,
                    group_sizes=geo.group_sizes,
                    k=int(k),
                    alpha_geo=1.0,
                    min_one_per_group=True,
                )
                vals.append(float(kl_min))
            vals = np.asarray(vals, dtype=float)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(ks, vals, marker="o", linewidth=1.5)
        ax.set_xlabel("Coreset size $k$")
        ax.set_ylabel(r"$\mathrm{KL}_{\min}(k)$")
        ax.grid(True, alpha=0.3)
        return self._save_pdf(fig, "klmin_vs_k.pdf")

    def figure_pareto_r1_k300(self) -> List[str]:
        """figures/pareto3d_k300_R1.pdf + 2D projections."""
        import matplotlib.pyplot as plt

        paths: List[str] = []
        # Collect all R1_k300 replicate pareto fronts
        pf_files = glob.glob(os.path.join(self.runs_root, "R1_k300", "rep*", "results", "vae_pareto.npz"))
        if not pf_files:
            # Try alternative run naming
            pf_files = glob.glob(os.path.join(self.runs_root, "R1", "rep*", "results", "vae_pareto.npz"))
        if not pf_files:
            # Placeholder figures to satisfy main.tex paths
            for fname in [
                "pareto3d_k300_R1.pdf",
                "pareto2d_skl_mmd_k300_R1.pdf",
                "pareto2d_skl_sd_k300_R1.pdf",
                "pareto2d_mmd_sd_k300_R1.pdf",
            ]:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.text(0.5, 0.5, "No R1_k300 pareto fronts found", ha="center", va="center")
                ax.set_axis_off()
                paths.append(self._save_pdf(fig, fname))
            return paths

        fronts = []
        obj_names = None
        for p in pf_files:
            try:
                pf = load_pareto_front(p)
                fronts.append(pf)
                obj_names = obj_names or list(pf.objectives)
            except Exception:
                continue

        if not fronts or obj_names is None:
            return []

        # Map objectives to indices
        name_to_idx = {str(n): i for i, n in enumerate(obj_names)}
        i_skl = name_to_idx.get("skl", 0)
        i_mmd = name_to_idx.get("mmd", 1 if len(obj_names) > 1 else 0)
        i_sd = name_to_idx.get("sinkhorn", 2 if len(obj_names) > 2 else (len(obj_names) - 1))

        # 3D plot
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(111, projection="3d")
        for pf in fronts:
            F = np.asarray(pf.F)
            ax.scatter(F[:, i_skl], F[:, i_mmd], F[:, i_sd], s=12, alpha=0.6)
        ax.set_xlabel("SKL")
        ax.set_ylabel("MMD")
        ax.set_zlabel("SD")
        paths.append(self._save_pdf(fig, "pareto3d_k300_R1.pdf"))

        # 2D projections
        def _scatter2(i, j, fname, xl, yl):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for pf in fronts:
                F = np.asarray(pf.F)
                ax.scatter(F[:, i], F[:, j], s=12, alpha=0.6)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
            return self._save_pdf(fig, fname)

        paths.append(_scatter2(i_skl, i_mmd, "pareto2d_skl_mmd_k300_R1.pdf", "SKL", "MMD"))
        paths.append(_scatter2(i_skl, i_sd, "pareto2d_skl_sd_k300_R1.pdf", "SKL", "SD"))
        paths.append(_scatter2(i_mmd, i_sd, "pareto2d_mmd_sd_k300_R1.pdf", "MMD", "SD"))
        return paths

    def figure_distortion_cardinality_r1(self, df_all: pd.DataFrame) -> str:
        r"""Fig 2: ``distortion_cardinality_R1.pdf`` (Section VIII.C).

        Phase 8: 2×2 panel with (a)–(d) labels, error bands, envelope line,
        IEEE double-column sizing (~7 in), ≥ 10 pt labels.
        """
        import matplotlib.pyplot as plt

        df = df_all.copy()
        df = df[df.get("run_id", "").astype(str).str.startswith("R1")]
        if df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(0.5, 0.5, "No R1 results found", ha="center", va="center")
            ax.set_axis_off()
            return self._save_pdf(fig, "distortion_cardinality_R1.pdf")

        if "k" not in df.columns:
            df["k"] = df["run_id"].astype(str).str.extract(r"_k(\d+)").astype(float)
        df["k"] = df["k"].astype(int)

        metrics = [
            ("nystrom_error",    r"$e_{\mathrm{Nys}}$"),
            ("kpca_distortion",  r"$e_{\mathrm{kPCA}}$"),
            ("krr_rmse_4G",      r"RMSE$_{4\mathrm{G}}$"),
            ("krr_rmse_5G",      r"RMSE$_{5\mathrm{G}}$"),
        ]
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
        for idx, (ax, (mcol, ylabel)) in enumerate(zip(axes.flatten(), metrics)):
            if mcol not in df.columns:
                ax.text(0.5, 0.5, f"{mcol} n/a", ha="center", va="center",
                        fontsize=8, color="gray")
                ax.set_axis_off()
                continue
            # Envelope (best per k)
            env = df.groupby("k")[mcol].min().reset_index().sort_values("k")
            ax.plot(env["k"], env[mcol], "o-", linewidth=1.5, markersize=4,
                    color="#1f77b4", label="Envelope (best)")
            # Error band
            agg = df.groupby("k")[mcol].agg(["mean", "std"]).reset_index().sort_values("k")
            if agg["std"].max() > 0:
                ax.fill_between(agg["k"], agg["mean"] - agg["std"],
                                agg["mean"] + agg["std"], alpha=0.15, color="#1f77b4")
            ax.set_xlabel("$k$", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.25)
            ax.annotate(panel_labels[idx], xy=(0.03, 0.95),
                        xycoords="axes fraction", fontsize=10, fontweight="bold",
                        va="top")
        fig.suptitle("R1: Raw-space metrics vs.\\ coreset size $k$",
                      fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save_pdf(fig, "distortion_cardinality_R1.pdf")

    def figure_baseline_comparison_k300(self, df_all: pd.DataFrame) -> str:
        """figures/baseline_comparison_k300.pdf"""
        import matplotlib.pyplot as plt
        import pandas as pd

        df = df_all.copy()
        # Normalize k
        if "k" not in df.columns:
            df["k"] = df["run_id"].astype(str).str.extract(r"_k(\d+)").astype(float)
        df = df[df["k"].fillna(300).astype(int) == 300]
        if df.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.text(0.5, 0.5, "No k=300 results found", ha="center", va="center")
            ax.set_axis_off()
            return self._save_pdf(fig, "baseline_comparison_k300.pdf")

        # Methods: NSGA2 reps and baselines
        metric = "nystrom_error" if "nystrom_error" in df.columns else df.columns[df.columns.str.contains("nystrom")].tolist()[0]
        df_plot = df[["run_id", "rep_id", "space", "method", "rep_name", "constraint_regime", "k", metric]].copy()

        # Compact method label
        df_plot["label"] = df_plot["method"].astype(str)
        if "space" in df_plot.columns:
            df_plot["label"] = df_plot["label"] + "(" + df_plot["space"].astype(str) + ")"

        # Two panels: quota vs exactk
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, regime in zip(axes, ["quota", "exactk"]):
            d = df_plot[df_plot["constraint_regime"].astype(str) == regime]
            if d.empty:
                ax.set_axis_off()
                continue
            g = d.groupby("label")[metric].mean().sort_values()
            ax.barh(g.index, g.values)
            ax.set_title(regime)
            ax.grid(True, axis="x", alpha=0.3)
        axes[0].set_ylabel("method(space)")
        axes[0].set_xlabel(metric)
        axes[1].set_xlabel(metric)
        return self._save_pdf(fig, "baseline_comparison_k300.pdf")

    def figure_geo_ablation_scatter(self, df_all: pd.DataFrame) -> str:
        r"""Fig 1: ``geo_ablation_tradeoff_scatter.pdf`` (Section VIII.B).

        Phase 8: log-scaled y-axis, marker shapes by regime, R1 constrained
        knee-point overlay, IEEE single-column sizing (~3.5 in).
        """
        import matplotlib.pyplot as plt

        df = df_all.copy()
        if "k" not in df.columns:
            df["k"] = df["run_id"].astype(str).str.extract(r"_k(\d+)").astype(float)
        df = df[df["k"].fillna(300).astype(int) == 300]
        if "geo_l1" not in df.columns or "nystrom_error" not in df.columns:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "geo_l1 or nystrom_error missing",
                    ha="center", va="center", fontsize=9)
            ax.set_axis_off()
            return self._save_pdf(fig, "geo_ablation_tradeoff_scatter.pdf")

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        regime_styles = {
            "quota": ("s", "#ff7f0e", "Quota-matched"),
            "exactk": ("o", "#1f77b4", "Exact-$k$ only"),
        }
        cr = df.get("constraint_regime", pd.Series(["unknown"] * len(df), index=df.index))
        for regime in cr.unique():
            mk, col, lab = regime_styles.get(str(regime), ("o", "#7f7f7f", str(regime)))
            sub = df[cr == regime]
            ax.scatter(sub["geo_l1"], sub["nystrom_error"], s=22, alpha=0.65,
                       marker=mk, color=col, label=lab, edgecolors="k", linewidths=0.3)

        # Overlay R1 constrained knee-point if available
        r1 = df_all[df_all.get("run_id", "").astype(str).str.startswith("R1")]
        r1 = r1[r1["k"].fillna(300).astype(int) == 300]
        if not r1.empty and "geo_l1" in r1.columns:
            knee = r1[r1.get("rep_name", pd.Series()).astype(str) == "knee"]
            if knee.empty:
                knee = r1
            if not knee.empty:
                row = knee.loc[knee["nystrom_error"].idxmin()]
                ax.scatter(row["geo_l1"], row["nystrom_error"],
                           s=90, marker="*", color="#d62728", edgecolors="k",
                           linewidths=0.5, zorder=10, label="R1 knee (constrained)")

        ax.set_xlabel(r"Geographic $\ell_1$ drift", fontsize=10)
        ax.set_ylabel(r"Nystr\"{o}m error", fontsize=10)
        ax.set_yscale("log")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_pdf(fig, "geo_ablation_tradeoff_scatter.pdf")

    def figure_objective_ablation_fronts_k300(self) -> str:
        """figures/objective_ablation_fronts_k300.pdf"""
        import matplotlib.pyplot as plt

        configs = [
            ("R3", "vae", "mmd", "sinkhorn", "MMD", "SD"),
            ("R4", "vae", "skl", "mmd", "SKL", "MMD"),
            ("R5", "vae", "skl", "sinkhorn", "SKL", "SD"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (rid, space, o1, o2, l1, l2) in zip(axes, configs):
            pf_files = glob.glob(os.path.join(self.runs_root, rid, "rep*", "results", f"{space}_pareto.npz"))
            if not pf_files:
                ax.set_title(f"{rid} (missing)")
                ax.set_axis_off()
                continue
            for p in pf_files:
                pf = load_pareto_front(p)
                names = list(pf.objectives)
                if o1 not in names or o2 not in names:
                    continue
                i1, i2 = names.index(o1), names.index(o2)
                F = np.asarray(pf.F)
                ax.scatter(F[:, i1], F[:, i2], s=10, alpha=0.6)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.set_title(rid)
            ax.grid(True, alpha=0.3)
        return self._save_pdf(fig, "objective_ablation_fronts_k300.pdf")

    def figure_representation_transfer_k300(self, df_all: pd.DataFrame) -> str:
        """figures/representation_transfer_k300.pdf"""
        import matplotlib.pyplot as plt

        df = df_all.copy()
        df = df[df.get("run_id", "").astype(str).isin(["R3", "R7", "R8"])]
        if df.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.text(0.5, 0.5, "No R3/R7/R8 rows found", ha="center", va="center")
            ax.set_axis_off()
            return self._save_pdf(fig, "representation_transfer_k300.pdf")

        # Metric panel: Nyström error and mean RMSE if available
        m1 = "nystrom_error"
        m2 = "krr_rmse_mean" if "krr_rmse_mean" in df.columns else ("krr_rmse_4G" if "krr_rmse_4G" in df.columns else None)

        fig, axes = plt.subplots(1, 2 if m2 else 1, figsize=(11, 4), sharey=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, metric in zip(axes, [m1] + ([m2] if m2 else [])):
            g = df.groupby("run_id")[metric].mean().reindex(["R3", "R7", "R8"])
            ax.bar(g.index, g.values)
            ax.set_title(metric)
            ax.grid(True, axis="y", alpha=0.3)
        return self._save_pdf(fig, "representation_transfer_k300.pdf")

    def figure_pareto_biobjective_grid_k300(self) -> str:
        """figures/pareto2d_biobjective_k300.pdf"""
        import matplotlib.pyplot as plt

        run_specs = [
            ("R3", "vae"),
            ("R4", "vae"),
            ("R5", "vae"),
            ("R7", "pca"),
            ("R8", "raw"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.flatten()
        for ax, (rid, space) in zip(axes, run_specs + [(None, None)]):
            if rid is None:
                ax.set_axis_off()
                continue
            pf_files = glob.glob(os.path.join(self.runs_root, rid, "rep*", "results", f"{space}_pareto.npz"))
            if not pf_files:
                ax.set_title(f"{rid} (missing)")
                ax.set_axis_off()
                continue
            for p in pf_files:
                pf = load_pareto_front(p)
                F = np.asarray(pf.F)
                if F.shape[1] < 2:
                    continue
                ax.scatter(F[:, 0], F[:, 1], s=8, alpha=0.6)
            ax.set_title(f"{rid} ({space})")
            ax.grid(True, alpha=0.3)
        return self._save_pdf(fig, "pareto2d_biobjective_k300.pdf")

    def figure_r7_surrogate_sensitivity(self) -> List[str]:
        """figures/surrogate_rankcorr.pdf and figures/surrogate_scatter_reference_vs_alt.pdf"""
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        from ..objectives.mmd import compute_rff_features
        from ..objectives.sinkhorn import AnchorSinkhorn
        from ..config.dataclasses import SinkhornConfig, MMDConfig
        from ..utils.math import median_sq_dist

        # Collect a manageable set of subsets from all saved coreset files
        coreset_files = glob.glob(os.path.join(self.runs_root, "**", "coresets", "*.npz"), recursive=True)
        subsets = []
        for p in coreset_files:
            try:
                d = np.load(p, allow_pickle=True)
                idx = np.asarray(d["indices"], dtype=int)
                if idx.size > 0:
                    subsets.append((p, idx))
            except Exception:
                continue
        if not subsets:
            # Placeholders
            figs = []
            for fname in ["surrogate_rankcorr.pdf", "surrogate_scatter_reference_vs_alt.pdf"]:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.text(0.5, 0.5, "No saved coresets found", ha="center", va="center")
                ax.set_axis_off()
                figs.append(self._save_pdf(fig, fname))
            return figs

        # Limit to avoid pathological runtime
        subsets = subsets[:120]

        # Use rep0 VAE space as reference for sensitivity plots
        cache_files = glob.glob(os.path.join(self.cache_root, "rep00", "assets.npz"))
        if not cache_files:
            cache_files = glob.glob(os.path.join(self.cache_root, "rep*", "assets.npz"))
        assets = load_replicate_cache(cache_files[0])
        X = np.asarray(assets.Z_vae if assets.Z_vae is not None else assets.X_scaled, dtype=np.float64)
        seed = 123

        # Reference settings
        sigma_sq = median_sq_dist(X, sample_size=2048, seed=seed) / 2.0
        ref_m = 2000
        Phi_ref = compute_rff_features(X, rff_dim=ref_m, sigma_sq=sigma_sq, seed=seed)
        mu_ref = Phi_ref.mean(axis=0)

        def mmd_from_phi(Phi, idx):
            mu_s = Phi[idx].mean(axis=0)
            diff = mu_s - Phi.mean(axis=0)
            return float(np.dot(diff, diff))

        m_grid = [500, 1000, 2000, 4000]
        mmd_ref_vals = []
        for _, idx in subsets:
            mmd_ref_vals.append(mmd_from_phi(Phi_ref, idx))
        mmd_ref_vals = np.asarray(mmd_ref_vals)

        mmd_rhos = []
        for m in m_grid:
            Phi = compute_rff_features(X, rff_dim=int(m), sigma_sq=sigma_sq, seed=seed)
            vals = np.asarray([mmd_from_phi(Phi, idx) for _, idx in subsets])
            rho = spearmanr(mmd_ref_vals, vals).correlation
            mmd_rhos.append(rho)

        # Sinkhorn settings
        sd_ref_cfg = SinkhornConfig(n_anchors=200, eta=0.05, max_iter=100)
        sd_ref = AnchorSinkhorn.build(X, sd_ref_cfg, seed=seed)
        sd_ref_vals = np.asarray([sd_ref.sinkhorn_divergence_subset(X, idx) for _, idx in subsets])

        A_grid = [50, 100, 200, 400]
        it_grid = [100, 200]
        sd_rhos = {it: [] for it in it_grid}
        for it in it_grid:
            for A in A_grid:
                cfg = SinkhornConfig(n_anchors=int(A), eta=0.05, max_iter=int(it))
                est = AnchorSinkhorn.build(X, cfg, seed=seed)
                vals = np.asarray([est.sinkhorn_divergence_subset(X, idx) for _, idx in subsets])
                rho = spearmanr(sd_ref_vals, vals).correlation
                sd_rhos[it].append(rho)

        # Rank-correlation figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(m_grid, mmd_rhos, marker="o")
        axes[0].set_xlabel("RFF dim m")
        axes[0].set_ylabel("Spearman \u03c1")
        axes[0].set_title("MMD sensitivity")
        axes[0].grid(True, alpha=0.3)

        for it in it_grid:
            axes[1].plot(A_grid, sd_rhos[it], marker="o", label=f"iters={it}")
        axes[1].set_xlabel("Anchors A")
        axes[1].set_ylabel("Spearman \u03c1")
        axes[1].set_title("Sinkhorn SD sensitivity")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        p1 = self._save_pdf(fig, "surrogate_rankcorr.pdf")

        # Scatter example (reference vs alt)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # MMD: m=500
        Phi_m = compute_rff_features(X, rff_dim=500, sigma_sq=sigma_sq, seed=seed)
        mmd_alt = np.asarray([mmd_from_phi(Phi_m, idx) for _, idx in subsets])
        axes[0].scatter(mmd_ref_vals, mmd_alt, s=12, alpha=0.7)
        axes[0].set_xlabel("MMD (m=2000)")
        axes[0].set_ylabel("MMD (m=500)")
        axes[0].grid(True, alpha=0.3)
        # SD: A=50 it=100
        est_alt = AnchorSinkhorn.build(X, SinkhornConfig(n_anchors=50, eta=0.05, max_iter=100), seed=seed)
        sd_alt = np.asarray([est_alt.sinkhorn_divergence_subset(X, idx) for _, idx in subsets])
        axes[1].scatter(sd_ref_vals, sd_alt, s=12, alpha=0.7)
        axes[1].set_xlabel("SD (A=200)")
        axes[1].set_ylabel("SD (A=50)")
        axes[1].grid(True, alpha=0.3)
        p2 = self._save_pdf(fig, "surrogate_scatter_reference_vs_alt.pdf")
        return [p1, p2]

    def figure_r7_repair_histograms(self) -> str:
        """figures/repair_magnitude_histograms.pdf"""
        import matplotlib.pyplot as plt

        log_files = glob.glob(os.path.join(self.runs_root, "**", "results", "*_repair_log.npz"), recursive=True)
        if not log_files:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.text(0.5, 0.5, "No repair logs found", ha="center", va="center")
            ax.set_axis_off()
            return self._save_pdf(fig, "repair_magnitude_histograms.pdf")

        fig, ax = plt.subplots(figsize=(7, 4))
        for p in sorted(log_files)[:6]:
            try:
                d = np.load(p)
                mag = np.asarray(d["repair_magnitude"], dtype=int)
                if mag.size == 0:
                    continue
                label = os.path.basename(p).replace("_repair_log.npz", "")
                ax.hist(mag, bins=30, alpha=0.35, label=label)
            except Exception:
                continue
        ax.set_xlabel("Repair magnitude (# bit flips)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        return self._save_pdf(fig, "repair_magnitude_histograms.pdf")

    def figure_r7_objective_metric_alignment(self) -> List[str]:
        """figures/objective_metric_alignment_heatmap.pdf + objective_metric_scatter_examples.pdf"""
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        import pandas as pd

        metric_files = glob.glob(os.path.join(self.runs_root, "**", "results", "front_metrics_*.csv"), recursive=True)
        if not metric_files:
            figs = []
            for fname in ["objective_metric_alignment_heatmap.pdf", "objective_metric_scatter_examples.pdf"]:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.text(0.5, 0.5, "No front_metrics_*.csv found", ha="center", va="center")
                ax.set_axis_off()
                figs.append(self._save_pdf(fig, fname))
            return figs

        # Focus on R1_k300 if available
        pick = [p for p in metric_files if "/R1_k300/" in p or os.path.sep + "R1_k300" + os.path.sep in p]
        path = pick[0] if pick else metric_files[0]
        df = pd.read_csv(path)

        obj_cols = [c for c in df.columns if c.startswith("f_")]
        met_cols = [c for c in df.columns if c in ["nystrom_error", "kpca_distortion", "krr_rmse_4G", "krr_rmse_5G", "geo_kl", "geo_l1"]]
        if not obj_cols or not met_cols:
            figs = []
            for fname in ["objective_metric_alignment_heatmap.pdf", "objective_metric_scatter_examples.pdf"]:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.text(0.5, 0.5, "Missing objective/metric columns", ha="center", va="center")
                ax.set_axis_off()
                figs.append(self._save_pdf(fig, fname))
            return figs

        corr = np.zeros((len(obj_cols), len(met_cols)), dtype=float)
        for i, oc in enumerate(obj_cols):
            for j, mc in enumerate(met_cols):
                corr[i, j] = spearmanr(df[oc].values, df[mc].values).correlation

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_yticks(range(len(obj_cols)))
        ax.set_yticklabels(obj_cols)
        ax.set_xticks(range(len(met_cols)))
        ax.set_xticklabels(met_cols, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Spearman correlations (objectives vs raw metrics)")
        p1 = self._save_pdf(fig, "objective_metric_alignment_heatmap.pdf")

        # Scatter examples
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes = np.atleast_1d(axes)
        x = obj_cols[0]
        y = met_cols[0]
        axes[0].scatter(df[x], df[y], s=12, alpha=0.7)
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)
        axes[0].grid(True, alpha=0.3)
        if len(obj_cols) > 1 and len(met_cols) > 1:
            x2 = obj_cols[min(1, len(obj_cols)-1)]
            y2 = met_cols[min(1, len(met_cols)-1)]
            axes[1].scatter(df[x2], df[y2], s=12, alpha=0.7)
            axes[1].set_xlabel(x2)
            axes[1].set_ylabel(y2)
            axes[1].grid(True, alpha=0.3)
        p2 = self._save_pdf(fig, "objective_metric_scatter_examples.pdf")
        return [p1, p2]
