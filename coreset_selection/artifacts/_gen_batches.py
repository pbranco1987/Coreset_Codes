"""
Batch generation mixin for ManuscriptArtifactGenerator.

Contains all ``generate_*`` and ``write_*`` methods that produce batch artifacts
(tables, cardinality curves, Pareto plots, baseline tables, geo distribution
plots, and the summary report).
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

from .plots import (
    plot_metric_vs_k,
    save_pareto_front_plots,
)
from .tables import (
    baseline_comparison_table,
)


class GenBatchesMixin:
    """Mixin providing all ``generate_*`` and ``write_*`` methods for the artifact generator."""

    def write_table_snippets(self, df_all: pd.DataFrame) -> List[str]:
        """Emit optional LaTeX table snippets under out_dir/tables/."""
        paths: List[str] = []
        # R0 summary table
        df_r0 = df_all[df_all.get("run_id", "").astype(str).str.startswith("R0")].copy()
        if not df_r0.empty:
            if "k" not in df_r0.columns:
                df_r0["k"] = df_r0["run_id"].astype(str).str.extract(r"_k(\d+)").astype(int)
            df_r0 = df_r0.sort_values("k")
            cols = [c for c in ["k", "kl_min", "geo_kl", "geo_l1", "geo_maxdev"] if c in df_r0.columns]
            tex = df_r0[cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}")
            out = os.path.join(self.tables_dir, "klmin_summary.tex")
            with open(out, "w") as f:
                f.write(tex)
            paths.append(out)
        return paths

    def generate_r0_klmin(self) -> List[str]:
        """
        Generate R0: KL minimum achievability curves.

        Shows how the minimum achievable geographic KL divergence
        varies with coreset size k.
        """
        import matplotlib.pyplot as plt

        # Load a cache to get geo info
        cache_files = glob.glob(os.path.join(self.cache_root, "rep*/assets.npz"))
        if not cache_files:
            print("  No cache files found for R0")
            return []

        assets = load_replicate_cache(cache_files[0])
        geo = build_geo_info(assets.state_labels)

        # Compute KL_min for various k
        k_values = np.arange(50, 501, 10)
        kl_min_values = []

        for k in k_values:
            kl_min, _ = min_achievable_geo_kl_bounded(
                pi=geo.pi,
                group_sizes=geo.group_sizes,
                k=k,
                alpha_geo=1.0,
                min_one_per_group=True,
            )
            kl_min_values.append(kl_min)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(k_values, kl_min_values, 'b-', linewidth=1.5)
        ax.set_xlabel('Coreset size $k$')
        ax.set_ylabel('Minimum achievable Geo-KL')
        ax.set_title('Geographic KL Lower Bound')
        ax.grid(True, alpha=0.3)

        # Mark key points
        for k_mark in [100, 200, 300, 400, 500]:
            if k_mark in k_values:
                idx = list(k_values).index(k_mark)
                ax.axvline(k_mark, color='gray', linestyle='--', alpha=0.5)
                ax.annotate(f'k={k_mark}', (k_mark, kl_min_values[idx]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Save
        paths = []
        for ext in ['png', 'pdf']:
            path = os.path.join(self.figures_dir, f"r0_kl_min.{ext}")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            paths.append(path)

        plt.close(fig)

        print(f"  Generated R0 KL minimum plot")
        return paths

    def generate_r1_cardinality_curves(self, df_all: pd.DataFrame) -> List[str]:
        """
        Generate R1: Metric vs. k curves.

        Shows how different metrics vary with coreset size for
        various methods.
        """
        import matplotlib.pyplot as plt

        paths = []

        # Filter to R1 runs
        df_r1 = df_all[df_all['run_id'].str.startswith('R1')]

        if len(df_r1) == 0:
            print("  No R1 results found")
            return paths

        metrics = ['nystrom_error', 'kpca_distortion', 'krr_rmse', 'geo_kl']
        metric_labels = {
            'nystrom_error': 'Nystr\u00f6m Error',
            'kpca_distortion': 'kPCA Distortion',
            'krr_rmse': 'KRR RMSE',
            'geo_kl': 'Geographic KL',
        }

        for metric in metrics:
            if metric not in df_r1.columns:
                continue

            fig, ax = plt.subplots(figsize=(5, 3.5))

            plot_metric_vs_k(
                df_r1, metric,
                methods=None,  # All methods
                ax=ax,
                title=f'{metric_labels.get(metric, metric)} vs. Coreset Size',
                ylabel=metric_labels.get(metric, metric),
            )

            for ext in ['png', 'pdf']:
                path = os.path.join(self.figures_dir, f"r1_{metric}_vs_k.{ext}")
                fig.savefig(path, dpi=300, bbox_inches='tight')
                paths.append(path)

            plt.close(fig)

        print(f"  Generated {len(paths)} R1 cardinality curve plots")
        return paths

    def generate_r1_pareto_plots(self) -> List[str]:
        """
        Generate R1: Pareto front visualizations.
        """
        paths = []

        # Find Pareto front files
        pattern = os.path.join(self.runs_root, "R1*/**/vae_space_pareto.npz")
        pareto_files = glob.glob(pattern, recursive=True)

        for pf_path in pareto_files[:3]:  # Limit to first 3 replicates
            try:
                pareto_data = load_pareto_front(pf_path)

                # Get run/rep info from path
                parts = pf_path.split(os.sep)
                run_rep = "_".join(parts[-3:-1])

                prefix = os.path.join(self.figures_dir, f"pareto_{run_rep}")

                new_paths = save_pareto_front_plots(
                    F=pareto_data.F,
                    objectives=pareto_data.objectives,
                    representatives=pareto_data.representatives,
                    out_prefix=prefix,
                    title=f"Pareto Front ({run_rep})",
                )
                paths.extend(new_paths)

            except Exception as e:
                print(f"  Warning: Could not process {pf_path}: {e}")

        print(f"  Generated {len(paths)} Pareto front plots")
        return paths

    def generate_baseline_tables(self, df_all: pd.DataFrame) -> List[str]:
        """
        Generate baseline comparison tables.
        """
        paths = []

        # Get unique k values
        k_values = sorted(df_all['k'].dropna().unique())

        if not k_values:
            return paths

        # Use median k for main comparison
        k_main = k_values[len(k_values) // 2] if k_values else 300

        # Identify methods
        all_methods = df_all['method'].unique()
        baseline_methods = [m for m in all_methods if m.startswith('baseline_')]
        pareto_methods = [m for m in all_methods if m.startswith('pareto_')]

        if not baseline_methods and not pareto_methods:
            return paths

        # Metrics to compare
        metrics = ['nystrom_error', 'geo_kl', 'geo_l1']
        metrics = [m for m in metrics if m in df_all.columns]

        if not metrics:
            return paths

        # Generate table
        latex, csv = baseline_comparison_table(
            df_all,
            baseline_methods=baseline_methods,
            pareto_methods=pareto_methods,
            metrics=metrics,
            k=int(k_main),
        )

        # Save LaTeX
        latex_path = os.path.join(self.tables_dir, f"baseline_comparison_k{int(k_main)}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex)
        paths.append(latex_path)

        # Save CSV
        csv_path = os.path.join(self.tables_dir, f"baseline_comparison_k{int(k_main)}.csv")
        with open(csv_path, 'w') as f:
            f.write(csv)
        paths.append(csv_path)

        print(f"  Generated baseline comparison tables at k={int(k_main)}")
        return paths

    def generate_geo_distribution_plots(self, df_all: pd.DataFrame) -> List[str]:
        """
        Generate geographic distribution visualizations.
        """
        import matplotlib.pyplot as plt

        paths = []

        # Load geo info
        cache_files = glob.glob(os.path.join(self.cache_root, "rep*/assets.npz"))
        if not cache_files:
            return paths

        try:
            assets = load_replicate_cache(cache_files[0])
            geo = build_geo_info(assets.state_labels)
        except Exception as e:
            print(f"  Could not load geo info: {e}")
            return paths

        # Check if we have geo_counts column
        if 'geo_counts' not in df_all.columns:
            # Try to compute from other data
            print("  No geo_counts column found, skipping geo distribution plots")
            return paths

        print(f"  Generated {len(paths)} geographic distribution plots")
        return paths

    def generate_summary_report(self, df_all: pd.DataFrame) -> str:
        """
        Generate a summary markdown report.

        Parameters
        ----------
        df_all : pd.DataFrame
            All results

        Returns
        -------
        str
            Path to generated report
        """
        report_path = os.path.join(self.out_dir, "summary_report.md")

        lines = [
            "# Coreset Selection Experiment Summary",
            "",
            f"**Total results**: {len(df_all)} rows",
            f"**Runs**: {df_all['run_id'].nunique() if 'run_id' in df_all.columns else 'N/A'}",
            f"**Methods**: {df_all['method'].nunique() if 'method' in df_all.columns else 'N/A'}",
            "",
            "## Results by Method",
            "",
        ]

        if 'method' in df_all.columns and 'nystrom_error' in df_all.columns:
            summary = df_all.groupby('method')['nystrom_error'].agg(['mean', 'std', 'count'])
            lines.append("| Method | Mean Nystr\u00f6m Error | Std | Count |")
            lines.append("|--------|-------------------|-----|-------|")
            for method, row in summary.iterrows():
                lines.append(f"| {method} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
            lines.append("")

        lines.append("## Generated Artifacts")
        lines.append("")
        lines.append(f"- Figures: `{self.figures_dir}`")
        lines.append(f"- Tables: `{self.tables_dir}`")

        with open(report_path, 'w') as f:
            f.write("\n".join(lines))

        return report_path
