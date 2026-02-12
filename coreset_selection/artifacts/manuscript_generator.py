"""
Comprehensive manuscript artifact generator.

Produces all figures, tables, and summary documents for the coreset selection paper.
Integrates with experiment results to generate:

Figures (16 total):
- pareto3d_triobjective.pdf - 3D Pareto front for tri-objective R1
- pareto2d_biobjective.pdf - 2D Pareto fronts for bi-objective ablations (R2-R5)
- metrics_vs_k.pdf - Performance metrics vs coreset size
- krr_vs_k.pdf - KRR RMSE breakdown by target (4G vs 5G)
- baseline_vae.pdf - NSGA-II vs baselines in VAE space
- baseline_pca.pdf - NSGA-II vs baselines in PCA space
- baseline_raw.pdf - NSGA-II vs baselines in Raw space
- baseline_nsga_comparison.pdf - Compare NSGA-II across spaces
- objective_ablation.pdf - Objective ablation study
- representation_transfer.pdf - Space comparison (VAE vs PCA vs Raw)
- r7_cross_space.pdf - Cross-space transfer correlations
- objective_metric_alignment.pdf - Objective-metric alignment scatter
- repair_histograms.pdf - Geographic repair activity
- surrogate_sensitivity.pdf - Surrogate approximation sensitivity
- surrogate_scatter.pdf - Surrogate vs true objective scatter

Tables (5 total):
- all_runs_summary.csv
- R1_summary_by_k.csv
- method_comparison.csv
- R6_cross_space_correlations.csv
- R6_objective_metric_alignment.csv

Documents:
- RESULTS_SUMMARY.md
- FIGURE_PREMISES.md
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Re-export everything from the three sub-modules so that
#   from .manuscript_generator import X
# keeps working unchanged.
# ---------------------------------------------------------------------------

from ._mg_data_gen import (  # noqa: F401  -- re-exports
    HAS_MATPLOTLIB,
    METHOD_COLORS,
    SPACE_COLORS,
    METRIC_LABELS,
    set_manuscript_style,
    ensure_dir,
    ExperimentResults,
    load_experiment_results,
    generate_all_runs_summary,
    generate_r1_summary_by_k,
    generate_method_comparison,
    generate_cross_space_correlations,
    generate_objective_metric_alignment,
)

from ._mg_plots import (  # noqa: F401  -- re-exports
    plot_metrics_vs_k,
    plot_krr_vs_k,
    plot_baseline_comparison,
    plot_pareto3d_triobjective,
    plot_pareto2d_biobjective,
    plot_objective_ablation,
    plot_representation_transfer,
    plot_r7_cross_space,
    plot_objective_metric_alignment,
    plot_repair_histograms,
    plot_surrogate_sensitivity,
    plot_surrogate_scatter,
    plot_baseline_nsga_comparison,
)

from ._mg_docs import (  # noqa: F401  -- re-exports
    generate_results_summary_md,
    generate_figure_premises_md,
)


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class ManuscriptArtifactGenerator:
    """
    Generate all manuscript artifacts (figures, tables, documents).
    
    Usage:
        generator = ManuscriptArtifactGenerator(
            runs_root="runs_out",
            out_dir="artifacts",
        )
        generated = generator.generate_all()
    """
    
    def __init__(
        self,
        runs_root: str = "runs_out",
        cache_root: str = "replicate_cache",
        out_dir: str = "artifacts",
        rep_folder: str = "rep00",
    ):
        self.runs_root = runs_root
        self.cache_root = cache_root
        self.out_dir = ensure_dir(out_dir)
        self.rep_folder = rep_folder
        
        self.figures_dir = ensure_dir(os.path.join(out_dir, "figures"))
        self.tables_dir = ensure_dir(os.path.join(out_dir, "tables"))
        
        self.results: Optional[ExperimentResults] = None
    
    def load_results(self) -> ExperimentResults:
        """Load experiment results."""
        if self.results is None:
            self.results = load_experiment_results(self.runs_root, self.rep_folder)
        return self.results
    
    def generate_all(self) -> Dict[str, List[str]]:
        """
        Generate all manuscript artifacts.
        
        Returns
        -------
        Dict[str, List[str]]
            Mapping from artifact type to list of generated file paths
        """
        if HAS_MATPLOTLIB:
            set_manuscript_style()
        
        generated = {
            "figures": [],
            "tables": [],
            "documents": [],
        }
        
        results = self.load_results()
        
        # Generate figures
        figure_generators = [
            ("metrics_vs_k.pdf", lambda: plot_metrics_vs_k(
                results, os.path.join(self.figures_dir, "metrics_vs_k.pdf"))),
            ("krr_vs_k.pdf", lambda: plot_krr_vs_k(
                results, os.path.join(self.figures_dir, "krr_vs_k.pdf"))),
            ("baseline_vae.pdf", lambda: plot_baseline_comparison(
                results, 'vae', os.path.join(self.figures_dir, "baseline_vae.pdf"))),
            ("baseline_pca.pdf", lambda: plot_baseline_comparison(
                results, 'pca', os.path.join(self.figures_dir, "baseline_pca.pdf"))),
            ("baseline_raw.pdf", lambda: plot_baseline_comparison(
                results, 'raw', os.path.join(self.figures_dir, "baseline_raw.pdf"))),
            ("baseline_nsga_comparison.pdf", lambda: plot_baseline_nsga_comparison(
                results, os.path.join(self.figures_dir, "baseline_nsga_comparison.pdf"))),
            ("objective_ablation.pdf", lambda: plot_objective_ablation(
                results, os.path.join(self.figures_dir, "objective_ablation.pdf"))),
            ("representation_transfer.pdf", lambda: plot_representation_transfer(
                results, os.path.join(self.figures_dir, "representation_transfer.pdf"))),
            ("pareto3d_triobjective.pdf", lambda: plot_pareto3d_triobjective(
                results, os.path.join(self.figures_dir, "pareto3d_triobjective.pdf"))),
            ("pareto2d_biobjective.pdf", lambda: plot_pareto2d_biobjective(
                results, os.path.join(self.figures_dir, "pareto2d_biobjective.pdf"))),
            ("r7_cross_space.pdf", lambda: plot_r7_cross_space(
                results, os.path.join(self.figures_dir, "r7_cross_space.pdf"))),
            ("objective_metric_alignment.pdf", lambda: plot_objective_metric_alignment(
                results, os.path.join(self.figures_dir, "objective_metric_alignment.pdf"))),
            ("repair_histograms.pdf", lambda: plot_repair_histograms(
                results, os.path.join(self.figures_dir, "repair_histograms.pdf"))),
            ("surrogate_sensitivity.pdf", lambda: plot_surrogate_sensitivity(
                results, os.path.join(self.figures_dir, "surrogate_sensitivity.pdf"))),
            ("surrogate_scatter.pdf", lambda: plot_surrogate_scatter(
                results, os.path.join(self.figures_dir, "surrogate_scatter.pdf"))),
        ]
        
        for name, generator in figure_generators:
            try:
                path = generator()
                if path:
                    generated["figures"].append(path)
                    print(f"  Generated: {name}")
            except Exception as e:
                print(f"  WARNING: {name} failed: {e}")
        
        # Generate tables
        table_generators = [
            ("all_runs_summary.csv", generate_all_runs_summary),
            ("R1_summary_by_k.csv", generate_r1_summary_by_k),
            ("method_comparison.csv", generate_method_comparison),
            ("R6_cross_space_correlations.csv", generate_cross_space_correlations),
            ("R6_objective_metric_alignment.csv", generate_objective_metric_alignment),
        ]
        
        for name, generator in table_generators:
            try:
                df = generator(results)
                if not df.empty:
                    path = os.path.join(self.tables_dir, name)
                    df.to_csv(path, index=False)
                    generated["tables"].append(path)
                    print(f"  Generated: {name}")
            except Exception as e:
                print(f"  WARNING: {name} failed: {e}")
        
        # Generate markdown documents
        try:
            summary_path = os.path.join(self.out_dir, "RESULTS_SUMMARY.md")
            with open(summary_path, 'w') as f:
                f.write(generate_results_summary_md(results))
            generated["documents"].append(summary_path)
            print("  Generated: RESULTS_SUMMARY.md")
        except Exception as e:
            print(f"  WARNING: RESULTS_SUMMARY.md failed: {e}")
        
        try:
            premises_path = os.path.join(self.out_dir, "FIGURE_PREMISES.md")
            with open(premises_path, 'w') as f:
                f.write(generate_figure_premises_md())
            generated["documents"].append(premises_path)
            print("  Generated: FIGURE_PREMISES.md")
        except Exception as e:
            print(f"  WARNING: FIGURE_PREMISES.md failed: {e}")
        
        return generated


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line entry point for artifact generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate manuscript artifacts")
    parser.add_argument("--runs-root", default="runs_out", help="Runs output directory")
    parser.add_argument("--cache-root", default="replicate_cache", help="Cache directory")
    parser.add_argument("--out-dir", default="artifacts", help="Output directory")
    parser.add_argument("--rep-folder", default="rep00", help="Replicate folder name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MANUSCRIPT ARTIFACT GENERATOR")
    print("=" * 60)
    
    generator = ManuscriptArtifactGenerator(
        runs_root=args.runs_root,
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        rep_folder=args.rep_folder,
    )
    
    generated = generator.generate_all()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Figures: {len(generated['figures'])}")
    print(f"Tables: {len(generated['tables'])}")
    print(f"Documents: {len(generated['documents'])}")
    print(f"\nOutput directory: {args.out_dir}")


if __name__ == "__main__":
    main()
