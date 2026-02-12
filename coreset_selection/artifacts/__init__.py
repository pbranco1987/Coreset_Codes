"""
Artifact generation module for manuscript figures and tables.

This module provides:
- ManuscriptArtifactGenerator: Main artifact generation orchestration
- ComprehensiveArtifactGenerator: Full manuscript figure/table generation
- Plot functions for Pareto fronts, metrics, distributions
- Table generators for LaTeX and CSV output
"""

from .generator import ManuscriptArtifactGenerator

from .manuscript_artifacts import ManuscriptArtifacts

from .manuscript_generator import (
    ManuscriptArtifactGenerator as ComprehensiveArtifactGenerator,
    ExperimentResults,
    load_experiment_results,
    set_manuscript_style,
    # Figure generators
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
    # Table generators
    generate_all_runs_summary,
    generate_r1_summary_by_k,
    generate_method_comparison,
    generate_cross_space_correlations,
    generate_objective_metric_alignment,
    # Document generators
    generate_results_summary_md,
    generate_figure_premises_md,
)

from .plots import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_metric_vs_k,
    plot_geographic_distribution,
    save_pareto_front_plots,
    plot_convergence,
)

from .tables import (
    format_number,
    format_mean_std,
    generate_latex_table,
    generate_csv_table,
    results_summary_table,
    baseline_comparison_table,
    run_specs_table,
    geographic_distribution_table,
    # New tables for manuscript compliance
    klmin_summary_table,
    front_stats_table,
    cardinality_metrics_table,
    objective_ablations_table,
    representation_transfer_table,
    surrogate_sensitivity_table,
    repair_activity_table,
    crossspace_objectives_table,
    baseline_quota_table,
    baseline_unconstrained_table,
    objective_metric_alignment_table,
)

__all__ = [
    # Generators
    "ManuscriptArtifactGenerator",
    "ManuscriptArtifacts",
    "ComprehensiveArtifactGenerator",
    "ExperimentResults",
    "load_experiment_results",
    "set_manuscript_style",
    # Comprehensive figure generators
    "plot_metrics_vs_k",
    "plot_krr_vs_k",
    "plot_baseline_comparison",
    "plot_pareto3d_triobjective",
    "plot_pareto2d_biobjective",
    "plot_objective_ablation",
    "plot_representation_transfer",
    "plot_r7_cross_space",
    "plot_objective_metric_alignment",
    "plot_repair_histograms",
    "plot_surrogate_sensitivity",
    "plot_surrogate_scatter",
    "plot_baseline_nsga_comparison",
    # Comprehensive table generators
    "generate_all_runs_summary",
    "generate_r1_summary_by_k",
    "generate_method_comparison",
    "generate_cross_space_correlations",
    "generate_objective_metric_alignment",
    # Document generators
    "generate_results_summary_md",
    "generate_figure_premises_md",
    # Legacy plots
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_metric_vs_k",
    "plot_geographic_distribution",
    "save_pareto_front_plots",
    "plot_convergence",
    # Legacy tables
    "format_number",
    "format_mean_std",
    "generate_latex_table",
    "generate_csv_table",
    "results_summary_table",
    "baseline_comparison_table",
    "run_specs_table",
    "geographic_distribution_table",
    # New tables for manuscript compliance
    "klmin_summary_table",
    "front_stats_table",
    "cardinality_metrics_table",
    "objective_ablations_table",
    "representation_transfer_table",
    "surrogate_sensitivity_table",
    "repair_activity_table",
    "crossspace_objectives_table",
    "baseline_quota_table",
    "baseline_unconstrained_table",
    "objective_metric_alignment_table",
]
