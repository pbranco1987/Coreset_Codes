"""Table generation utilities (re-export facade)."""

from ._tables_core import (
    format_number,
    format_mean_std,
    generate_latex_table,
    generate_csv_table,
    bold_best_in_column,
    generate_wrapped_latex_table,
    generate_sectioned_latex_table,
)
from ._tables_legacy import (
    results_summary_table,
    baseline_comparison_table,
    run_specs_table,
    geographic_distribution_table,
)
from ._tables_manuscript import (
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
    # Core formatting
    "format_number",
    "format_mean_std",
    "generate_latex_table",
    "generate_csv_table",
    "bold_best_in_column",
    "generate_wrapped_latex_table",
    "generate_sectioned_latex_table",
    # Legacy tables
    "results_summary_table",
    "baseline_comparison_table",
    "run_specs_table",
    "geographic_distribution_table",
    # Manuscript tables
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
