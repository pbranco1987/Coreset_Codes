"""
Evaluation module for coreset quality assessment.

This module provides metrics for evaluating coreset quality:
- Raw space metrics: Nyström error, kernel PCA, KRR
- Geographic metrics: KL divergence, quota satisfaction
- Coverage and diversity metrics
- R11 diagnostics: Proxy stability, cross-space evaluation,
  objective–metric alignment (manuscript Section VIII.K)
"""

from .raw_space import RawSpaceEvaluator

from .geo_diagnostics import (
    geo_diagnostics,
    geo_diagnostics_weighted,
    dual_geo_diagnostics,
    compute_quota_satisfaction,
    state_coverage_report,
    geographic_entropy,
    geographic_concentration_index,
)

from .metrics import (
    coverage_stats,
    k_center_cost,
    k_median_cost,
    diversity_score,
    min_pairwise_distance,
    representation_error,
    all_metrics,
)

from .r7_diagnostics import (
    SurrogateSensitivityConfig,
    surrogate_sensitivity_analysis,
    cross_space_evaluation,
    objective_metric_alignment,
    compute_alignment_heatmap_data,
    RepairDiagnostics,
    aggregate_repair_diagnostics,
    R11Results,
    R6Results,  # backward-compatible alias
    run_r7_diagnostics,
    # Phase 7 CSV export functions
    export_proxy_stability_csv,
    export_objective_metric_alignment_csv,
    export_alignment_heatmap_csv,
    # Phase 7 baseline loading helpers
    load_baseline_indices_from_dir,
    find_r10_results_dir,
)

from .kpi_stability import (
    state_kpi_stability,
    state_krr_stability,
    per_state_kpi_drift_matrix,
    export_state_kpi_drift_csv,
)

from .method_comparison import (
    load_result_rows,
    group_by_method,
    mean_per_method,
    effect_isolation_table,
    rank_table,
    pairwise_dominance_matrix,
    stability_summary,
    build_comparison_report,
    DOWNSTREAM_LOWER,
    DOWNSTREAM_HIGHER,
)

from .classification_metrics import (
    infer_target_type,
    accuracy,
    cohens_kappa,
    macro_precision,
    macro_recall,
    macro_f1,
    weighted_f1,
    confusion_matrix_dict,
    full_classification_evaluation,
    multitarget_classification_evaluation,
)

from .qos_tasks import (
    QoSConfig,
    Demeaner,
    build_lagged_features,
    qos_coreset_evaluation,
    qos_fullset_reference,
    qos_summary_table,
)

__all__ = [
    # Raw space evaluation
    "RawSpaceEvaluator",
    # Geographic diagnostics
    "geo_diagnostics",
    "geo_diagnostics_weighted",
    "dual_geo_diagnostics",
    "compute_quota_satisfaction",
    "state_coverage_report",
    "geographic_entropy",
    "geographic_concentration_index",
    # Coverage and diversity metrics
    "coverage_stats",
    "k_center_cost",
    "k_median_cost",
    "diversity_score",
    "min_pairwise_distance",
    "representation_error",
    "all_metrics",
    # R11 diagnostics (manuscript Section VIII.K)
    "SurrogateSensitivityConfig",
    "surrogate_sensitivity_analysis",
    "cross_space_evaluation",
    "objective_metric_alignment",
    "compute_alignment_heatmap_data",
    "RepairDiagnostics",
    "aggregate_repair_diagnostics",
    "R11Results",
    "R6Results",
    "run_r7_diagnostics",
    "export_proxy_stability_csv",
    "export_objective_metric_alignment_csv",
    "export_alignment_heatmap_csv",
    "load_baseline_indices_from_dir",
    "find_r10_results_dir",
    # State-conditioned KPI stability (manuscript Section VII)
    "state_kpi_stability",
    "state_krr_stability",
    # G8: Per-state KPI drift matrix and CSV export
    "per_state_kpi_drift_matrix",
    "export_state_kpi_drift_csv",
    # Cross-method comparison (kernel k-means analysis protocol)
    "load_result_rows",
    "group_by_method",
    "mean_per_method",
    "effect_isolation_table",
    "rank_table",
    "pairwise_dominance_matrix",
    "stability_summary",
    "build_comparison_report",
    "DOWNSTREAM_LOWER",
    "DOWNSTREAM_HIGHER",
    # Phase 2: Classification metrics for categorical targets
    "infer_target_type",
    "accuracy",
    "cohens_kappa",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "confusion_matrix_dict",
    "full_classification_evaluation",
    "multitarget_classification_evaluation",
    # QoS downstream tasks (ISG / IQS composite-index evaluation)
    "QoSConfig",
    "Demeaner",
    "build_lagged_features",
    "qos_coreset_evaluation",
    "qos_fullset_reference",
    "qos_summary_table",
]
