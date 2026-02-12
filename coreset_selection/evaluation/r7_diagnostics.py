"""
R11 Post-hoc Diagnostics Module -- Re-export facade.

Implements all diagnostic analyses required by manuscript Section VIII.K:
1. Proxy stability analysis (Table IV):
   - RFF dimension sweep for MMD: m in {500, 1000, 2000, 4000}
   - Anchor count sweep for Sinkhorn: A in {50, 100, 200, 400}
   - Cross-representation Spearman rho (VAE-vs-raw, VAE-vs-PCA, PCA-vs-raw)
2. Objective-metric alignment (Fig 4):
   - Spearman rank correlations between optimization objectives
     (f_MMD, f_SD, optionally f_SKL) and raw-space evaluation metrics
     (e_Nys, e_kPCA, RMSE_4G, RMSE_5G, geo_kl, geo_l1)
3. Repair operator activity statistics

Per manuscript Table II (tab:run-matrix), R11 is "Diagnostics: proxy
stability + objective/metric alignment (k=300)".  It loads candidate subsets
from R1 (Pareto front) and R10 (baselines) and produces:
  - proxy_stability.csv   -> Table IV data
  - objective_metric_alignment.csv -> Fig 4 data

Implementation is split across three private sub-modules for
maintainability; all public names are re-exported here so that
existing ``from .r7_diagnostics import ...`` continues to work.
"""

from __future__ import annotations

# -- Objective computation functions --
from ._r7_objectives import (
    SurrogateSensitivityConfig,
    compute_rff_mmd,
    compute_anchored_sinkhorn,
)

# -- Analysis functions --
from ._r7_analysis import (
    surrogate_sensitivity_analysis,
    cross_space_evaluation,
    objective_metric_alignment,
    compute_alignment_heatmap_data,
)

# -- Export, IO, orchestration --
from ._r7_export import (
    RepairDiagnostics,
    aggregate_repair_diagnostics,
    export_proxy_stability_csv,
    export_objective_metric_alignment_csv,
    export_alignment_heatmap_csv,
    load_baseline_indices_from_dir,
    find_r10_results_dir,
    R11Results,
    R6Results,
    run_r7_diagnostics,
)

__all__ = [
    # Objectives
    "SurrogateSensitivityConfig",
    "compute_rff_mmd",
    "compute_anchored_sinkhorn",
    # Analysis
    "surrogate_sensitivity_analysis",
    "cross_space_evaluation",
    "objective_metric_alignment",
    "compute_alignment_heatmap_data",
    # Export / IO / orchestration
    "RepairDiagnostics",
    "aggregate_repair_diagnostics",
    "export_proxy_stability_csv",
    "export_objective_metric_alignment_csv",
    "export_alignment_heatmap_csv",
    "load_baseline_indices_from_dir",
    "find_r10_results_dir",
    "R11Results",
    "R6Results",
    "run_r7_diagnostics",
]
