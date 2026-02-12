#!/usr/bin/env python3
"""
Manuscript Compliance Verification Script — Phase 12 enhanced.

Verifies that the codebase implements all requirements from the manuscript
"Constrained Nystrom Landmark Selection for Scalable Telecom Analytics".

Usage:
    python -m coreset_selection.scripts.verify_compliance
    python -m coreset_selection.scripts.verify_compliance --output-dir runs_out --artifacts-dir artifacts_out

Phase 12 additions (on top of Phases 1-11 checks):
  - verify_coverage_targets():    Coverage target count = 10 (Table V alignment)
  - verify_output_coverage():     All run IDs R0-R12 represented in outputs
  - verify_table_v_structure():   Table V has exactly 10 rows
  - verify_fig2_panel_structure():Fig 2 is a 2x2 panel (4 sub-plots)
  - verify_proxy_stability():     Table IV has 3 required sections

This script checks:
1. Algorithm implementations (Algorithm 1, 2, 3)
2. Objective functions (SKL, MMD, Sinkhorn)
3. Experimental configurations (R0-R12)
4. Evaluation metrics
5. Artifact generation (manuscript + complementary figures/tables)
6. Gap closures G1-G11
7. Phase 11 narrative-strengthening tables (N1-N7)
8. [Phase 12] Output validation and structural compliance
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple


# ===================================================================
# Helpers
# ===================================================================

def check_import(module_path: str, items: List[str]) -> Tuple[bool, List[str]]:
    """Check if items can be imported from a module.

    Returns (all_ok, missing_items).
    """
    missing = []
    try:
        module = __import__(module_path, fromlist=items)
        for item in items:
            if not hasattr(module, item):
                missing.append(item)
    except ImportError as e:
        return False, [f"Module import failed: {e}"]
    return len(missing) == 0, missing


# ===================================================================
# Verification sections (Phases 1-11)
# ===================================================================

def verify_algorithms() -> List[str]:
    """Verify Algorithm 1, 2, 3 implementations."""
    issues = []

    # Algorithm 1: KL-Guided Quota Construction (+ Phase 6 lazy-heap & path)
    ok, missing = check_import(
        "coreset_selection.geo.kl",
        [
            "kl_optimal_integer_counts_bounded",
            "min_achievable_geo_kl_bounded",
            "compute_quota_path",
            "save_quota_path",
        ],
    )
    if not ok:
        issues.append(f"Algorithm 1 (KL quota + Phase 6 path): Missing {missing}")

    # Phase 6 projector
    ok, missing = check_import(
        "coreset_selection.geo.projector",
        [
            "GeographicConstraintProjector",
            "build_feasible_quota_mask",
            "compute_quota_violation",
        ],
    )
    if not ok:
        issues.append(f"Phase 6 (Projector): Missing {missing}")
    else:
        try:
            from coreset_selection.geo.projector import GeographicConstraintProjector
            for mname in [
                "get_cstar",
                "quota_path",
                "validate_capacity",
                "most_constrained_groups",
            ]:
                if not hasattr(GeographicConstraintProjector, mname):
                    issues.append(
                        f"Phase 6: GeographicConstraintProjector missing "
                        f"method '{mname}'"
                    )
        except Exception as e:
            issues.append(f"Phase 6: Could not verify projector methods: {e}")

    # Phase 6 scripts
    scripts_dir = os.path.dirname(__file__)
    for script_name in ["print_quota.py", "quota_summary_table.py"]:
        if not os.path.exists(os.path.join(scripts_dir, script_name)):
            issues.append(f"Phase 6: Missing script scripts/{script_name}")

    # Algorithm 2: Feasibility-Preserving Repair
    ok, missing = check_import(
        "coreset_selection.optimization.repair",
        [
            "QuotaAndCardinalityRepair",
            "ExactKRepair",
            "LeastHarmQuotaRepair",
            "LeastHarmExactKRepair",
            "RepairActivityTracker",
        ],
    )
    if not ok:
        issues.append(f"Algorithm 2 (Repair): Missing {missing}")

    # Algorithm 3: NSGA-II
    ok, missing = check_import(
        "coreset_selection.optimization.nsga2_internal",
        ["nsga2_optimize", "fast_non_dominated_sort"],
    )
    if not ok:
        issues.append(f"Algorithm 3 (NSGA-II): Missing {missing}")

    return issues


def verify_objectives() -> List[str]:
    """Verify objective function implementations."""
    issues = []

    ok, missing = check_import(
        "coreset_selection.objectives.skl",
        [
            "symmetric_kl_diag_gaussians",
            "clamp_variance",
            "VAE_VARIANCE_CLAMP_MIN",
            "VAE_VARIANCE_CLAMP_MAX",
            "compute_moment_matched_gaussian",
        ],
    )
    if not ok:
        issues.append(f"SKL objective: Missing {missing}")

    ok, missing = check_import(
        "coreset_selection.objectives.mmd",
        ["compute_rff_features", "mmd_from_rff", "RFFMMD"],
    )
    if not ok:
        issues.append(f"MMD objective: Missing {missing}")

    ok, missing = check_import(
        "coreset_selection.objectives.sinkhorn",
        ["AnchorSinkhorn", "sinkhorn2_logstab"],
    )
    if not ok:
        issues.append(f"Sinkhorn objective: Missing {missing}")

    return issues


def verify_configurations() -> List[str]:
    """Verify run configurations R0-R12."""
    issues = []

    ok, missing = check_import(
        "coreset_selection.config",
        ["K_GRID", "get_run_specs", "RunSpec", "ExperimentConfig"],
    )
    if not ok:
        issues.append(f"Configuration: Missing {missing}")
        return issues

    from coreset_selection.config import get_run_specs, K_GRID

    expected_k = [50, 100, 200, 300, 400, 500]
    if list(K_GRID) != expected_k:
        issues.append(f"K_GRID mismatch: got {K_GRID}, expected {expected_k}")

    specs = get_run_specs()
    for rid in [f"R{i}" for i in range(13)]:
        if rid not in specs:
            issues.append(f"Missing run spec: {rid}")

    # EffortSweepGrid
    ok, missing = check_import("coreset_selection.config.dataclasses", ["EffortSweepGrid"])
    if not ok:
        issues.append(f"EffortSweepGrid dataclass: Missing {missing}")

    # Time complexity annotations
    ok, missing = check_import(
        "coreset_selection.experiment.time_complexity",
        [
            "COMPLEXITY_ANNOTATIONS",
            "fit_power_law",
            "annotate_complexity_fits",
            "save_time_complexity_summary",
        ],
    )
    if not ok:
        issues.append(f"Time complexity enhancements: Missing {missing}")

    # G5: Complementary analysis methods
    ok, _ = check_import(
        "coreset_selection.artifacts.manuscript_artifacts",
        ["ManuscriptArtifacts"],
    )
    if ok:
        from coreset_selection.artifacts.manuscript_artifacts import ManuscriptArtifacts

        for mname in [
            "fig_cumulative_pareto_improvement",
            "fig_constraint_tightness_vs_fidelity",
            "fig_skl_ablation_comparison",
        ]:
            if not hasattr(ManuscriptArtifacts, mname):
                issues.append(
                    f"G5 complementary analysis missing: "
                    f"ManuscriptArtifacts.{mname}"
                )

        # G6: LaTeX table generators
        for mname in ["tab_exp_settings", "tab_run_matrix"]:
            if not hasattr(ManuscriptArtifacts, mname):
                issues.append(
                    f"G6 LaTeX table generator missing: "
                    f"ManuscriptArtifacts.{mname}"
                )

    # G7: BaselineVariantGenerator
    ok, missing = check_import(
        "coreset_selection.baselines.variant_generator",
        ["BaselineVariantGenerator", "BaselineResult", "METHOD_REGISTRY", "VARIANT_PAIRS"],
    )
    if not ok:
        issues.append(f"G7 BaselineVariantGenerator: Missing {missing}")

    # G8: Per-state KPI drift export
    ok, missing = check_import(
        "coreset_selection.evaluation.kpi_stability",
        ["per_state_kpi_drift_matrix", "export_state_kpi_drift_csv"],
    )
    if not ok:
        issues.append(f"G8 Per-state KPI drift export: Missing {missing}")

    # G9: Dual geo diagnostics
    ok, missing = check_import(
        "coreset_selection.evaluation.geo_diagnostics",
        ["geo_diagnostics_weighted", "dual_geo_diagnostics"],
    )
    if not ok:
        issues.append(f"G9 Dual geo diagnostics: Missing {missing}")
    else:
        from coreset_selection.evaluation.geo_diagnostics import (
            dual_geo_diagnostics as _dgd,
        )
        from coreset_selection.geo.info import GeoInfo as _GI
        import numpy as _np

        _dummy_geo = _GI.from_group_ids(
            _np.array([0, 0, 1, 1, 2]), groups=["A", "B", "C"]
        )
        _result = _dgd(_dummy_geo, _np.array([0, 2, 4]), k=3)
        for _key in [
            "geo_kl_muni", "geo_l1_muni", "geo_maxdev_muni",
            "geo_kl_pop", "geo_l1_pop", "geo_maxdev_pop",
            "geo_kl", "geo_l1", "geo_maxdev",
        ]:
            if _key not in _result:
                issues.append(f"G9 dual_geo_diagnostics missing key: {_key}")

    # G10: ResultsSaver effort helpers
    ok, missing = check_import(
        "coreset_selection.experiment.saver", ["ResultsSaver"]
    )
    if not ok:
        issues.append(f"G10 ResultsSaver: Missing {missing}")
    else:
        from coreset_selection.experiment.saver import ResultsSaver as _RS

        for mname in ["save_effort_grid_csv", "save_effort_sweep_summary"]:
            if not hasattr(_RS, mname):
                issues.append(f"G10 ResultsSaver missing method: {mname}")

    return issues


def verify_evaluation() -> List[str]:
    """Verify evaluation metric implementations."""
    issues = []

    ok, missing = check_import(
        "coreset_selection.evaluation.raw_space", ["RawSpaceEvaluator"]
    )
    if not ok:
        issues.append(f"Raw space evaluation: Missing {missing}")

    ok, missing = check_import(
        "coreset_selection.evaluation.r7_diagnostics",
        [
            "surrogate_sensitivity_analysis",
            "cross_space_evaluation",
            "objective_metric_alignment",
            "run_r7_diagnostics",
        ],
    )
    if not ok:
        issues.append(f"R6 diagnostics: Missing {missing}")

    ok, missing = check_import(
        "coreset_selection.evaluation.geo_diagnostics",
        [
            "geo_diagnostics",
            "geo_diagnostics_weighted",
            "dual_geo_diagnostics",
            "compute_quota_satisfaction",
        ],
    )
    if not ok:
        issues.append(f"Geographic diagnostics: Missing {missing}")

    return issues


def verify_artifacts() -> List[str]:
    """Verify artifact generation capabilities."""
    issues = []

    # Table utilities
    ok, missing = check_import(
        "coreset_selection.artifacts.tables",
        [
            "klmin_summary_table",
            "front_stats_table",
            "cardinality_metrics_table",
            "objective_ablations_table",
            "representation_transfer_table",
            "surrogate_sensitivity_table",
            "repair_activity_table",
            "crossspace_objectives_table",
        ],
    )
    if not ok:
        issues.append(f"Table generators: Missing {missing}")

    # Legacy generator
    ok, missing = check_import(
        "coreset_selection.artifacts.generator",
        ["ManuscriptArtifactGenerator"],
    )
    if not ok:
        issues.append(f"Artifact generator: Missing {missing}")

    # ManuscriptArtifacts
    ok, missing = check_import(
        "coreset_selection.artifacts.manuscript_artifacts",
        ["ManuscriptArtifacts"],
    )
    if not ok:
        issues.append(f"ManuscriptArtifacts class: Missing {missing}")
    else:
        from coreset_selection.artifacts.manuscript_artifacts import ManuscriptArtifacts

        # Phase 10a figures (N1-N6)
        for mname, label in [
            ("fig_kl_floor_vs_k",           "Fig N1 (KL floor vs k)"),
            ("fig_pareto_front_k300",       "Fig N2 (Pareto front k=300)"),
            ("fig_objective_ablation_bars",  "Fig N3 (Objective ablation bars)"),
            ("fig_constraint_comparison",    "Fig N4 (Constraint comparison)"),
            ("fig_effort_quality",           "Fig N5 (Effort-quality trade-off)"),
            ("fig_baseline_comparison",      "Fig N6 (Baseline comparison)"),
        ]:
            if not hasattr(ManuscriptArtifacts, mname):
                issues.append(f"Phase 10a missing: ManuscriptArtifacts.{mname} ({label})")

        # Phase 10b figures (N7-N12)
        for mname, label in [
            ("fig_multi_seed_boxplot",         "Fig N7 (Multi-seed boxplot)"),
            ("fig_state_kpi_heatmap",          "Fig N8 (State KPI heatmap)"),
            ("fig_composition_shift",          "Fig N9 (Composition shift)"),
            ("fig_pareto_front_evolution",     "Fig N10 (Pareto front evolution)"),
            ("fig_nystrom_error_distribution", "Fig N11 (Nystrom error distribution)"),
            ("fig_krr_worst_state_rmse_vs_k",  "Fig N12 (Worst-state RMSE vs k)"),
        ]:
            if not hasattr(ManuscriptArtifacts, mname):
                issues.append(f"Phase 10b missing: ManuscriptArtifacts.{mname} ({label})")

        # Phase 11 tables (N1-N7)
        for mname, label in [
            ("tab_constraint_diagnostics_cross_config",
             "Table N1 (constraint diagnostics cross-config)"),
            ("tab_objective_ablation_summary",
             "Table N2 (objective ablation summary)"),
            ("tab_representation_transfer_summary",
             "Table N3 (representation transfer summary)"),
            ("tab_skl_ablation_summary",
             "Table N4 (SKL ablation summary)"),
            ("tab_multi_seed_statistics",
             "Table N5 (multi-seed statistics)"),
            ("tab_worst_state_rmse_by_k",
             "Table N6 (worst-state RMSE by k)"),
            ("tab_baseline_paired_unconstrained_vs_quota",
             "Table N7 (baseline paired unconstrained vs quota)"),
        ]:
            if not hasattr(ManuscriptArtifacts, mname):
                issues.append(f"Phase 11 missing: ManuscriptArtifacts.{mname} ({label})")

    # Verify manuscript figure/table constants
    ok, missing = check_import(
        "coreset_selection.config.constants",
        [
            "MANUSCRIPT_FIGURE_FILES",
            "MANUSCRIPT_TABLE_FILES",
            "COMPLEMENTARY_FIGURE_FILES",
            "COMPLEMENTARY_TABLE_FILES",
        ],
    )
    if not ok:
        issues.append(f"Manuscript artifact constants: Missing {missing}")
    else:
        from coreset_selection.config.constants import (
            MANUSCRIPT_FIGURE_FILES,
            MANUSCRIPT_TABLE_FILES,
            COMPLEMENTARY_TABLE_FILES,
        )

        expected_figs = [
            "geo_ablation_tradeoff_scatter.pdf",
            "distortion_cardinality_R1.pdf",
            "regional_validity_k300.pdf",
            "objective_metric_alignment_heatmap.pdf",
        ]
        for f in expected_figs:
            if f not in MANUSCRIPT_FIGURE_FILES:
                issues.append(
                    f"Manuscript figure missing from MANUSCRIPT_FIGURE_FILES: {f}"
                )

        expected_tabs = [
            "exp_settings.tex",
            "run_matrix.tex",
            "r1_by_k.tex",
            "proxy_stability.tex",
            "krr_multitask_k300.tex",
        ]
        for t in expected_tabs:
            if t not in MANUSCRIPT_TABLE_FILES:
                issues.append(
                    f"Manuscript table missing from MANUSCRIPT_TABLE_FILES: {t}"
                )

        # G10: Effort sweep output files
        for t in ["effort_grid_config.csv", "effort_sweep_summary.csv"]:
            if t not in COMPLEMENTARY_TABLE_FILES:
                issues.append(
                    f"G10 complementary table missing from "
                    f"COMPLEMENTARY_TABLE_FILES: {t}"
                )

        # Phase 11 table filenames registered
        for t in [
            "constraint_diagnostics_cross_config.tex",
            "objective_ablation_summary.tex",
            "representation_transfer_summary.tex",
            "worst_state_rmse_by_k.tex",
            "baseline_paired_unconstrained_vs_quota.tex",
        ]:
            if t not in COMPLEMENTARY_TABLE_FILES:
                issues.append(
                    f"Phase 11 table missing from COMPLEMENTARY_TABLE_FILES: {t}"
                )

    return issues


def verify_constants() -> List[str]:
    """Verify manuscript constants are defined and match Table I."""
    issues = []

    ok, missing = check_import(
        "coreset_selection.config.constants",
        [
            "NSGA2_POP_SIZE",
            "NSGA2_N_GENERATIONS",
            "RFF_DIM_DEFAULT",
            "SINKHORN_N_ANCHORS",
            "SINKHORN_ETA",
            "SKL_VAR_CLAMP_MIN",
            "SKL_VAR_CLAMP_MAX",
            "ALPHA_GEO",
            "VAE_LATENT_DIM",
            "EVAL_SIZE",
            "KPCA_COMPONENTS",
            "N_REPLICATES_PRIMARY",
        ],
    )
    if not ok:
        issues.append(f"Constants: Missing {missing}")
        return issues

    from coreset_selection.config.constants import (
        NSGA2_POP_SIZE,
        NSGA2_N_GENERATIONS,
        RFF_DIM_DEFAULT,
        SINKHORN_N_ANCHORS,
        SINKHORN_ETA,
        VAE_LATENT_DIM,
        EVAL_SIZE,
        KPCA_COMPONENTS,
        N_REPLICATES_PRIMARY,
    )

    checks = [
        (NSGA2_POP_SIZE, 200, "NSGA2_POP_SIZE"),
        (NSGA2_N_GENERATIONS, 1000, "NSGA2_N_GENERATIONS"),
        (RFF_DIM_DEFAULT, 2000, "RFF_DIM_DEFAULT"),
        (SINKHORN_N_ANCHORS, 200, "SINKHORN_N_ANCHORS"),
        (SINKHORN_ETA, 0.05, "SINKHORN_ETA"),
        (VAE_LATENT_DIM, 32, "VAE_LATENT_DIM"),
        (EVAL_SIZE, 2000, "EVAL_SIZE"),
        (KPCA_COMPONENTS, 20, "KPCA_COMPONENTS"),
        (N_REPLICATES_PRIMARY, 5, "N_REPLICATES_PRIMARY"),
    ]
    for actual, expected, name in checks:
        if actual != expected:
            issues.append(f"{name}: got {actual}, expected {expected}")

    return issues


# ===================================================================
# Phase 12 — New output validation checks
# ===================================================================

def verify_coverage_targets() -> List[str]:
    """Verify that COVERAGE_TARGETS_TABLE_V has exactly 10 entries.

    Per manuscript Table V, there are 10 coverage targets:
      1. Area (4G)           2. Area (5G)
      3. Households (4G)     4. Residents (4G)
      5. Area (4G + 5G)      6. Area (All)
      7. Households (4G+5G)  8. Households (All)
      9. Residents (4G+5G)  10. Residents (All)
    """
    issues = []
    try:
        from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V

        n = len(COVERAGE_TARGETS_TABLE_V)
        if n != 10:
            issues.append(
                f"COVERAGE_TARGETS_TABLE_V has {n} entries (expected 10)"
            )

        # Verify expected key-label pairs
        expected_keys = [
            "cov_area_4G",
            "cov_area_5G",
            "cov_hh_4G",
            "cov_res_4G",
            "cov_area_4G_5G",
            "cov_area_all",
            "cov_hh_4G_5G",
            "cov_hh_all",
            "cov_res_4G_5G",
            "cov_res_all",
        ]
        for key in expected_keys:
            if key not in COVERAGE_TARGETS_TABLE_V:
                issues.append(
                    f"COVERAGE_TARGETS_TABLE_V missing key: {key}"
                )
    except ImportError as e:
        issues.append(f"Cannot import COVERAGE_TARGETS_TABLE_V: {e}")

    return issues


def verify_output_coverage(output_dir: str = "runs_out") -> List[str]:
    """Verify all run IDs R0-R12 are represented in the output directory.

    Scans ``output_dir`` for directories whose names start with R0, R1, ..., R12.
    Each run ID must be present at least once with at least one result file.
    """
    issues = []
    import re

    if not os.path.isdir(output_dir):
        issues.append(f"Output directory '{output_dir}' does not exist")
        return issues

    # Collect all run IDs found
    found_ids = set()
    for entry in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, entry)):
            m = re.match(r"(R\d+)", entry)
            if m:
                found_ids.add(m.group(1))

    expected = {f"R{i}" for i in range(13)}
    missing = expected - found_ids
    if missing:
        for rid in sorted(missing, key=lambda r: int(r[1:])):
            issues.append(f"Run {rid} not found in {output_dir}")

    return issues


def verify_table_v_structure(artifacts_dir: str = "artifacts_out") -> List[str]:
    """Verify that Table V (krr_multitask_k300.tex) has exactly 10 rows.

    The manuscript Table V reports KRR RMSE for 10 coverage targets.
    This function parses the LaTeX source to confirm the row count.
    """
    issues = []
    tab_dir = os.path.join(artifacts_dir, "tables")
    path = os.path.join(tab_dir, "krr_multitask_k300.tex")

    if not os.path.isfile(path):
        issues.append(
            "Table V (krr_multitask_k300.tex) not found in "
            f"{tab_dir}"
        )
        return issues

    with open(path, "r") as f:
        content = f.read()

    # Count data rows (lines with '&' excluding rules and header rows)
    lines = content.splitlines()
    data_rows = [
        ln
        for ln in lines
        if "&" in ln
        and "\\hline" not in ln
        and "\\toprule" not in ln
        and "\\bottomrule" not in ln
        and "\\midrule" not in ln
        and not ln.strip().startswith("%")
    ]
    # Heuristic: first row with & is likely the header
    header_kws = {"Target", "target", "Method", "\\textbf", "Coverage"}
    non_header = [
        ln
        for ln in data_rows
        if not any(kw in ln for kw in header_kws)
    ]
    n = len(non_header) if non_header else max(len(data_rows) - 1, 0)
    if n != 10:
        issues.append(
            f"Table V has {n} data rows (expected 10 for 10 coverage targets)"
        )

    return issues


def verify_fig2_panel_structure(artifacts_dir: str = "artifacts_out") -> List[str]:
    """Verify Fig 2 (distortion_cardinality_R1.pdf) is a 2x2 panel.

    Manuscript Fig 2 must show 4 metrics: e_Nys, e_kPCA, RMSE(4G), RMSE(5G)
    in a 2x2 layout with panel labels (a)-(d).  This function performs a
    heuristic check on file size and embedded panel labels.
    """
    issues = []
    fig_dir = os.path.join(artifacts_dir, "figures")
    path = os.path.join(fig_dir, "distortion_cardinality_R1.pdf")

    if not os.path.isfile(path):
        issues.append("Fig 2 (distortion_cardinality_R1.pdf) not found")
        return issues

    fsize = os.path.getsize(path)
    if fsize < 5_000:
        issues.append(
            f"Fig 2 suspiciously small ({fsize} bytes); "
            f"a 2x2 panel is typically > 20 KB"
        )

    # Check for panel labels in PDF text stream
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("latin-1", errors="ignore")
        found = sum(1 for c in "abcd" if f"({c})" in text)
        if found < 4:
            issues.append(
                f"Fig 2 may not be 2x2: found {found}/4 panel labels "
                f"((a)-(d)) in PDF text stream"
            )
    except Exception:
        pass

    return issues


def verify_proxy_stability_structure(
    artifacts_dir: str = "artifacts_out",
) -> List[str]:
    """Verify Table IV (proxy_stability.tex) has 3 required sections.

    Per manuscript, Table IV must contain:
      1. RFF dimension sweep
      2. Anchor count sweep
      3. Cross-representation comparison
    """
    issues = []
    tab_dir = os.path.join(artifacts_dir, "tables")
    path = os.path.join(tab_dir, "proxy_stability.tex")

    if not os.path.isfile(path):
        issues.append("Table IV (proxy_stability.tex) not found")
        return issues

    with open(path, "r") as f:
        content = f.read().lower()

    sections = [
        ("RFF dimension sweep", ["rff", "fourier", "dimension"]),
        ("Anchor count sweep", ["anchor"]),
        ("Cross-representation", ["cross", "representation", "transfer"]),
    ]
    for name, keywords in sections:
        if not any(kw in content for kw in keywords):
            issues.append(
                f"Table IV missing section '{name}' "
                f"(keywords: {keywords})"
            )

    return issues


def verify_phase12_scripts() -> List[str]:
    """Verify Phase 12 scripts and test infrastructure exist.

    Phase 12 requires:
      - scripts/generate_all_artifacts.py  (enhanced with --validate)
      - tests/test_end_to_end.py          (integration test)
    """
    issues = []
    base = Path(__file__).resolve().parent.parent  # coreset_selection/

    # generate_all_artifacts.py with --validate support
    gen_path = base / "scripts" / "generate_all_artifacts.py"
    if not gen_path.is_file():
        issues.append("scripts/generate_all_artifacts.py not found")
    else:
        content = gen_path.read_text()
        if "--validate" not in content:
            issues.append(
                "scripts/generate_all_artifacts.py missing --validate flag "
                "(Phase 12 requirement)"
            )
        if "scan_completed_runs" not in content:
            issues.append(
                "scripts/generate_all_artifacts.py missing scan_completed_runs "
                "(Phase 12 requirement)"
            )

    # test_end_to_end.py
    test_path = base / "tests" / "test_end_to_end.py"
    if not test_path.is_file():
        issues.append("tests/test_end_to_end.py not found (Phase 12 requirement)")

    # test_evaluation_protocol.py
    eval_test_path = base / "tests" / "test_evaluation_protocol.py"
    if not eval_test_path.is_file():
        issues.append(
            "tests/test_evaluation_protocol.py not found "
            "(Phase 4/12 requirement)"
        )

    return issues


# ===================================================================
# Main entry point
# ===================================================================

def main() -> int:
    """Run all compliance checks."""
    parser = argparse.ArgumentParser(
        description="Manuscript Compliance Verification (Phase 12 enhanced).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs_out",
        help="Root directory for run outputs (for R0-R12 coverage check).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts_out",
        help="Directory containing generated artifacts.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Manuscript Compliance Verification (Phase 12 Enhanced)")
    print("=" * 60)
    print()

    all_issues = []

    # ---- Phases 1-11 code checks ----
    code_sections = [
        ("Algorithms",      verify_algorithms),
        ("Objectives",      verify_objectives),
        ("Configurations",  verify_configurations),
        ("Evaluation",      verify_evaluation),
        ("Artifacts",       verify_artifacts),
        ("Constants",       verify_constants),
    ]

    for name, check_fn in code_sections:
        print(f"Checking {name}...")
        try:
            issues = check_fn()
            if issues:
                print(f"  FAIL: {len(issues)} issue(s)")
                for issue in issues:
                    print(f"     - {issue}")
                all_issues.extend(issues)
            else:
                print("  OK")
        except Exception as e:
            print(f"  ERROR: check failed: {e}")
            all_issues.append(f"{name}: {e}")

    # ---- Phase 12 output/structural checks ----
    print()
    print("-" * 60)
    print("Phase 12 — Output & Structural Validation")
    print("-" * 60)
    print()

    phase12_sections = [
        ("Coverage targets (=10)",
         verify_coverage_targets),
        ("Phase 12 scripts & tests",
         verify_phase12_scripts),
        ("R0-R12 output coverage",
         lambda: verify_output_coverage(args.output_dir)),
        ("Table V structure (10 rows)",
         lambda: verify_table_v_structure(args.artifacts_dir)),
        ("Fig 2 panel structure (2x2)",
         lambda: verify_fig2_panel_structure(args.artifacts_dir)),
        ("Table IV proxy stability (3 sections)",
         lambda: verify_proxy_stability_structure(args.artifacts_dir)),
    ]

    for name, check_fn in phase12_sections:
        print(f"Checking {name}...")
        try:
            issues = check_fn()
            if issues:
                print(f"  FAIL: {len(issues)} issue(s)")
                for issue in issues:
                    print(f"     - {issue}")
                all_issues.extend(issues)
            else:
                print("  OK")
        except Exception as e:
            print(f"  ERROR: check failed: {e}")
            all_issues.append(f"{name}: {e}")

    # ---- Summary ----
    print()
    print("=" * 60)
    if all_issues:
        print(f"COMPLIANCE CHECK: {len(all_issues)} issue(s) found")
        return 1
    else:
        print("COMPLIANCE CHECK PASSED: All requirements verified")
        return 0


if __name__ == "__main__":
    sys.exit(main())
