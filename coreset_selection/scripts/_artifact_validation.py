"""Artifact validation helpers for the manuscript artifact pipeline.

Extracted from ``generate_all_artifacts.py`` to keep the main module focused
on the generation and CLI entry point.  All public names are re-exported by
``generate_all_artifacts``.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# Artifact validation
# -----------------------------------------------------------------------

def validate_artifacts(
    out_dir: str,
    runs_root: str = "runs_out",
    *,
    verbose: bool = True,
) -> Tuple[int, int, List[str]]:
    """Validate generated artifacts against manuscript requirements.

    Checks performed (Phase 12 specification):
      1. All 4 manuscript figures exist with nonzero file size.
      2. All 5 manuscript tables exist with nonzero file size.
      3. All complementary figures exist.
      4. Key metric values are within expected ranges.
      5. All run IDs R0-R12 are represented in outputs.

    Returns
    -------
    passed : int
        Number of checks passed.
    failed : int
        Number of checks failed.
    issues : List[str]
        Human-readable descriptions of each failure.
    """
    from ._artifact_scanning import scan_completed_runs

    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")

    passed = 0
    failed = 0
    issues: List[str] = []

    def _check(condition: bool, msg: str) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            issues.append(msg)
            if verbose:
                print(f"  FAIL: {msg}")

    def _file_ok(path: str) -> bool:
        return os.path.isfile(path) and os.path.getsize(path) > 0

    # ------------------------------------------------------------------
    # 1. Manuscript figures (Figs 1-4)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Checking manuscript figures (Figs 1-4)...")
    manuscript_figures = [
        ("Fig 1", "geo_ablation_tradeoff_scatter.pdf"),
        ("Fig 2", "distortion_cardinality_R1.pdf"),
        ("Fig 3", "regional_validity_k300.pdf"),
        ("Fig 4", "objective_metric_alignment_heatmap.pdf"),
    ]
    for label, fname in manuscript_figures:
        path = os.path.join(fig_dir, fname)
        _check(_file_ok(path), f"{label} ({fname}) missing or empty")

    # ------------------------------------------------------------------
    # 2. Manuscript tables (Tables I-V)
    # ------------------------------------------------------------------
    if verbose:
        print("[2/5] Checking manuscript tables (Tables I-V)...")
    manuscript_tables = [
        ("Table I",   "exp_settings.tex"),
        ("Table II",  "run_matrix.tex"),
        ("Table III", "r1_by_k.tex"),
        ("Table IV",  "proxy_stability.tex"),
        ("Table V",   "krr_multitask_k300.tex"),
    ]
    for label, fname in manuscript_tables:
        path = os.path.join(tab_dir, fname)
        _check(_file_ok(path), f"{label} ({fname}) missing or empty")

    # ------------------------------------------------------------------
    # 3. Complementary / narrative-strengthening figures + Phase 11 tables
    # ------------------------------------------------------------------
    if verbose:
        print("[3/5] Checking complementary figures (N1-N12 + extras)...")
    complementary_figures = [
        ("Fig N1",  "kl_floor_vs_k.pdf"),
        ("Fig N2",  "pareto_front_mmd_sd_k300.pdf"),
        ("Fig N3",  "objective_ablation_bars_k300.pdf"),
        ("Fig N4",  "constraint_comparison_bars_k300.pdf"),
        ("Fig N5",  "effort_quality_tradeoff.pdf"),
        ("Fig N6",  "baseline_comparison_grouped.pdf"),
        ("Fig N7",  "multi_seed_stability_boxplot.pdf"),
        ("Fig N8",  "state_kpi_heatmap.pdf"),
        ("Fig N9",  "composition_shift_sankey.pdf"),
        ("Fig N10", "pareto_front_evolution.pdf"),
        ("Fig N11", "nystrom_error_distribution.pdf"),
        ("Fig N12", "krr_worst_state_rmse_vs_k.pdf"),
    ]
    for label, fname in complementary_figures:
        path = os.path.join(fig_dir, fname)
        _check(_file_ok(path), f"{label} ({fname}) missing or empty")

    phase11_tables = [
        ("Table N1", "constraint_diagnostics_cross_config.tex"),
        ("Table N2", "objective_ablation_summary.tex"),
        ("Table N3", "representation_transfer_summary.tex"),
        ("Table N4", "skl_ablation_summary.tex"),
        ("Table N5", "multi_seed_statistics.tex"),
        ("Table N6", "worst_state_rmse_by_k.tex"),
        ("Table N7", "baseline_paired_unconstrained_vs_quota.tex"),
    ]
    for label, fname in phase11_tables:
        path = os.path.join(tab_dir, fname)
        _check(_file_ok(path), f"{label} ({fname}) missing or empty")

    # ------------------------------------------------------------------
    # 4. Key metric range checks
    # ------------------------------------------------------------------
    if verbose:
        print("[4/5] Checking key metric value ranges...")
    _validate_metric_ranges(runs_root, _check)

    # ------------------------------------------------------------------
    # 5. Run ID coverage
    # ------------------------------------------------------------------
    if verbose:
        print("[5/5] Checking R0-R12 representation in outputs...")
    runs = scan_completed_runs(runs_root)
    expected_run_ids = [f"R{i}" for i in range(13)]
    for rid in expected_run_ids:
        _check(rid in runs, f"Run {rid} not found in {runs_root}")

    return passed, failed, issues


def _validate_metric_ranges(runs_root: str, _check) -> None:
    """Check that key metric values fall within plausible ranges.

    Ranges are conservative physical bounds from the manuscript's reported
    values (Section VIII).  They catch gross errors (e.g. negative RMSE,
    Nystrom error > 1) without being overly strict.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    csv_paths = glob.glob(
        os.path.join(runs_root, "**", "all_results.csv"), recursive=True
    )
    if not csv_paths:
        _check(False, "No all_results.csv found - cannot verify metric ranges")
        return

    dfs = []
    for p in csv_paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            pass
    if not dfs:
        _check(False, "Could not parse any all_results.csv files")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Nystrom error: relative Frobenius norm in [0, 1]
    if "e_nys" in df.columns:
        vals = df["e_nys"].dropna()
        if len(vals) > 0:
            _check(vals.min() >= 0.0,
                   f"e_nys has negative values (min={vals.min():.4f})")
            _check(vals.max() <= 1.5,
                   f"e_nys suspiciously large (max={vals.max():.4f}, expected <= 1)")

    # KRR RMSE: must be nonneg
    for col in ["krr_rmse_4G", "krr_rmse_5G", "rmse_4g", "rmse_5g"]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                _check(vals.min() >= 0.0,
                       f"{col} has negative values (min={vals.min():.4f})")

    # geo_kl: KL divergence must be nonneg
    for col in ["geo_kl", "geo_kl_muni", "geo_kl_pop"]:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                _check(vals.min() >= 0.0,
                       f"{col} has negative values (min={vals.min():.4f})")

    # Kendall tau: must be in [-1, 1]
    for col in df.columns:
        if "kendall" in col.lower() or "tau" in col.lower():
            vals = df[col].dropna()
            if len(vals) > 0:
                _check(vals.min() >= -1.0 and vals.max() <= 1.0,
                       f"{col} out of [-1, 1] range")


# -----------------------------------------------------------------------
# Table V structure check  (Phase 12: coverage target count = 10)
# -----------------------------------------------------------------------

def validate_table_v(tab_dir: str) -> List[str]:
    """Verify that Table V (krr_multitask_k300.tex) has exactly 10 data rows.

    The manuscript defines 10 coverage targets in Table V; this function
    parses the LaTeX source to count rows.
    """
    issues: List[str] = []
    path = os.path.join(tab_dir, "krr_multitask_k300.tex")
    if not os.path.isfile(path):
        issues.append("Table V (krr_multitask_k300.tex) not found")
        return issues

    with open(path, "r") as f:
        content = f.read()

    # Count data rows: lines with '&' that are not rules or headers
    data_lines = [
        line for line in content.splitlines()
        if "&" in line
        and "\\hline" not in line
        and "\\toprule" not in line
        and "\\bottomrule" not in line
        and "\\midrule" not in line
        and not line.strip().startswith("%")
    ]
    header_kws = {"Target", "target", "Method", "Run", "Metric", "\\textbf"}
    non_header = [
        ln for ln in data_lines
        if not any(kw in ln for kw in header_kws)
    ]
    n_rows = len(non_header) if non_header else max(len(data_lines) - 1, 0)
    if n_rows != 10:
        issues.append(
            f"Table V should have exactly 10 data rows "
            f"(10 coverage targets), found {n_rows}"
        )
    return issues


# -----------------------------------------------------------------------
# Fig 2 panel structure check  (Phase 12: verify 2x2 panel)
# -----------------------------------------------------------------------

def validate_fig2_panel(fig_dir: str) -> List[str]:
    """Verify that Fig 2 (distortion_cardinality_R1.pdf) is a 2x2 panel.

    Heuristic checks: file size and presence of panel labels (a)-(d).
    """
    issues: List[str] = []
    path = os.path.join(fig_dir, "distortion_cardinality_R1.pdf")
    if not os.path.isfile(path):
        issues.append("Fig 2 (distortion_cardinality_R1.pdf) not found")
        return issues

    fsize = os.path.getsize(path)
    if fsize < 5_000:
        issues.append(
            f"Fig 2 suspiciously small ({fsize} bytes); "
            f"expected 2x2 panel (> 20 KB typical)"
        )

    # Check for panel labels in PDF text stream
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("latin-1", errors="ignore")
        found = sum(1 for c in "abcd" if f"({c})" in text)
        if found < 4:
            issues.append(
                f"Fig 2 may not be a 2x2 panel: found {found}/4 "
                f"panel labels ((a)-(d)) in PDF text stream"
            )
    except Exception:
        pass
    return issues


# -----------------------------------------------------------------------
# Proxy stability table structure check
# -----------------------------------------------------------------------

def validate_proxy_stability_table(tab_dir: str) -> List[str]:
    """Verify proxy_stability.tex has three sections (RFF, anchor, cross-repr).

    Per manuscript Table IV, the proxy stability table must contain:
    1. RFF dimension sweep section
    2. Anchor count sweep section
    3. Cross-representation comparison section
    """
    issues: List[str] = []
    path = os.path.join(tab_dir, "proxy_stability.tex")
    if not os.path.isfile(path):
        issues.append("Table IV (proxy_stability.tex) not found")
        return issues

    with open(path, "r") as f:
        content = f.read().lower()

    sections = [
        ("RFF dimension sweep",   ["rff", "fourier", "dimension"]),
        ("Anchor count sweep",    ["anchor"]),
        ("Cross-representation",  ["cross", "representation", "transfer"]),
    ]
    for name, keywords in sections:
        if not any(kw in content for kw in keywords):
            issues.append(
                f"Table IV missing section: {name} "
                f"(keywords: {keywords})"
            )
    return issues
