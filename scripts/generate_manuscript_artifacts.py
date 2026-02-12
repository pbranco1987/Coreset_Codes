#!/usr/bin/env python
"""Generate all manuscript figures and tables from experiment outputs.

Usage (from project root):
    python scripts/generate_manuscript_artifacts.py \\
        --runs-root runs_out_labgele/k300 \\
        --cache-root replicate_cache \\
        --data-dir data \\
        --out-dir manuscript_artifacts

All figures referenced by the LaTeX manuscript are produced under
<out-dir>/figures/ and all tables under <out-dir>/tables/.

The generator gracefully handles missing experiments: each artifact
is attempted independently and failures are logged without aborting.

Manuscript figures (8, directly referenced via \\includegraphics):
  Fig 1: kl_floor_vs_k.pdf               — KL feasibility floor
  Fig 2: geo_ablation_tradeoff_scatter.pdf — Geographic ablation scatter
  Fig 3: distortion_cardinality_R1.pdf    — Raw-space metrics vs k
  Fig 4: krr_worst_state_rmse_vs_k.pdf    — Worst-state RMSE equity
  Fig 5: regional_validity_k300.pdf       — Regional KPI validity
  Fig 6: baseline_comparison_grouped.pdf  — Baseline method comparison
  Fig 7: representation_transfer_bars.pdf — Representation transfer
  Fig 8: objective_metric_alignment_heatmap.pdf — Obj-metric Spearman

Manuscript tables (6, directly referenced via \\label):
  Table I:   exp_settings.tex             — Experiment settings
  Table II:  run_matrix.tex               — Run matrix
  Table III: r1_by_k.tex                  — R1 metric envelope by k
  Table IV:  repr_timing.tex              — Representation timing
  Table V:   proxy_stability.tex          — Proxy stability diagnostics
  Table VI:  krr_multitask_k300.tex       — Multi-target KRR RMSE

Plus ~15 complementary figures and ~12 complementary tables for
narrative-strengthening and reviewer defense.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure the coreset_selection package is importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Manuscript-referenced artifact filenames for quick validation
MANUSCRIPT_FIGURES = [
    "kl_floor_vs_k.pdf",
    "geo_ablation_tradeoff_scatter.pdf",
    "distortion_cardinality_R1.pdf",
    "krr_worst_state_rmse_vs_k.pdf",
    "regional_validity_k300.pdf",
    "baseline_comparison_grouped.pdf",
    "representation_transfer_bars.pdf",
    "objective_metric_alignment_heatmap.pdf",
]

MANUSCRIPT_TABLES = [
    "exp_settings.tex",
    "run_matrix.tex",
    "r1_by_k.tex",
    "repr_timing.tex",
    "proxy_stability.tex",
    "krr_multitask_k300.tex",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures and tables from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After running K=300 on LABGELE:
  python scripts/generate_manuscript_artifacts.py \\
      --runs-root runs_out_labgele/k300 \\
      --cache-root replicate_cache

  # Only figures:
  python scripts/generate_manuscript_artifacts.py \\
      --runs-root runs_out_labgele/k300 \\
      --figures-only

  # Only tables:
  python scripts/generate_manuscript_artifacts.py \\
      --runs-root runs_out_labgele/k300 \\
      --tables-only

  # Custom output directory:
  python scripts/generate_manuscript_artifacts.py \\
      --runs-root runs_out_labgele/k300 \\
      --out-dir paper/figures_tables

  # Validate existing artifacts (check which are present):
  python scripts/generate_manuscript_artifacts.py \\
      --runs-root runs_out_labgele/k300 \\
      --validate-only
        """,
    )
    parser.add_argument(
        "--runs-root",
        default="runs_out",
        help="Root directory containing experiment outputs (e.g. runs_out_labgele/k300)",
    )
    parser.add_argument(
        "--cache-root",
        default="replicate_cache",
        help="Root directory for replicate caches (VAE/PCA)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input data files (smp_main.csv, etc.)",
    )
    parser.add_argument(
        "--out-dir",
        default="manuscript_artifacts",
        help="Output directory for generated figures and tables",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check which manuscript artifacts exist; do not generate",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    for attr in ("runs_root", "cache_root", "data_dir", "out_dir"):
        path = getattr(args, attr)
        if not os.path.isabs(path):
            setattr(args, attr, os.path.join(_project_root, path))

    print("=" * 60)
    print("MANUSCRIPT ARTIFACT GENERATOR")
    print("=" * 60)
    print(f"  Runs root : {args.runs_root}")
    print(f"  Cache root: {args.cache_root}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Output dir: {args.out_dir}")
    print("=" * 60)

    # ---- Validate-only mode ----
    if args.validate_only:
        _validate_artifacts(args.out_dir)
        return

    # Check that runs root exists
    if not os.path.isdir(args.runs_root):
        print(f"\n[WARNING] Runs root does not exist: {args.runs_root}")
        print("  Artifacts will be generated with placeholder data where possible.")

    from coreset_selection.artifacts.manuscript_artifacts import ManuscriptArtifacts

    gen = ManuscriptArtifacts(
        runs_root=args.runs_root,
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
    )

    t0 = time.time()
    results = gen.generate_all()
    elapsed = time.time() - t0

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("ARTIFACT GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Figures: {len(results.get('figures', []))}")
    for f in results.get("figures", []):
        basename = os.path.basename(f)
        tag = "  [MANUSCRIPT]" if basename in MANUSCRIPT_FIGURES else ""
        print(f"    {basename}{tag}")
    print(f"  Tables:  {len(results.get('tables', []))}")
    for t in results.get("tables", []):
        basename = os.path.basename(t)
        tag = "  [MANUSCRIPT]" if basename in MANUSCRIPT_TABLES else ""
        print(f"    {basename}{tag}")

    # ---- Manuscript completeness check ----
    print(f"\n{'=' * 60}")
    print("MANUSCRIPT COMPLETENESS CHECK")
    print(f"{'=' * 60}")
    _validate_artifacts(args.out_dir)


def _validate_artifacts(out_dir: str) -> None:
    """Check which manuscript-referenced artifacts exist on disk."""
    fig_dir = os.path.join(out_dir, "figures")
    tab_dir = os.path.join(out_dir, "tables")

    print("\n  Manuscript figures:")
    n_figs_ok = 0
    for fname in MANUSCRIPT_FIGURES:
        path = os.path.join(fig_dir, fname)
        exists = os.path.isfile(path)
        if exists:
            size_kb = os.path.getsize(path) / 1024
            status = f"OK ({size_kb:.0f} KB)"
            n_figs_ok += 1
        else:
            status = "MISSING"
        print(f"    [{status:>15s}] {fname}")

    print("\n  Manuscript tables:")
    n_tabs_ok = 0
    for fname in MANUSCRIPT_TABLES:
        path = os.path.join(tab_dir, fname)
        exists = os.path.isfile(path)
        if exists:
            # Check if it contains placeholder "---" values
            with open(path) as f:
                content = f.read()
            has_placeholder = "---" in content
            size_kb = os.path.getsize(path) / 1024
            if has_placeholder:
                status = f"TEMPLATE ({size_kb:.0f} KB)"
            else:
                status = f"OK ({size_kb:.0f} KB)"
                n_tabs_ok += 1
        else:
            status = "MISSING"
        print(f"    [{status:>15s}] {fname}")

    total = len(MANUSCRIPT_FIGURES) + len(MANUSCRIPT_TABLES)
    total_ok = n_figs_ok + n_tabs_ok
    print(f"\n  Score: {total_ok}/{total} manuscript artifacts fully populated")
    if total_ok < total:
        print("  (Run more experiments to fill in missing/template artifacts)")
    else:
        print("  All manuscript artifacts are ready for compilation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
