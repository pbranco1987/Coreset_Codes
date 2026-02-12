#!/usr/bin/env python3
r"""Generate all manuscript artifacts (figures + tables) -- Phase 12 compliant.

Top-level analysis script that:
1. Scans the ``runs_out/`` directory for all completed runs.
2. Instantiates :class:`ManuscriptArtifacts`.
3. Calls ``generate_all()`` to produce every figure and table.
4. Reports missing data, failed figures, and incomplete tables.
5. (With ``--validate``) checks artifact existence, sizes, and metric ranges.

Usage
-----
::

    # Generate all artifacts
    python -m coreset_selection.scripts.generate_all_artifacts \
        --runs-root runs_out \
        --cache-root cache \
        --out-dir artifacts_out

    # Generate + validate
    python -m coreset_selection.scripts.generate_all_artifacts \
        --runs-root runs_out \
        --out-dir artifacts_out \
        --validate

    # Validation only (skip generation)
    python -m coreset_selection.scripts.generate_all_artifacts \
        --out-dir artifacts_out \
        --validate-only

Manuscript figure references (Phase 8):
    Fig 1  -> ``figures/geo_ablation_tradeoff_scatter.pdf``
    Fig 2  -> ``figures/distortion_cardinality_R1.pdf``
    Fig 3  -> ``figures/regional_validity_k300.pdf``
    Fig 4  -> ``figures/objective_metric_alignment_heatmap.pdf``

Manuscript table references (Phase 9):
    Tab I   -> ``tables/exp_settings.tex``
    Tab II  -> ``tables/run_matrix.tex``
    Tab III -> ``tables/r1_by_k.tex``
    Tab IV  -> ``tables/proxy_stability.tex``
    Tab V   -> ``tables/krr_multitask_k300.tex``

Narrative-strengthening figures (Phase 10a: Figs N1-N6):
    Fig N1  -> ``figures/kl_floor_vs_k.pdf``
    Fig N2  -> ``figures/pareto_front_mmd_sd_k300.pdf``
    Fig N3  -> ``figures/objective_ablation_bars_k300.pdf``
    Fig N4  -> ``figures/constraint_comparison_bars_k300.pdf``
    Fig N5  -> ``figures/effort_quality_tradeoff.pdf``
    Fig N6  -> ``figures/baseline_comparison_grouped.pdf``

Narrative-strengthening figures (Phase 10b: Figs N7-N12):
    Fig N7  -> ``figures/multi_seed_stability_boxplot.pdf``
    Fig N8  -> ``figures/state_kpi_heatmap.pdf``
    Fig N9  -> ``figures/composition_shift_sankey.pdf``
    Fig N10 -> ``figures/pareto_front_evolution.pdf``
    Fig N11 -> ``figures/nystrom_error_distribution.pdf``
    Fig N12 -> ``figures/krr_worst_state_rmse_vs_k.pdf``

Narrative-strengthening tables (Phase 11: Tables N1-N7):
    Tab N1  -> ``tables/constraint_diagnostics_cross_config.tex``
    Tab N2  -> ``tables/objective_ablation_summary.tex``
    Tab N3  -> ``tables/representation_transfer_summary.tex``
    Tab N4  -> ``tables/skl_ablation_summary.tex``
    Tab N5  -> ``tables/multi_seed_statistics.tex``
    Tab N6  -> ``tables/worst_state_rmse_by_k.tex``
    Tab N7  -> ``tables/baseline_paired_unconstrained_vs_quota.tex``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Import scanning helpers from the private sub-module
from ._artifact_scanning import (
    scan_completed_runs,
    print_run_summary,
)

# Import validation helpers from the private sub-module
from ._artifact_validation import (
    validate_artifacts,
    _validate_metric_ranges,
    validate_table_v,
    validate_fig2_panel,
    validate_proxy_stability_table,
)


# -----------------------------------------------------------------------
# Main generation + validation pipeline
# -----------------------------------------------------------------------

def generate_artifacts(
    runs_root: str,
    cache_root: str,
    out_dir: str,
    generator: str = "manuscript_artifacts",
    data_dir: str = "data",
) -> Dict[str, List[str]]:
    """Run the full artifact generation pipeline.

    Parameters
    ----------
    runs_root : str
        Root directory containing run outputs.
    cache_root : str
        Root directory containing replicate caches.
    out_dir : str
        Output directory for generated artifacts.
    generator : str
        Which generator to use: "manuscript_artifacts", "generator", or "both".
    data_dir : str
        Root directory for input data files (needed for choropleth maps).

    Returns
    -------
    Dict[str, List[str]]
        ``{"figures": [...], "tables": [...], "failed": [...]}``
    """
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, List[str]] = {"figures": [], "tables": [], "failed": []}

    # ---- Step 1: Scan runs ----
    print("=" * 70)
    print("Phase 12 -- Analysis Script & Artifact Generation")
    print("=" * 70)

    runs = scan_completed_runs(runs_root)
    if runs:
        print(f"\nDiscovered {len(runs)} run group(s) in {runs_root}:")
        print_run_summary(runs)
    else:
        print(f"\n[!] No completed runs found in {runs_root}")
        print("    Artifact generation will proceed with available data.")

    # ---- Step 2: Generate via ManuscriptArtifacts ----
    if generator in ("manuscript_artifacts", "both"):
        try:
            from coreset_selection.artifacts.manuscript_artifacts import (
                ManuscriptArtifacts,
            )

            print("\n" + "=" * 70)
            print("ManuscriptArtifacts (Phases 8-11 compliant)")
            print("=" * 70)

            gen = ManuscriptArtifacts(
                runs_root=runs_root,
                cache_root=cache_root,
                out_dir=out_dir,
                data_dir=data_dir,
            )
            gen_result = gen.generate_all()
            result["figures"].extend(gen_result.get("figures", []))
            result["tables"].extend(gen_result.get("tables", []))

            print(f"\nGenerated {len(gen_result.get('figures', []))} figures:")
            for p in gen_result.get("figures", []):
                print(f"  + {os.path.basename(p)}")
            print(f"\nGenerated {len(gen_result.get('tables', []))} tables:")
            for p in gen_result.get("tables", []):
                print(f"  + {os.path.basename(p)}")

        except Exception as e:
            print(f"\n[X] ManuscriptArtifacts generation failed: {e}")
            result["failed"].append(f"ManuscriptArtifacts: {e}")

    # ---- Step 3: Legacy generator ----
    if generator in ("generator", "both"):
        try:
            from coreset_selection.artifacts.generator import (
                ManuscriptArtifactGenerator,
            )

            out2 = (
                os.path.join(out_dir, "generator_output")
                if generator == "both"
                else out_dir
            )
            print("\n" + "=" * 70)
            print("ManuscriptArtifactGenerator (legacy)")
            print("=" * 70)

            gen2 = ManuscriptArtifactGenerator(
                runs_root=runs_root,
                cache_root=cache_root,
                out_dir=out2,
            )
            gen2_result = gen2.generate_all()
            result["figures"].extend(gen2_result.get("figures", []))
            result["tables"].extend(gen2_result.get("tables", []))

        except Exception as e:
            print(f"\n[X] Legacy generator failed: {e}")
            result["failed"].append(f"LegacyGenerator: {e}")

    return result


def main() -> int:
    """Entry point: parse args, generate artifacts, optionally validate."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 12 -- Generate all manuscript artifacts and validate "
            "end-to-end compliance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs-root", default="runs_out",
        help="Root directory containing run outputs (default: runs_out).",
    )
    parser.add_argument(
        "--cache-root", default="cache",
        help="Root directory containing replicate caches (default: cache).",
    )
    parser.add_argument(
        "--out-dir", default="artifacts_out",
        help="Output directory for generated artifacts (default: artifacts_out).",
    )
    parser.add_argument(
        "--generator", default="manuscript_artifacts",
        choices=["manuscript_artifacts", "generator", "both"],
        help="Which generator to use (default: manuscript_artifacts).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help=(
            "After generation, validate that all required artifacts exist, "
            "have nonzero size, and key metrics are in expected ranges."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip generation; only run validation on existing artifacts.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with nonzero status if any validation check fails.",
    )
    args = parser.parse_args()

    t0 = time.time()
    exit_code = 0

    # ---- Generation phase ----
    if not args.validate_only:
        result = generate_artifacts(
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            out_dir=args.out_dir,
            generator=args.generator,
        )
        elapsed = time.time() - t0
        n_figs = len(result["figures"])
        n_tabs = len(result["tables"])
        n_fail = len(result["failed"])

        print(f"\n{'=' * 70}")
        print(
            f"Generation complete: {n_figs} figures, {n_tabs} tables "
            f"({n_fail} failure(s)) in {elapsed:.1f}s"
        )
        print(f"Output: {os.path.abspath(args.out_dir)}")
        print(f"{'=' * 70}")

        if n_fail > 0:
            print("\nFailed artifacts:")
            for f in result["failed"]:
                print(f"  [X] {f}")
    else:
        print("=" * 70)
        print("Phase 12 -- Validation Only (skipping generation)")
        print("=" * 70)

    # ---- Validation phase ----
    if args.validate or args.validate_only:
        print(f"\n{'=' * 70}")
        print("Artifact Validation (Phase 12)")
        print(f"{'=' * 70}")

        passed, failed, all_issues = validate_artifacts(
            out_dir=args.out_dir,
            runs_root=args.runs_root,
            verbose=True,
        )

        # Additional structural checks
        tab_dir = os.path.join(args.out_dir, "tables")
        fig_dir = os.path.join(args.out_dir, "figures")

        print("\n[Extra] Checking Table V structure (10 rows)...")
        for issue in validate_table_v(tab_dir):
            print(f"  FAIL: {issue}")
            failed += 1
            all_issues.append(issue)
        else:
            passed += 1

        print("[Extra] Checking Fig 2 panel structure (2x2)...")
        f2_issues = validate_fig2_panel(fig_dir)
        for issue in f2_issues:
            print(f"  FAIL: {issue}")
            failed += 1
            all_issues.append(issue)
        if not f2_issues:
            passed += 1

        print("[Extra] Checking Table IV structure (3 sections)...")
        ps_issues = validate_proxy_stability_table(tab_dir)
        for issue in ps_issues:
            print(f"  FAIL: {issue}")
            failed += 1
            all_issues.append(issue)
        if not ps_issues:
            passed += 1

        # Coverage target count
        print("[Extra] Checking coverage target count = 10...")
        try:
            from coreset_selection.config.constants import COVERAGE_TARGETS_TABLE_V
            n_targets = len(COVERAGE_TARGETS_TABLE_V)
            if n_targets == 10:
                passed += 1
            else:
                msg = (
                    f"COVERAGE_TARGETS_TABLE_V has {n_targets} entries "
                    f"(expected 10)"
                )
                print(f"  FAIL: {msg}")
                failed += 1
                all_issues.append(msg)
        except ImportError as e:
            msg = f"Cannot import COVERAGE_TARGETS_TABLE_V: {e}"
            print(f"  FAIL: {msg}")
            failed += 1
            all_issues.append(msg)

        # Summary
        total = passed + failed
        print(f"\n{'=' * 70}")
        print(f"Validation: {passed}/{total} checks passed, {failed} failed")
        if all_issues:
            print("\nAll issues:")
            for i, issue in enumerate(all_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("All validation checks passed.")
        print(f"{'=' * 70}")

        if failed > 0 and args.strict:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
