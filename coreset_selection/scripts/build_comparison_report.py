#!/usr/bin/env python3
r"""Post-hoc comparison report builder.

Reads saved experiment result CSVs (``all_results.csv``, baseline summaries,
etc.) and produces the incremental comparison outputs recommended by the
kernel k-means vs MMD+Sinkhorn analysis:

    - effect_isolation.csv  (constraint vs objective attribution)
    - rank_table_lower.csv / rank_table_higher.csv
    - pairwise_dominance.csv
    - stability_summary.csv
    - comparison_summary.json

Usage::

    python scripts/build_comparison_report.py --results-dir runs_out/ --output-dir comparison/

Or from within the experiment runner (e.g. at the end of an R10 run)::

    from evaluation.method_comparison import build_comparison_report, load_result_rows
    rows = load_result_rows("runs_out/R10/rep00/results/all_results.csv")
    build_comparison_report(rows, output_dir="runs_out/R10/rep00/comparison")
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.method_comparison import (
    load_result_rows,
    build_comparison_report,
)


def find_result_csvs(results_dir: str) -> list:
    """Find all result CSV files in the output tree."""
    p = Path(results_dir)
    targets = []
    # Prefer all_results.csv → then baseline_summary.csv → then any results/*.csv
    for pattern in ["**/all_results.csv", "**/baseline_summary.csv", "**/results/*.csv"]:
        found = sorted(p.rglob(pattern.split("/")[-1]))
        targets.extend(f for f in found if f not in targets)
    return targets


def main():
    parser = argparse.ArgumentParser(description="Build comparison report from saved results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Root directory containing experiment outputs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {results-dir}/comparison)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "comparison")

    csvs = find_result_csvs(args.results_dir)
    if not csvs:
        print(f"No result CSVs found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(csvs)} result CSV(s):")
    all_rows = []
    for c in csvs:
        rows = load_result_rows(str(c))
        print(f"  {c} — {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No data rows found.")
        sys.exit(1)

    print(f"\nTotal: {len(all_rows)} result rows across {len(csvs)} files")
    print(f"Building comparison report in {output_dir}...\n")
    build_comparison_report(all_rows, output_dir)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
