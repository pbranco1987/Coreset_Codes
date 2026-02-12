#!/usr/bin/env python
"""
merge_results.py — Combine experiment results from multiple servers.

Usage:
    python scripts/merge_results.py
    python scripts/merge_results.py --results-dir results --output-dir results/combined

What it does:
    1. Discovers all all_results.csv files from both server result trees
    2. Tags each row with server name and seed
    3. Produces a combined CSV
    4. Runs the comparison report (rank tables, stability, effect isolation)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def find_result_csvs(base_dir: str) -> List[Dict[str, str]]:
    """Find all all_results.csv files and tag with server metadata."""
    entries = []
    base = Path(base_dir)

    for server_dir in sorted(base.iterdir()):
        if not server_dir.is_dir():
            continue
        server_name = server_dir.name  # "labgele" or "lessonia"

        # Find all all_results.csv files recursively
        for csv_path in server_dir.rglob("all_results.csv"):
            entries.append({
                "server": server_name,
                "csv_path": str(csv_path),
            })

    return entries


def load_and_tag_rows(entries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Load CSV rows from all found files and tag with server metadata."""
    all_rows = []

    for entry in entries:
        server = entry["server"]
        csv_path = entry["csv_path"]

        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse numeric values
                    clean = {}
                    for k, v in row.items():
                        try:
                            clean[k] = float(v)
                        except (ValueError, TypeError):
                            clean[k] = v

                    # Tag with server metadata
                    clean["server"] = server
                    all_rows.append(clean)
        except Exception as e:
            print(f"  WARNING: Failed to read {csv_path}: {e}")

    return all_rows


def write_combined_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write all rows to a single combined CSV."""
    if not rows:
        print("  No rows to write.")
        return

    # Collect all keys in stable order
    all_keys = []
    seen = set()
    # Priority columns first
    priority = ["server", "run_id", "rep_id", "method", "space", "k",
                "constraint_regime", "rep_name"]
    for k in priority:
        if any(k in r for r in rows) and k not in seen:
            all_keys.append(k)
            seen.add(k)
    # Then all remaining keys alphabetically
    for r in rows:
        for k in sorted(r.keys()):
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in r.items()})

    print(f"  Combined CSV: {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge experiment results from multiple servers",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing server result subdirectories (default: results/)",
    )
    parser.add_argument(
        "--output-dir", default="results/combined",
        help="Output directory for merged results (default: results/combined/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir

    print("============================================")
    print("  Results Merger")
    print("============================================")
    print(f"  Source: {results_dir}/")
    print(f"  Output: {output_dir}/")
    print("")

    # Step 1: Find all CSVs
    entries = find_result_csvs(results_dir)
    if not entries:
        print("  ERROR: No all_results.csv files found!")
        print(f"  Searched in: {results_dir}/")
        print("  Make sure you ran collect_results.sh first.")
        sys.exit(1)

    print(f"  Found {len(entries)} result files:")
    for e in entries:
        print(f"    [{e['server']}] {e['csv_path']}")
    print("")

    # Step 2: Load and tag
    print("  Loading and tagging rows...")
    rows = load_and_tag_rows(entries)
    print(f"  Total rows: {len(rows)}")

    # Count per server
    server_counts = {}
    for r in rows:
        s = r.get("server", "unknown")
        server_counts[s] = server_counts.get(s, 0) + 1
    for s, c in sorted(server_counts.items()):
        print(f"    {s}: {c} rows")
    print("")

    # Step 3: Write combined CSV
    combined_path = os.path.join(output_dir, "all_results_combined.csv")
    write_combined_csv(rows, combined_path)
    print("")

    # Step 4: Run comparison report
    print("  Generating comparison report...")
    try:
        # Add project to path
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from coreset_selection.evaluation.method_comparison import (
            build_comparison_report,
        )

        comparison_dir = os.path.join(output_dir, "comparison")
        build_comparison_report(rows, comparison_dir)
        print(f"  Comparison report: {comparison_dir}/")
    except ImportError as e:
        print(f"  WARNING: Could not import coreset_selection: {e}")
        print("  Skipping comparison report. Install the package first:")
        print("    cd coreset_selection && pip install -e .")
    except Exception as e:
        print(f"  WARNING: Comparison report failed: {e}")

    print("")
    print("============================================")
    print("  Merge Complete")
    print("============================================")
    print(f"  Combined CSV:      {combined_path}")
    print(f"  Comparison report: {os.path.join(output_dir, 'comparison')}/")
    print("============================================")


if __name__ == "__main__":
    main()
