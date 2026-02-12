"""Run scanning helpers for the manuscript artifact pipeline.

Extracted from ``generate_all_artifacts.py`` to keep the main module focused
on the generation and CLI entry point.  All public names are re-exported by
``generate_all_artifacts``.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict


# -----------------------------------------------------------------------
# Run discovery
# -----------------------------------------------------------------------

def scan_completed_runs(runs_root: str) -> Dict[str, Dict[str, Any]]:
    """Scan the runs_out/ directory tree for all completed run outputs.

    A "completed run" is any directory that contains at least one
    ``all_results.csv`` or a recognisable ``rep*/results/`` sub-tree.

    Parameters
    ----------
    runs_root : str
        Root directory (e.g. ``runs_out/``).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from run_id -> { 'path': ..., 'reps': int,
        'has_results_csv': bool, 'has_pareto': bool, ... }.
    """
    runs: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(runs_root):
        return runs

    for entry in sorted(os.listdir(runs_root)):
        run_path = os.path.join(runs_root, entry)
        if not os.path.isdir(run_path):
            continue

        # Extract run ID from directory name (e.g. R1_k300 -> R1, R10_baselines -> R10)
        m = re.match(r"(R\d+)", entry)
        run_id = m.group(1) if m else entry

        # Count replicates
        rep_dirs = sorted(glob.glob(os.path.join(run_path, "rep*")))
        n_reps = len(rep_dirs)

        # Check for results artefacts
        has_results_csv = bool(
            glob.glob(os.path.join(run_path, "**", "all_results.csv"), recursive=True)
        )
        has_pareto = bool(
            glob.glob(os.path.join(run_path, "**", "*pareto*.npz"), recursive=True)
        )
        has_metrics = bool(
            glob.glob(os.path.join(run_path, "**", "metrics*.json"), recursive=True)
            or glob.glob(os.path.join(run_path, "**", "metrics*.csv"), recursive=True)
        )

        info: Dict[str, Any] = {
            "path": run_path,
            "dir_name": entry,
            "reps": n_reps,
            "has_results_csv": has_results_csv,
            "has_pareto": has_pareto,
            "has_metrics": has_metrics,
        }

        # Merge with existing entry for same run_id (multiple dirs like R1_k50, R1_k100, ...)
        if run_id in runs:
            prev = runs[run_id]
            prev["reps"] += n_reps
            prev["has_results_csv"] = prev["has_results_csv"] or has_results_csv
            prev["has_pareto"] = prev["has_pareto"] or has_pareto
            prev["has_metrics"] = prev["has_metrics"] or has_metrics
            prev.setdefault("extra_dirs", []).append(entry)
        else:
            runs[run_id] = info

    return runs


def print_run_summary(runs: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print the discovered run summary."""
    print(f"{'Run':>6}  {'Reps':>4}  {'CSV':>3}  {'Pareto':>6}  {'Metrics':>7}  Path")
    print("-" * 70)
    for rid in sorted(runs, key=lambda r: (int(re.sub(r"\\D", "", r) or "99"), r)):
        info = runs[rid]
        csv_ok = "Y" if info["has_results_csv"] else "-"
        par_ok = "Y" if info["has_pareto"] else "-"
        met_ok = "Y" if info["has_metrics"] else "-"
        print(
            f"{rid:>6}  {info['reps']:>4}  {csv_ok:>3}  {par_ok:>6}"
            f"  {met_ok:>7}  {info.get('path', '')}"
        )
