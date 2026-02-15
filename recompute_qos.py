#!/usr/bin/env python
"""Post-hoc QoS recomputation with corrected target (qf_mean).

The original QoS evaluation incorrectly used cov_area_4G as the target.
The correct target is `qf_mean` (Qualidade do Funcionamento — technical
quality of service from Anatel's ISG satisfaction survey).

This script:
  1. Loads qf_mean from smp_main.csv (once, shared across all runs)
  2. Scans runs_out/ for completed experiment directories
  3. For each coreset, re-runs QoS evaluation with the corrected target
  4. Writes corrected metrics to a new CSV alongside the originals

Usage:
  python recompute_qos.py --data-dir data --runs-dir runs_out --cache-dir cache
  python recompute_qos.py --data-dir data --runs-dir runs_out --cache-dir cache --run-ids R1 R5
  python recompute_qos.py --data-dir data --runs-dir /remote/machine1/runs_out --cache-dir /remote/machine1/cache

The script is designed to be safe:
  - It NEVER modifies original result files
  - It writes corrected QoS metrics to `qos_corrected.csv` alongside `all_results.csv`
  - It can be re-run safely (skips already-processed entries)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# QoS target extraction (standalone — does NOT require a cache rebuild)
# ---------------------------------------------------------------------------

def load_qf_mean(data_dir: str) -> np.ndarray:
    """Load qf_mean from smp_main.csv, returning (N,) float64 array.

    NaN values are filled with 0.0 to match the standard target treatment.
    """
    # Find smp_main.csv (handles fuzzy naming)
    candidates = list(Path(data_dir).glob("smp_main*.csv"))
    if not candidates:
        # Try inside ZIP bundles
        candidates = list(Path(data_dir).glob("*.csv"))
        candidates = [c for c in candidates if "smp_main" in c.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find smp_main.csv in {data_dir}. "
            f"Available files: {list(Path(data_dir).glob('*'))}"
        )

    csv_path = candidates[0]
    print(f"[qf_mean] Loading from {csv_path}")

    # Read only the qf_mean column for efficiency
    df = pd.read_csv(csv_path, usecols=["qf_mean"])
    arr = pd.to_numeric(df["qf_mean"], errors="coerce").to_numpy(dtype=np.float64)
    n_valid = int(np.isfinite(arr).sum())
    arr = np.nan_to_num(arr, nan=0.0)
    print(f"[qf_mean] Loaded: N={len(arr)}, valid={n_valid}, "
          f"mean={arr.mean():.4f}, range=[{arr.min():.4f}, {arr.max():.4f}]")
    return arr


# ---------------------------------------------------------------------------
# Cache loading (lightweight — only what QoS needs)
# ---------------------------------------------------------------------------

def load_cache_for_qos(cache_path: str) -> dict:
    """Load minimal fields from a replicate cache for QoS re-evaluation.

    Returns dict with: X_scaled, eval_test_idx, state_labels, entity_ids, time_ids
    """
    data = np.load(cache_path, allow_pickle=True)

    result = {}
    result["X_scaled"] = np.asarray(data["X_scaled"], dtype=np.float64)

    # Eval indices
    if "eval_test_idx" in data.files:
        result["eval_test_idx"] = np.asarray(data["eval_test_idx"], dtype=int)
    elif "eval_test" in data.files:
        result["eval_test_idx"] = np.asarray(data["eval_test"], dtype=int)
    else:
        raise KeyError(f"No eval_test_idx found in {cache_path}. Keys: {data.files}")

    # State labels
    if "state_labels" in data.files:
        result["state_labels"] = data["state_labels"]
    else:
        result["state_labels"] = None

    # Entity and time IDs (optional, for fixed-effects models)
    metadata = {}
    if "entity_ids" in data.files:
        metadata["entity_ids"] = data["entity_ids"]
    if "time_ids" in data.files:
        metadata["time_ids"] = data["time_ids"]
    result["metadata"] = metadata

    return result


# ---------------------------------------------------------------------------
# Coreset discovery
# ---------------------------------------------------------------------------

def find_coresets(rep_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Find all coreset .npz files in a replicate directory.

    Returns list of (name, indices_array) tuples.
    """
    coresets_dir = os.path.join(rep_dir, "coresets")
    if not os.path.isdir(coresets_dir):
        return []

    results = []
    for npz_file in sorted(Path(coresets_dir).glob("*.npz")):
        try:
            data = np.load(npz_file, allow_pickle=True)
            if "indices" in data.files:
                idx = np.asarray(data["indices"], dtype=int)
                name = npz_file.stem
                results.append((name, idx))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# QoS evaluation (uses the project's own qos_tasks module)
# ---------------------------------------------------------------------------

def run_qos_evaluation(
    X_full: np.ndarray,
    qf_mean: np.ndarray,
    S_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    entity_ids: Optional[np.ndarray] = None,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run QoS evaluation with the corrected qf_mean target."""
    from coreset_selection.evaluation.qos_tasks import QoSConfig, qos_coreset_evaluation

    qos_cfg = QoSConfig(
        models=["ols", "ridge", "elastic_net", "pls", "constrained", "heuristic"],
        run_fixed_effects=True,
    )

    metrics = qos_coreset_evaluation(
        X_full=X_full,
        y_full=qf_mean,
        S_idx=S_idx,
        eval_test_idx=eval_test_idx,
        entity_ids=entity_ids,
        time_ids=time_ids,
        state_labels=state_labels,
        config=qos_cfg,
    )
    return metrics


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def discover_runs(runs_dir: str, run_ids: Optional[List[str]] = None) -> List[Tuple[str, str, int]]:
    """Discover all (run_id, rep_dir, rep_id) from runs_out/.

    Returns sorted list of (run_id, full_rep_dir_path, rep_id_int).
    """
    found = []
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        print(f"[warn] runs_dir not found: {runs_dir}")
        return found

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_ids and run_id not in run_ids:
            continue

        for rep_dir in sorted(run_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            name = rep_dir.name
            if name.startswith("rep") and name[3:].isdigit():
                rep_id = int(name[3:])
                found.append((run_id, str(rep_dir), rep_id))

    return found


def find_cache_path(cache_dir: str, rep_id: int) -> Optional[str]:
    """Find the cache .npz for a given replicate ID."""
    candidates = [
        os.path.join(cache_dir, f"rep{rep_id:02d}", "assets.npz"),
        os.path.join(cache_dir, f"rep{rep_id:02d}.npz"),
        os.path.join(cache_dir, f"rep{rep_id:d}", "assets.npz"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-compute QoS metrics with corrected qf_mean target"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing smp_main.csv"
    )
    parser.add_argument(
        "--runs-dir", required=True,
        help="Root output directory (e.g., runs_out/)"
    )
    parser.add_argument(
        "--cache-dir", required=True,
        help="Directory containing replicate caches (e.g., cache/)"
    )
    parser.add_argument(
        "--run-ids", nargs="*", default=None,
        help="Specific run IDs to process (default: all)"
    )
    parser.add_argument(
        "--output-name", default="qos_corrected.csv",
        help="Filename for corrected QoS results (default: qos_corrected.csv)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if qos_corrected.csv already exists"
    )
    args = parser.parse_args()

    # (1) Load the corrected target (once for all runs)
    print("=" * 70)
    print("Post-hoc QoS Recomputation (qf_mean target)")
    print("=" * 70)
    qf_mean = load_qf_mean(args.data_dir)

    # (2) Discover runs
    runs = discover_runs(args.runs_dir, args.run_ids)
    print(f"\n[discovery] Found {len(runs)} replicate directories")
    if not runs:
        print("[done] Nothing to process.")
        return

    # (3) Process each replicate
    cache_store: Dict[int, dict] = {}  # Cache loaded assets by rep_id
    total_coresets = 0
    total_errors = 0
    all_rows = []

    for run_id, rep_dir, rep_id in runs:
        coresets = find_coresets(rep_dir)
        if not coresets:
            continue

        # Check if already processed
        out_csv = os.path.join(rep_dir, "results", args.output_name)
        if os.path.exists(out_csv) and not args.force:
            print(f"  [{run_id}/rep{rep_id:02d}] Already processed ({args.output_name} exists), skipping")
            continue

        # Load cache (shared across coresets in same replicate)
        if rep_id not in cache_store:
            cache_path = find_cache_path(args.cache_dir, rep_id)
            if cache_path is None:
                print(f"  [{run_id}/rep{rep_id:02d}] Cache not found, skipping")
                continue
            print(f"  [{run_id}/rep{rep_id:02d}] Loading cache: {cache_path}")
            try:
                cache_store[rep_id] = load_cache_for_qos(cache_path)
            except Exception as e:
                print(f"  [{run_id}/rep{rep_id:02d}] Cache load failed: {e}")
                continue

        cache = cache_store[rep_id]
        X_scaled = cache["X_scaled"]
        eval_test_idx = cache["eval_test_idx"]
        state_labels = cache["state_labels"]
        entity_ids = cache["metadata"].get("entity_ids")
        time_ids = cache["metadata"].get("time_ids")

        # Validate shapes
        if len(qf_mean) != X_scaled.shape[0]:
            print(f"  [{run_id}/rep{rep_id:02d}] Shape mismatch: qf_mean={len(qf_mean)}, "
                  f"X_scaled={X_scaled.shape[0]}. Skipping.")
            continue

        rep_rows = []
        print(f"  [{run_id}/rep{rep_id:02d}] Processing {len(coresets)} coresets...")

        for coreset_name, indices in coresets:
            t0 = time.time()
            try:
                metrics = run_qos_evaluation(
                    X_full=X_scaled,
                    qf_mean=qf_mean,
                    S_idx=indices,
                    eval_test_idx=eval_test_idx,
                    state_labels=state_labels,
                    entity_ids=entity_ids,
                    time_ids=time_ids,
                )

                row = {
                    "run_id": run_id,
                    "rep_id": rep_id,
                    "coreset_name": coreset_name,
                    "k": len(indices),
                    "target": "qf_mean",
                }
                row.update(metrics)
                rep_rows.append(row)
                total_coresets += 1

                dt = time.time() - t0
                # Show a key metric for quick sanity checking
                rmse_key = "qos_ols_pooled_rmse"
                rmse_val = metrics.get(rmse_key, float("nan"))
                print(f"    {coreset_name} (k={len(indices)}): "
                      f"OLS RMSE={rmse_val:.4f} ({dt:.1f}s)")

            except Exception as e:
                total_errors += 1
                print(f"    {coreset_name} FAILED: {e}")
                traceback.print_exc()

        # Save per-replicate CSV
        if rep_rows:
            results_dir = os.path.join(rep_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rep_rows)
            df.to_csv(out_csv, index=False)
            print(f"    -> Saved {len(rep_rows)} rows to {out_csv}")
            all_rows.extend(rep_rows)

    # (4) Also save a global summary CSV
    if all_rows:
        global_csv = os.path.join(args.runs_dir, "qos_corrected_all.csv")
        pd.DataFrame(all_rows).to_csv(global_csv, index=False)
        print(f"\n[summary] Global results: {global_csv}")

    print(f"\n[done] Processed {total_coresets} coresets, {total_errors} errors")


if __name__ == "__main__":
    main()
