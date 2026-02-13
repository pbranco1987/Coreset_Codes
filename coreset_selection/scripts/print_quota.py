#!/usr/bin/env python
"""Print per-state quota allocation and deviation diagnostics.

Phase 6 §6.3 — Quota inspection utility.

Usage
-----
    python -m coreset_selection.scripts.print_quota --k 300
    python -m coreset_selection.scripts.print_quota --k 50 100 200 300 400 500
    python -m coreset_selection.scripts.print_quota --k 300 --top 10
    python -m coreset_selection.scripts.print_quota --k 300 --save quota_summary.csv

If no real data is available, uses a synthetic G=27 dataset to demonstrate
the quota planner output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional, Sequence

import numpy as np


def _build_geo_from_data(data_dir: Optional[str] = None):
    """Try to build GeoInfo from real dataset; fall back to synthetic."""
    try:
        from ..data.manager import DataManager
        from ..config.dataclasses import FilesConfig
        files_cfg = FilesConfig(data_dir=data_dir or "data")
        dm = DataManager(files_cfg, seed=0)
        dm.load()
        from ..geo.info import build_geo_info
        state_labels = np.asarray(dm.state_labels())
        pop = dm.population()
        return build_geo_info(state_labels, population_weights=pop)
    except Exception:
        return _synthetic_geo()


def _synthetic_geo():
    """Generate a synthetic G=27 GeoInfo mimicking Brazil's state structure."""
    from ..geo.info import build_geo_info

    rng = np.random.default_rng(42)
    G = 27
    state_names = [f"S{i:02d}" for i in range(G)]

    # Simulate heterogeneous state sizes (some very small, some large)
    raw_sizes = rng.lognormal(mean=4.5, sigma=1.2, size=G)
    raw_sizes = np.clip(raw_sizes, 5, None).astype(int)
    N = int(raw_sizes.sum())

    state_labels = np.concatenate(
        [np.full(int(s), name) for s, name in zip(raw_sizes, state_names)]
    )
    pop = rng.exponential(scale=50_000, size=N)

    return build_geo_info(state_labels, population_weights=pop)


def print_quota_table(
    k_values: Sequence[int],
    alpha_geo: float = 1.0,
    top_n: int = 0,
    save_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> None:
    """Print quota allocation table for given k values."""
    from ..geo.projector import GeographicConstraintProjector
    from ..geo.kl import compute_quota_path, kl_pi_hat_from_counts

    geo = _build_geo_from_data(data_dir)
    projector = GeographicConstraintProjector(
        geo=geo, alpha_geo=alpha_geo, min_one_per_group=True,
    )

    k_values = sorted(set(int(k) for k in k_values))

    # Compute full quota path
    path = compute_quota_path(
        pi=geo.pi,
        group_sizes=geo.group_sizes,
        k_grid=k_values,
        alpha_geo=alpha_geo,
        min_one_per_group=True,
    )

    # --- Summary table ---
    print("=" * 72)
    print("QUOTA PATH SUMMARY: c*(k) and KL_min(k)")
    print("=" * 72)
    print(f"{'k':>6s}  {'KL_min':>10s}  {'L1':>8s}  {'MaxDev':>8s}  {'Saturated':>10s}")
    print("-" * 72)
    for row in path:
        k = row["k"]
        cap = projector.validate_capacity(k)
        n_sat = len(cap["saturated_groups"])
        print(
            f"{k:6d}  {row['kl_min']:10.6f}  {row['geo_l1']:8.4f}  "
            f"{row['geo_maxdev']:8.4f}  {n_sat:10d}"
        )
    print()

    # --- Per-state detail for each k ---
    all_rows_csv: List[dict] = []
    for row in path:
        k = row["k"]
        cstar = np.array(row["cstar"], dtype=int)
        pi = geo.pi

        print(f"--- k = {k} ---")
        print(
            f"  {'Group':>8s}  {'n_g':>6s}  {'pi_g':>8s}  {'c*(k)':>6s}  "
            f"{'pi_hat':>8s}  {'Dev':>8s}  {'Util%':>6s}"
        )

        detail_rows = []
        for g in range(geo.G):
            n_g = int(geo.group_sizes[g])
            c_g = int(cstar[g])
            pi_g = float(pi[g])
            pi_hat_g = c_g / max(k, 1)
            dev = pi_hat_g - pi_g
            util = c_g / max(n_g, 1) * 100.0
            detail_rows.append({
                "k": k,
                "group": geo.groups[g],
                "n_g": n_g,
                "pi_g": pi_g,
                "cstar": c_g,
                "pi_hat": pi_hat_g,
                "deviation": dev,
                "utilisation_pct": util,
            })

        # Sort by utilisation for display
        detail_rows.sort(key=lambda r: r["utilisation_pct"], reverse=True)

        display_rows = detail_rows[:top_n] if top_n > 0 else detail_rows
        for dr in display_rows:
            print(
                f"  {dr['group']:>8s}  {dr['n_g']:6d}  {dr['pi_g']:8.4f}  "
                f"{dr['cstar']:6d}  {dr['pi_hat']:8.4f}  "
                f"{dr['deviation']:+8.4f}  {dr['utilisation_pct']:6.1f}"
            )
        if top_n > 0 and len(detail_rows) > top_n:
            print(f"  ... ({len(detail_rows) - top_n} more groups)")
        print()

        all_rows_csv.extend(detail_rows)

    # --- Most constrained groups ---
    if any(k >= geo.G for k in k_values):
        k_ref = max(k_values)
        mc = projector.most_constrained_groups(k_ref, top_n=10)
        print(f"TOP 10 MOST CONSTRAINED GROUPS (k={k_ref}):")
        print(f"  {'Group':>8s}  {'c*':>4s}  {'n_g':>5s}  {'Util':>6s}  {'pi_g':>8s}")
        for m in mc:
            print(
                f"  {m['group']:>8s}  {m['cstar']:4d}  {m['n_g']:5d}  "
                f"{m['utilisation']:6.2%}  {m['pi_g']:8.6f}"
            )
        print()

    # --- Optional CSV save ---
    if save_path:
        fieldnames = [
            "k", "group", "n_g", "pi_g", "cstar",
            "pi_hat", "deviation", "utilisation_pct",
        ]
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows_csv)
        print(f"Saved detailed quota table to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Print per-state quota allocation c*(k) and KL_min(k)."
    )
    parser.add_argument(
        "--k", nargs="+", type=int, required=True,
        help="Coreset sizes to inspect.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Dirichlet smoothing α (default: 1.0).",
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Show only top N groups per k (0 = all).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save detailed table to CSV.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data directory (omit for synthetic demo).",
    )
    args = parser.parse_args()

    print_quota_table(
        k_values=args.k,
        alpha_geo=args.alpha,
        top_n=args.top,
        save_path=args.save,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
