#!/usr/bin/env python
"""
Compute the theoretical KL-divergence floor `KL_min(k)` for the default
cardinality grid under both weight modes (municipality-share and
population-share).

What is this?
-------------
`KL_min(k)` is the minimum achievable KL divergence between the target
group distribution `π` and the achieved distribution `π̂(c)` over all
feasible integer count vectors `c` of size `k`. It is computed in closed
form by Algorithm 2 (manuscript appendix) via
:func:`coreset_selection.geo.compute_quota_path`.

This tool is the **Batch 0 prereq-quota step** of the experiment waves
plan. It runs once per dataset, produces a small JSON + CSV pair, and
does **not** require a trained VAE or any optimiser — only the raw
`metadata.csv` and `city_populations.csv`.

Output: an audit artifact showing, for every `k` in the default grid,
whether the default tolerance `τ = 0.02` is achievable at all. For small
`k` (typically `k ≤ 100`), `KL_min(k) > 0.02`, which is what motivates
the adaptive-τ calibration protocol in `adaptive_tau.py`.

Usage
-----
    python scripts/launchers/compute_kl_floor.py \\
        --output-dir EXPERIMENTS-tau_fixed-.../prereq-quota-k_all/

    python scripts/launchers/compute_kl_floor.py \\
        --k-grid 30 50 100 200 300 400 500 \\
        --alpha 1.0 \\
        --output-dir /tmp/prereq/

Outputs
-------
    {output-dir}/kl_floor_muni.csv
    {output-dir}/kl_floor_pop.csv
    {output-dir}/quota_path_muni.json
    {output-dir}/quota_path_pop.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Project root lookup so that ``coreset_selection`` imports regardless of
# the working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from coreset_selection.geo import compute_quota_path, save_quota_path


DEFAULT_K_GRID: List[int] = [30, 50, 100, 200, 300, 400, 500]


def build_pi_and_group_sizes(
    metadata_path: Path, populations_path: Path
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build state list, group sizes, and both π vectors from raw CSVs.

    Parameters
    ----------
    metadata_path : Path
        Path to ``metadata.csv`` (columns include ``codigo_ibge`` and ``UF``).
    populations_path : Path
        Path to ``city_populations.csv`` (columns include ``CODIGO_IBGE``
        and ``POPULACAO``).

    Returns
    -------
    (states, group_sizes, pi_muni, pi_pop) : Tuple[List[str], ndarray, ndarray, ndarray]
        - states: ordered list of state codes (length G).
        - group_sizes: municipalities per state (shape (G,)).
        - pi_muni: municipality-share target distribution.
        - pi_pop: population-share target distribution.
    """
    meta = pd.read_csv(metadata_path)
    meta.columns = [c.lower() for c in meta.columns]
    state_col = "uf"
    states = sorted(meta[state_col].dropna().unique().tolist())
    group_sizes = np.array(
        [int((meta[state_col] == st).sum()) for st in states], dtype=int,
    )

    pi_muni = group_sizes / int(group_sizes.sum())

    pop_df = pd.read_csv(populations_path)
    pop_df.columns = [c.lower() for c in pop_df.columns]
    merged = meta.merge(
        pop_df[["codigo_ibge", "populacao"]], on="codigo_ibge", how="left",
    )
    # Median-impute missing populations so no state collapses to zero weight.
    merged["populacao"] = merged["populacao"].fillna(merged["populacao"].median())
    pop_per_state = np.array(
        [
            float(merged.loc[merged[state_col] == st, "populacao"].sum())
            for st in states
        ],
        dtype=float,
    )
    pi_pop = pop_per_state / pop_per_state.sum()

    return states, group_sizes, pi_muni, pi_pop


def main() -> None:
    """CLI entry point for the prereq-quota step."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute KL_min(k) table for the default cardinality grid. "
            "This is the Batch 0 prereq-quota step of the experiment "
            "waves plan."
        ),
    )
    parser.add_argument(
        "--k-grid",
        type=int,
        nargs="+",
        default=DEFAULT_K_GRID,
        help=(
            "Cardinalities at which to compute KL_min(k). "
            f"Default: {DEFAULT_K_GRID}."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing alpha for KL computation. Default: 1.0.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "metadata.csv",
        help="Path to metadata.csv.",
    )
    parser.add_argument(
        "--populations",
        type=Path,
        default=PROJECT_ROOT / "data" / "city_populations.csv",
        help="Path to city_populations.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where quota_path_*.json and kl_floor_*.csv are written.",
    )
    parser.add_argument(
        "--min-one-per-group",
        action="store_true",
        default=True,
        help="Enforce at least one coreset point per state. Default: True.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {args.metadata}")
    print(f"Loading populations from {args.populations}")
    states, group_sizes, pi_muni, pi_pop = build_pi_and_group_sizes(
        args.metadata, args.populations,
    )
    G = len(states)
    N = int(group_sizes.sum())
    print(f"G = {G} states, N = {N} municipalities")
    print(f"K grid = {args.k_grid}")
    print()

    for mode, pi in (("muni", pi_muni), ("pop", pi_pop)):
        print("=" * 70)
        print(f"{mode.upper()}-SHARE QUOTA PATH")
        print("=" * 70)
        rows = compute_quota_path(
            pi=pi,
            group_sizes=group_sizes,
            k_grid=args.k_grid,
            alpha_geo=args.alpha,
            min_one_per_group=args.min_one_per_group,
        )
        for row in rows:
            print(
                f"  k={row['k']:>3d}   KL_min = {row['kl_min']:.6f}   "
                f"feasible = {row.get('is_feasible', True)}"
            )

        json_path, csv_path = save_quota_path(
            path_rows=rows,
            output_dir=str(args.output_dir),
            json_name=f"quota_path_{mode}.json",
            csv_name=f"kl_floor_{mode}.csv",
        )
        print(f"  -> {json_path}")
        print(f"  -> {csv_path}")
        print()


if __name__ == "__main__":
    main()
