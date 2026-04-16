#!/usr/bin/env python
"""
Post-hoc soft-KL repair for baseline coresets.

For each baseline experiment whose constraint_mode includes a SOFT KL
component (population_share, municipality_share, joint_soft_hard,
joint_hard_soft), loads saved baseline indices, applies the same swap-based
proportionality repair that NSGA-II uses, then re-evaluates all metrics.

This makes the baselines directly comparable ("apples to apples") to the
NSGA-II runs that enforce the same soft constraints during optimization.

Constraint modes and their soft components:
    population_share      (ps)  → pop-share KL repair
    municipality_share    (ms)  → muni-share KL repair
    joint_soft_hard       (sh)  → pop-share KL repair  (muni is hard quota)
    joint_hard_soft       (hs)  → muni-share KL repair (pop is hard quota)

Modes that do NOT need repair (already deterministic or unconstrained):
    none                  (0)   → no constraint
    population_share_quota(ph)  → hard quota already applied
    municipality_share_quota(mh)→ hard quota already applied
    joint                 (hh)  → both hard quotas already applied

Usage:
    python scripts/launchers/run_baselines.py
    python scripts/launchers/run_baselines.py --modes ps ms
    python scripts/launchers/run_baselines.py --spaces vae raw pca
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root: this script lives at scripts/launchers/run_baselines.py,
# so parents[2] gives us the repository root (Coreset_Codes/).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from coreset_selection.geo.info import build_geo_info, GeoInfo
from coreset_selection.geo.projector import GeographicConstraintProjector
from coreset_selection.constraints.proportionality import (
    build_population_share_constraint,
    build_municipality_share_constraint,
    ProportionalityConstraintSet,
)
from coreset_selection.optimization._nsga2_operators import _repair_mask
from coreset_selection.evaluation.geo_diagnostics import dual_geo_diagnostics
from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
from coreset_selection.evaluation.qos_tasks import QoSConfig, qos_coreset_evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "labgele_experiments_v2" / "experiments_v2"
CACHE_DIR = PROJECT_ROOT / "replicate_cache"
OUTPUT_DIR = PROJECT_ROOT / "results" / "repaired_baselines"
SEED = 2026
TAU_POP = 0.02
TAU_MUNI = 0.02
ALPHA_GEO = 1.0
K = 100
N_REPS = 5
MAX_REPAIR_ITERS = 200

# Mapping: mode abbreviation → (constraint_mode, which soft constraints to build)
#   "soft_components" lists which KL constraints the repair enforces.
#   Hard-quota components are already baked into the baseline's quota regime
#   and don't need repair.
MODE_CONFIG = {
    "ps": {
        "constraint_mode": "population_share",
        "dir_suffix": "ps",
        "soft_components": ["pop"],         # pop-share KL
        "preserve_group_counts": False,
    },
    "ms": {
        "constraint_mode": "municipality_share",
        "dir_suffix": "ms",
        "soft_components": ["muni"],        # muni-share KL
        "preserve_group_counts": False,
    },
    "sh": {
        "constraint_mode": "joint_soft_hard",
        "dir_suffix": "sh",
        "soft_components": ["pop"],         # pop is SOFT; muni is hard quota
        "preserve_group_counts": True,      # joint → preserve group counts
    },
    "hs": {
        "constraint_mode": "joint_hard_soft",
        "dir_suffix": "hs",
        "soft_components": ["muni"],        # muni is SOFT; pop is hard quota
        "preserve_group_counts": True,      # joint → preserve group counts
    },
}


def load_cache(rep_id: int) -> dict:
    """Load replicate cache as a dict of arrays."""
    cache_path = CACHE_DIR / f"rep{rep_id:02d}" / "assets.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    data = np.load(str(cache_path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_geo_and_constraints(
    state_labels: np.ndarray,
    population: np.ndarray,
    mode_key: str,
) -> Tuple[GeoInfo, ProportionalityConstraintSet, GeographicConstraintProjector]:
    """Build GeoInfo, ProportionalityConstraintSet, and projector for the given mode."""
    geo = build_geo_info(state_labels, population_weights=population)
    cfg = MODE_CONFIG[mode_key]

    # Determine weight_type for projector based on soft component
    weight_type = "pop" if "pop" in cfg["soft_components"] else "uniform"

    constraints = []
    for component in cfg["soft_components"]:
        if component == "pop":
            constraints.append(
                build_population_share_constraint(
                    geo=geo, population=population, alpha=ALPHA_GEO, tau=TAU_POP,
                )
            )
        elif component == "muni":
            constraints.append(
                build_municipality_share_constraint(
                    geo=geo, alpha=ALPHA_GEO, tau=TAU_MUNI,
                )
            )

    cs = ProportionalityConstraintSet(
        geo=geo,
        constraints=constraints,
        min_one_per_group=True,
        preserve_group_counts=cfg["preserve_group_counts"],
        max_iters=MAX_REPAIR_ITERS,
    )

    projector = GeographicConstraintProjector(
        geo=geo, alpha_geo=ALPHA_GEO, min_one_per_group=True,
        weight_type=weight_type,
    )

    return geo, cs, projector


def repair_selection(
    indices: np.ndarray,
    N: int,
    k: int,
    projector: GeographicConstraintProjector,
    constraint_set: ProportionalityConstraintSet,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the same repair pipeline as NSGA-II: quota projection + swap repair.

    This calls _repair_mask(projector, constraint_set, rng, mask, k, True, True)
    which first projects to per-state quota counts, then applies the swap-based
    soft-KL repair — identical to what NSGA-II does internally.
    """
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    mask_repaired = _repair_mask(projector, constraint_set, rng, mask, k, True, True)
    return np.where(mask_repaired)[0]


def build_evaluator(cache: dict, seed: int) -> RawSpaceEvaluator:
    """Build RawSpaceEvaluator from cache arrays."""
    X_scaled = np.asarray(cache["X_scaled"], dtype=np.float64)
    y_4G = cache["y_4G"]
    y_5G = cache["y_5G"]
    y = np.column_stack([y_4G, y_5G])
    target_names = ["4G", "5G"]

    # Add extra coverage targets if present
    for key in sorted(cache.keys()):
        if key.startswith("y_extra_"):
            tname = key[len("y_extra_"):]
            y = np.column_stack([y, cache[key]])
            target_names.append(tname)

    return RawSpaceEvaluator.build(
        X_raw=X_scaled,
        y=y,
        eval_idx=cache["eval_idx"],
        eval_train_idx=cache["eval_train_idx"],
        eval_test_idx=cache["eval_test_idx"],
        seed=seed,
        target_names=target_names,
    )


def collect_downstream_targets(cache: dict) -> tuple:
    """Collect extra regression and classification targets from cache."""
    extra_reg = {}
    for key in sorted(cache.keys()):
        if key.startswith("y_extreg_"):
            tname = key[len("y_extreg_"):]
            extra_reg[tname] = np.asarray(cache[key])

    cls_targets = {}
    for key in sorted(cache.keys()):
        if key.startswith("y_cls_"):
            tname = key[len("y_cls_"):]
            cls_targets[tname] = np.asarray(cache[key])

    return extra_reg, cls_targets


def evaluate_coreset(
    idx_sel: np.ndarray,
    geo: GeoInfo,
    evaluator: RawSpaceEvaluator,
    state_labels: np.ndarray,
    extra_reg: Dict[str, np.ndarray],
    cls_targets: Dict[str, np.ndarray],
    cache: dict,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a coreset selection with the full metric suite."""
    row: Dict[str, Any] = {}

    # 1. Geographic diagnostics (dual: muni + pop)
    geo_all = dual_geo_diagnostics(geo, idx_sel, K, alpha=ALPHA_GEO)
    row.update(geo_all)

    # 2. Raw-space metrics (Nystrom, kPCA, KRR)
    row.update(evaluator.all_metrics_with_state_stability(idx_sel, state_labels))

    # 3. Multi-model downstream (KNN, RF, LR, GBT on regression + classification)
    try:
        multi_metrics = evaluator.multi_model_downstream(
            S_idx=idx_sel,
            regression_targets=extra_reg,
            classification_targets=cls_targets,
            seed=seed,
        )
        row.update(multi_metrics)
    except Exception as e:
        print(f"      multi-model failed: {e}")

    # 4. QoS downstream (OLS, Ridge, Elastic Net, PLS)
    try:
        qos_y = cache.get("qos_target", None)
        if qos_y is None:
            qos_y = evaluator.y[:, 0] if evaluator.y.ndim == 2 else evaluator.y
        qos_cfg = QoSConfig(
            models=["ols", "ridge", "elastic_net", "pls", "constrained", "heuristic"],
            run_fixed_effects=True,
        )
        qos_metrics = qos_coreset_evaluation(
            X_full=evaluator.X_raw,
            y_full=np.asarray(qos_y),
            S_idx=idx_sel,
            eval_test_idx=evaluator.eval_test_idx,
            entity_ids=None,
            time_ids=None,
            state_labels=state_labels,
            config=qos_cfg,
        )
        row.update(qos_metrics)
    except Exception as e:
        print(f"      QoS failed: {e}")

    return row


def process_mode(
    mode_key: str,
    spaces: List[str],
    all_rows: List[Dict[str, Any]],
):
    """Process all baselines for a given constraint mode across specified spaces."""
    cfg = MODE_CONFIG[mode_key]
    dir_suffix = cfg["dir_suffix"]
    constraint_mode = cfg["constraint_mode"]

    print(f"\n{'#'*70}")
    print(f"# MODE: {mode_key} ({constraint_mode})")
    print(f"#   Soft components: {cfg['soft_components']}")
    print(f"#   preserve_group_counts: {cfg['preserve_group_counts']}")
    print(f"{'#'*70}")

    for space in spaces:
        space_prefix = {"vae": "v", "raw": "r", "pca": "p"}[space]
        baseline_dir = EXPERIMENTS_DIR / f"B_{space_prefix}_{dir_suffix}"

        if not baseline_dir.exists():
            print(f"\n  WARNING: {baseline_dir} not found, skipping")
            continue

        print(f"\n  Space: {space} (dir: {baseline_dir.name})")

        for rep_id in range(N_REPS):
            rep_str = f"rep{rep_id:02d}"
            print(f"\n  {'='*60}")
            print(f"  {rep_str} — {mode_key}/{space}")
            print(f"  {'='*60}")

            # Load cache
            t0 = time.time()
            cache = load_cache(rep_id)
            N = len(cache["state_labels"])
            print(f"    Cache loaded: N={N} ({time.time()-t0:.1f}s)")

            # Build geo + constraints + projector for this mode
            state_labels = cache["state_labels"]
            population = np.asarray(cache["population"], dtype=np.float64)
            geo, constraint_set, projector = build_geo_and_constraints(
                state_labels, population, mode_key,
            )
            soft_desc = ", ".join(
                f"tau_{c}={TAU_POP if c == 'pop' else TAU_MUNI}"
                for c in cfg["soft_components"]
            )
            print(f"    Geo: {geo.G} groups, {soft_desc}, alpha={ALPHA_GEO}")

            # Build evaluator
            t0 = time.time()
            evaluator = build_evaluator(cache, SEED)
            print(f"    Evaluator built ({time.time()-t0:.1f}s)")

            # Collect downstream targets
            extra_reg, cls_targets = collect_downstream_targets(cache)
            print(f"    Downstream targets: {len(extra_reg)} reg + {len(cls_targets)} cls")

            # Find baseline coreset files
            coreset_dir = baseline_dir / rep_str / "coresets"
            if not coreset_dir.exists():
                print(f"    WARNING: {coreset_dir} not found, skipping")
                continue

            # Process both exactk and quota baselines
            for regime in ["exactk", "quota"]:
                npz_pattern = str(coreset_dir / f"*_{space}_{regime}.npz")
                npz_files = sorted(glob.glob(npz_pattern))
                print(f"\n    --- {regime} baselines ({len(npz_files)} files) ---")

                for npz_path in npz_files:
                    fname = os.path.basename(npz_path)
                    method = fname.split(f"_{space}_{regime}")[0]

                    data = np.load(npz_path, allow_pickle=True)
                    indices_orig = data["indices"]
                    k_actual = len(indices_orig)

                    # Pre-repair KL for each soft component
                    mask_orig = np.zeros(N, dtype=bool)
                    mask_orig[indices_orig] = True
                    kl_before = {
                        c.name: c.value(mask_orig, geo)
                        for c in constraint_set.constraints
                    }

                    # Apply repair (same pipeline as NSGA-II: quota + swap)
                    rng = np.random.default_rng(SEED + rep_id)
                    t0 = time.time()
                    indices_repaired = repair_selection(
                        indices_orig, N, k_actual, projector, constraint_set, rng,
                    )
                    repair_time = time.time() - t0

                    # Post-repair KL
                    mask_rep = np.zeros(N, dtype=bool)
                    mask_rep[indices_repaired] = True
                    kl_after = {
                        c.name: c.value(mask_rep, geo)
                        for c in constraint_set.constraints
                    }

                    n_changed = int((~np.isin(indices_repaired, indices_orig)).sum())
                    kl_str = "  ".join(
                        f"{name} {kl_before[name]:.4f}->{kl_after[name]:.4f}"
                        for name in kl_before
                    )
                    print(
                        f"      {method:6s} ({regime}): "
                        f"{kl_str}  "
                        f"changed={n_changed}/{k_actual}  "
                        f"({repair_time:.2f}s)"
                    )

                    # Full evaluation of repaired coreset
                    t0 = time.time()
                    row = evaluate_coreset(
                        indices_repaired, geo, evaluator,
                        state_labels, extra_reg, cls_targets, cache, SEED,
                    )
                    eval_time = time.time() - t0

                    # Add metadata
                    row["method"] = method
                    row["constraint_mode"] = constraint_mode
                    row["constraint_mode_abbrev"] = mode_key
                    row["constraint_regime"] = f"{regime}_repaired"
                    row["space"] = space
                    row["k"] = k_actual
                    row["rep_id"] = rep_id
                    row["n_items_changed"] = n_changed
                    row["repair_time_s"] = repair_time
                    row["eval_time_s"] = eval_time

                    # Store per-constraint KL before/after
                    for cname in kl_before:
                        row[f"kl_{cname}_before_repair"] = kl_before[cname]
                        row[f"kl_{cname}_after_repair"] = kl_after[cname]

                    all_rows.append(row)
                    print(f"        evaluated ({eval_time:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc soft-KL repair for baseline coresets",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(MODE_CONFIG.keys()),
        choices=list(MODE_CONFIG.keys()),
        help="Which constraint modes to repair (default: all soft modes)",
    )
    parser.add_argument(
        "--spaces", nargs="+", default=["vae"],
        choices=["vae", "raw", "pca"],
        help="Which feature spaces to process (default: vae)",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for mode_key in args.modes:
        process_mode(mode_key, args.spaces, all_rows)

    # Save results
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = OUTPUT_DIR / "all_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to {out_path}")

        # Summary per mode
        print(f"\n{'='*70}")
        print("REPAIR SUMMARY")
        print(f"{'='*70}")
        for mode_key in df["constraint_mode_abbrev"].unique():
            cfg = MODE_CONFIG[mode_key]
            sub = df[df["constraint_mode_abbrev"] == mode_key]
            print(f"\n  {mode_key} ({cfg['constraint_mode']}):")
            for regime in sorted(sub["constraint_regime"].unique()):
                rsub = sub[sub["constraint_regime"] == regime]
                print(f"    {regime}:")
                for method in sorted(rsub["method"].unique()):
                    msub = rsub[rsub["method"] == method]
                    # Show KL before/after for each soft constraint
                    parts = []
                    for cname in cfg["soft_components"]:
                        col_b = f"kl_{cname}_before_repair"
                        col_a = f"kl_{cname}_after_repair"
                        if col_b in msub.columns:
                            parts.append(
                                f"KL_{cname} "
                                f"{msub[col_b].mean():.4f}->"
                                f"{msub[col_a].mean():.4f}"
                            )
                    nc = msub["n_items_changed"].mean()
                    kl_str = "  ".join(parts)
                    print(f"      {method:6s}: {kl_str}  avg_changed={nc:.1f}")
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    main()
