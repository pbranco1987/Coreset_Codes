#!/usr/bin/env python3
"""
Bootstrap target-variable re-evaluation of coreset quality.

Experiment-driven: takes a single --run-id (experiment directory name) and
--rep-id, reads config.json to auto-discover space, k, constraint_mode,
cache_dir, and evaluates ALL NSGA-II Pareto front solutions + 8 baselines
under B bootstrap target draws (default B=30, 5 reg + 5 cls targets).

Output safety:
- A metadata sidecar JSON is saved alongside each final CSV, recording all
  parameters (B, n_reg, n_cls, seed, methods, timing, etc.).
- If a previous output exists with different parameters, it is archived
  to ``{output_dir}/_prev/`` before proceeding — results are never
  silently overwritten or lost.
- Atomic writes + checkpointing ensure crash-safe durability.
- The CSV contains one row per (bootstrap_draw, method), with ALL
  evaluations on ALL selections (entire Pareto front + baselines) for
  ALL models and ALL B draws — nothing is summarized or discarded.

Baselines are computed on-the-fly at the experiment's exact k value using
the same representation space, kernel settings, and geographic constraints.

Usage:
    python bootstrap_reeval.py --run-id K_vae_k100 --rep-id 0
    python bootstrap_reeval.py --run-id N_v_ph --rep-id 2
    python bootstrap_reeval.py --run-id A_mmd --rep-id 0 --n-bootstrap 30
    python bootstrap_reeval.py --run-id K_vae_k100 --rep-id 0 --n-reg 5 --n-cls 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project imports
from coreset_selection.evaluation.bootstrap_targets import (
    BootstrapSample,
    build_target_pools,
    generate_bootstrap_samples,
    build_bootstrap_features,
)
from coreset_selection.evaluation._raw_kernels import _rbf_kernel, _median_sq_dist
from coreset_selection.evaluation._raw_nystrom import _nystrom_components, _nystrom_features
from coreset_selection.evaluation.multi_model_evaluator import evaluate_all_downstream_models
from coreset_selection.experiment.saver import load_pareto_front

# Baseline imports
from coreset_selection.baselines import (
    baseline_uniform,
    baseline_kmeans_reps,
    baseline_kernel_herding,
    baseline_farthest_first,
    baseline_rls,
    baseline_dpp,
    baseline_kernel_thinning,
    baseline_kkmeans_nystrom,
)
from coreset_selection.baselines.utils import rff_features
from coreset_selection.utils.math import median_sq_dist
from coreset_selection.geo import build_geo_info, GeographicConstraintProjector


# ============================================================================
# Constants
# ============================================================================

BASELINE_METHODS = ["U", "KM", "KH", "FF", "RLS", "DPP", "KT", "KKN"]

BASELINE_DISPLAY = {
    "U": "Uniform", "KM": "k-Medoids", "KH": "Kernel Herding",
    "FF": "FastForward", "RLS": "RLS-Nystrom", "DPP": "k-DPP",
    "KT": "Kernel Thinning", "KKN": "KKN",
}

SPACE_CODE = {"vae": "v", "raw": "r", "pca": "p"}

# Representation key in assets.npz for each space
SPACE_TO_ASSET_KEY = {"vae": "Z_vae", "raw": "X_scaled", "pca": "Z_pca"}

# Constraint mode → projector weight_type (from runner.py lines 188-195)
CONSTRAINT_TO_WEIGHT_TYPE = {
    "population_share": "pop",
    "population_share_quota": "pop",
    "joint_hard_soft": "pop",
    "municipality_share": "muni",
    "municipality_share_quota": "muni",
    "joint": "muni",
    "joint_soft_hard": "muni",
    "none": "muni",  # fallback (no constraints applied anyway)
}

# Constraint mode → B_ directory suffix
CONSTRAINT_TO_BASELINE_SUFFIX = {
    "population_share": "ps",
    "population_share_quota": "ph",
    "municipality_share": "ms",
    "municipality_share_quota": "mh",
    "joint": "hh",
    "joint_hard_soft": "hs",
    "joint_soft_hard": "sh",
    "none": "0",
}


# ============================================================================
# Experiment config reader
# ============================================================================

def read_experiment_config(
    experiments_dir: str, run_id: str, rep_id: int,
) -> Dict:
    """Read config.json for an experiment and extract key settings."""
    config_path = os.path.join(
        experiments_dir, run_id, f"rep{rep_id:02d}", "config.json"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    space = cfg["space"]
    k = cfg["solver"]["k"]
    constraint_mode = cfg.get("geo", {}).get("constraint_mode", "population_share")
    alpha_geo = cfg.get("geo", {}).get("alpha_geo", 1.0)
    min_one_per_group = cfg.get("geo", {}).get("min_one_per_group", True)
    seed = cfg.get("seed", 2026)
    cache_dir = cfg.get("files", {}).get("cache_dir", "replicate_cache_seed2026")
    objectives = cfg.get("solver", {}).get("objectives", ["mmd", "sinkhorn"])

    # Normalize cache_dir: configs may contain absolute paths from another
    # server (e.g. /home/jupyter-pbranco/...).  If the stored path doesn't
    # exist locally, resolve the cache directory name relative to the
    # project root (parent of experiments_dir).
    if not os.path.isdir(cache_dir):
        project_root = os.path.dirname(os.path.abspath(experiments_dir))
        cache_name = os.path.basename(cache_dir.rstrip("/"))
        local_candidate = os.path.join(project_root, cache_name)
        if os.path.isdir(local_candidate):
            cache_dir = local_candidate

    return {
        "space": space,
        "k": k,
        "constraint_mode": constraint_mode,
        "alpha_geo": float(alpha_geo),
        "min_one_per_group": bool(min_one_per_group),
        "seed": int(seed),
        "cache_dir": cache_dir,
        "objectives": objectives,
    }


# ============================================================================
# Loading utilities
# ============================================================================

def _resolve_base_cache(cache_dir: str) -> str:
    """Derive the base replicate cache from a dim-specific cache path.

    Dimension-sweep caches (e.g. ``replicate_cache_seed2026_pca_d8``) only
    contain ``assets.npz`` — ``splits.npz`` lives in the base cache
    (``replicate_cache_seed2026``).  Strip the ``_{space}_d{dim}`` suffix
    to find the base directory.
    """
    import re
    base = re.sub(r"_(vae|pca|raw)_d\d+$", "", cache_dir)
    if base != cache_dir and os.path.isdir(base):
        return base
    return cache_dir


def load_splits(cache_dir: str, rep_id: int) -> Dict[str, np.ndarray]:
    """Load eval train/test split indices from replicate cache."""
    inner = os.path.join(cache_dir, os.path.basename(cache_dir))
    if os.path.isdir(inner):
        cache_dir = inner

    splits_path = os.path.join(cache_dir, f"rep{rep_id:02d}", "splits.npz")

    # Dimension-specific caches may not contain splits — fall back to
    # the base cache (e.g. replicate_cache_seed2026).
    if not os.path.isfile(splits_path):
        base_cache = _resolve_base_cache(cache_dir)
        if base_cache != cache_dir:
            inner_base = os.path.join(base_cache, os.path.basename(base_cache))
            if os.path.isdir(inner_base):
                base_cache = inner_base
            splits_path = os.path.join(base_cache, f"rep{rep_id:02d}", "splits.npz")

    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"Splits not found: {splits_path}")

    data = np.load(splits_path, allow_pickle=True)
    return {
        "eval_idx": data["eval_idx"],
        "eval_train_idx": data["eval_train_idx"],
        "eval_test_idx": data["eval_test_idx"],
    }


def load_representation(
    cache_dir: str, rep_id: int, space: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load representation + geo data from replicate cache."""
    inner = os.path.join(cache_dir, os.path.basename(cache_dir))
    if os.path.isdir(inner):
        cache_dir = inner

    assets_path = os.path.join(cache_dir, f"rep{rep_id:02d}", "assets.npz")
    if not os.path.isfile(assets_path):
        raise FileNotFoundError(f"Assets not found: {assets_path}")

    assets = np.load(assets_path, allow_pickle=True)

    key = SPACE_TO_ASSET_KEY[space]
    X_repr = assets[key].astype(np.float64)
    state_labels = assets["state_labels"]
    population = assets["population"].astype(np.float64)

    return X_repr, state_labels, population


# ============================================================================
# On-the-fly baseline computation
# ============================================================================

def compute_baselines(
    X_repr: np.ndarray,
    state_labels: np.ndarray,
    population: np.ndarray,
    k: int,
    space: str,
    constraint_mode: str = "population_share",
    seed: int = 2026,
    rff_dim: int = 2000,
    alpha_geo: float = 1.0,
    min_one_per_group: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute all 8 baseline indices using the exactk regime.

    Replicates the exact logic from BaselineVariantGenerator._build_exact_methods
    + project_to_exact_k_mask, using the MATCHING constraint_mode.
    """
    n = X_repr.shape[0]

    # Determine weight_type from constraint_mode
    weight_type = CONSTRAINT_TO_WEIGHT_TYPE.get(constraint_mode, "muni")

    # Build GeoInfo and projector (matching NSGA-II experiment)
    geo = build_geo_info(
        state_labels=state_labels,
        population_weights=population,
    )

    use_projector = (constraint_mode != "none")
    projector = None
    if use_projector:
        projector = GeographicConstraintProjector(
            geo=geo,
            alpha_geo=alpha_geo,
            min_one_per_group=min_one_per_group,
            weight_type=weight_type,
        )

    # Bandwidth via median heuristic
    sigma_sq = median_sq_dist(X_repr, sample_size=2048, seed=seed) / 2.0
    sigma_sq = float(max(sigma_sq, 1e-12))

    # Pre-compute RFF features (shared by kernel-based methods)
    Phi = rff_features(X_repr, m=rff_dim, sigma_sq=sigma_sq, seed=seed + 17)
    mean_phi = Phi.mean(axis=0)
    meanK_approx = Phi @ mean_phi

    rng = np.random.default_rng(seed)

    # Compute each baseline (same seed offsets as variant_generator)
    raw_methods = {
        "U":   lambda: baseline_uniform(n, k=k, seed=seed + 1),
        "KM":  lambda: baseline_kmeans_reps(X_repr, k=k, seed=seed + 2),
        "KH":  lambda: baseline_kernel_herding(
                    X_repr, k=k, sigma_sq=sigma_sq,
                    rff_dim=rff_dim, seed=seed + 3),
        "FF":  lambda: baseline_farthest_first(X_repr, k=k, seed=seed + 4),
        "RLS": lambda: baseline_rls(
                    X_repr, k=k, sigma_sq=sigma_sq,
                    rff_dim=rff_dim, seed=seed + 5),
        "DPP": lambda: baseline_dpp(
                    X_repr, k=k, sigma_sq=sigma_sq,
                    rff_dim=rff_dim, seed=seed + 6),
        "KT":  lambda: baseline_kernel_thinning(
                    X_repr, k=k, sigma_sq=sigma_sq, seed=seed + 7,
                    meanK=meanK_approx, meanK_rff_dim=rff_dim, unique=True),
        "KKN": lambda: baseline_kkmeans_nystrom(
                    X_repr, k=k, seed=seed + 8, sigma_sq=sigma_sq),
    }

    indices: Dict[str, np.ndarray] = {}

    for bm, fn in raw_methods.items():
        t0 = time.time()
        try:
            sel = np.asarray(fn(), dtype=int)

            if use_projector and projector is not None:
                # Apply exactk projection (enforces exact k + min per group)
                mask = np.zeros(n, dtype=bool)
                mask[sel] = True
                mask = projector.project_to_exact_k_mask(mask, k=k, rng=rng)
                sel = np.flatnonzero(mask)
            else:
                # No constraints — just take first k if needed
                if len(sel) > k:
                    sel = sel[:k]

            if len(sel) == k:
                indices[bm] = sel
                dt = time.time() - t0
                print(f"      {bm} ({BASELINE_DISPLAY[bm]}): "
                      f"computed {k} indices ({dt:.1f}s)")
            else:
                print(f"      {bm}: wrong size {len(sel)} != {k}, skipping")
        except Exception as e:
            print(f"      {bm}: FAILED ({e})")

    return indices


def save_baseline_indices(
    indices: Dict[str, np.ndarray],
    cache_path: str,
    run_id: str,
) -> None:
    """Cache computed baseline indices to disk."""
    os.makedirs(cache_path, exist_ok=True)
    for bm, idx in indices.items():
        out = os.path.join(cache_path, f"{bm}_exactk.npz")
        np.savez_compressed(out, indices=idx)


def load_cached_baselines(
    cache_path: str,
    k: int,
) -> Dict[str, np.ndarray]:
    """Try to load previously cached baseline indices."""
    indices: Dict[str, np.ndarray] = {}
    if not os.path.isdir(cache_path):
        return indices
    for bm in BASELINE_METHODS:
        npz_path = os.path.join(cache_path, f"{bm}_exactk.npz")
        if os.path.isfile(npz_path):
            try:
                data = np.load(npz_path)
                idx = data["indices"]
                if len(idx) == k:
                    indices[bm] = idx
            except Exception:
                pass
    return indices


def load_all_method_indices(
    experiments_dir: str,
    run_id: str,
    rep_id: int,
    exp_cfg: Dict,
) -> Dict[str, np.ndarray]:
    """Load coreset indices for ALL methods (NSGA-II + baselines).

    NSGA-II: loads from {run_id}/rep{NN}/results/{space}_pareto.npz.
    Baselines: tries pre-computed B_{code}_{suffix} first, then cached
    on-the-fly indices, then computes on-the-fly if needed.
    """
    space = exp_cfg["space"]
    k = exp_cfg["k"]
    constraint_mode = exp_cfg["constraint_mode"]
    cache_dir = exp_cfg["cache_dir"]
    seed = exp_cfg["seed"]
    alpha_geo = exp_cfg["alpha_geo"]
    min_one_per_group = exp_cfg["min_one_per_group"]

    indices: Dict[str, np.ndarray] = {}

    # 1. NSGA-II — load ALL solutions from Pareto front
    pareto_path = os.path.join(
        experiments_dir, run_id,
        f"rep{rep_id:02d}", "results", f"{space}_pareto.npz"
    )
    rep_map: Dict[int, str] = {}  # front index → representative name
    if os.path.isfile(pareto_path):
        try:
            data = np.load(pareto_path, allow_pickle=True)
            X_front = data["X"]  # (n_solutions, N) binary masks
            # Build rep_index → name mapping from stored representatives
            rep_names = [str(n) for n in data.get("rep_names", [])]
            rep_indices = list(data.get("rep_indices", []))
            for rname, ridx in zip(rep_names, rep_indices):
                rep_map[int(ridx)] = rname
            # Load ALL solutions on the front
            n_front = X_front.shape[0]
            for i in range(n_front):
                sol_idx = np.flatnonzero(X_front[i])
                if len(sol_idx) != k:
                    continue
                # Use representative name if this is a named rep,
                # otherwise use pf_XX
                if i in rep_map:
                    indices[rep_map[i]] = sol_idx
                else:
                    indices[f"pf_{i:02d}"] = sol_idx
            print(f"    Loaded {len(indices)} front solutions "
                  f"(reps: {rep_map})")
        except Exception as e:
            print(f"  [warn] Failed to load pareto {pareto_path}: {e}")

    if not indices:
        return indices  # No NSGA-II results, skip

    # 2. Baselines — try pre-computed from matching B_ directory
    space_code = SPACE_CODE.get(space, space[0])
    baseline_suffix = CONSTRAINT_TO_BASELINE_SUFFIX.get(constraint_mode, "ps")
    baseline_dir = os.path.join(
        experiments_dir, f"B_{space_code}_{baseline_suffix}",
        f"rep{rep_id:02d}", "coresets"
    )
    for bm in BASELINE_METHODS:
        npz_path = os.path.join(baseline_dir, f"{bm}_{space}_exactk.npz")
        if os.path.isfile(npz_path):
            try:
                data = np.load(npz_path)
                idx = data["indices"]
                if len(idx) == k:
                    indices[bm] = idx
            except Exception:
                pass

    # 3. Check cached on-the-fly baselines
    onthefly_cache = os.path.join(
        experiments_dir, "bootstrap_baselines",
        run_id, f"rep{rep_id:02d}"
    )
    missing = [bm for bm in BASELINE_METHODS if bm not in indices]
    if missing:
        cached = load_cached_baselines(onthefly_cache, k)
        for bm, idx in cached.items():
            if bm not in indices:
                indices[bm] = idx
                missing = [m for m in missing if m != bm]

    # 4. Compute remaining baselines on-the-fly
    if missing:
        print(f"    Computing {len(missing)} baselines on-the-fly "
              f"(k={k}, space={space}, constraint={constraint_mode})...")
        try:
            X_repr, state_labels, population = load_representation(
                cache_dir, rep_id, space
            )
            computed = compute_baselines(
                X_repr, state_labels, population,
                k=k, space=space,
                constraint_mode=constraint_mode,
                seed=seed,
                alpha_geo=alpha_geo,
                min_one_per_group=min_one_per_group,
            )
            # Cache for reuse
            save_baseline_indices(computed, onthefly_cache, run_id)
            for bm, idx in computed.items():
                if bm not in indices:
                    indices[bm] = idx
        except Exception as e:
            print(f"    [warn] Baseline computation failed: {e}")

    return indices


# ============================================================================
# Core evaluation loop
# ============================================================================

def _checkpoint_path(output_dir: str, run_id: str, rep_id: int) -> str:
    """Path for checkpoint file tracking completed bootstrap draws."""
    return os.path.join(
        output_dir, f".ckpt_{run_id}_rep{rep_id:02d}.json"
    )


def _partial_csv_path(output_dir: str, run_id: str, rep_id: int) -> str:
    """Path for partial CSV (rows appended incrementally)."""
    return os.path.join(
        output_dir, f".partial_{run_id}_rep{rep_id:02d}.csv"
    )


def load_checkpoint(
    output_dir: str, run_id: str, rep_id: int,
) -> Tuple[set, Optional[Dict[str, Any]]]:
    """Load completed boot_ids and config from checkpoint file.

    Returns
    -------
    completed_boots : set
        Set of completed boot_id integers.
    config : dict or None
        The ``config`` dict stored in the checkpoint (contains n_bootstrap,
        n_reg, n_cls, seed).  None if missing or unreadable.
    """
    ckpt = _checkpoint_path(output_dir, run_id, rep_id)
    if not os.path.isfile(ckpt):
        return set(), None
    try:
        with open(ckpt) as f:
            data = json.load(f)
        boots = set(data.get("completed_boots", []))
        cfg = data.get("config", None)
        return boots, cfg
    except Exception:
        return set(), None


def save_checkpoint(
    output_dir: str, run_id: str, rep_id: int,
    completed_boots: set, fieldnames: List[str],
    *,
    n_bootstrap: int,
    n_reg: int,
    n_cls: int,
    seed: int,
) -> None:
    """Save checkpoint atomically (write tmp → fsync → rename).

    Stores the config parameters alongside completed draws so that a
    subsequent resume can detect configuration mismatches and avoid
    mixing rows from different configurations.
    """
    ckpt = _checkpoint_path(output_dir, run_id, rep_id)
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({
                "completed_boots": sorted(completed_boots),
                "fieldnames": fieldnames,
                "run_id": run_id,
                "rep_id": rep_id,
                "config": {
                    "n_bootstrap": n_bootstrap,
                    "n_reg": n_reg,
                    "n_cls": n_cls,
                    "seed": seed,
                },
            }, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ckpt)          # atomic on Linux
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_rows_to_partial(
    output_dir: str, run_id: str, rep_id: int,
    rows: List[Dict], fieldnames: List[str],
) -> None:
    """Append rows to partial CSV (with fsync to guarantee durability)."""
    partial = _partial_csv_path(output_dir, run_id, rep_id)
    write_header = not os.path.isfile(partial)
    with open(partial, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            restval="", extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def finalize_output(
    output_dir: str, run_id: str, rep_id: int,
    n_bootstrap: int = 0, n_methods: int = 0,
) -> Optional[str]:
    """Deduplicate, validate, verify, and move partial CSV → final output.

    Safety net for power-loss scenarios:
    1. Corrupted rows (truncated by power-loss mid-append) are discarded
       — a row is valid only if both ``boot_id`` and ``method`` are non-empty.
    2. Duplicate (boot_id, method) rows (from checkpoint lag) are deduped
       — last occurrence wins.
    3. Final CSV is written atomically (tmp → fsync → rename).
    4. Partial + checkpoint are NOT deleted here — caller is responsible
       for cleanup after the metadata sidecar has been safely written.
    """
    partial = _partial_csv_path(output_dir, run_id, rep_id)
    output_csv = os.path.join(
        output_dir, f"bootstrap_raw_{run_id}_rep{rep_id:02d}.csv"
    )

    if not os.path.isfile(partial):
        return None

    # ── Read all rows ──
    with open(partial, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # ── Discard corrupted / truncated rows ──
    # A power loss mid-append can leave a partially-written last line.
    # Such rows will have empty or missing boot_id / method fields.
    valid_rows: list = []
    n_corrupt = 0
    for row in all_rows:
        bid = (row.get("boot_id") or "").strip()
        method = (row.get("method") or "").strip()
        if bid and method:
            valid_rows.append(row)
        else:
            n_corrupt += 1
    if n_corrupt > 0:
        print(f"  [finalize] Discarded {n_corrupt} corrupted/truncated rows")

    # ── Deduplicate on (boot_id, method) ──
    seen: dict = {}   # (boot_id, method) -> row
    for row in valid_rows:
        key = (row["boot_id"], row["method"])
        seen[key] = row  # last occurrence wins

    deduped = list(seen.values())
    n_removed = len(valid_rows) - len(deduped)
    if n_removed > 0:
        print(f"  [finalize] Removed {n_removed} duplicate rows "
              f"({len(valid_rows)} → {len(deduped)})")

    # ── Verify expected row count ──
    if n_bootstrap > 0 and n_methods > 0:
        expected = n_bootstrap * n_methods
        if len(deduped) != expected:
            print(f"  [WARNING] Expected {expected} rows "
                  f"({n_bootstrap} draws × {n_methods} methods) "
                  f"but got {len(deduped)}")

    # ── Write deduplicated CSV to temp → fsync → atomic rename ──
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames,
                restval="", extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(deduped)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_csv)     # atomic on Linux
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    # NOTE: partial + checkpoint are NOT deleted here.
    # Caller must call cleanup_intermediate_files() after metadata is saved
    # to avoid losing the ability to detect incomplete finalization on
    # power loss between CSV write and metadata write.
    return output_csv


def cleanup_intermediate_files(
    output_dir: str, run_id: str, rep_id: int,
) -> None:
    """Delete checkpoint and partial files after finalization + metadata save.

    Call this ONLY after both the final CSV and metadata sidecar have been
    safely written to disk.  This ordering ensures that a power loss at any
    point leaves the system in a recoverable state.
    """
    for path in [
        _partial_csv_path(output_dir, run_id, rep_id),
        _checkpoint_path(output_dir, run_id, rep_id),
    ]:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


def _meta_path(output_dir: str, run_id: str, rep_id: int) -> str:
    """Path for the metadata sidecar JSON accompanying the final CSV."""
    return os.path.join(
        output_dir, f"bootstrap_meta_{run_id}_rep{rep_id:02d}.json"
    )


def save_metadata(
    output_dir: str,
    run_id: str,
    rep_id: int,
    *,
    exp_cfg: Dict[str, Any],
    n_bootstrap: int,
    n_reg: int,
    n_cls: int,
    seed: int,
    methods: List[str],
    n_pareto_front: int,
    n_baselines: int,
    total_rows: int,
    elapsed_s: float,
    csv_path: str,
) -> str:
    """Save a metadata sidecar JSON alongside the final CSV.

    Captures all parameters needed to interpret the results and detect
    configuration mismatches on subsequent runs.
    """
    meta = {
        "run_id": run_id,
        "rep_id": rep_id,
        "space": exp_cfg.get("space"),
        "k": exp_cfg.get("k"),
        "constraint_mode": exp_cfg.get("constraint_mode"),
        "n_bootstrap": n_bootstrap,
        "n_reg": n_reg,
        "n_cls": n_cls,
        "seed": seed,
        "n_methods": len(methods),
        "methods": sorted(methods),
        "n_pareto_front": n_pareto_front,
        "n_baselines": n_baselines,
        "total_rows": total_rows,
        "csv_file": os.path.basename(csv_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "elapsed_s": round(elapsed_s, 1),
    }

    meta_file = _meta_path(output_dir, run_id, rep_id)
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix=".meta.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, meta_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return meta_file


def load_metadata(
    output_dir: str, run_id: str, rep_id: int,
) -> Optional[Dict[str, Any]]:
    """Load metadata sidecar if it exists."""
    meta_file = _meta_path(output_dir, run_id, rep_id)
    if not os.path.isfile(meta_file):
        return None
    try:
        with open(meta_file) as f:
            return json.load(f)
    except Exception:
        return None


def _backup_existing_output(
    output_dir: str, run_id: str, rep_id: int,
) -> None:
    """Back up an existing final CSV + metadata to a timestamped archive.

    Moves files to ``{output_dir}/_prev/{run_id}_rep{NN}_{timestamp}/``
    so nothing is ever lost.
    """
    csv_path = os.path.join(
        output_dir, f"bootstrap_raw_{run_id}_rep{rep_id:02d}.csv"
    )
    meta_file = _meta_path(output_dir, run_id, rep_id)

    if not os.path.isfile(csv_path):
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(
        output_dir, "_prev",
        f"{run_id}_rep{rep_id:02d}_{ts}",
    )
    os.makedirs(archive_dir, exist_ok=True)

    shutil.move(csv_path, os.path.join(archive_dir, os.path.basename(csv_path)))
    print(f"  [backup] Moved existing CSV to {archive_dir}/")

    if os.path.isfile(meta_file):
        shutil.move(meta_file, os.path.join(archive_dir, os.path.basename(meta_file)))

    # Also move checkpoint/partial if they exist (stale from old run)
    for pattern in [
        _checkpoint_path(output_dir, run_id, rep_id),
        _partial_csv_path(output_dir, run_id, rep_id),
    ]:
        if os.path.isfile(pattern):
            shutil.move(pattern, os.path.join(archive_dir, os.path.basename(pattern)))


def _backup_stale_checkpoint(
    output_dir: str, run_id: str, rep_id: int,
) -> None:
    """Back up stale checkpoint + partial files when no final CSV exists.

    This handles the case where a previous run crashed mid-way with
    different config, leaving orphaned .ckpt_ and .partial_ files.
    """
    ckpt = _checkpoint_path(output_dir, run_id, rep_id)
    partial = _partial_csv_path(output_dir, run_id, rep_id)

    has_any = os.path.isfile(ckpt) or os.path.isfile(partial)
    if not has_any:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(
        output_dir, "_prev",
        f"{run_id}_rep{rep_id:02d}_stale_{ts}",
    )
    os.makedirs(archive_dir, exist_ok=True)

    for path in [ckpt, partial]:
        if os.path.isfile(path):
            shutil.move(
                path, os.path.join(archive_dir, os.path.basename(path))
            )

    print(f"  [safety] Archived stale checkpoint/partial to {archive_dir}/")


def _check_existing_and_handle(
    output_dir: str,
    run_id: str,
    rep_id: int,
    n_bootstrap: int,
    n_reg: int,
    n_cls: int,
    seed: int,
) -> bool:
    """Check if final output already exists and decide whether to skip or back up.

    Returns True if the run should be skipped (output exists and matches).
    Returns False if the run should proceed (no output or old output backed up).
    """
    csv_path = os.path.join(
        output_dir, f"bootstrap_raw_{run_id}_rep{rep_id:02d}.csv"
    )
    if not os.path.isfile(csv_path):
        return False

    # Final CSV exists — check metadata to see if params match
    meta = load_metadata(output_dir, run_id, rep_id)
    if meta is None:
        # No metadata.  Two possibilities:
        # (a) Power died between finalize_output and save_metadata — the CSV
        #     is valid but the checkpoint/partial may still exist alongside it.
        #     Check if a checkpoint exists with matching config → trust the CSV.
        # (b) Legacy file from an older run.  Back up and re-run.
        _, ckpt_cfg = load_checkpoint(output_dir, run_id, rep_id)
        if ckpt_cfg is not None:
            ckpt_match = (
                ckpt_cfg.get("n_bootstrap") == n_bootstrap
                and ckpt_cfg.get("n_reg") == n_reg
                and ckpt_cfg.get("n_cls") == n_cls
                and ckpt_cfg.get("seed") == seed
            )
            if ckpt_match:
                # Power died between finalize_output and save_metadata.
                # The CSV is likely valid, but metadata was never written.
                # Do NOT return True here — the dispatcher requires both
                # CSV + metadata to exist.  Instead, return False to let
                # the normal flow resume: checkpoint has all draws done,
                # for-loop skips everything, finalize re-writes CSV,
                # and save_metadata finally writes the missing sidecar.
                # Crucially, do NOT clean up intermediates — the
                # checkpoint is needed for the resume to work.
                print(
                    f"  [recovery] Final CSV exists without metadata but "
                    f"checkpoint config matches — resuming to write metadata"
                )
                return False

        print(f"  [safety] Found existing CSV without metadata — backing up")
        _backup_existing_output(output_dir, run_id, rep_id)
        return False

    # Compare key parameters
    params_match = (
        meta.get("n_bootstrap") == n_bootstrap
        and meta.get("n_reg") == n_reg
        and meta.get("n_cls") == n_cls
        and meta.get("seed") == seed
    )

    if params_match:
        print(f"  [skip] Final output already exists with matching params: {csv_path}")
        return True

    # Params differ — back up old results and re-run with new params
    print(
        f"  [safety] Existing output has different params "
        f"(B={meta.get('n_bootstrap')}, r={meta.get('n_reg')}, "
        f"c={meta.get('n_cls')}, seed={meta.get('seed')}) "
        f"vs requested (B={n_bootstrap}, r={n_reg}, c={n_cls}, seed={seed})"
    )
    _backup_existing_output(output_dir, run_id, rep_id)
    return False


def run_bootstrap_evaluation(
    *,
    data_dir: str,
    experiments_dir: str,
    output_dir: str,
    run_id: str,
    rep_id: int,
    n_bootstrap: int,
    n_reg: int,
    n_cls: int,
    seed: int,
) -> bool:
    """Run bootstrap target-variable evaluation for a single (run_id, rep_id).

    Checkpoint system: after each bootstrap draw, rows are appended to a
    partial CSV and the completed draw ID is saved to a checkpoint file.
    On restart, completed draws are skipped automatically.

    Output safety: a metadata sidecar JSON is saved alongside the final CSV.
    If a previous output exists with different parameters, it is archived
    to ``{output_dir}/_prev/`` before proceeding.  Results are never
    silently overwritten.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_csv_path = os.path.join(data_dir, "smp_main.csv")
    if not os.path.isfile(raw_csv_path):
        print(f"[ERROR] Raw CSV not found: {raw_csv_path}")
        sys.exit(1)

    # ── Step 0: Read experiment config ──
    print("=" * 70)
    print(f"  BOOTSTRAP RE-EVALUATION: {run_id} rep{rep_id:02d}")
    print(f"  B={n_bootstrap}, n_reg={n_reg}, n_cls={n_cls}, seed={seed}")
    print("=" * 70)
    t0 = time.time()

    exp_cfg = read_experiment_config(experiments_dir, run_id, rep_id)
    space = exp_cfg["space"]
    k = exp_cfg["k"]
    constraint_mode = exp_cfg["constraint_mode"]
    cache_dir = exp_cfg["cache_dir"]

    print(f"  space={space}, k={k}, constraint={constraint_mode}")
    print(f"  cache_dir={cache_dir}")

    # Final output file — unique per (run_id, rep_id)
    output_csv = os.path.join(
        output_dir,
        f"bootstrap_raw_{run_id}_rep{rep_id:02d}.csv"
    )

    # Check existing output: skip if matching, back up if params differ
    if _check_existing_and_handle(
        output_dir, run_id, rep_id,
        n_bootstrap, n_reg, n_cls, seed,
    ):
        return True   # already complete — report success to dispatcher

    # Load checkpoint — which draws are already done?
    completed_boots, ckpt_cfg = load_checkpoint(output_dir, run_id, rep_id)
    if completed_boots:
        # Validate that checkpoint config matches current params
        if ckpt_cfg is not None:
            cfg_match = (
                ckpt_cfg.get("n_bootstrap") == n_bootstrap
                and ckpt_cfg.get("n_reg") == n_reg
                and ckpt_cfg.get("n_cls") == n_cls
                and ckpt_cfg.get("seed") == seed
            )
            if not cfg_match:
                print(
                    f"  [safety] Stale checkpoint has different config "
                    f"(B={ckpt_cfg.get('n_bootstrap')}, r={ckpt_cfg.get('n_reg')}, "
                    f"c={ckpt_cfg.get('n_cls')}, seed={ckpt_cfg.get('seed')}) "
                    f"vs current (B={n_bootstrap}, r={n_reg}, c={n_cls}, seed={seed})"
                )
                _backup_stale_checkpoint(output_dir, run_id, rep_id)
                completed_boots = set()
        else:
            # Legacy checkpoint without config — cannot trust it
            print(f"  [safety] Stale checkpoint without config metadata — backing up")
            _backup_stale_checkpoint(output_dir, run_id, rep_id)
            completed_boots = set()

        if completed_boots:
            print(f"  [checkpoint] Resuming: {len(completed_boots)}/{n_bootstrap} "
                  f"draws already completed")

    if not completed_boots:
        # No valid checkpoint — ensure no orphaned partial CSV exists.
        # This catches the case where power died during the very first draw
        # (after partial append but before checkpoint write).  An orphaned
        # partial may have a different header (from a different config) which
        # would silently drop new columns via DictWriter(extrasaction="ignore").
        partial_path = _partial_csv_path(output_dir, run_id, rep_id)
        if os.path.isfile(partial_path):
            print(f"  [safety] Orphaned partial CSV without valid checkpoint — backing up")
            _backup_stale_checkpoint(output_dir, run_id, rep_id)

    # ── Step 1: Build target pools ──
    reg_pool, cls_pool, substantive_cols = build_target_pools(raw_csv_path)
    print(f"[bootstrap] Target pools: {len(reg_pool)} reg, {len(cls_pool)} cls "
          f"({time.time() - t0:.1f}s)")

    # ── Step 2: Load raw CSV and standardize ──
    print("[bootstrap] Loading feature matrix...", flush=True)
    df = pd.read_csv(raw_csv_path)
    N = len(df)

    X_full_raw = df[substantive_cols].values.astype(np.float64)
    for j in range(X_full_raw.shape[1]):
        col_vals = X_full_raw[:, j]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            X_full_raw[nan_mask, j] = np.nanmedian(col_vals)

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full_raw).astype(np.float64)
    print(f"[bootstrap] X_full: {X_full.shape}")

    # ── Step 3: Generate bootstrap samples ──
    print(f"[bootstrap] Generating {n_bootstrap} bootstrap samples...",
          flush=True)
    samples = generate_bootstrap_samples(
        n_bootstrap=n_bootstrap,
        n_reg=n_reg,
        n_cls=n_cls,
        reg_pool=reg_pool,
        cls_pool=cls_pool,
        all_feature_cols=substantive_cols,
        seed=seed,
    )

    for b in range(min(3, len(samples))):
        s = samples[b]
        print(f"  Boot {b}: {len(s.reg_targets)} reg + "
              f"{len(s.cls_targets)} cls, "
              f"{len(s.excluded_cols)} excluded cols")

    # ── Step 4: Load ALL method indices ──
    print(f"\n[bootstrap] Loading method indices...", flush=True)
    methods = load_all_method_indices(
        experiments_dir, run_id, rep_id, exp_cfg,
    )
    if not methods:
        print(f"[ERROR] No method indices found for {run_id} rep{rep_id:02d}")
        sys.exit(1)

    nsga_keys = [m for m in methods if m not in BASELINE_METHODS]
    bl_keys = [m for m in methods if m in BASELINE_METHODS]
    pf_keys = [m for m in nsga_keys if m.startswith("pf_")]
    rep_keys = [m for m in nsga_keys if not m.startswith("pf_")]
    print(f"  NSGA-II front: {len(pf_keys)} unnamed + {rep_keys} = {len(nsga_keys)} total")
    print(f"  Baselines: {bl_keys}")
    print(f"  Total: {len(methods)} methods")

    # ── Step 5: Load train/test splits ──
    try:
        splits = load_splits(cache_dir, rep_id)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    eval_idx = splits["eval_idx"]
    eval_train_idx = splits["eval_train_idx"]
    eval_test_idx = splits["eval_test_idx"]

    pos_map = np.full(N, -1, dtype=int)
    pos_map[eval_idx] = np.arange(len(eval_idx))
    tr_pos = pos_map[eval_train_idx]
    te_pos = pos_map[eval_test_idx]
    tr_pos = tr_pos[tr_pos >= 0]
    te_pos = te_pos[te_pos >= 0]

    n_methods = len(methods)
    remaining = n_bootstrap - len(completed_boots)
    print(f"\n  {n_methods} methods x {remaining} remaining draws",
          flush=True)

    # ── Step 6: Evaluate with per-draw checkpointing ──
    # Pre-compute ALL possible fieldnames across all bootstrap samples
    # (each draw samples different targets → different metric column names)
    from coreset_selection.evaluation.multi_model_evaluator import (
        _regression_models, _classification_models,
    )
    reg_model_names = list(_regression_models(seed).keys())
    cls_model_names = list(_classification_models(seed).keys())
    reg_metrics = ["rmse", "mae", "r2"]
    cls_metrics = ["accuracy", "bal_accuracy", "macro_f1"]

    # Collect ALL target names across ALL bootstrap samples
    all_reg_targets: set = set()
    all_cls_targets: set = set()
    for s in samples:
        all_reg_targets.update(s.reg_targets)
        all_cls_targets.update(s.cls_targets)

    # Build complete metric column set
    metric_cols: set = set()
    for t in sorted(all_reg_targets):
        for m in reg_model_names:
            for met in reg_metrics:
                metric_cols.add(f"{m}_{met}_{t}")
    for t in sorted(all_cls_targets):
        for m in cls_model_names:
            for met in cls_metrics:
                metric_cols.add(f"{m}_{met}_{t}")

    priority = [
        "boot_id", "run_id", "space", "k", "constraint_mode",
        "rep_id", "method", "n_features", "n_excluded", "sigma_sq",
        "eval_time_s", "reg_targets", "cls_targets",
    ]
    fieldnames = priority + sorted(metric_cols)
    print(f"  Pre-computed {len(fieldnames)} columns "
          f"({len(all_reg_targets)} reg + {len(all_cls_targets)} cls unique targets)")

    for b_idx, sample in enumerate(samples):
        # Skip already completed draws
        if sample.boot_id in completed_boots:
            continue

        t_draw = time.time()

        X_eval, kept_names = build_bootstrap_features(
            X_full, substantive_cols, sample.excluded_cols
        )
        n_features = len(kept_names)

        X_E = X_eval[eval_idx]
        med_d2 = _median_sq_dist(X_E, seed=seed)
        sigma_sq = max(float(med_d2) / 2.0, 1e-12)

        boot_reg_targets: Dict[str, np.ndarray] = {
            t: reg_pool[t] for t in sample.reg_targets
        }
        boot_cls_targets: Dict[str, np.ndarray] = {
            t: cls_pool[t] for t in sample.cls_targets
        }

        draw_rows: List[Dict] = []

        for method_name, S_idx in methods.items():
            t_method = time.time()

            X_S = X_eval[S_idx]
            C, W, lam = _nystrom_components(
                X_E, X_S, sigma_sq
            )
            Phi = _nystrom_features(C, W, lam)

            Phi_train = Phi[tr_pos]
            Phi_test = Phi[te_pos]

            metrics = evaluate_all_downstream_models(
                Phi_train=Phi_train,
                Phi_test=Phi_test,
                eval_train_idx=eval_train_idx,
                eval_test_idx=eval_test_idx,
                regression_targets=boot_reg_targets,
                classification_targets=boot_cls_targets,
                seed=seed,
            )

            dt_m = time.time() - t_method

            row = {
                "boot_id": sample.boot_id,
                "run_id": run_id,
                "space": space,
                "k": k,
                "constraint_mode": constraint_mode,
                "rep_id": rep_id,
                "method": method_name,
                "n_features": n_features,
                "n_excluded": len(sample.excluded_cols),
                "sigma_sq": sigma_sq,
                "eval_time_s": round(dt_m, 2),
            }
            row.update(metrics)
            row["reg_targets"] = ";".join(sample.reg_targets)
            row["cls_targets"] = ";".join(sample.cls_targets)

            draw_rows.append(row)

        # Append rows to partial CSV + update checkpoint
        if draw_rows:
            append_rows_to_partial(
                output_dir, run_id, rep_id, draw_rows, fieldnames
            )
            completed_boots.add(sample.boot_id)
            save_checkpoint(
                output_dir, run_id, rep_id, completed_boots, fieldnames,
                n_bootstrap=n_bootstrap, n_reg=n_reg,
                n_cls=n_cls, seed=seed,
            )

        dt_draw = time.time() - t_draw
        n_done = len(completed_boots)
        if n_done % 5 == 0 or n_done == 1 or n_done == n_bootstrap:
            print(
                f"    boot {n_done}/{n_bootstrap} "
                f"({n_features} feat, {n_methods} methods, "
                f"{dt_draw:.1f}s) [checkpointed]",
                flush=True,
            )

    # ── Finalize: deduplicate, verify, rename partial → final ──
    final_path = finalize_output(
        output_dir, run_id, rep_id,
        n_bootstrap=n_bootstrap, n_methods=n_methods,
    )

    total_time = time.time() - t0

    if final_path:
        # Count rows and methods for the log
        with open(final_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_methods_total = len(set(r["method"] for r in rows))
        n_boots_total = len(set(r["boot_id"] for r in rows))
        print(f"\n  [saved] {final_path} "
              f"({len(rows)} rows = {n_boots_total} draws × "
              f"{n_methods_total} methods)")

        # Save metadata sidecar JSON
        method_names = sorted(methods.keys())
        n_pf = len([m for m in method_names if m not in BASELINE_METHODS])
        n_bl = len([m for m in method_names if m in BASELINE_METHODS])

        meta_file = save_metadata(
            output_dir, run_id, rep_id,
            exp_cfg=exp_cfg,
            n_bootstrap=n_bootstrap,
            n_reg=n_reg,
            n_cls=n_cls,
            seed=seed,
            methods=method_names,
            n_pareto_front=n_pf,
            n_baselines=n_bl,
            total_rows=len(rows),
            elapsed_s=total_time,
            csv_path=final_path,
        )
        print(f"  [meta] {meta_file}")

        # Only now — AFTER both CSV and metadata are safely on disk —
        # delete the intermediate checkpoint + partial files.
        cleanup_intermediate_files(output_dir, run_id, rep_id)

    # ── Mandatory pre-exit verification ──
    # Before declaring success, verify that all output files are present,
    # internally consistent, and safe from overwrites.  This is a hard
    # gate: if verification fails, the function returns False so the CLI
    # can exit with a non-zero code.  The dispatcher uses this to decide
    # whether to mark the job as complete or re-queue it.
    ok = verify_final_output(
        output_dir, run_id, rep_id,
        n_bootstrap=n_bootstrap, n_methods=n_methods,
        n_reg=n_reg, n_cls=n_cls, seed=seed,
    )

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    if ok:
        print(f"  DONE (VERIFIED): {run_id} rep{rep_id:02d} ({total_time / 60:.1f} min)")
    else:
        print(f"  DONE (VERIFICATION FAILED): {run_id} rep{rep_id:02d} ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")
    return ok


# ============================================================================
# Pre-exit output verification
# ============================================================================

def verify_final_output(
    output_dir: str,
    run_id: str,
    rep_id: int,
    *,
    n_bootstrap: int,
    n_methods: int,
    n_reg: int,
    n_cls: int,
    seed: int,
) -> bool:
    """Verify that all output files are present, consistent, and complete.

    This is called as the LAST step before a job declares success.
    The dispatcher uses the return value (reflected in the exit code)
    to decide whether to mark the job as complete or re-queue it.

    Checks performed:
    1. Final CSV exists and is readable.
    2. Metadata sidecar JSON exists and is readable.
    3. Metadata params match the current run's params.
    4. Row count == n_bootstrap * n_methods.
    5. All boot_ids 0..B-1 are present in the CSV.
    6. No intermediate files remain (partial, checkpoint).
    """
    csv_path = os.path.join(
        output_dir, f"bootstrap_raw_{run_id}_rep{rep_id:02d}.csv"
    )
    meta_file = _meta_path(output_dir, run_id, rep_id)
    partial = _partial_csv_path(output_dir, run_id, rep_id)
    ckpt = _checkpoint_path(output_dir, run_id, rep_id)

    errors: List[str] = []

    # 1. Final CSV exists and is readable
    if not os.path.isfile(csv_path):
        errors.append(f"Final CSV missing: {csv_path}")
    else:
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            errors.append(f"Final CSV unreadable: {e}")
            rows = []

        if rows:
            # 4. Row count
            expected_rows = n_bootstrap * n_methods
            if len(rows) != expected_rows:
                errors.append(
                    f"Row count mismatch: got {len(rows)}, "
                    f"expected {expected_rows} ({n_bootstrap} x {n_methods})"
                )

            # 5. All boot_ids present
            boot_ids = set(int(r["boot_id"]) for r in rows if r.get("boot_id"))
            expected_boots = set(range(n_bootstrap))
            missing_boots = expected_boots - boot_ids
            if missing_boots:
                errors.append(
                    f"Missing boot_ids: {sorted(missing_boots)} "
                    f"({len(missing_boots)} of {n_bootstrap})"
                )

            # Check all rows have non-empty method
            empty_method = sum(1 for r in rows if not (r.get("method") or "").strip())
            if empty_method:
                errors.append(f"{empty_method} rows with empty method field")

    # 2. Metadata sidecar exists and is readable
    if not os.path.isfile(meta_file):
        errors.append(f"Metadata sidecar missing: {meta_file}")
    else:
        meta = load_metadata(output_dir, run_id, rep_id)
        if meta is None:
            errors.append("Metadata sidecar unreadable")
        else:
            # 3. Metadata params match
            if meta.get("n_bootstrap") != n_bootstrap:
                errors.append(
                    f"Metadata n_bootstrap={meta.get('n_bootstrap')} "
                    f"!= expected {n_bootstrap}"
                )
            if meta.get("n_reg") != n_reg:
                errors.append(
                    f"Metadata n_reg={meta.get('n_reg')} != expected {n_reg}"
                )
            if meta.get("n_cls") != n_cls:
                errors.append(
                    f"Metadata n_cls={meta.get('n_cls')} != expected {n_cls}"
                )
            if meta.get("seed") != seed:
                errors.append(
                    f"Metadata seed={meta.get('seed')} != expected {seed}"
                )

    # 6. No stale intermediate files
    if os.path.isfile(partial):
        errors.append(f"Stale partial CSV still exists: {partial}")
    if os.path.isfile(ckpt):
        errors.append(f"Stale checkpoint still exists: {ckpt}")

    if errors:
        print(f"\n  [VERIFY FAILED] {len(errors)} issue(s):")
        for e in errors:
            print(f"    - {e}")
        return False

    print(f"  [VERIFY OK] CSV + metadata present, "
          f"{n_bootstrap}x{n_methods}={n_bootstrap * n_methods} rows, "
          f"all boot_ids 0..{n_bootstrap - 1} present")
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap target-variable re-evaluation "
                    "(experiment-driven, one run_id + rep_id per invocation)."
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Experiment directory name (e.g., K_vae_k100, N_v_ph, A_mmd)",
    )
    parser.add_argument(
        "--rep-id", type=int, required=True,
        help="Replicate ID (0-4)",
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Path to data directory containing smp_main.csv",
    )
    parser.add_argument(
        "--experiments-dir", default="experiments_v2/",
        help="Path to experiments_v2 output directory",
    )
    parser.add_argument(
        "--output-dir", default="bootstrap_results/",
        help="Output directory for bootstrap result CSVs",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=30,
        help="Number of bootstrap draws (B). Default: 30",
    )
    parser.add_argument(
        "--n-reg", type=int, default=5,
        help="Number of regression targets per draw. Default: 5",
    )
    parser.add_argument(
        "--n-cls", type=int, default=5,
        help="Number of classification targets per draw. Default: 5",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap sampling",
    )

    args = parser.parse_args()

    ok = run_bootstrap_evaluation(
        data_dir=args.data_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        rep_id=args.rep_id,
        n_bootstrap=args.n_bootstrap,
        n_reg=args.n_reg,
        n_cls=args.n_cls,
        seed=args.seed,
    )
    # Exit code 0 = verified success, 1 = verification failed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
