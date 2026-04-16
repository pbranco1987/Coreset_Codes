#!/usr/bin/env python
"""
Standalone coreset evaluation -- computes ALL downstream metrics from saved
coreset indices.

Architecture
------------
This script is intentionally decoupled from coreset *construction* (NSGA-II
and baseline runners).  The separation enforces a clean contract:

    Construction  -->  coreset.npz (indices only)  -->  This script  -->  metrics.csv

Keeping evaluation standalone has three benefits:

1. **Reproducibility**: metrics can be recomputed without re-running the
   optimiser.  If a metric definition changes, we re-evaluate existing
   coresets rather than re-optimising from scratch.
2. **Auditability**: a single code path for all metrics means every row in
   every results table traces back to the same functions, regardless of
   whether the coreset came from NSGA-II, random sampling, or a greedy
   baseline.
3. **Parallelism**: batch mode lets us evaluate hundreds of experiment
   directories with a single invocation, which is useful for the full
   factorial sweep (R1--R14) described in the manuscript.

Evaluation stages (per coreset)
-------------------------------
The five-stage pipeline mirrors ``_runner_eval.EvalMixin._evaluate_coreset``
but runs without any runner or config object:

    [1/5]  Geographic diagnostics  -- KL, L1, max-deviation under both
           municipality-share and population-share weighting.
    [2/5]  Raw-space operator metrics  -- Nystrom error, kernel-PCA
           distortion, KRR RMSE (4G/5G), and state-conditioned stability.
    [3/5]  KPI stability  -- per-state target-mean drift and ranking
           stability (Kendall's tau).
    [4/5]  Multi-model downstream  -- KNN, RF, LR, GBT on Nystrom features
           for extra regression/classification targets.
    [5/5]  (Reserved for QoS -- disabled in the clean pipeline.)

Inputs
------
- ``{experiment-dir}/coreset.npz`` -- saved coreset indices.  May contain
  a boolean mask matrix ``X`` (Pareto front, one row per solution), a 1-D
  ``indices`` array (single coreset), or a boolean ``mask`` array.
- ``{experiment-dir}/representatives/*.npz`` -- named Pareto selections
  (``knee.npz``, ``best-mmd.npz``, etc.), each containing ``indices``.
- ``{cache-path}/assets.npz`` -- replicate cache produced during the
  data-preparation phase (contains ``X_scaled``, eval splits, targets, state
  labels, population weights, and metadata).

Outputs
-------
- ``{experiment-dir}/metrics.csv`` -- one row per coreset (all Pareto-front
  members plus named representatives).
- ``{experiment-dir}/metrics-representatives.csv`` -- subset of metrics.csv
  containing only the named selections (knee, best-mmd, ...).

Usage examples
--------------
Evaluate a single experiment::

    python scripts/analysis/evaluate_coresets.py \\
        --experiment-dir runs_out/nsga2-vae-popsoft-k100-rep0 \\
        --cache-path replicate_cache_seed4200/rep00/assets.npz

Batch-evaluate all experiments matching a pattern::

    python scripts/analysis/evaluate_coresets.py \\
        --experiment-dir runs_out/ \\
        --cache-path replicate_cache_seed4200/rep00/assets.npz \\
        --batch --pattern "nsga2-vae-*"

Force re-evaluation (overwrite existing metrics.csv)::

    python scripts/analysis/evaluate_coresets.py \\
        --experiment-dir runs_out/ \\
        --cache-path replicate_cache_seed4200/rep00/assets.npz \\
        --batch --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project root discovery -- this script lives two levels below the repo root
# (scripts/analysis/), so we go up twice to find the package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from coreset_selection.config._dc_results import ReplicateAssets
from coreset_selection.data.cache import load_replicate_cache
from coreset_selection.evaluation.geo_diagnostics import dual_geo_diagnostics
from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
from coreset_selection.geo.info import build_geo_info


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_coreset_indices(experiment_dir: Path) -> Dict[str, np.ndarray]:
    """Load coreset indices from an experiment directory.

    Scans two locations for saved coresets:

    1. ``coreset.npz`` in the experiment root -- may use any of three
       storage conventions (see below).
    2. ``representatives/*.npz`` -- named Pareto-front selections produced
       by the NSGA-II runner (e.g., ``knee.npz``, ``best-mmd.npz``).

    Storage conventions inside ``coreset.npz``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Key ``X``: boolean mask matrix of shape ``(n_solutions, N)``.
      Each row is a binary decision vector from the Pareto front; we
      convert to integer indices via ``np.flatnonzero``.
    - Key ``indices``: 1-D integer array (single coreset, e.g., baselines).
    - Key ``mask``: 1-D boolean array (single coreset, alternative format).
    - Fallback: the first array key is inspected -- bool arrays are treated
      as masks, integer arrays as direct indices.

    Parameters
    ----------
    experiment_dir : Path
        Directory containing ``coreset.npz`` and/or ``representatives/``.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from coreset name to 1-D integer index array.  Names follow
        the convention ``pf_000``, ``pf_001``, ... for Pareto-front members
        and the stem of the .npz filename for named representatives (e.g.,
        ``knee``, ``best-mmd``, ``selection``).
    """
    coreset_path = experiment_dir / "coreset.npz"
    results: Dict[str, np.ndarray] = {}

    if coreset_path.exists():
        data = np.load(coreset_path, allow_pickle=True)

        if "X" in data:
            # ---- Pareto front: each row is a boolean decision vector ----
            # The NSGA-II runner stores the entire non-dominated set as a
            # binary matrix X of shape (n_solutions, N).  Row i is 1 where
            # municipality i is selected.
            X = data["X"]
            if X.ndim == 2:
                for i in range(X.shape[0]):
                    mask = X[i].astype(bool)
                    # Convert boolean mask -> sorted integer indices
                    results[f"pf_{i:03d}"] = np.flatnonzero(mask)
            elif X.ndim == 1:
                # Edge case: a single-solution front stored as a 1-D vector
                mask = X.astype(bool)
                results["pf_000"] = np.flatnonzero(mask)

        elif "indices" in data:
            # ---- Baseline/single coreset: indices stored directly ----
            results["selection"] = np.asarray(data["indices"], dtype=int)

        elif "mask" in data:
            # ---- Alternative single-coreset format: boolean mask ----
            results["selection"] = np.flatnonzero(
                np.asarray(data["mask"], dtype=bool)
            )
        else:
            # ---- Fallback: inspect the first available array key ----
            for key in data.files:
                arr = data[key]
                if arr.dtype == bool:
                    results[key] = np.flatnonzero(arr)
                elif np.issubdtype(arr.dtype, np.integer):
                    results[key] = arr
                break  # only use the first key

    # ------------------------------------------------------------------
    # Named representatives (produced by the NSGA-II post-processing)
    # ------------------------------------------------------------------
    # These are standalone .npz files saved under representatives/ with
    # meaningful names.  They are always evaluated in addition to the
    # full Pareto front because downstream tables report only the named
    # selections (not all front members).
    reps_dir = experiment_dir / "representatives"
    if reps_dir.is_dir():
        for npz_file in sorted(reps_dir.glob("*.npz")):
            name = npz_file.stem  # e.g., "knee", "best-mmd"
            rep_data = np.load(npz_file, allow_pickle=True)
            if "indices" in rep_data:
                results[name] = np.asarray(rep_data["indices"], dtype=int)
            elif "mask" in rep_data:
                results[name] = np.flatnonzero(
                    np.asarray(rep_data["mask"], dtype=bool)
                )
            else:
                # Same fallback logic as above
                for key in rep_data.files:
                    arr = rep_data[key]
                    if arr.dtype == bool:
                        results[name] = np.flatnonzero(arr)
                    elif np.issubdtype(arr.dtype, np.integer):
                        results[name] = arr
                    break

    return results


def load_cache_assets(cache_path: str) -> ReplicateAssets:
    """Load a replicate cache from disk.

    The replicate cache (``assets.npz``) is created during the
    data-preparation phase and contains everything needed for evaluation:
    the standardized feature matrix ``X_scaled``, evaluation-set indices
    and their train/test split, coverage targets, state labels, and
    population weights.  By loading it once here we avoid recomputing
    VAE embeddings or re-splitting the data.

    Parameters
    ----------
    cache_path : str
        Path to the replicate cache directory (the one containing
        ``assets.npz``).

    Returns
    -------
    ReplicateAssets
        Dataclass with fields ``X_scaled``, ``eval_idx``,
        ``eval_train_idx``, ``eval_test_idx``, ``y``, ``state_labels``,
        ``population``, and ``metadata``.
    """
    return load_replicate_cache(cache_path)


# ---------------------------------------------------------------------------
# Build multi-target y (matches _runner_eval._build_multitarget_y)
# ---------------------------------------------------------------------------

def build_multitarget_y(
    assets: ReplicateAssets,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Build the multi-target response matrix from replicate assets.

    The base targets (typically 4G and 5G coverage area) live in
    ``assets.y``.  Extra coverage targets (e.g., additional bands or
    QoS indicators) may be stored in ``assets.metadata["extra_targets"]``.
    This function concatenates them column-wise so that downstream
    evaluators (KRR, state-conditioned stability) receive a single
    ``(N, T)`` matrix.

    The logic intentionally mirrors ``_runner_eval._build_multitarget_y``
    to guarantee identical target ordering between the live runner and
    this standalone evaluator.

    Parameters
    ----------
    assets : ReplicateAssets
        Replicate cache containing ``y`` and ``metadata``.

    Returns
    -------
    y_multi : np.ndarray or None
        Combined target matrix of shape ``(N, T)`` where T is the total
        number of base + extra targets.  None if ``assets.y`` is None.
    target_names : list of str or None
        Human-readable column names matching the columns of ``y_multi``.
        None if ``assets.y`` is None.
    """
    y_base = assets.y
    if y_base is None:
        return None, None

    y_base = np.asarray(y_base)
    if y_base.ndim == 1:
        y_base = y_base.reshape(-1, 1)

    # Assign column names based on the number of base targets.
    # The two-column case (4G + 5G coverage area) is the standard layout.
    n_base = y_base.shape[1]
    if n_base == 2:
        base_names = ["cov_area_4G", "cov_area_5G"]
    elif n_base == 1:
        base_names = ["target"]
    else:
        base_names = [f"target_{i}" for i in range(n_base)]

    # ------------------------------------------------------------------
    # Append extra coverage targets stored in cache metadata.
    # Legacy key names (e.g., "cov_4g" vs. "cov_area_4G") are normalised
    # through the _LEGACY_TARGET_KEY_MAP so that old and new caches are
    # handled identically.
    # ------------------------------------------------------------------
    extra: Dict[str, Any] = {}
    if hasattr(assets, "metadata") and isinstance(assets.metadata, dict):
        extra = assets.metadata.get("extra_targets", {})

    if not extra:
        return y_base, base_names

    from coreset_selection.config.constants import (
        COVERAGE_TARGET_NAMES,
        _LEGACY_TARGET_KEY_MAP,
    )

    # Normalise legacy keys to canonical names
    normalized_extra: Dict[str, Any] = {}
    for k, v in extra.items():
        canonical = _LEGACY_TARGET_KEY_MAP.get(k, k)
        normalized_extra[canonical] = v

    # Preserve the canonical ordering defined in COVERAGE_TARGET_NAMES and
    # skip any target already present in the base columns.
    extra_names: List[str] = [
        k for k in COVERAGE_TARGET_NAMES
        if k in normalized_extra and k not in base_names
    ]
    if not extra_names:
        return y_base, base_names

    extra_arrays = [
        np.asarray(normalized_extra[k], dtype=np.float64) for k in extra_names
    ]
    y_multi = np.column_stack([y_base] + extra_arrays)
    return y_multi, base_names + extra_names


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_single_coreset(
    idx_sel: np.ndarray,
    *,
    assets: ReplicateAssets,
    raw_evaluator: RawSpaceEvaluator,
    geo: Any,
    k: int,
    state_labels: np.ndarray,
    extra_regression_targets: Optional[Dict[str, np.ndarray]] = None,
    classification_targets: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 4200,
) -> Dict[str, Any]:
    """Evaluate a single coreset on all downstream metrics.

    Runs the five-stage evaluation pipeline on the municipality indices
    in ``idx_sel``.  This is a standalone equivalent of
    ``_runner_eval.EvalMixin._evaluate_coreset`` -- it requires no runner
    or experiment config, only the pre-built evaluator objects.

    Index semantics
    ~~~~~~~~~~~~~~~
    ``idx_sel`` contains *absolute* row indices into the full dataset
    (shape ``(N,)``).  These are the same indices stored in ``coreset.npz``.
    Internally, each evaluator maps them to its own index space:

    - ``RawSpaceEvaluator`` intersects ``idx_sel`` with the evaluation set E
      and applies the S-cap-E exclusion (see note below).
    - ``dual_geo_diagnostics`` computes group histograms over ``idx_sel``.
    - ``state_kpi_stability`` filters ``y`` and ``state_labels`` by
      ``idx_sel`` to compute per-state means.

    The S-cap-E fix
    ~~~~~~~~~~~~~~~
    When a selected landmark falls inside the evaluation set (i.e.,
    S intersect E is non-empty), the Nystrom approximation trivially
    achieves zero error at those points, biasing all operator metrics
    downward.  ``RawSpaceEvaluator`` removes S-cap-E from E before
    computing any metric, yielding E_clean = E \\ S.  The number of
    excluded points is logged as ``n_excluded`` in the Nystrom cache
    for diagnostics.

    Parameters
    ----------
    idx_sel : np.ndarray
        1-D integer array of selected municipality indices (absolute).
    assets : ReplicateAssets
        Replicate cache (used here only for metadata access).
    raw_evaluator : RawSpaceEvaluator
        Pre-built evaluator bound to the replicate's feature matrix and
        eval splits.  Constructed via ``RawSpaceEvaluator.build()``.
    geo : GeoInfo
        Geographic group information (state labels + population weights).
    k : int
        Coreset cardinality (number of selected municipalities).
    state_labels : np.ndarray
        Per-municipality state/group labels, shape ``(N,)``.
    extra_regression_targets : dict, optional
        Additional named regression targets for multi-model evaluation.
    classification_targets : dict, optional
        Named classification targets for multi-model evaluation.
    seed : int
        Random seed for reproducibility in multi-model downstream.

    Returns
    -------
    Dict[str, Any]
        Flat dictionary of metric names to values.  Key prefixes:
        ``geo_`` for geographic, ``nystrom_`` / ``kpca_`` / ``krr_`` for
        raw-space, ``kpi_`` for stability, and model-specific prefixes
        (``knn_``, ``rf_``, ``lr_``, ``gbt_``) for multi-model downstream.
    """
    from coreset_selection.evaluation.kpi_stability import state_kpi_stability

    idx_sel = np.asarray(idx_sel, dtype=int)
    row: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # [1/5] Geographic diagnostics
    # ------------------------------------------------------------------
    # Computes KL divergence, L1 deviation, and max deviation between
    # the full-dataset geographic distribution and the coreset's
    # distribution, under both municipality-share and population-share
    # weighting (dual_geo_diagnostics returns both _muni and _pop keys).
    t0 = time.perf_counter()
    geo_all = dual_geo_diagnostics(geo, idx_sel, k, alpha=1.0)
    row.update(geo_all)
    dt_geo = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # [2/5] Nystrom + KRR + state-conditioned stability
    # ------------------------------------------------------------------
    # This single call computes:
    #   - Nystrom Gram-matrix approximation error  e_Nys(S)
    #   - Kernel PCA spectral distortion           e_kPCA(S)
    #   - KRR test RMSE for each target (4G, 5G)
    #   - Worst-group (per-state) RMSE
    #   - State-conditioned ranking stability (Kendall tau, drift)
    # All metrics use the S-cap-E exclusion so landmarks never evaluate
    # themselves (see docstring above for details).
    t0 = time.perf_counter()
    row.update(
        raw_evaluator.all_metrics_with_state_stability(idx_sel, state_labels)
    )
    dt_raw = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # [3/5] KPI stability (per-state target means)
    # ------------------------------------------------------------------
    # Measures how well the coreset preserves per-state mean coverage
    # values.  Complements the operator metrics by checking whether
    # aggregate KPIs (e.g., mean 4G coverage in Minas Gerais) are
    # faithfully represented in the subset.
    t0 = time.perf_counter()
    if raw_evaluator.y is not None:
        try:
            kpi_stab = state_kpi_stability(
                y=raw_evaluator.y,
                state_labels=state_labels,
                S_idx=idx_sel,
            )
            row.update(kpi_stab)
        except Exception:
            # Gracefully skip if KPI stability fails (e.g., too few
            # points per state in the coreset).
            pass
    dt_kpi = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # [4/5] Multi-model downstream (KNN, RF, LR, GBT)
    # ------------------------------------------------------------------
    # Trains four model families on Nystrom features derived from the
    # coreset landmarks, then evaluates on the held-out test portion of E.
    # This tests whether the Nystrom feature space induced by S is a
    # generally useful representation, not just good for KRR.
    t0 = time.perf_counter()
    if extra_regression_targets or classification_targets:
        try:
            multi_metrics = raw_evaluator.multi_model_downstream(
                S_idx=idx_sel,
                regression_targets=extra_regression_targets or {},
                classification_targets=classification_targets or {},
                seed=seed,
            )
            row.update(multi_metrics)
        except Exception as e:
            print(f"    [WARN] Multi-model failed: {e}")
    dt_multi = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # [5/5] QoS downstream (reserved, currently disabled)
    # ------------------------------------------------------------------
    # QoS evaluation is non-standard and rarely used in the clean pipeline.
    # If needed in the future, add it here following the same pattern.

    # Print per-stage timing for profiling and debugging
    dt_total = dt_geo + dt_raw + dt_kpi + dt_multi
    print(
        f"    eval: {dt_total:.1f}s "
        f"(geo={dt_geo:.1f} raw={dt_raw:.1f} kpi={dt_kpi:.1f} "
        f"multi={dt_multi:.1f})"
    )

    return row


# ---------------------------------------------------------------------------
# Evaluate an entire experiment directory
# ---------------------------------------------------------------------------

def evaluate_experiment(
    experiment_dir: Path,
    cache_path: str,
    *,
    force: bool = False,
) -> bool:
    """Evaluate all coresets in an experiment directory.

    This is the main entry point for single-experiment evaluation.  It
    loads coreset indices, builds the evaluator objects once (expensive),
    then loops over every coreset (Pareto-front members and named
    representatives), writing results to CSV.

    The evaluator objects -- ``RawSpaceEvaluator``, ``GeoInfo``, and the
    multi-target matrix -- are constructed once and shared across all
    coresets in the directory.  This is efficient because:

    - ``RawSpaceEvaluator.build()`` computes the RBF bandwidth via the
      median heuristic on E_train (O(|E_train|^2) pairwise distances)
      and pre-allocates the full Gram matrix K_EE.  Doing this once
      amortises the cost across dozens of Pareto-front members.
    - ``build_geo_info()`` constructs group histograms that are reused
      for every coreset.

    Parameters
    ----------
    experiment_dir : Path
        Directory containing ``coreset.npz`` and/or ``representatives/``.
    cache_path : str
        Path to the replicate cache (passed through to
        ``load_replicate_cache``).
    force : bool
        If True, overwrite existing ``metrics.csv``.  If False, skip
        directories that already have results.

    Returns
    -------
    bool
        True if evaluation ran successfully (or was skipped because
        metrics already exist), False if no coresets were found or
        evaluation produced no rows.
    """
    metrics_path = experiment_dir / "metrics.csv"
    if metrics_path.exists() and not force:
        print(f"  [SKIP] {experiment_dir.name} -- metrics.csv already exists")
        return True

    # ------------------------------------------------------------------
    # Load coreset indices from coreset.npz and/or representatives/
    # ------------------------------------------------------------------
    coresets = load_coreset_indices(experiment_dir)
    if not coresets:
        print(f"  [SKIP] {experiment_dir.name} -- no coreset.npz found")
        return False

    print(f"\n  [EVAL] {experiment_dir.name} -- {len(coresets)} coresets")

    # ------------------------------------------------------------------
    # Load replicate cache and extract shared data
    # ------------------------------------------------------------------
    assets = load_cache_assets(cache_path)
    state_labels: np.ndarray = assets.state_labels
    population: Optional[np.ndarray] = getattr(assets, "population", None)

    # ------------------------------------------------------------------
    # Build GeoInfo: encapsulates group labels and population weights
    # for geographic proportionality diagnostics.
    # ------------------------------------------------------------------
    geo = build_geo_info(state_labels, population_weights=population)

    # ------------------------------------------------------------------
    # Build multi-target y: combines base targets (4G, 5G) with any
    # extra coverage targets stored in cache metadata.
    # ------------------------------------------------------------------
    y_multi, target_names = build_multitarget_y(assets)

    # ------------------------------------------------------------------
    # Build RawSpaceEvaluator (the most expensive setup step)
    # ------------------------------------------------------------------
    # RawSpaceEvaluator.build() is a factory that:
    #   1. Computes the RBF bandwidth sigma^2 via the median heuristic
    #      on eval_train_idx (O(|E_train|^2) pairwise distances).
    #   2. Stores references to X_scaled, y, and the eval split indices.
    #   3. Lazily computes K_EE (the true Gram matrix on E) on first use.
    #
    # The eval splits (eval_idx / eval_train_idx / eval_test_idx) are
    # absolute indices into X_scaled.  They were fixed at cache-creation
    # time via stratified sampling to ensure every state is represented.
    raw_evaluator = RawSpaceEvaluator.build(
        X_raw=assets.X_scaled,
        y=y_multi,
        eval_idx=assets.eval_idx,
        eval_train_idx=assets.eval_train_idx,
        eval_test_idx=assets.eval_test_idx,
        seed=4200,
        target_names=target_names,
    )

    # ------------------------------------------------------------------
    # Extract optional extra targets for multi-model downstream [4/5]
    # ------------------------------------------------------------------
    meta: Dict[str, Any] = (
        assets.metadata if isinstance(assets.metadata, dict) else {}
    )
    extra_reg: Dict[str, np.ndarray] = meta.get(
        "extra_regression_targets", {}
    )
    cls_targets: Dict[str, np.ndarray] = meta.get(
        "classification_targets", {}
    )

    # Infer k from the first coreset (all coresets in a single experiment
    # directory normally share the same cardinality).
    first_key = next(iter(coresets))
    k: int = len(coresets[first_key])

    # ------------------------------------------------------------------
    # Evaluate every coreset in the directory
    # ------------------------------------------------------------------
    all_rows: List[Dict[str, Any]] = []
    rep_rows: List[Dict[str, Any]] = []

    # Named representatives are the coresets that appear in manuscript
    # tables (Table V, VI, etc.).  We track them separately so we can
    # write a smaller CSV for quick inspection.
    representative_names = {
        "knee", "best-mmd", "best-sinkhorn", "best-skl",
        "chebyshev", "selection",
    }

    for name, idx_sel in coresets.items():
        print(f"    {name} (k={len(idx_sel)})...")
        row = evaluate_single_coreset(
            idx_sel,
            assets=assets,
            raw_evaluator=raw_evaluator,
            geo=geo,
            k=len(idx_sel),
            state_labels=state_labels,
            extra_regression_targets=extra_reg,
            classification_targets=cls_targets,
            seed=4200,
        )
        # Attach identifiers so every CSV row is self-describing
        row["coreset_name"] = name
        row["k"] = len(idx_sel)
        all_rows.append(row)

        if name in representative_names:
            rep_rows.append(row)

    if not all_rows:
        print(f"  [WARN] No coresets evaluated for {experiment_dir.name}")
        return False

    # ------------------------------------------------------------------
    # Write metrics.csv (all Pareto-front members + representatives)
    # ------------------------------------------------------------------
    # Column order: identifiers first (coreset_name, k), then all metric
    # columns in sorted order for reproducible diffs.
    fieldnames: List[str] = ["coreset_name", "k"] + sorted(
        k_name
        for k_name in all_rows[0].keys()
        if k_name not in ("coreset_name", "k")
    )
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"    -> {metrics_path.name} ({len(all_rows)} rows)")

    # ------------------------------------------------------------------
    # Write metrics-representatives.csv (named selections only)
    # ------------------------------------------------------------------
    # This smaller file lets downstream tools focus on the named
    # selections (knee, best-mmd, best-sinkhorn) without filtering the
    # full Pareto front.
    if rep_rows:
        rep_path = experiment_dir / "metrics-representatives.csv"
        with open(rep_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rep_rows:
                writer.writerow(row)
        print(f"    -> {rep_path.name} ({len(rep_rows)} rows)")

    return True


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    base_dir: Path,
    cache_path: str,
    *,
    force: bool = False,
    pattern: str = "*",
) -> None:
    """Evaluate all experiment subdirectories in a base directory.

    Iterates over every subdirectory of ``base_dir`` that contains a
    ``coreset.npz`` file and calls ``evaluate_experiment`` on each.
    Directories that already have a ``metrics.csv`` are skipped unless
    ``force=True``.

    Parameters
    ----------
    base_dir : Path
        Parent directory containing multiple experiment subdirectories
        (e.g., ``runs_out/``).
    cache_path : str
        Path to the replicate cache (shared across all experiments).
    force : bool
        If True, re-evaluate even when ``metrics.csv`` already exists.
    pattern : str
        Glob pattern to filter subdirectory names (e.g., ``"nsga2-vae-*"``).
        Only directories whose name matches the pattern are processed.
    """
    # Discover experiment directories: any subdirectory containing coreset.npz
    subdirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "coreset.npz").exists()
    )

    # Apply optional name filter
    if pattern != "*":
        import fnmatch
        subdirs = [d for d in subdirs if fnmatch.fnmatch(d.name, pattern)]

    total = len(subdirs)
    success = 0
    skipped = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Batch evaluation: {total} experiments in {base_dir.name}")
    print(f"Cache: {cache_path}")
    print(f"Force: {force}")
    print(f"{'='*70}")

    for i, exp_dir in enumerate(subdirs, 1):
        print(f"\n[{i}/{total}] {exp_dir.name}")
        try:
            ok = evaluate_experiment(exp_dir, cache_path, force=force)
            if ok:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Done: {success} evaluated, {skipped} skipped, {failed} failed")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and dispatch to single or batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate coresets -- compute all downstream metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to a single experiment directory or the base directory "
             "containing multiple experiments (use with --batch).",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        required=True,
        help="Path to replicate cache assets.npz.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate all experiment subdirectories, not just one.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if metrics.csv already exists.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to filter experiment subdirectories (batch mode).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()

    if args.batch:
        evaluate_batch(
            exp_dir, args.cache_path, force=args.force, pattern=args.pattern
        )
    else:
        evaluate_experiment(exp_dir, args.cache_path, force=args.force)


if __name__ == "__main__":
    main()
