"""Time complexity analysis for coreset selection vs k.

Enhanced experiment (complementary to the manuscript) that measures:
  - Per-phase timing: quota computation, objective setup, NSGA-II selection,
    evaluation (Nystrom, kPCA, KRR), geo diagnostics
  - Baseline method times vs k
  - Theoretical complexity annotations: O(k^2), O(km), O(N*k), etc.
  - Total pipeline time scaling

The per-phase breakdown enables the time-budget pie charts and stacked-area
plots used in the manuscript supplement.

Run standalone:
    python -m coreset_selection.experiment.time_complexity --data-dir data
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import plotting/analysis helpers from the private sub-module
from ._time_plotting import (
    fit_power_law,
    annotate_complexity_fits,
    plot_time_complexity,
    _plot_phase_stacked,
    save_time_complexity_summary,
)


# ---------------------------------------------------------------------------
# Theoretical complexity annotations
# ---------------------------------------------------------------------------
# These annotations describe the dominant term for each method/phase as a
# function of N (dataset size), k (coreset size), m (RFF dimension), and D
# (feature dimension).  They are recorded alongside the empirical timings so
# that the artifact generator can overlay fitted curves.

COMPLEXITY_ANNOTATIONS: Dict[str, str] = {
    # Selection methods
    "nsga2":          r"$O(P \cdot T \cdot k)$",
    "uniform":        r"$O(k)$",
    "kmeans":         r"$O(N \cdot k \cdot D \cdot I)$",
    "herding":        r"$O(k \cdot m)$",
    "farthest_first": r"$O(N \cdot k)$",
    "kernel_thinning":r"$O(N \cdot m + k^2)$",
    "rls":            r"$O(N \cdot m)$",
    "dpp":            r"$O(N \cdot m + k^3)$",
    # Evaluation phases
    "quota_computation": r"$O(G \cdot k)$",
    "objective_setup":   r"$O(N \cdot m)$",
    "nystrom_eval":      r"$O(N \cdot k^2)$",
    "kpca_eval":         r"$O(k^3 + N \cdot k)$",
    "krr_eval":          r"$O(k^3)$",
    "geo_eval":          r"$O(G)$",
    "all_eval":          r"$O(N \cdot k^2 + k^3)$",
}


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------

def run_time_complexity_experiment(
    cfg,
    assets,
    geo,
    projector,
    constraint_set,
    seed: int = 42,
    k_grid: Optional[List[int]] = None,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Run time complexity analysis varying k with per-phase instrumentation.

    For each k in k_grid, measures wall-clock time for:
      1. **Quota computation** -- computing c*(k) from the geographic projector
      2. **Objective setup** -- building the RFF / Sinkhorn anchor cache
      3. **NSGA-II selection** -- full optimisation loop
      4. **Evaluation** -- Nystrom error, kPCA distortion, KRR RMSE (separately)
      5. **Geo diagnostics** -- KL/L1 geographic compliance
      6. **Baseline methods** -- each classical baseline independently

    Parameters
    ----------
    cfg : ExperimentConfig
        Base configuration.
    assets : ReplicateAssets
        Cached data assets.
    geo : GeoInfo
        Geographic grouping info.
    projector : GeographicConstraintProjector
        Constraint projector.
    constraint_set : ProportionalityConstraintSet or None
        Active proportionality constraints.
    seed : int
        Random seed.
    k_grid : list of int, optional
        Cardinality values to test. Default: [50, 100, 200, 300, 400, 500].
    n_repeats : int
        Number of timing repetitions per k (for averaging).

    Returns
    -------
    pd.DataFrame
        Columns: k, method, phase, time_mean_s, time_std_s, time_min_s,
        time_max_s, n_repeats, N, D, m, complexity
    """
    from ..objectives.computer import build_space_objective_computer
    from ..optimization.nsga2_internal import nsga2_optimize
    from ..optimization.selection import select_pareto_representatives
    from ..evaluation.raw_space import RawSpaceEvaluator
    from ..evaluation.geo_diagnostics import geo_diagnostics
    from ..baselines import (
        baseline_uniform,
        baseline_kmeans_reps,
        baseline_kernel_herding,
        baseline_farthest_first,
        baseline_kernel_thinning,
    )
    from ..baselines.utils import rff_features
    from ..utils.math import median_sq_dist

    if k_grid is None:
        k_grid = [50, 100, 200, 300, 400, 500]

    rows: List[Dict[str, Any]] = []
    X = np.asarray(assets.X_scaled, dtype=np.float64)
    N, D = X.shape

    # Build RFF features once (shared across k)
    sigma_sq = max(median_sq_dist(X, sample_size=2048, seed=seed) / 2.0, 1e-12)
    rff_dim = int(cfg.mmd.rff_dim)

    print(f"[TimeComplexity] N={N}, D={D}, m={rff_dim}")
    print(f"[TimeComplexity] Running time analysis for k in {k_grid}, "
          f"n_repeats={n_repeats}", flush=True)

    for k in k_grid:
        print(f"\n[TimeComplexity] --- k = {k} ---", flush=True)

        # Shared metadata injected into every row for this k
        meta = {"N": N, "D": D, "m": rff_dim}

        # ---- Phase 1: Quota computation ----
        quota_times = []
        for rep in range(n_repeats):
            t0 = time.perf_counter()
            projector.compute_quota(k)
            quota_times.append(time.perf_counter() - t0)
        rows.append(_make_time_row(k, "quota_computation", "quota", quota_times, meta))

        # ---- Phase 2: Objective setup (RFF + Sinkhorn anchors) ----
        setup_times = []
        for rep in range(n_repeats):
            t0 = time.perf_counter()
            _ = build_space_objective_computer(
                X=X,
                logvars=None,
                mmd_cfg=cfg.mmd,
                sinkhorn_cfg=cfg.sinkhorn,
                seed=seed + rep,
            )
            setup_times.append(time.perf_counter() - t0)
        rows.append(_make_time_row(k, "objective_setup", "setup", setup_times, meta))

        # ---- Phase 3: NSGA-II selection ----
        nsga_times = []
        for rep in range(n_repeats):
            # Use reduced effort for timing (don't spend hours)
            pop_size = min(cfg.solver.pop_size, 100)
            n_gen = min(cfg.solver.n_gen, 200)

            computer = build_space_objective_computer(
                X=X,
                logvars=None,
                mmd_cfg=cfg.mmd,
                sinkhorn_cfg=cfg.sinkhorn,
                seed=seed + rep,
            )

            t0 = time.perf_counter()
            nsga2_optimize(
                computer=computer,
                projector=projector,
                constraint_set=constraint_set,
                k=k,
                objectives=tuple(cfg.solver.objectives),
                pop_size=pop_size,
                n_gen=n_gen,
                crossover_prob=cfg.solver.crossover_prob,
                mutation_prob=cfg.solver.mutation_prob,
                use_quota=cfg.geo.use_quota_constraints,
                enforce_exact_k=cfg.solver.enforce_exact_k,
                seed=seed + rep + 1000,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            nsga_times.append(elapsed)

        rows.append(_make_time_row(k, "nsga2", "selection", nsga_times, meta))

        # ---- Phase 4: Baseline methods ----
        baseline_funcs = {
            "uniform": lambda kk: baseline_uniform(N, k=kk, seed=seed),
            "kmeans": lambda kk: baseline_kmeans_reps(X, k=kk, seed=seed),
            "herding": lambda kk: baseline_kernel_herding(
                X, k=kk, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed
            ),
            "farthest_first": lambda kk: baseline_farthest_first(X, k=kk, seed=seed),
        }

        # Add kernel_thinning if available
        try:
            baseline_funcs["kernel_thinning"] = lambda kk: baseline_kernel_thinning(
                X, k=kk, sigma_sq=sigma_sq, seed=seed,
                meanK_rff_dim=min(512, rff_dim), unique=True,
            )
        except Exception:
            pass

        for name, fn in baseline_funcs.items():
            times_b = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                try:
                    fn(k)
                except Exception:
                    pass
                times_b.append(time.perf_counter() - t0)
            rows.append(_make_time_row(k, name, "selection", times_b, meta))

        # ---- Phase 5: Evaluation sub-phases ----
        if cfg.eval.enabled and assets.eval_idx is not None:
            evaluator = RawSpaceEvaluator.build(
                X_raw=assets.X_scaled,
                y=assets.y,
                eval_idx=assets.eval_idx,
                eval_train_idx=assets.eval_train_idx,
                eval_test_idx=assets.eval_test_idx,
                seed=seed,
            )

            # Use a dummy selection for timing
            rng = np.random.default_rng(seed)
            dummy_idx = rng.choice(N, size=k, replace=False)

            # 5a: Full evaluation
            eval_times = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                evaluator.all_metrics(dummy_idx)
                eval_times.append(time.perf_counter() - t0)
            rows.append(_make_time_row(k, "all_eval", "eval", eval_times, meta))

            # 5b: Nystrom approximation only
            nys_times = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                evaluator.nystrom_error(dummy_idx)
                nys_times.append(time.perf_counter() - t0)
            rows.append(_make_time_row(k, "nystrom_eval", "eval", nys_times, meta))

            # 5c: kPCA distortion only
            kpca_times = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                try:
                    evaluator.kpca_distortion(dummy_idx)
                except Exception:
                    pass
                kpca_times.append(time.perf_counter() - t0)
            rows.append(_make_time_row(k, "kpca_eval", "eval", kpca_times, meta))

            # 5d: KRR only
            krr_times = []
            for rep in range(n_repeats):
                t0 = time.perf_counter()
                evaluator.krr_rmse(dummy_idx)
                krr_times.append(time.perf_counter() - t0)
            rows.append(_make_time_row(k, "krr_eval", "eval", krr_times, meta))

        # ---- Phase 6: Geo diagnostics ----
        geo_times = []
        dummy_idx = np.random.default_rng(seed).choice(N, size=k, replace=False)
        for rep in range(n_repeats):
            t0 = time.perf_counter()
            geo_diagnostics(geo, dummy_idx, k, alpha=cfg.geo.alpha_geo)
            geo_times.append(time.perf_counter() - t0)
        rows.append(_make_time_row(k, "geo_eval", "eval", geo_times, meta))

    df = pd.DataFrame(rows)
    print(f"\n[TimeComplexity] Collected {len(df)} timing rows")
    return df


def _make_time_row(
    k: int,
    method: str,
    phase: str,
    times: List[float],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one timing row with complexity annotation."""
    arr = np.array(times, dtype=np.float64)
    row = {
        "k": k,
        "method": method,
        "phase": phase,
        "time_mean_s": float(arr.mean()),
        "time_std_s": float(arr.std()),
        "time_min_s": float(arr.min()),
        "time_max_s": float(arr.max()),
        "n_repeats": len(times),
        "complexity": COMPLEXITY_ANNOTATIONS.get(method, ""),
    }
    if meta:
        row.update(meta)
    return row
