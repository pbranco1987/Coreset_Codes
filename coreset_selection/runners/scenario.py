#!/usr/bin/env python
"""
Standalone script for running a single experiment scenario (R0-R12) independently.

This script is designed to enable truly parallel execution of experiment configurations.
Each scenario can be run in a separate process, on a separate machine, or in a separate
container without any coordination with other scenarios.

Usage:
    # Run R1 scenario with all default k values and replicates
    python -m coreset_selection.run_scenario R1 --data-dir /path/to/data

    # Run R3 with specific k values and replicates
    python -m coreset_selection.run_scenario R3 --k-values 100,300 --rep-ids 0,1,2

    # Run all replicates but only k=300
    python -m coreset_selection.run_scenario R1 --k-values 300

    # Run R12 effort sweep with custom pop sizes and generation counts
    python -m coreset_selection.run_scenario R12 --effort-pop-sizes 50,100,200 --effort-n-gens 100,500

    # Run with specific output directory for this scenario
    python -m coreset_selection.run_scenario R2 --output-dir runs_out/R2

Key Features for Independent Execution:
- Self-contained: Each scenario builds/augments its own cache as needed
- Thread-safe caching: Uses file locks to prevent concurrent cache corruption
- Independent output: Results can be written to scenario-specific directories
- No inter-scenario dependencies during execution (except R11 which requires R1 outputs)
"""

from __future__ import annotations

# =============================================================================
# CRITICAL: Set thread limits BEFORE importing torch (via any other module)
#
# We need to parse --parallel-experiments FIRST to calculate thread limits,
# but we can't import argparse's heavy dependencies. So we do minimal parsing.
# =============================================================================
import os
import sys

# Import helper functions from the private sub-module
from ._scenario_helpers import (
    _get_parallel_experiments_from_argv,
    _get_cpu_count,
    parse_int_list,
)

# Calculate thread limit
_parallel = _get_parallel_experiments_from_argv()
if _parallel and _parallel > 0:
    _n_threads = max(1, _get_cpu_count() // _parallel)
else:
    # Default: check env var or use 2 threads per experiment
    _n_threads = int(os.environ.get("CORESET_NUM_THREADS", "2"))

# Set environment variables
os.environ["OMP_NUM_THREADS"] = str(_n_threads)
os.environ["MKL_NUM_THREADS"] = str(_n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(_n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(_n_threads)
os.environ["NUMEXPR_MAX_THREADS"] = str(_n_threads)

# MKL / OpenMP threading behaviour
# - MKL_THREADING_LAYER=GNU: use GNU OpenMP (libgomp) so MKL and PyTorch share
#   the same thread pool rather than spawning a second one.
# - OMP_MAX_ACTIVE_LEVELS=1: disable nested parallelism so the thread count
#   stays exactly at _n_threads.
# - MKL_DYNAMIC=TRUE (default): allow MKL to use fewer threads when the matrix
#   is small enough that the threading overhead exceeds the benefit.  With the
#   small matrices in this workload (N=5569, D=621) this is often the case.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

# IMPORTANT: do NOT set OMP_PROC_BIND / OMP_PLACES / KMP_AFFINITY.
#
# These pin each process's threads to specific cores starting from core 0.
# When multiple experiment processes run simultaneously they ALL bind to the
# same first _n_threads cores, causing massive contention while the remaining
# cores sit idle.  Without these, the OS scheduler distributes threads across
# all available cores, which is exactly what we want for independent processes.
#
# Only safe to set affinity when a single process owns all cores (or when an
# external job scheduler such as SLURM/cgroups handles core assignment).

# Intel OpenMP hint: keep worker threads briefly spinning between parallel
# regions (~200 ms) rather than sleeping immediately (KMP_BLOCKTIME=0).
# This avoids the sleep/wake overhead for the many short BLAS calls in the
# NSGA objective evaluation loop and VAE training.
os.environ.setdefault("KMP_BLOCKTIME", "200")

# Report what we configured
if _parallel:
    print(f"[threads] Auto-configured: {_n_threads} threads ({_get_cpu_count()} cores / {_parallel} parallel experiments)", flush=True)
else:
    print(f"[threads] Using {_n_threads} threads (default)", flush=True)
# =============================================================================

import argparse
import time
from dataclasses import replace
from typing import Dict, List, Optional

from ..cli import build_base_config
from ..config.dataclasses import ExperimentConfig
from ..config.run_specs import get_run_specs, apply_run_spec, K_GRID, D_GRID, RunSpec
from ..data.cache import ensure_replicate_cache, prebuild_full_cache
from ..experiment.runner import run_single_experiment
from ..experiment.saver import claim_next_rep_id, is_rep_complete, scan_existing_reps
from ..utils.debug_timing import timer, DEBUG_ENABLED


def run_scenario_standalone(
    run_id: str,
    *,
    k_values: Optional[List[int]] = None,
    rep_ids: Optional[List[int]] = None,
    n_replicates: Optional[int] = None,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    fail_fast: bool = False,
    verbose: bool = True,
    # Thread control for parallel execution
    num_threads: Optional[int] = None,
    parallel_experiments: Optional[int] = None,
    # R6-specific options
    source_run: str = "R1",
    source_space: str = "vae",
    source_k: int = None,
    # Cache control
    force_rebuild_cache: bool = False,
    # Resume control
    resume: bool = False,
) -> dict:
    """
    Run a single experiment scenario (R0-R14) as a standalone job.

    This function is the recommended entry point for parallel execution.
    Each scenario can be launched in a separate process or machine.

    **Auto-increment behaviour** (default): when neither ``rep_ids`` nor
    ``n_replicates`` is specified, the runner scans the output directory
    for existing ``repNN`` folders and starts from the next available ID.
    Re-running the same command therefore *appends* new replicates instead
    of overwriting previous ones.

    Parameters
    ----------
    run_id : str
        Scenario identifier (R0, R1, ..., R14)
    k_values : Optional[List[int]]
        Override k values for the sweep. If None, uses values from RunSpec.
    rep_ids : Optional[List[int]]
        Specific replicate IDs to run.  Takes precedence over n_replicates
        and auto-increment.  **Warning**: specifying an existing ID will
        overwrite those results.
    n_replicates : Optional[int]
        Number of *new* replicates to add (auto-incremented from the next
        available ID).  If None, uses default from RunSpec (typically 1).
        Ignored if rep_ids is specified.
    data_dir : str
        Directory containing input data files
    output_dir : str
        Output directory for results
    cache_dir : str
        Directory for replicate caches (shared across scenarios)
    seed : int
        Base random seed
    device : str
        Compute device ('cpu' or 'cuda')
    fail_fast : bool
        Stop on first failure
    verbose : bool
        Print progress messages
    num_threads : Optional[int]
        Number of CPU threads per experiment. For parallel execution, set to
        (total_cores / num_experiments). If None, auto-detects.
    parallel_experiments : Optional[int]
        Number of experiments running in parallel. If specified, threads are
        auto-calculated as (total_cores / parallel_experiments).
    source_run : str
        (R6 only) Source run base ID
    source_space : str
        (R6 only) Source space: vae|pca|raw
    source_k : int
        (R6 only) k value for source run
    resume : bool
        Resume mode: skip completed ``(run_name, rep)`` combinations,
        re-run incomplete ones (reusing the same rep ID and seed), and
        only create new rep IDs when the target replicate count has not
        been reached.  Preserves seed and cache consistency.

    Returns
    -------
    dict
        Summary with keys: run_id, n_completed, n_failed, elapsed_seconds, results

    Examples
    --------
    # First run  -> creates rep00
    run_scenario_standalone("R1", data_dir="data")

    # Second run -> auto-creates rep01
    run_scenario_standalone("R1", data_dir="data")

    # Add 3 new replicates at once (e.g. rep02, rep03, rep04)
    run_scenario_standalone("R1", data_dir="data", n_replicates=3)

    # Explicit rep_ids (overwrites if they exist)
    run_scenario_standalone("R1", data_dir="data", rep_ids=[0, 1])
    """
    start_time = time.time()

    run_specs = get_run_specs()
    if run_id not in run_specs:
        raise ValueError(f"Unknown run ID: {run_id}. Valid IDs: {sorted(run_specs.keys())}")

    spec = run_specs[run_id]

    # Build base configuration
    base_cfg = build_base_config(
        output_dir=output_dir,
        cache_dir=cache_dir,
        data_dir=data_dir,
        seed=seed,
        device=device,
    )

    # Determine k values to sweep
    if k_values is None:
        if spec.sweep_k is not None:
            k_values = list(spec.sweep_k)
        elif spec.k is not None:
            k_values = [int(spec.k)]
        else:
            raise ValueError(
                f"No k value specified for {run_id}. "
                f"Pass --k-values on the command line (e.g., --k-values 300)."
            )

    # ------------------------------------------------------------------
    # Determine replicate mode
    # ------------------------------------------------------------------
    # Priority: explicit rep_ids > n_replicates > spec.get_n_reps_for_k(k)
    #
    # When no rep_ids are given, we AUTO-INCREMENT: scan the output
    # directory for existing repNN folders and start from the next
    # available ID.  This means re-running the same command always
    # appends new replicates instead of overwriting.
    #
    # When n_replicates is None (not explicitly set by CLI), we respect
    # per-k overrides via spec.get_n_reps_for_k(k).
    # ------------------------------------------------------------------
    if rep_ids is not None:
        # Explicit IDs - use as-is for every k value
        _explicit_rep_ids = rep_ids
        _cli_n_replicates = None        # sentinel: not in auto mode
    else:
        _explicit_rep_ids = None        # sentinel: auto-increment mode
        # If user explicitly passed --n-replicates, honour that flat override;
        # otherwise defer to per-k logic (resolved later in the k-loop).
        _cli_n_replicates = int(n_replicates) if n_replicates is not None else None

    # R6 special handling: depends on outputs from prior runs
    if run_id == "R6":
        # Use source_k if explicitly provided, otherwise keep the
        # k_values already determined above (e.g. from --k-values).
        if source_k is not None:
            k_values = [source_k]
        os.environ["CORESET_R6_SOURCE_RUN"] = source_run
        os.environ["CORESET_R6_SOURCE_SPACE"] = source_space
        os.environ["CORESET_R6_K"] = str(k_values[0])

        # Validate that dependent runs have completed for the specific k value
        effective_k = k_values[0]
        missing_deps = []
        for dep_run in spec.depends_on_runs:
            # Check if output directory exists for the dependency at effective_k
            dep_dir = os.path.join(output_dir, f"{dep_run}_k{effective_k}")
            dep_dir_alt = os.path.join(output_dir, dep_run)
            if not os.path.exists(dep_dir) and not os.path.exists(dep_dir_alt):
                missing_deps.append(f"{dep_run} (k={effective_k})")

        if missing_deps:
            print(f"\n{'='*60}")
            print(f"ERROR: R6 depends on runs that have not been completed!")
            print(f"{'='*60}")
            print(f"  Missing dependencies: {missing_deps}")
            print(f"  Looked in: {output_dir}/")
            print(f"\nPlease run the following first:")
            for dep_run in spec.depends_on_runs:
                print(f"  python -m coreset_selection.run_scenario {dep_run} --data-dir {data_dir} --k-values {effective_k}")
            print(f"{'='*60}\n")
            return {
                "run_id": run_id,
                "status": "SKIPPED",
                "reason": f"Missing dependencies: {missing_deps}",
            }

    # ------------------------------------------------------------------
    # Progress estimation
    # ------------------------------------------------------------------
    # Build a mapping from k -> number of replicates for that k.
    def _n_reps_for(k: int) -> int:
        if _explicit_rep_ids is not None:
            return len(_explicit_rep_ids)
        if _cli_n_replicates is not None:
            return _cli_n_replicates
        return spec.get_n_reps_for_k(k)

    n_dims = len(spec.sweep_dim) if spec.sweep_dim else 1
    n_total = sum(_n_reps_for(k) for k in k_values) * n_dims
    n_completed = 0
    n_failed = 0
    results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {run_id}")
        print(f"{'='*60}")
        print(f"  Description: {spec.description}")
        if spec.sweep_k is not None:
            print(f"  k values: {k_values}  (sweep)")
        elif len(k_values) == 1:
            print(f"  k value: {k_values[0]}  (fixed)")
        else:
            print(f"  k values: {k_values}")
        if spec.sweep_dim:
            print(f"  Dimension sweep: D in {{{', '.join(str(d) for d in spec.sweep_dim)}}}")
        if _explicit_rep_ids is not None:
            print(f"  Replicates (explicit): {_explicit_rep_ids}")
        else:
            reps_detail = {k: _n_reps_for(k) for k in k_values}
            if len(set(reps_detail.values())) == 1:
                print(f"  New replicates per k: {list(reps_detail.values())[0]} (auto-incremented)")
            else:
                print(f"  New replicates per k: {reps_detail} (auto-incremented)")
        print(f"  Total runs: {n_total}")
        print(f"  Output dir: {output_dir}")
        print(f"  Cache dir: {cache_dir}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Dimension sweep (R13/R14): if sweep_dim is set, iterate over
    # dimension values as an outer loop.  Each D gets its own cache
    # directory (because VAE/PCA embeddings have different shapes).
    # For non-sweep scenarios, dim_values = [None] and the loop
    # executes once with no dimension override.
    # ------------------------------------------------------------------
    dim_values: List[Optional[int]] = (
        [int(d) for d in spec.sweep_dim] if spec.sweep_dim else [None]
    )

    for dim in dim_values:
        # ---- Build dimension-specific base config ----
        if dim is not None:
            space_tag = str(spec.space)  # "vae" or "pca"
            dim_cache_dir = f"{cache_dir}_{space_tag}_d{dim}"
            dim_base_cfg = replace(
                base_cfg,
                files=replace(base_cfg.files, cache_dir=dim_cache_dir),
            )
            # Set the correct VAE latent_dim or PCA n_components.
            # Zero out the unused representation so the cache builder
            # doesn't waste time training a model that won't be used
            # (e.g. no VAE training for a PCA-only sweep).
            if space_tag == "vae":
                dim_base_cfg = replace(
                    dim_base_cfg,
                    vae=replace(dim_base_cfg.vae, latent_dim=int(dim)),
                    pca=replace(dim_base_cfg.pca, n_components=0),
                )
            elif space_tag == "pca":
                dim_base_cfg = replace(
                    dim_base_cfg,
                    pca=replace(dim_base_cfg.pca, n_components=int(dim)),
                    vae=replace(dim_base_cfg.vae, epochs=0),
                )
            dim_suffix = f"_d{dim}"
            if verbose:
                print(f"\n[{run_id}] === Dimension D={dim} ({space_tag}) ===")
        else:
            dim_base_cfg = base_cfg
            dim_suffix = ""

        # --------------------------------------------------------------
        # PHASE 1: Collect all rep_ids and PRE-BUILD caches
        # --------------------------------------------------------------
        # Determine all rep_ids across all k values FIRST so we can
        # pre-build caches once per replicate before running experiments.
        # This guarantees that every experiment configuration (different k,
        # objectives, constraints) uses the EXACT SAME VAE/PCA
        # representation for a given replicate -- essential for fair
        # cross-experiment comparison.
        # --------------------------------------------------------------
        k_to_rep_ids: Dict[int, List[int]] = {}
        n_skipped = 0   # track reps skipped by resume
        for k in k_values:
            # Always include _k{k} suffix so that parallel single-k
            # launches (e.g. k=100 and k=300) never collide.
            run_name = f"{run_id}_k{k}{dim_suffix}"
            n_reps_per_k = _n_reps_for(k)
            if _explicit_rep_ids is not None:
                k_to_rep_ids[k] = list(_explicit_rep_ids)
            elif resume:
                # ----------------------------------------------------------
                # Resume mode: scan existing reps, skip complete, reuse
                # incomplete, create new only if target count not reached.
                # ----------------------------------------------------------
                existing = scan_existing_reps(output_dir, run_name)
                complete = [r for r in existing
                            if is_rep_complete(output_dir, run_name, r)]
                incomplete = [r for r in existing
                              if not is_rep_complete(output_dir, run_name, r)]
                needed = n_reps_per_k - len(complete)

                if needed <= 0:
                    # All target reps are done for this k
                    if verbose:
                        print(f"[resume] {run_name}: "
                              f"{len(complete)}/{n_reps_per_k} complete, "
                              f"skipping")
                    k_to_rep_ids[k] = []
                    n_skipped += len(complete)
                else:
                    # Reuse incomplete reps first, then create new ones
                    reps_to_run = incomplete[:needed]
                    still_needed = needed - len(reps_to_run)
                    for _ in range(still_needed):
                        reps_to_run.append(
                            claim_next_rep_id(output_dir, run_name))
                    k_to_rep_ids[k] = reps_to_run
                    n_skipped += len(complete)
                    if verbose:
                        print(f"[resume] {run_name}: "
                              f"{len(complete)} done, "
                              f"{len(incomplete)} incomplete, "
                              f"{still_needed} new "
                              f"→ running {len(reps_to_run)}")
            else:
                k_to_rep_ids[k] = [
                    claim_next_rep_id(output_dir, run_name)
                    for _ in range(n_reps_per_k)
                ]

        # Unique rep_ids across all k values (usually the same set).
        all_rep_ids = sorted(set(r for reps in k_to_rep_ids.values() for r in reps))

        # Adjust total progress count when resume skipped some reps.
        if resume and n_skipped > 0:
            n_total = max(0, n_total - n_skipped)
            if verbose:
                print(f"[resume] Skipped {n_skipped} already-complete reps "
                      f"(remaining: {n_total})")

        if spec.cache_build_mode != "skip" and all_rep_ids:
            if verbose:
                dim_desc = f" (D={dim})" if dim is not None else ""
                print(f"\n[{run_id}] Pre-building caches for replicates{dim_desc}: {all_rep_ids}")

            if dim is not None:
                # Dimension sweep: seed dimension-specific caches from the
                # base cache (which Phase 1 / prep already built).  This
                # avoids re-doing data loading, preprocessing, splits, etc.
                # — only the representation at the new dimension is trained.
                from ..data.cache import _seed_dim_cache
                for rep in all_rep_ids:
                    _seed_dim_cache(
                        base_cache_dir=cache_dir,
                        dim_cache_dir=dim_cache_dir,
                        rep_id=rep,
                        space_tag=space_tag,
                    )
                    rep_cfg = replace(
                        dim_base_cfg,
                        rep_id=int(rep),
                        seed=int(seed + rep),
                    )
                    ensure_replicate_cache(rep_cfg, rep)
            else:
                # Standard: use prebuild_full_cache for the shared cache
                for rep in all_rep_ids:
                    prebuild_full_cache(dim_base_cfg, rep, seed=seed, force_rebuild=force_rebuild_cache)

            if verbose:
                print(f"[{run_id}] Cache pre-build complete.\n")

        # --------------------------------------------------------------
        # PHASE 2: Run experiments (caches are guaranteed to exist)
        # --------------------------------------------------------------
        for k in k_values:
            # Always include _k{k} suffix (mirrors Phase 1 logic).
            run_name = f"{run_id}_k{k}{dim_suffix}"
            current_rep_ids = k_to_rep_ids[k]

            for rep in current_rep_ids:
                # Resume guard: double-check completion (another process
                # may have finished this rep between Phase 1 and Phase 2).
                if resume and is_rep_complete(output_dir, run_name, rep):
                    if verbose:
                        print(f"[resume] Skipping {run_name} rep={rep} "
                              f"(already complete)")
                    n_completed += 1
                    continue

                if verbose:
                    print(f"[{run_id}] Running {run_name} rep={rep} ({n_completed + n_failed + 1}/{n_total})...")

                try:
                    # Apply run spec and configure
                    cfg = apply_run_spec(dim_base_cfg, spec, rep, dim_override=dim)
                    cfg = replace(
                        cfg,
                        run_id=run_name,
                        rep_id=int(rep),
                        seed=int(seed + rep),
                        solver=replace(cfg.solver, k=int(k)),
                    )

                    # NOTE: Cache was pre-built in Phase 1.  The call to
                    # ensure_replicate_cache inside run_single_experiment will
                    # hit the fast path (all keys already present).

                    # Run experiment
                    result = run_single_experiment(cfg)
                    results.append(result)
                    n_completed += 1

                    if verbose:
                        print(f"[{run_id}] Completed: {run_name} rep={rep}")

                except Exception as e:
                    n_failed += 1
                    if verbose:
                        print(f"[{run_id}] FAILED: {run_name} rep={rep}: {e}")

                    if fail_fast:
                        elapsed = time.time() - start_time
                        return {
                            "run_id": run_id,
                            "n_completed": n_completed,
                            "n_failed": n_failed,
                            "elapsed_seconds": elapsed,
                            "results": results,
                            "status": "failed",
                        }

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO {run_id} COMPLETE")
        print(f"  Completed: {n_completed}/{n_total}")
        print(f"  Failed: {n_failed}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'='*60}\n")

    # Print timing summary if debug mode is enabled
    if DEBUG_ENABLED:
        timer.print_summary()

    return {
        "run_id": run_id,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
        "results": results,
        "status": "success" if n_failed == 0 else "partial",
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run a single experiment scenario (R0-R11) independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run R1 with default settings (1 replicate, all k values)
  python -m coreset_selection.run_scenario R1

  # Run R3 with specific k values
  python -m coreset_selection.run_scenario R3 --k-values 100,200,300

  # Run with custom directories
  python -m coreset_selection.run_scenario R1 --output-dir runs_out/R1 --cache-dir cache

  # Use GPU
  python -m coreset_selection.run_scenario R1 --device cuda

===============================================================================
PARALLEL EXECUTION (running R0-R11 in separate terminals)
===============================================================================
When running multiple experiments simultaneously, use --parallel-experiments
to avoid thread contention:

  Terminal 1:  python -m coreset_selection.run_scenario R0 --parallel-experiments 10
  Terminal 2:  python -m coreset_selection.run_scenario R1 --parallel-experiments 10
  ...
  Terminal 10: python -m coreset_selection.run_scenario R8 --parallel-experiments 10

ALTERNATIVE: Set environment variable once before running all experiments:
  export CORESET_NUM_THREADS=20   # 200 cores / 10 experiments = 20 threads each
  # Then run without --parallel-experiments flag

This auto-calculates the optimal thread count per experiment.
===============================================================================

Scenarios (all run with 1 replicate by default):
  R0  - Quota computation: c*(k) and KL_min(k) for all k in K
  R1  - NSGA-II main: tri-objective (SKL, MMD, SD) with quota, VAE space
  R2  - Geography ablation: exact-k only, no quota constraints
  R3  - Objective ablation: bi-objective (MMD, SD), removes SKL
  R4  - Objective ablation: bi-objective (SKL, SD), removes MMD
  R5  - Objective ablation: bi-objective (SKL, MMD), removes Sinkhorn
  R6  - Baselines: quota-matched and unconstrained
  R7  - Post-hoc diagnostics on fixed subsets from R1-R6
  R8  - Representation transfer: tri-objective (SKL, MMD, SD) in PCA space
  R9  - Representation transfer: tri-objective (SKL, MMD, SD) in raw space
  R10 - Constraint ablation: no quota + no exact-k
  R11 - Constraint ablation: quota only (no exact-k repair)
        """
    )

    parser.add_argument(
        "run_id",
        choices=sorted(get_run_specs().keys()),
        help="Scenario identifier (e.g., R0-R11)"
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated k values to run (overrides default from RunSpec)"
    )

    parser.add_argument(
        "--rep-ids",
        type=str,
        default=None,
        help="Comma-separated replicate IDs to run (overrides auto-increment; WARNING: overwrites existing)"
    )

    parser.add_argument(
        "--n-replicates",
        type=int,
        default=None,
        help="Number of NEW replicates to add (auto-incremented from next available ID; default: 1)"
    )

    parser.add_argument(
        "--parallel-experiments",
        type=int,
        default=None,
        help=(
            "Number of experiments running in parallel (e.g., 10 if running R0-R11 "
            "in separate terminals). Auto-calculates optimal thread count per experiment."
        )
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing input data files"
    )

    parser.add_argument(
        "--output-dir",
        default="runs_out",
        help="Output directory for results"
    )

    parser.add_argument(
        "--cache-dir",
        default="replicate_cache",
        help="Directory for replicate caches"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device (cpu/cuda)"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Force rebuild of replicate caches even if they already exist"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume mode: skip completed (run, k, rep) combinations, "
            "re-run incomplete ones reusing the same rep ID and seed, "
            "and create new reps only if the target count is not reached."
        )
    )

    # R6-specific options
    parser.add_argument(
        "--source-run",
        default="R1",
        help="(R6 only) Source run base ID"
    )

    parser.add_argument(
        "--source-space",
        default="vae",
        help="(R6 only) Source space: vae|pca|raw"
    )

    parser.add_argument(
        "--source-k",
        type=int,
        default=None,
        help="(R6 only) k value for source run"
    )

    # R12 effort sweep parameters
    parser.add_argument(
        "--effort-pop-sizes",
        default=None,
        help="(R12 only) Comma-separated NSGA-II population sizes, e.g. '50,100,150,200'"
    )
    parser.add_argument(
        "--effort-n-gens",
        default=None,
        help="(R12 only) Comma-separated NSGA-II generation counts, e.g. '100,300,500,700'"
    )

    args = parser.parse_args()

    # Set R12 effort parameters via environment variables (read by runner)
    if args.effort_pop_sizes is not None:
        os.environ["CORESET_R12_POP_SIZES"] = str(args.effort_pop_sizes)
    if args.effort_n_gens is not None:
        os.environ["CORESET_R12_N_GENS"] = str(args.effort_n_gens)

    # Parse k values and rep ids
    k_values = parse_int_list(args.k_values)
    rep_ids = parse_int_list(args.rep_ids)

    try:
        summary = run_scenario_standalone(
            run_id=args.run_id,
            k_values=k_values,
            rep_ids=rep_ids,
            n_replicates=args.n_replicates,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            fail_fast=args.fail_fast,
            verbose=not args.quiet,
            parallel_experiments=args.parallel_experiments,
            source_run=args.source_run,
            source_space=args.source_space,
            source_k=args.source_k,
            force_rebuild_cache=args.force_rebuild_cache,
            resume=args.resume,
        )

        if summary["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
