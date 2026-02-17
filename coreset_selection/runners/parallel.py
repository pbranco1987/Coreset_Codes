#!/usr/bin/env python
"""
Parallel runner for executing multiple experiment scenarios (R0-R11) concurrently.

This script orchestrates parallel execution of independent scenarios using either
subprocess-based parallelism (recommended for distributed execution) or
process-pool parallelism (for single-machine multi-core execution).

Usage:
    # Run all scenarios in parallel (default: all R0-R11)
    python -m coreset_selection.parallel_runner --data-dir /path/to/data

    # Run specific scenarios in parallel
    python -m coreset_selection.parallel_runner --scenarios R1,R2,R3

    # Run with specific number of parallel workers
    python -m coreset_selection.parallel_runner --n-workers 4

    # Generate shell commands for external parallel execution (e.g., SLURM, SGE)
    python -m coreset_selection.parallel_runner --generate-commands > run_parallel.sh

Parallelization Strategies:
1. subprocess mode (default): Each scenario runs in a separate subprocess.
   Best for: distributed systems, job schedulers, reproducibility.

2. multiprocessing mode: Uses Python's ProcessPoolExecutor.
   Best for: single-machine multi-core execution with simpler setup.

3. command generation mode: Outputs shell commands for external scheduling.
   Best for: HPC clusters with SLURM, SGE, PBS, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from ..config.run_specs import get_run_specs, K_GRID

# Re-export helpers so existing callers see them here
from ._parallel_helpers import (
    get_scenario_dependencies,
    topological_sort_scenarios,
    build_scenario_command,
    generate_shell_commands,
    generate_slurm_script,
)


def run_scenario_subprocess(
    run_id: str,
    cmd: List[str],
    verbose: bool = True,
    log_dir: str = "",
) -> Tuple[str, int, float, str, str]:
    """
    Run a scenario in a subprocess.

    Returns: (run_id, return_code, elapsed_seconds, stdout, stderr)
    """
    start = time.time()

    if verbose:
        print(f"[parallel] Starting {run_id}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start

        if verbose:
            status = "completed" if result.returncode == 0 else "FAILED"
            print(f"[parallel] {run_id} {status} in {elapsed:.1f}s")

        # Write per-scenario log files for monitoring
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{run_id}.log")
            with open(log_path, "w") as f:
                f.write(f"=== {run_id} ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Status: {status} (exit code {result.returncode})\n")
                f.write(f"Elapsed: {elapsed:.1f}s\n")
                f.write(f"\n{'='*60}\nSTDOUT\n{'='*60}\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"\n{'='*60}\nSTDERR\n{'='*60}\n")
                    f.write(result.stderr)

        return (run_id, result.returncode, elapsed, result.stdout, result.stderr)

    except Exception as e:
        elapsed = time.time() - start
        return (run_id, -1, elapsed, "", str(e))


def run_scenarios_parallel_subprocess(
    scenarios: List[str],
    *,
    n_workers: int = 4,
    n_replicates: Optional[int] = None,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    fail_fast: bool = False,
    verbose: bool = True,
    python_executable: str = "python",
    k_override: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, dict]:
    """
    Run scenarios in parallel using subprocesses.

    Respects dependency ordering (R7 waits for R1).

    Phase 0 — Pre-build replicate caches (VAE + PCA) SEQUENTIALLY in the
              main process so that every parallel subprocess finds a fully
              populated cache and skips representation training entirely.
              This avoids redundant VAE/PCA builds that would otherwise
              occur when multiple subprocesses race for the cache lock.
    """
    from ..cli import build_base_config
    from ..data.cache import prebuild_full_cache

    # ------------------------------------------------------------------
    # Phase 0: Pre-build caches for all replicates needed by any scenario
    # ------------------------------------------------------------------
    run_specs = get_run_specs()
    rep_ids_needed: set = set()
    any_needs_cache = False
    for sid in scenarios:
        spec = run_specs.get(sid)
        if spec and spec.cache_build_mode != "skip":
            # Dimension sweep scenarios (R13/R14) build dimension-specific
            # caches inside run_scenario_standalone — skip the shared pre-build.
            if spec.sweep_dim is not None:
                continue
            any_needs_cache = True
            n_reps_for_spec = n_replicates if n_replicates is not None else spec.max_n_reps
            rep_ids_needed.update(range(int(n_reps_for_spec)))

    if any_needs_cache and rep_ids_needed:
        base_cfg = build_base_config(
            output_dir=output_dir,
            cache_dir=cache_dir,
            data_dir=data_dir,
            seed=seed,
            device=device,
        )
        sorted_reps = sorted(rep_ids_needed)
        if verbose:
            print(f"\n[Phase 0] Pre-building caches for replicates: {sorted_reps}")
            print(f"[Phase 0] This ensures ALL parallel scenarios share the SAME VAE/PCA.\n")
        for rep in sorted_reps:
            prebuild_full_cache(base_cfg, rep, seed=seed)
        if verbose:
            print(f"[Phase 0] Cache pre-build complete — launching parallel scenarios.\n")

    # ------------------------------------------------------------------
    # Phase 1: Launch parallel subprocesses (caches are ready)
    # ------------------------------------------------------------------
    # k-specific subfolder removed — scenario.py now always appends _k{k}
    # when the RunSpec defines sweep_k.
    effective_output_dir = output_dir

    waves = topological_sort_scenarios(scenarios)
    all_results = {}

    for wave_idx, wave in enumerate(waves):
        if verbose:
            print(f"\n{'='*60}")
            print(f"WAVE {wave_idx + 1}/{len(waves)}: {', '.join(wave)}")
            print(f"{'='*60}\n")

        # Build commands for this wave
        commands = {}
        effective_workers = min(n_workers, len(wave))
        for run_id in wave:
            cmd = build_scenario_command(
                run_id,
                data_dir=data_dir,
                output_dir=effective_output_dir,
                cache_dir=cache_dir,
                seed=seed,
                device=device,
                k_values=[k_override] if k_override is not None else None,
                n_replicates=n_replicates,
                fail_fast=fail_fast,
                parallel_experiments=effective_workers,
                python_executable=python_executable,
                resume=resume,
            )
            commands[run_id] = cmd

        # Run this wave in parallel
        wave_results = {}
        scenario_log_dir = os.path.join(effective_output_dir, "logs")

        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(run_scenario_subprocess, run_id, cmd, verbose, scenario_log_dir): run_id
                for run_id, cmd in commands.items()
            }

            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                    run_id, return_code, elapsed, stdout, stderr = result
                    wave_results[run_id] = {
                        "return_code": return_code,
                        "elapsed_seconds": elapsed,
                        "stdout": stdout,
                        "stderr": stderr,
                        "status": "success" if return_code == 0 else "failed",
                    }

                    if return_code != 0 and fail_fast:
                        if verbose:
                            print(f"[parallel] Stopping due to failure in {run_id}")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                except Exception as e:
                    wave_results[run_id] = {
                        "return_code": -1,
                        "elapsed_seconds": 0,
                        "stdout": "",
                        "stderr": str(e),
                        "status": "error",
                    }

        all_results.update(wave_results)

        # Check if we should stop
        if fail_fast and any(r["status"] != "success" for r in wave_results.values()):
            break

    return all_results


def run_scenarios_sequential(
    scenarios: List[str],
    *,
    data_dir: str = "data",
    output_dir: str = "runs_out",
    cache_dir: str = "replicate_cache",
    seed: int = 123,
    device: str = "cpu",
    fail_fast: bool = False,
    verbose: bool = True,
    k_override: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, dict]:
    """Run scenarios sequentially (for testing or single-core systems)."""
    from .scenario import run_scenario_standalone

    # k-specific subfolder removed — scenario.py now always appends _k{k}
    # when the RunSpec defines sweep_k.
    effective_output_dir = output_dir

    results = {}

    for run_id in scenarios:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {run_id}")
            print(f"{'='*60}\n")

        try:
            summary = run_scenario_standalone(
                run_id=run_id,
                data_dir=data_dir,
                output_dir=effective_output_dir,
                cache_dir=cache_dir,
                seed=seed,
                device=device,
                k_values=[k_override] if k_override is not None else None,
                fail_fast=fail_fast,
                verbose=verbose,
                resume=resume,
            )
            results[run_id] = summary

            if fail_fast and summary.get("status") != "success":
                break

        except Exception as e:
            results[run_id] = {
                "run_id": run_id,
                "status": "error",
                "error": str(e),
            }
            if fail_fast:
                break

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run multiple experiment scenarios in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios in parallel
  python -m coreset_selection.parallel_runner

  # Run specific scenarios
  python -m coreset_selection.parallel_runner --scenarios R1,R2,R3

  # Run with 8 parallel workers
  python -m coreset_selection.parallel_runner --n-workers 8

  # Generate shell commands for manual/HPC execution
  python -m coreset_selection.parallel_runner --generate-commands

  # Generate SLURM array job script
  python -m coreset_selection.parallel_runner --generate-slurm

  # Run sequentially (for debugging)
  python -m coreset_selection.parallel_runner --sequential
        """
    )

    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario IDs (default: all R0-R11)"
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--n-replicates",
        type=int,
        default=None,
        help="Number of replicates per scenario (default: from RunSpec, typically 5)"
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
        "--k",
        type=int,
        default=None,
        help="Override coreset size K for all scenarios. "
             "Output goes into <output-dir>/k<K>/. "
             "(default: use RunSpec defaults, typically 300)"
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
        "--sequential",
        action="store_true",
        help="Run sequentially instead of in parallel"
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

    parser.add_argument(
        "--generate-commands",
        action="store_true",
        help="Generate shell commands for external parallel execution"
    )

    parser.add_argument(
        "--generate-slurm",
        action="store_true",
        help="Generate SLURM array job script"
    )

    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use"
    )

    # SLURM options
    parser.add_argument("--slurm-partition", default="standard")
    parser.add_argument("--slurm-time", default="24:00:00")
    parser.add_argument("--slurm-memory", default="32G")

    args = parser.parse_args()

    # Parse scenarios
    if args.scenarios:
        scenarios = [s.strip().upper() for s in args.scenarios.split(",")]
    else:
        scenarios = sorted(get_run_specs().keys())

    # Validate scenarios
    valid_scenarios = set(get_run_specs().keys())
    invalid = [s for s in scenarios if s not in valid_scenarios]
    if invalid:
        print(f"[ERROR] Invalid scenarios: {invalid}")
        print(f"Valid scenarios: {sorted(valid_scenarios)}")
        sys.exit(1)

    # Generate commands mode
    if args.generate_commands:
        print(generate_shell_commands(
            scenarios,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            python_executable=args.python,
        ))
        sys.exit(0)

    # Generate SLURM script mode
    if args.generate_slurm:
        print(generate_slurm_script(
            scenarios,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            partition=args.slurm_partition,
            time_limit=args.slurm_time,
            memory=args.slurm_memory,
            python_executable=args.python,
        ))
        sys.exit(0)

    # Run scenarios
    verbose = not args.quiet
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"CORESET SELECTION PARALLEL RUNNER")
        print(f"{'='*60}")
        print(f"Scenarios: {scenarios}")
        print(f"Workers: {args.n_workers if not args.sequential else 1}")
        print(f"Mode: {'sequential' if args.sequential else 'parallel'}")
        if args.k is not None:
            print(f"K override: {args.k}  (output → {args.output_dir}/k{args.k}/)")
        if args.resume:
            print(f"Resume: ON (skipping completed runs)")
        print(f"{'='*60}\n")

    if args.sequential:
        results = run_scenarios_sequential(
            scenarios,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            fail_fast=args.fail_fast,
            verbose=verbose,
            k_override=args.k,
            resume=args.resume,
        )
    else:
        results = run_scenarios_parallel_subprocess(
            scenarios,
            n_workers=args.n_workers,
            n_replicates=args.n_replicates,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            fail_fast=args.fail_fast,
            verbose=verbose,
            python_executable=args.python,
            k_override=args.k,
            resume=args.resume,
        )

    elapsed = time.time() - start_time

    # Summary
    n_success = sum(1 for r in results.values() if r.get("status") == "success")
    n_failed = len(results) - n_success

    if verbose:
        print(f"\n{'='*60}")
        print(f"PARALLEL EXECUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total scenarios: {len(results)}")
        print(f"Successful: {n_success}")
        print(f"Failed: {n_failed}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"{'='*60}\n")

        if n_failed > 0:
            print("Failed scenarios:")
            for run_id, result in results.items():
                if result.get("status") != "success":
                    print(f"  - {run_id}: {result.get('stderr', 'Unknown error')[:200]}")

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
