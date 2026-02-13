"""All cmd_* command handlers for the CLI."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ..config.run_specs import get_run_specs, apply_run_spec
from ..data.cache import build_replicate_cache
from ..experiment.runner import ExperimentRunner, run_single_experiment

from ._config import build_base_config, _parse_int_list


def cmd_prep(args: argparse.Namespace) -> int:
    """
    Prepare replicate caches.

    Builds cached assets (VAE embeddings, train/test splits) for each replicate.
    """
    print(f"[prep] Building replicate caches...")
    print(f"  Data directory: {args.data_dir}")

    base_cfg = build_base_config(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
    )

    for rep_id in range(args.n_replicates):
        print(f"[prep] Building replicate {rep_id}...")

        # Update config for this replicate
        cfg = replace(base_cfg, rep_id=rep_id, seed=args.seed + rep_id)

        try:
            cache_path = build_replicate_cache(cfg, rep_id)
            print(f"[prep] Saved cache to {cache_path}")
        except Exception as e:
            print(f"[prep] Failed for replicate {rep_id}: {e}")
            if args.fail_fast:
                return 1

    print(f"[prep] Done. Built {args.n_replicates} replicate caches.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """
    Run an experiment.
    """
    print(f"[run] Starting {args.run_id} rep={args.rep_id}...")

    # Build base config
    base_cfg = build_base_config(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
    )

    # Apply run spec if available
    run_specs = get_run_specs()
    if args.run_id in run_specs:
        cfg = apply_run_spec(base_cfg, run_specs[args.run_id], args.rep_id)
    else:
        cfg = replace(base_cfg, run_id=args.run_id, rep_id=args.rep_id)

    # Update k if specified
    if args.k is not None:
        cfg = replace(cfg, solver=replace(cfg.solver, k=args.k))

    # Run experiment
    try:
        result = run_single_experiment(cfg)
        print(f"[run] Completed: {result}")
        return 0
    except Exception as e:
        print(f"[run] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """
    Run a sweep over k values and replicates.
    """
    print(f"[sweep] Starting sweep: k={args.k_values}, reps={args.n_replicates}")

    base_cfg = build_base_config(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
    )

    # Apply run spec
    run_specs = get_run_specs()
    if args.run_id in run_specs:
        base_cfg = apply_run_spec(base_cfg, run_specs[args.run_id], 0)
    else:
        base_cfg = replace(base_cfg, run_id=args.run_id)

    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]

    n_total = len(k_values) * args.n_replicates
    n_done = 0
    n_failed = 0

    for k in k_values:
        for rep in range(args.n_replicates):
            print(f"[sweep] Running k={k}, rep={rep} ({n_done+1}/{n_total})...")

            cfg = replace(
                base_cfg,
                solver=replace(base_cfg.solver, k=k),
                rep_id=rep,
                run_id=f"{args.run_id}_k{k}",
            )

            try:
                run_single_experiment(cfg)
                n_done += 1
            except Exception as e:
                print(f"[sweep] Failed k={k}, rep={rep}: {e}")
                n_failed += 1
                if args.fail_fast:
                    return 1

    print(f"[sweep] Done. Completed {n_done}, failed {n_failed}")
    return 0 if n_failed == 0 else 1


def cmd_parallel(args: argparse.Namespace) -> int:
    """
    Run multiple experiments in parallel with optimal thread allocation.

    Automatically distributes CPU cores across experiments.
    Logs output to files for monitoring with tail -f.

    Execution follows three phases:

    Phase 0 — Pre-build replicate caches (VAE + PCA) SEQUENTIALLY so that
              every parallel scenario reuses the exact same learned
              representations.  This avoids redundant VAE trainings that
              would otherwise happen when the first scenario to acquire the
              lock trains a partial cache and later scenarios augment it.

    Phase 1 — Launch all scenarios in parallel (caches are ready).

    Phase 2 — Collect results and print summary.
    """
    import subprocess
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    from ..data.cache import prebuild_full_cache
    from ..config.run_specs import get_run_specs

    runs = args.runs
    if not runs or runs == ['all']:
        runs = [f'r{i}' for i in range(15)]  # r0-r14

    n_cores = multiprocessing.cpu_count()
    n_jobs = len(runs)
    threads_per_job = args.threads or max(2, min(16, n_cores // n_jobs))

    # Create log directory
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("PARALLEL EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"  CPU cores:      {n_cores}")
    print(f"  Experiments:    {n_jobs} ({', '.join(runs)})")
    print(f"  Threads/job:    {threads_per_job}")
    print(f"  Log directory:  {log_dir}/")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 0: Pre-build replicate caches (VAE + PCA) SEQUENTIALLY
    # ------------------------------------------------------------------
    # Determine the union of all rep_ids needed across scenarios and
    # build the full cache (VAE + PCA) for each one.  This guarantees
    # that when the parallel Phase 1 starts, every scenario finds a
    # complete cache and hits the fast path — no redundant VAE trainings.
    # ------------------------------------------------------------------
    all_run_specs = get_run_specs()
    rep_ids_needed: set = set()
    any_needs_cache = False
    for r in runs:
        rid = r.upper()
        spec = all_run_specs.get(rid)
        if spec and spec.cache_build_mode != "skip":
            any_needs_cache = True
            # Use max_n_reps to ensure caches exist for all per-k replicate counts
            rep_ids_for_spec = list(range(int(spec.max_n_reps)))
            rep_ids_needed.update(rep_ids_for_spec)

    if any_needs_cache and rep_ids_needed:
        base_cfg = build_base_config(
            output_dir=getattr(args, 'output_dir', 'runs_out'),
            cache_dir=getattr(args, 'cache_dir', 'replicate_cache'),
            data_dir=getattr(args, 'data_dir', 'data'),
            seed=getattr(args, 'seed', 123),
            device=getattr(args, 'device', 'cpu'),
        )
        sorted_reps = sorted(rep_ids_needed)
        print()
        print(f"[Phase 0] Pre-building caches for replicates: {sorted_reps}")
        print(f"[Phase 0] This ensures ALL parallel scenarios share the SAME VAE/PCA.")
        print()
        for rep in sorted_reps:
            prebuild_full_cache(
                base_cfg, rep,
                seed=getattr(args, 'seed', 123),
            )
        print()
        print(f"[Phase 0] Cache pre-build complete — launching parallel scenarios.")
        print()

    # ------------------------------------------------------------------
    # PHASE 1: Launch all scenarios in parallel
    # ------------------------------------------------------------------
    print()
    print("Monitor progress with:")
    print(f"  tail -f {log_dir}/*.log")
    print()
    print("Or watch a specific experiment:")
    for r in runs[:3]:
        print(f"  tail -f {log_dir}/{r}.log")
    if len(runs) > 3:
        print(f"  ...")
    print()
    print("=" * 60)

    def run_experiment(run_id: str) -> tuple:
        """Run a single experiment in subprocess with thread limits."""
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads_per_job)
        env['MKL_NUM_THREADS'] = str(threads_per_job)
        env['OPENBLAS_NUM_THREADS'] = str(threads_per_job)
        env['NUMEXPR_MAX_THREADS'] = str(threads_per_job)

        cmd = [sys.executable, '-m', 'coreset_selection', run_id]

        # Add common arguments
        if hasattr(args, 'output_dir') and args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        if hasattr(args, 'cache_dir') and args.cache_dir:
            cmd.extend(['--cache-dir', args.cache_dir])
        if hasattr(args, 'data_dir') and args.data_dir:
            cmd.extend(['--data-dir', args.data_dir])

        log_file = os.path.join(log_dir, f"{run_id}.log")

        start = time.time()
        with open(log_file, 'w') as f:
            f.write(f"=== {run_id} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Threads: {threads_per_job}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

            elapsed = time.time() - start
            f.write(f"\n{'=' * 60}\n")
            f.write(f"=== {run_id} finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"=== Exit code: {result.returncode}, Duration: {elapsed:.1f}s ===\n")

        return run_id, result.returncode == 0, elapsed

    # Run experiments in parallel using process pool
    results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(run_experiment, r): r for r in runs}

        for future in as_completed(futures):
            run_id, success, elapsed = future.result()
            status = "\u2713" if success else "\u2717"
            print(f"  [{status}] {run_id} completed in {elapsed:.1f}s")
            results[run_id] = success

    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # PHASE 2: Summary
    # ------------------------------------------------------------------
    n_success = sum(results.values())
    n_failed = len(results) - n_success

    print()
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    if n_failed == 0:
        print(f"SUCCESS: All {n_success} experiments completed")
    else:
        failed = [r for r, ok in results.items() if not ok]
        print(f"FAILED: {n_failed} experiments: {', '.join(failed)}")
        print(f"Check logs: {log_dir}/<run_id>.log")
    print("=" * 60)

    return 0 if n_failed == 0 else 1


def cmd_artifacts(args: argparse.Namespace) -> int:
    """
    Generate manuscript artifacts (figures and tables).

    Auto-detects runs_out/ directory and generates all figures/tables.

    Usage:
        python -m coreset_selection figs              # auto-detect
        python -m coreset_selection figs runs_out/    # explicit path
        python -m coreset_selection figs -o outputs/  # custom output
    """
    # Auto-detect runs directory
    runs_dir = getattr(args, 'runs_dir', None)
    if runs_dir is None:
        # Try common locations
        for candidate in ['runs_out', 'runs', 'output', 'results', '.']:
            if os.path.isdir(candidate):
                # Check if it has run folders
                subdirs = [d for d in os.listdir(candidate)
                          if os.path.isdir(os.path.join(candidate, d))
                          and d.startswith('R')]
                if subdirs:
                    runs_dir = candidate
                    break
        if runs_dir is None:
            runs_dir = 'runs_out'

    # Auto-detect output directory
    output_dir = getattr(args, 'output', None) or 'artifacts'
    rep_folder = getattr(args, 'rep', 'rep00')
    data_dir = getattr(args, 'data_dir', 'data')

    print(f"[figs] Generating manuscript artifacts")
    print(f"  \u2192 Input:  {runs_dir}/")
    print(f"  \u2192 Output: {output_dir}/")

    from ..artifacts.manuscript_generator import ManuscriptArtifactGenerator as Generator

    generator = Generator(
        runs_root=runs_dir,
        cache_root='replicate_cache',
        out_dir=output_dir,
        rep_folder=rep_folder,
    )

    try:
        generated = generator.generate_all()

        n_figs = len(generated.get('figures', []))
        n_tables = len(generated.get('tables', []))
        n_docs = len(generated.get('documents', []))

        print(f"\n[figs] Legacy generator: {n_figs} figures, {n_tables} tables, {n_docs} docs")

        # Also run ManuscriptArtifacts (Phase 8-12 compliant)
        from ..artifacts.manuscript_artifacts import ManuscriptArtifacts
        ms_dir = os.path.join(output_dir, "manuscript")
        ms_gen = ManuscriptArtifacts(
            runs_root=runs_dir,
            cache_root='replicate_cache',
            out_dir=ms_dir,
            data_dir=data_dir,
        )
        ms_result = ms_gen.generate_all()
        n_ms_figs = len(ms_result.get('figures', []))
        n_ms_tabs = len(ms_result.get('tables', []))
        print(f"[figs] Manuscript generator: {n_ms_figs} figures, {n_ms_tabs} tables")

        print(f"\n[figs] Done!")
        print(f"  \u2192 {output_dir}/figures/")
        print(f"  \u2192 {output_dir}/tables/")
        print(f"  \u2192 {ms_dir}/figures/")
        print(f"  \u2192 {ms_dir}/tables/")

        return 0
    except Exception as e:
        print(f"[figs] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_list_runs(args: argparse.Namespace) -> int:
    """
    List available run specifications.
    """
    run_specs = get_run_specs()

    print("Available run specifications:")
    print("-" * 60)

    for run_id, spec in run_specs.items():
        desc = getattr(spec, 'description', 'No description')
        k = getattr(spec, 'k', None)
        k_str = f"k={k}" if k is not None else "k=user-defined"
        print(f"  {run_id}: {k_str}, {desc}")

    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    """Run a full manuscript scenario (R0-R11) as a standalone job.

    This is the recommended entry point for parallelization: you can launch
    multiple scenarios (e.g. R1, R2, ... R8) in separate processes.

    The function follows a two-phase design (matching run_scenario_standalone):

    Phase 1 — Pre-build replicate caches with ALL representations (VAE + PCA)
              so that every experiment configuration reuses the exact same
              learned representations.  The cache uses file-level locking, so
              multiple parallel processes safely share a single build.

    Phase 2 — Run the actual experiments.  ensure_replicate_cache inside
              run_single_experiment hits the fast path because the cache was
              fully pre-built in Phase 1.
    """
    from ..data.cache import prebuild_full_cache

    run_specs = get_run_specs()
    run_id = str(getattr(args, "run_id", ""))
    if run_id not in run_specs:
        print(f"[scenario] Unknown run ID: {run_id}. Use list-runs to see valid IDs.")
        return 1

    spec = run_specs[run_id]
    base_cfg = build_base_config(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        device=args.device,
    )

    # k list
    k_values = _parse_int_list(getattr(args, "k_values", None))
    if k_values is None:
        if spec.sweep_k is not None:
            k_values = list(spec.sweep_k)
        elif spec.k is not None:
            k_values = [int(spec.k)]
        else:
            print(f"[scenario] ERROR: No k value specified. Use -k <value> to set the coreset size.")
            return 1

    # replicate list
    _explicit_rep_ids = _parse_int_list(getattr(args, "rep_ids", None))

    # Helper: resolve rep_ids for a given k value.  When the user has not
    # provided explicit rep IDs, we use per-k counts from the RunSpec.
    def _rep_ids_for_k(k: int):
        if _explicit_rep_ids is not None:
            return _explicit_rep_ids
        return list(range(spec.get_n_reps_for_k(k)))

    # For cache pre-build, we need the union of all rep_ids across k values.
    _all_rep_ids_needed = sorted(set(
        r for k in k_values for r in _rep_ids_for_k(k)
    ))

    # Special handling: R6 depends on outputs from prior runs.
    if run_id == "R6":
        import os

        src = str(getattr(args, "source_run", "R1"))
        src_space = str(getattr(args, "source_space", "vae"))
        k_for_r7 = getattr(args, "k", None)
        if k_for_r7 is None:
            print("[scenario] ERROR: R6 requires --k to be specified.")
            return 1
        k_for_r7 = int(k_for_r7)

        # Set environment overrides for the underlying runner.
        os.environ["CORESET_R6_SOURCE_RUN"] = src
        os.environ["CORESET_R6_SOURCE_SPACE"] = src_space
        os.environ["CORESET_R6_K"] = str(k_for_r7)

        # R6 is single-k by definition.
        k_values = [k_for_r7]

    # ------------------------------------------------------------------
    # PHASE 1: Pre-build replicate caches with ALL representations
    # ------------------------------------------------------------------
    if spec.cache_build_mode != "skip" and _all_rep_ids_needed:
        print(f"\n[{run_id}] Phase 1: Pre-building caches for replicates: {_all_rep_ids_needed}")
        print(f"[{run_id}] This ensures all experiment configs share the SAME VAE/PCA per replicate.\n")

        seed = args.seed
        for rep in _all_rep_ids_needed:
            prebuild_full_cache(base_cfg, rep, seed=seed)

        print(f"[{run_id}] Phase 1 complete — caches ready.\n")

    # ------------------------------------------------------------------
    # PHASE 2: Run experiments (caches are guaranteed to exist)
    # ------------------------------------------------------------------
    n_total = sum(len(_rep_ids_for_k(k)) for k in k_values)
    n_done = 0
    n_failed = 0

    for k in k_values:
        # Use suffix only when multiple k values are being run for this scenario.
        run_name = f"{run_id}_k{k}" if len(k_values) > 1 else run_id
        for rep in _rep_ids_for_k(k):
            print(f"[scenario] {run_name} rep={rep} ({n_done+1}/{n_total})")
            try:
                cfg0 = apply_run_spec(base_cfg, spec, rep)
                cfg = replace(
                    cfg0,
                    run_id=run_name,
                    rep_id=int(rep),
                    seed=int(args.seed + rep),
                    solver=replace(cfg0.solver, k=int(k)),
                )
                # NOTE: Cache was pre-built in Phase 1.  The call to
                # ensure_replicate_cache inside run_single_experiment will
                # hit the fast path (all keys already present).
                run_single_experiment(cfg)
                n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"[scenario] FAILED {run_name} rep={rep}: {e}")
                if getattr(args, "fail_fast", False):
                    return 1

    print(f"[scenario] Completed {n_done}/{n_total} runs ({n_failed} failed)")
    return 0 if n_failed == 0 else 1


def _cmd_scenario_fixed(run_id: str):
    """Return a command handler that runs a fixed scenario run_id."""

    def _inner(args: argparse.Namespace) -> int:
        setattr(args, "run_id", run_id)

        # Merge -k (k_single) into k_values for unified handling
        k_single = getattr(args, "k_single", None)
        k_values = getattr(args, "k_values", None)

        # -k takes precedence over --k-values
        if k_single is not None:
            setattr(args, "k_values", k_single)
        elif k_values is not None:
            setattr(args, "k_values", k_values)

        return cmd_scenario(args)

    return _inner


def cmd_all(args: argparse.Namespace) -> int:
    """
    Run ALL experiment configurations sequentially (r0-r14).
    """
    runs = [f'r{i}' for i in range(15)]  # Always r0-r14
    return _run_sequential(runs, args)


def cmd_seq(args: argparse.Namespace) -> int:
    """
    Run selected experiments sequentially.
    """
    runs = args.runs
    if not runs:
        print("Error: specify which experiments to run, e.g.: seq r1 r2 r6")
        return 1
    return _run_sequential(runs, args)


def _run_sequential(runs: list, args: argparse.Namespace) -> int:
    """
    Internal: run a list of experiments sequentially.

    Pre-builds replicate caches (VAE + PCA) for the union of all
    replicates BEFORE running any experiment, ensuring every scenario
    shares the exact same learned representations.
    """
    import time
    from ..data.cache import prebuild_full_cache
    from ..config.run_specs import get_run_specs

    print("=" * 60)
    print(f"RUNNING {len(runs)} EXPERIMENTS SEQUENTIALLY")
    print(f"  {', '.join(r.upper() for r in runs)}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Phase 0: Pre-build caches for all replicates
    # ------------------------------------------------------------------
    all_run_specs = get_run_specs()
    rep_ids_needed: set = set()
    any_needs_cache = False
    for r in runs:
        rid = r.upper()
        spec = all_run_specs.get(rid)
        if spec and spec.cache_build_mode != "skip":
            any_needs_cache = True
            rep_ids_arg = _parse_int_list(getattr(args, 'rep_ids', None))
            if rep_ids_arg is not None:
                rep_ids_needed.update(rep_ids_arg)
            else:
                rep_ids_needed.update(range(int(spec.max_n_reps)))

    if any_needs_cache and rep_ids_needed:
        base_cfg = build_base_config(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            data_dir=getattr(args, 'data_dir', 'data'),
            seed=args.seed,
            device=args.device,
        )
        sorted_reps = sorted(rep_ids_needed)
        print(f"[Phase 0] Pre-building caches for replicates: {sorted_reps}")
        print(f"[Phase 0] This ensures ALL scenarios share the SAME VAE/PCA.\n")
        for rep in sorted_reps:
            prebuild_full_cache(base_cfg, rep, seed=args.seed)
        print(f"[Phase 0] Cache pre-build complete.\n")

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    results = {}
    total_start = time.time()

    for i, run_id in enumerate(runs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(runs)}] Starting {run_id.upper()}...")
        print('='*60)

        start = time.time()

        # Build args for this run
        run_args = argparse.Namespace(
            run_id=run_id.upper(),
            k_values=getattr(args, 'k_values', None),
            rep_ids=getattr(args, 'rep_ids', None),
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            device=args.device,
            fail_fast=getattr(args, 'fail_fast', False),
            source_run=getattr(args, 'source_run', 'R1'),
            source_space=getattr(args, 'source_space', 'vae'),
        )

        try:
            ret = cmd_scenario(run_args)
            elapsed = time.time() - start
            success = (ret == 0)
            results[run_id] = (success, elapsed)

            status = "\u2713 SUCCESS" if success else "\u2717 FAILED"
            print(f"\n[{run_id.upper()}] {status} in {elapsed:.1f}s")

            if not success and getattr(args, 'fail_fast', False):
                print("\n[seq] Stopping due to --fail-fast")
                break

        except Exception as e:
            elapsed = time.time() - start
            results[run_id] = (False, elapsed)
            print(f"\n[{run_id.upper()}] \u2717 EXCEPTION: {e}")
            if getattr(args, 'fail_fast', False):
                break

    total_time = time.time() - total_start

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_success = sum(1 for ok, _ in results.values() if ok)
    n_failed = len(results) - n_success

    for run_id, (success, elapsed) in results.items():
        status = "\u2713" if success else "\u2717"
        print(f"  [{status}] {run_id.upper():4s}  {elapsed:7.1f}s")

    print("-" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Success: {n_success}/{len(results)}")

    if n_failed > 0:
        failed = [r for r, (ok, _) in results.items() if not ok]
        print(f"  Failed: {', '.join(failed)}")

    print("=" * 60)

    return 0 if n_failed == 0 else 1
