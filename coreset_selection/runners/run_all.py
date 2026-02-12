#!/usr/bin/env python
"""
Main orchestration script for Brazil telecom coreset selection experiments.

This script runs the complete experiment pipeline as specified in the manuscript:
1. Build replicate caches (VAE training, data splits)
2. Run experiments R0-R12 across k grid
3. Generate manuscript artifacts (figures, tables)

Usage:
    python -m coreset_selection.run_all --data-dir /path/to/data
    python -m coreset_selection.run_all --prep-only  # Only build caches
    python -m coreset_selection.run_all --run R1 --k 300  # Single run

For parallel execution of scenarios, use the dedicated modules:
    # Run a single scenario independently
    python -m coreset_selection.run_scenario R1 --data-dir data

    # Run multiple scenarios in parallel
    python -m coreset_selection.parallel_runner --n-workers 4 --data-dir data

    # Generate shell commands for external parallel execution
    python -m coreset_selection.parallel_runner --generate-commands
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import List, Optional

from ..cli import build_base_config
from ..config.dataclasses import ExperimentConfig
from ..config.run_specs import get_run_specs, apply_run_spec, K_GRID
from ..data.cache import prebuild_full_cache
from ..experiment.runner import run_single_experiment
from ..experiment.saver import claim_next_rep_id
from ..artifacts.generator import ManuscriptArtifactGenerator


# Default configuration
DEFAULT_N_REPLICATES = 5  # R1/R5 use 5 seeds per manuscript
DEFAULT_K_GRID = [50, 100, 200, 300, 400, 500]


def run_prep(
    cfg: ExperimentConfig,
    n_replicates: int = DEFAULT_N_REPLICATES,
    fail_fast: bool = True,
) -> bool:
    """
    Build replicate caches (idempotent — skips if cache already valid).
    
    Each replicate cache is built with BOTH VAE and PCA representations
    so that all experiment configurations (R0-R12) share the exact same
    learned representations for each replicate.  This is critical for
    fair cross-experiment comparison.
    
    Parameters
    ----------
    cfg : ExperimentConfig
        Base configuration
    n_replicates : int
        Number of replicates to build
    fail_fast : bool
        Stop on first failure
        
    Returns
    -------
    bool
        True if all replicates built successfully
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Building {n_replicates} replicate caches (VAE + PCA)")
    print(f"         Existing valid caches will be reused.")
    print(f"{'='*60}")
    
    success = True
    
    for rep_id in range(n_replicates):
        print(f"\n[prep] Ensuring replicate {rep_id} cache...")
        
        try:
            cache_path = prebuild_full_cache(cfg, rep_id, seed=cfg.seed)
            print(f"[prep] Ready: {cache_path}")
        except Exception as e:
            print(f"[prep] FAILED rep {rep_id}: {e}")
            success = False
            if fail_fast:
                return False
    
    print(f"\n[prep] All {n_replicates} replicate caches ready")
    return success


def run_experiments(
    cfg: ExperimentConfig,
    run_ids: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    fail_fast: bool = False,
) -> bool:
    """
    Run experiments with auto-incrementing replicate IDs.
    
    Each call appends ``n_replicates`` new replicates starting from the
    next available ID found in the output directory.
    
    Parameters
    ----------
    cfg : ExperimentConfig
        Base configuration
    run_ids : Optional[List[str]]
        Run IDs to execute (None = all)
    k_values : Optional[List[int]]
        k values to sweep (None = from run spec)
    n_replicates : int
        Number of new replicates to add per (run, k) combination
    fail_fast : bool
        Stop on first failure
        
    Returns
    -------
    bool
        True if all experiments completed successfully
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Running experiments")
    print(f"{'='*60}")
    
    run_specs = get_run_specs()
    
    if run_ids is None:
        run_ids = list(run_specs.keys())
    
    success = True
    total_runs = 0
    failed_runs = 0
    
    for run_id in run_ids:
        if run_id not in run_specs:
            print(f"[run] WARNING: Unknown run ID {run_id}, skipping")
            continue
        
        spec = run_specs[run_id]
        
        # Determine k values
        if k_values is not None:
            ks = k_values
        elif hasattr(spec, 'sweep_k') and spec.sweep_k:
            ks = K_GRID
        else:
            ks = [spec.k] if hasattr(spec, 'k') else [300]
        
        for k in ks:
            run_name = f"{run_id}_k{k}" if len(ks) > 1 else run_id

            # Determine replicates for this specific k value
            n_reps = spec.get_n_reps_for_k(k)

            # Auto-increment: atomically claim rep IDs (concurrency-safe)
            rep_ids = [
                claim_next_rep_id(cfg.files.output_dir, run_name)
                for _ in range(n_reps)
            ]

            for rep in rep_ids:
                print(f"\n[run] {run_name} rep={rep}...")
                
                try:
                    run_cfg = apply_run_spec(cfg, spec, rep)
                    run_cfg = replace(
                        run_cfg,
                        run_id=run_name,
                        rep_id=int(rep),
                        seed=int(cfg.seed + rep),
                        solver=replace(run_cfg.solver, k=k),
                    )
                    
                    result = run_single_experiment(run_cfg)
                    total_runs += 1
                    print(f"[run] Completed: {run_name} rep={rep}")
                    
                except Exception as e:
                    print(f"[run] FAILED: {run_name} rep={rep}: {e}")
                    failed_runs += 1
                    success = False
                    if fail_fast:
                        return False
    
    print(f"\n[run] Completed {total_runs} runs, {failed_runs} failed")
    return success


def run_artifacts(
    cfg: ExperimentConfig,
    output_dir: str = "artifacts_out",
) -> bool:
    """
    Generate manuscript artifacts.
    
    Parameters
    ----------
    cfg : ExperimentConfig
        Configuration (for paths)
    output_dir : str
        Output directory for artifacts
        
    Returns
    -------
    bool
        True if artifacts generated successfully
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generating manuscript artifacts")
    print(f"{'='*60}")
    
    try:
        # Legacy generator (backward compatibility)
        generator = ManuscriptArtifactGenerator(
            runs_root=cfg.files.output_dir,
            cache_root=cfg.files.cache_dir,
            out_dir=output_dir,
        )
        generated = generator.generate_all()
        
        print(f"\n[artifacts] Legacy generator:")
        print(f"  Figures: {len(generated.get('figures', []))}")
        print(f"  Tables: {len(generated.get('tables', []))}")
        
        # New manuscript-aligned generator (comprehensive)
        from ..artifacts.manuscript_artifacts import ManuscriptArtifacts
        ms_dir = os.path.join(output_dir, "analysis_out_full_taxonomy")
        ms_gen = ManuscriptArtifacts(
            runs_root=cfg.files.output_dir,
            cache_root=cfg.files.cache_dir,
            out_dir=ms_dir,
            data_dir=cfg.files.data_dir,
        )
        ms_generated = ms_gen.generate_all()
        
        print(f"\n[artifacts] Manuscript-aligned generator:")
        print(f"  Figures: {len(ms_generated.get('figures', []))}")
        print(f"  Tables: {len(ms_generated.get('tables', []))}")
        
        return True
        
    except Exception as e:
        print(f"[artifacts] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Brazil telecom coreset selection experiments"
    )
    
    # Data arguments
    parser.add_argument("--data-dir", default="data",
                       help="Directory containing input data files")
    parser.add_argument("--output-dir", default="runs_out",
                       help="Output directory for results")
    parser.add_argument("--cache-dir", default="replicate_cache",
                       help="Directory for replicate caches")
    parser.add_argument("--artifacts-dir", default="artifacts_out",
                       help="Output directory for artifacts")
    
    # Run control
    parser.add_argument("--prep-only", action="store_true",
                       help="Only build replicate caches")
    parser.add_argument("--skip-prep", action="store_true",
                       help="Skip cache building (use existing caches)")
    parser.add_argument("--artifacts-only", action="store_true",
                       help="Only generate artifacts")
    parser.add_argument("--time-complexity", action="store_true",
                       help="Run time complexity analysis vs k (extra experiment)")
    parser.add_argument("--run", type=str, default=None,
                       help="Single run ID to execute (e.g., R1)")
    parser.add_argument("--k", type=int, default=None,
                       help="Single k value to test")
    
    # Experiment parameters
    parser.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES,
                       help="Number of replicates")
    parser.add_argument("--seed", type=int, default=123,
                       help="Base random seed")
    parser.add_argument("--device", default="cpu",
                       help="Compute device (cpu/cuda)")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    # Build base config
    cfg = build_base_config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        device=args.device,
    )
    
    print(f"Brazil Telecom Coreset Selection Experiments")
    print(f"=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Replicates: {args.n_replicates}")
    
    # Artifacts only mode
    if args.artifacts_only:
        success = run_artifacts(cfg, args.artifacts_dir)
        sys.exit(0 if success else 1)
    
    # Time complexity experiment (extra, not in manuscript)
    if args.time_complexity:
        print(f"\n{'='*60}")
        print(f"EXTRA: Running time complexity analysis vs k")
        print(f"{'='*60}")
        try:
            from ..data.cache import ensure_replicate_cache, load_replicate_cache
            from ..geo.info import build_geo_info
            from ..geo.projector import GeographicConstraintProjector
            from ..experiment.time_complexity import (
                run_time_complexity_experiment,
                plot_time_complexity,
                save_time_complexity_summary,
            )
            cache_path = ensure_replicate_cache(cfg, 0)
            assets = load_replicate_cache(cache_path)
            geo = build_geo_info(
                assets.state_labels,
                population_weights=getattr(assets, "population", None),
            )
            projector = GeographicConstraintProjector(
                geo=geo,
                alpha_geo=float(cfg.geo.alpha_geo),
                min_one_per_group=bool(cfg.geo.min_one_per_group),
            )
            tc_df = run_time_complexity_experiment(
                cfg, assets, geo, projector,
                constraint_set=None,
                seed=cfg.seed,
                k_grid=[50, 100, 200, 300, 400, 500],
            )
            tc_out = os.path.join(args.artifacts_dir, "time_complexity")
            os.makedirs(tc_out, exist_ok=True)

            # Save structured CSV with complexity annotations & power-law fits
            summary_csv = save_time_complexity_summary(tc_df, tc_out)
            print(f"[time-complexity] Summary CSV: {summary_csv}")

            # Also save the raw timing data
            tc_df.to_csv(os.path.join(tc_out, "time_complexity_raw.csv"), index=False)

            # Generate publication-quality figures
            fig_paths = plot_time_complexity(tc_df, tc_out)
            print(f"[time-complexity] Generated {len(fig_paths)} figures:")
            for fp in fig_paths:
                print(f"  {fp}")
            print("[time-complexity] Done.")
        except Exception as e:
            print(f"[time-complexity] FAILED: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(0)
    
    # Prep only mode
    if args.prep_only:
        success = run_prep(cfg, args.n_replicates, args.fail_fast)
        sys.exit(0 if success else 1)
    
    # Single run mode
    if args.run is not None:
        run_ids = [args.run]
        k_values = [args.k] if args.k else None
    else:
        run_ids = None
        k_values = None
    
    # Full pipeline
    success = True
    
    # Phase 1: Build caches (idempotent — skips if already valid)
    if not args.skip_prep:
        success = run_prep(cfg, args.n_replicates, args.fail_fast)
        if not success and args.fail_fast:
            sys.exit(1)
    else:
        print(f"\n[skip-prep] Skipping cache build (--skip-prep). Using existing caches.")
    
    # Phase 2: Run experiments
    success = run_experiments(
        cfg, run_ids, k_values, args.n_replicates, args.fail_fast
    ) and success
    if not success and args.fail_fast:
        sys.exit(1)
    
    # Phase 3: Generate artifacts
    success = run_artifacts(cfg, args.artifacts_dir) and success
    
    print(f"\n{'='*60}")
    print(f"COMPLETED: {'SUCCESS' if success else 'WITH FAILURES'}")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
