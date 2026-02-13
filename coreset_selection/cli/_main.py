"""Main entry point and argparse setup for the CLI."""

from __future__ import annotations

import argparse
import sys

from ..config.run_specs import get_run_specs
from ._commands import (
    cmd_prep, cmd_run, cmd_sweep, cmd_parallel,
    cmd_artifacts, cmd_list_runs, cmd_scenario,
    _cmd_scenario_fixed, cmd_all, cmd_seq,
)


def main() -> int:
    """
    Main entry point for CLI.
    """
    parser = argparse.ArgumentParser(
        prog="coreset_selection",
        description="Geographically-constrained Pareto coreset selection",
    )

    # Global thread control (already applied early, but keep for help text)
    parser.add_argument(
        "-j", "--threads", type=int, default=None, metavar="N",
        help="Limit threads per process (for parallel runs). "
             "E.g., -j4 for 4 threads. Set to cores/num_parallel_jobs."
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # prep command
    prep_parser = subparsers.add_parser("prep", help="Prepare replicate caches")
    prep_parser.add_argument("--n-replicates", type=int, default=10,
                            help="Number of replicates to prepare")
    prep_parser.add_argument("--output-dir", default="runs_out",
                            help="Output directory")
    prep_parser.add_argument("--cache-dir", default="replicate_cache",
                            help="Cache directory")
    prep_parser.add_argument("--data-dir", default="data",
                            help="Data directory containing input files")
    prep_parser.add_argument("--seed", type=int, default=123,
                            help="Base random seed")
    prep_parser.add_argument("--device", default="cpu",
                            help="Compute device (cpu/cuda)")
    prep_parser.add_argument("--fail-fast", action="store_true",
                            help="Stop on first failure")
    prep_parser.set_defaults(func=cmd_prep)

    # run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--run-id", required=True,
                           help="Run identifier (e.g., R1)")
    run_parser.add_argument("--rep-id", type=int, default=0,
                           help="Replicate ID")
    run_parser.add_argument("--k", type=int, required=True,
                           help="Coreset size k")
    run_parser.add_argument("--output-dir", default="runs_out",
                           help="Output directory")
    run_parser.add_argument("--cache-dir", default="replicate_cache",
                           help="Cache directory")
    run_parser.add_argument("--data-dir", default="data",
                           help="Data directory containing input files")
    run_parser.add_argument("--seed", type=int, default=123,
                           help="Base random seed")
    run_parser.add_argument("--device", default="cpu",
                           help="Compute device")
    run_parser.set_defaults(func=cmd_run)

    # scenario command (run full R# scenario independently)
    scenario_parser = subparsers.add_parser(
        "scenario",
        help="Run a full manuscript scenario (R0-R11) as a standalone job",
    )
    scenario_parser.add_argument("--run-id", required=True, choices=sorted(get_run_specs().keys()))
    scenario_parser.add_argument("--k-values", default=None, help="Comma-separated k values (overrides run spec)")
    scenario_parser.add_argument("--rep-ids", default=None, help="Comma-separated replicate IDs (overrides run spec)")
    scenario_parser.add_argument("--output-dir", default="runs_out")
    scenario_parser.add_argument("--cache-dir", default="replicate_cache")
    scenario_parser.add_argument("--data-dir", default="data")
    scenario_parser.add_argument("--seed", type=int, default=123)
    scenario_parser.add_argument("--device", default="cpu")
    scenario_parser.add_argument("--fail-fast", action="store_true")
    # R6 extras
    scenario_parser.add_argument("--source-run", default="R1", help="(R6) Source run base ID")
    scenario_parser.add_argument("--source-space", default="vae", help="(R6) Source space: vae|pca|raw")
    scenario_parser.add_argument("--k", type=int, default=None, help="(R6) k value for source run")
    scenario_parser.set_defaults(func=cmd_scenario)

    # Convenience aliases: r0, r1, ..., r10
    # Show default k values in help text
    _k_info = {
        "R0": "k-sweep 50-500, quota path + KL baseline",
        "R1": "k-sweep 50-500, VAE bi-obj MMD+Sinkhorn, pop-share, 5 seeds",
        "R2": "single k, MMD-only ablation, VAE",
        "R3": "single k, Sinkhorn-only ablation, VAE",
        "R4": "single k, municipality-quota constraint swap, VAE",
        "R5": "single k, joint constraints (pop+muni), VAE",
        "R6": "single k, unconstrained (exact-k only), VAE",
        "R7": "single k, tri-objective MMD+Sinkhorn+SKL, VAE",
        "R8": "k-sweep 50-500, raw space, MMD+Sinkhorn",
        "R9": "k-sweep 50-500, PCA space, MMD+Sinkhorn",
        "R10": "single k, baseline suite (8 methods)",
        "R11": "single k, diagnostics (needs R1 first)",
        "R12": "effort sweep (pop_size x n_gen)",
        "R13": "VAE latent dim sweep D in {4..128}",
        "R14": "PCA dim sweep D in {4..128}",
    }

    for _rid in [f"r{i}" for i in range(15)]:
        _run_id = _rid.upper()
        _info = _k_info.get(_run_id, "")
        help_text = f"Run scenario {_run_id}" + (f" ({_info})" if _info else "")

        p = subparsers.add_parser(_rid, help=help_text)

        # Add simple -k option (single value or comma-separated) — REQUIRED
        p.add_argument("-k", dest="k_single", type=str, required=True,
                      metavar="K",
                      help="Coreset size(s): single value (e.g., -k 200) or comma-separated (e.g., -k 100,200)")
        p.add_argument("--k-values", default=None,
                      help="[deprecated] Use -k instead")
        p.add_argument("--rep-ids", default=None, help="Comma-separated replicate IDs (overrides run spec)")
        p.add_argument("--output-dir", default="runs_out")
        p.add_argument("--cache-dir", default="replicate_cache")
        p.add_argument("--data-dir", default="data")
        p.add_argument("--seed", type=int, default=123)
        p.add_argument("--device", default="cpu")
        p.add_argument("--fail-fast", action="store_true")
        if _run_id == "R6":
            p.add_argument("--source-run", default="R1", help="Source run base ID")
            p.add_argument("--source-space", default="vae", help="Source space: vae|pca|raw")
        p.set_defaults(func=_cmd_scenario_fixed(_run_id))

    # sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run sweep over k values")
    sweep_parser.add_argument("--run-id", required=True,
                             help="Run identifier")
    sweep_parser.add_argument("--k-values", required=True,
                             help="Comma-separated k values (e.g., 100,200,300)")
    sweep_parser.add_argument("--n-replicates", type=int, default=10,
                             help="Number of replicates per k")
    sweep_parser.add_argument("--output-dir", default="runs_out",
                             help="Output directory")
    sweep_parser.add_argument("--cache-dir", default="replicate_cache",
                             help="Cache directory")
    sweep_parser.add_argument("--data-dir", default="data",
                             help="Data directory containing input files")
    sweep_parser.add_argument("--seed", type=int, default=123,
                             help="Base random seed")
    sweep_parser.add_argument("--device", default="cpu",
                             help="Compute device")
    sweep_parser.add_argument("--fail-fast", action="store_true",
                             help="Stop on first failure")
    sweep_parser.set_defaults(func=cmd_sweep)

    # parallel command - run multiple experiments with optimal threading
    parallel_parser = subparsers.add_parser(
        "parallel",
        aliases=["par"],
        help="Run multiple experiments in parallel (logs to files)")
    parallel_parser.add_argument("runs", nargs="*", default=["all"],
                                help="Experiments to run (e.g., r1 r2 r3) or 'all' for r0-r9")
    parallel_parser.add_argument("-t", "--threads", type=int, default=None,
                                help="Threads per job (auto-calculated if not set)")
    parallel_parser.add_argument("-l", "--log-dir", default="logs",
                                help="Directory for log files (default: logs/)")
    parallel_parser.add_argument("--output-dir", default="runs_out",
                                help="Output directory for results")
    parallel_parser.add_argument("--cache-dir", default="replicate_cache",
                                help="Cache directory")
    parallel_parser.add_argument("--data-dir", default="data",
                                help="Data directory")
    parallel_parser.set_defaults(func=cmd_parallel)

    # all command - run ALL experiments sequentially
    all_parser = subparsers.add_parser(
        "all",
        help="Run ALL experiments sequentially (r0-r10)")
    all_parser.add_argument("-k", dest="k_values", type=str, required=True,
                           metavar="K",
                           help="Coreset size(s): single value or comma-separated (e.g., -k 300)")
    all_parser.add_argument("--rep-ids", default=None,
                           help="Comma-separated replicate IDs")
    all_parser.add_argument("--output-dir", default="runs_out",
                           help="Output directory for results")
    all_parser.add_argument("--cache-dir", default="replicate_cache",
                           help="Cache directory")
    all_parser.add_argument("--data-dir", default="data",
                           help="Data directory")
    all_parser.add_argument("--seed", type=int, default=123,
                           help="Base random seed")
    all_parser.add_argument("--device", default="cpu",
                           help="Compute device")
    all_parser.add_argument("--fail-fast", action="store_true",
                           help="Stop on first failure")
    all_parser.set_defaults(func=cmd_all)

    # seq command - run SELECTED experiments sequentially
    seq_parser = subparsers.add_parser(
        "seq",
        help="Run selected experiments sequentially (e.g., seq r1 r2 r6)")
    seq_parser.add_argument("runs", nargs="+",
                           help="Experiments to run (e.g., r1 r2 r6)")
    seq_parser.add_argument("-k", dest="k_values", type=str, required=True,
                           metavar="K",
                           help="Coreset size(s): single value or comma-separated (e.g., -k 300)")
    seq_parser.add_argument("--rep-ids", default=None,
                           help="Comma-separated replicate IDs")
    seq_parser.add_argument("--output-dir", default="runs_out",
                           help="Output directory for results")
    seq_parser.add_argument("--cache-dir", default="replicate_cache",
                           help="Cache directory")
    seq_parser.add_argument("--data-dir", default="data",
                           help="Data directory")
    seq_parser.add_argument("--seed", type=int, default=123,
                           help="Base random seed")
    seq_parser.add_argument("--device", default="cpu",
                           help="Compute device")
    seq_parser.add_argument("--fail-fast", action="store_true",
                           help="Stop on first failure")
    seq_parser.set_defaults(func=cmd_seq)

    # artifacts command (with alias 'figs')
    artifacts_parser = subparsers.add_parser(
        "artifacts",
        aliases=["figs", "plots"],
        help="Generate manuscript figures and tables (auto-detects runs_out/)")
    artifacts_parser.add_argument("runs_dir", nargs="?", default=None,
                                  help="Directory with run results (auto-detects)")
    artifacts_parser.add_argument("-o", "--output", default=None,
                                  help="Output directory (default: artifacts/)")
    artifacts_parser.add_argument("--rep", default="rep00",
                                  help="Replicate folder (default: rep00)")
    artifacts_parser.add_argument("--data-dir", default="data",
                                  help="Data directory (for choropleth maps)")
    artifacts_parser.set_defaults(func=cmd_artifacts)

    # list-runs command
    list_parser = subparsers.add_parser("list-runs",
                                         help="List available run specifications")
    list_parser.set_defaults(func=cmd_list_runs)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


def cli():
    """Entry point for console script."""
    sys.exit(main())
