"""Backward-compatibility stub — real implementation in runners/parallel.py.

This module re-exports everything from ``runners.parallel`` so that
existing imports and ``python -m coreset_selection.parallel_runner`` continue
to work.
"""
from .runners.parallel import *  # noqa: F401,F403
from .runners.parallel import (
    run_scenario_subprocess,
    run_scenarios_parallel_subprocess,
    run_scenarios_sequential,
    get_scenario_dependencies,
    topological_sort_scenarios,
    build_scenario_command,
    generate_shell_commands,
    generate_slurm_script,
    main,
)

if __name__ == "__main__":
    main()
