"""Experiment runner orchestration modules.

This package contains modules for running experiment scenarios:
- ``scenario``: Standalone single-scenario runner (run_scenario_standalone)
- ``parallel``: Multi-scenario parallel execution (run_scenarios_parallel_subprocess)
- ``run_all``: Full pipeline orchestration (prep + experiments + artifacts)
"""

from __future__ import annotations

from .scenario import run_scenario_standalone, main as scenario_main
from .parallel import (
    run_scenario_subprocess,
    run_scenarios_parallel_subprocess,
    run_scenarios_sequential,
    main as parallel_main,
    get_scenario_dependencies,
    topological_sort_scenarios,
    build_scenario_command,
    generate_shell_commands,
    generate_slurm_script,
)
from .run_all import run_prep, run_experiments, run_artifacts, main as run_all_main

__all__ = [
    # scenario
    "run_scenario_standalone",
    "scenario_main",
    # parallel
    "run_scenario_subprocess",
    "run_scenarios_parallel_subprocess",
    "run_scenarios_sequential",
    "parallel_main",
    "get_scenario_dependencies",
    "topological_sort_scenarios",
    "build_scenario_command",
    "generate_shell_commands",
    "generate_slurm_script",
    # run_all
    "run_prep",
    "run_experiments",
    "run_artifacts",
    "run_all_main",
]
