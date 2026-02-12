"""
Experiment module for running coreset selection experiments.

This module provides:
- ExperimentRunner: Main experiment orchestration
- ResultsSaver: Unified output saving
- ParetoFrontData: Pareto front data structure
"""

from .saver import (
    ParetoFrontData,
    ResultsSaver,
    load_pareto_front,
)

from .runner import (
    ExperimentRunner,
    run_single_experiment,
    run_sweep,
)

__all__ = [
    # Saver
    "ParetoFrontData",
    "ResultsSaver",
    "load_pareto_front",
    # Runner
    "ExperimentRunner",
    "run_single_experiment",
    "run_sweep",
]
