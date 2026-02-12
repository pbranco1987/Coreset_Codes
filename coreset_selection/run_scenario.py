"""Backward-compatibility stub — real implementation in runners/scenario.py.

This module re-exports everything from ``runners.scenario`` so that
existing imports and ``python -m coreset_selection.run_scenario`` continue
to work.  The module-level thread-setup code in ``runners/scenario.py``
executes when this stub is imported.
"""
from .runners.scenario import *  # noqa: F401,F403
from .runners.scenario import run_scenario_standalone, main, parse_int_list

if __name__ == "__main__":
    main()
