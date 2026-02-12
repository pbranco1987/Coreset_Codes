"""Backward-compatibility stub — real implementation in runners/run_all.py.

This module re-exports everything from ``runners.run_all`` so that
existing imports and ``python -m coreset_selection.run_all`` continue
to work.
"""
from .runners.run_all import *  # noqa: F401,F403
from .runners.run_all import run_prep, run_experiments, run_artifacts, main

if __name__ == "__main__":
    main()
