"""CLI interface for coreset_selection.

All implementation lives in the cli subpackage modules. This __init__
re-exports every public name so that existing imports such as

    from coreset_selection.cli import cli
    from coreset_selection.cli import build_base_config

continue to work unchanged.

Thread Control
--------------
set_thread_limits and _parse_threads_early are executed eagerly
(below) so that thread caps are applied BEFORE numpy/torch are
imported by downstream modules.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# 1.  Thread-limit bootstrap — MUST happen before numpy/torch imports
# ------------------------------------------------------------------
from ._thread_setup import set_thread_limits, _parse_threads_early

# Apply thread limits early if specified
_early_threads = _parse_threads_early()
if _early_threads is not None:
    set_thread_limits(_early_threads)

# ------------------------------------------------------------------
# 2.  Re-export everything from the sub-modules
# ------------------------------------------------------------------
from ._config import build_base_config, _parse_int_list

from ._commands import (
    cmd_prep,
    cmd_run,
    cmd_sweep,
    cmd_parallel,
    cmd_artifacts,
    cmd_list_runs,
    cmd_scenario,
    _cmd_scenario_fixed,
    cmd_all,
    cmd_seq,
    _run_sequential,
)

from ._main import main, cli

__all__ = [
    # thread setup
    "set_thread_limits",
    "_parse_threads_early",
    # config helpers
    "build_base_config",
    "_parse_int_list",
    # command handlers
    "cmd_prep",
    "cmd_run",
    "cmd_sweep",
    "cmd_parallel",
    "cmd_artifacts",
    "cmd_list_runs",
    "cmd_scenario",
    "_cmd_scenario_fixed",
    "cmd_all",
    "cmd_seq",
    "_run_sequential",
    # main entry points
    "main",
    "cli",
]

if __name__ == "__main__":
    cli()
