"""Helper functions for the standalone scenario runner.

Extracted from ``run_scenario.py`` to keep the main module focused on the
experiment runner and CLI entry point.  All public names are re-exported by
``run_scenario``.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def _get_parallel_experiments_from_argv():
    """Quick parse of --parallel-experiments before full argparse."""
    for i, arg in enumerate(sys.argv):
        if arg == "--parallel-experiments" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
        if arg.startswith("--parallel-experiments="):
            try:
                return int(arg.split("=")[1])
            except (ValueError, IndexError):
                pass
    return None


def _get_cpu_count():
    """Get available CPU count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integer list."""
    if not s:
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out
