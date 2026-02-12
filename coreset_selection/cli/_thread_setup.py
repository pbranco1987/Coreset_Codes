"""Thread setup utilities — must run BEFORE numpy/torch imports."""

from __future__ import annotations

import os
import sys
from typing import Optional


def set_thread_limits(n_threads: int) -> None:
    """
    Limit threads for NumPy/BLAS/OpenMP BEFORE importing numpy.

    Must be called before any numpy import to take effect.
    """
    n = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_MAX_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n

    # Also limit PyTorch if used
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Try to set torch threads if already imported
    try:
        import torch
        torch.set_num_threads(n_threads)
    except (ImportError, RuntimeError):
        pass


# Parse --threads/-j early, before other imports
def _parse_threads_early() -> Optional[int]:
    """Parse -j/--threads before full argument parsing."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ('-j', '--threads') and i < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except (ValueError, IndexError):
                pass
        elif arg.startswith('-j') and len(arg) > 2:
            try:
                return int(arg[2:])
            except ValueError:
                pass
        elif arg.startswith('--threads='):
            try:
                return int(arg.split('=')[1])
            except (ValueError, IndexError):
                pass
    return None
