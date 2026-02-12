"""coreset_selection.utils.torch_perf

Lightweight performance helpers for PyTorch.

Context
-------
In some environments (Jupyter, shared servers, certain schedulers), thread
environment variables (e.g. OMP_NUM_THREADS / MKL_NUM_THREADS) may be set very
conservatively (sometimes to ``1``). That can make PyTorch CPU training appear
to "hang" at the first epoch because each epoch becomes extremely slow.

CRITICAL FOR PARALLEL EXECUTION:
When running multiple experiments (R0-R11) in parallel, each process must use
a LIMITED number of threads. Without limits, N processes each using all cores
causes massive thread contention and 10-100x slowdown.

Set the environment variable before running:
    export CORESET_NUM_THREADS=20  # for 10 parallel experiments on 200 cores

Or use --parallel-experiments flag which sets this automatically.

This module provides a single, safe helper to configure PyTorch to use the
available CPUs (respecting CPU affinity when available).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def available_cpu_count() -> int:
    """Return the number of CPUs available to the current process.

    We prefer ``os.sched_getaffinity`` (Linux) because it respects CPU affinity
    and container cgroup pinning. Falls back to ``os.cpu_count``.
    """

    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _get_thread_limit_from_env() -> Optional[int]:
    """
    Check environment variables for thread limits.
    
    Priority:
    1. CORESET_NUM_THREADS - explicit thread count
    2. CORESET_PARALLEL_EXPERIMENTS - auto-calculate from parallel count
    3. OMP_NUM_THREADS - respect existing OpenMP setting
    
    Returns None if no limit is set.
    """
    # Check explicit thread count
    num_threads = os.environ.get("CORESET_NUM_THREADS")
    if num_threads is not None:
        try:
            return max(1, int(num_threads))
        except ValueError:
            pass
    
    # Check parallel experiments count for auto-calculation
    n_parallel = os.environ.get("CORESET_PARALLEL_EXPERIMENTS")
    if n_parallel is not None:
        try:
            n_parallel = int(n_parallel)
            if n_parallel > 1:
                n_cpus = available_cpu_count()
                # Leave ~5% for system overhead
                return max(1, int(n_cpus * 0.95) // n_parallel)
        except ValueError:
            pass
    
    # Check if OMP_NUM_THREADS is already set (respect external config)
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if omp_threads is not None:
        try:
            return max(1, int(omp_threads))
        except ValueError:
            pass
    
    return None


def _set_env_threads(n: int, *, force: bool) -> None:
    """Set common thread env vars.

    If ``force`` is False, only sets vars that are currently unset.
    """

    n = max(1, int(n))
    vars_to_set = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    for var in vars_to_set:
        if force or (os.environ.get(var) is None):
            os.environ[var] = str(n)


def configure_torch_threads(
    *,
    torch_threads: Optional[int] = None,
    torch_interop_threads: Optional[int] = None,
    force_env: bool = True,
    verbose: bool = False,
) -> Tuple[int, Optional[int]]:
    """Configure PyTorch to use appropriate CPU threads.

    Parameters
    ----------
    torch_threads:
        Intra-op thread count (``torch.set_num_threads``). If None or <=0,
        checks environment variables, then falls back to available CPU count.
        
        IMPORTANT FOR PARALLEL EXECUTION:
        When running multiple experiments simultaneously, set CORESET_NUM_THREADS
        or CORESET_PARALLEL_EXPERIMENTS environment variable to avoid thread
        contention.
        
    torch_interop_threads:
        Inter-op thread count (``torch.set_num_interop_threads``). If None,
        leaves PyTorch default. If provided and >0, we attempt to set it.
        Note: PyTorch requires interop threads to be set early (before parallel
        work starts); if it's too late, we silently ignore the error.
    force_env:
        If True, overwrite common thread environment variables.
    verbose:
        If True, print the configured values.

    Returns
    -------
    (threads, interop_threads)
        The actual intra-op thread count used and the inter-op value requested
        (or None if not requested).
    """
    import sys
    
    n_avail = available_cpu_count()
    n_threads = int(torch_threads or 0)
    
    if n_threads <= 0:
        # Check environment for thread limits
        env_limit = _get_thread_limit_from_env()
        if env_limit is not None:
            n_threads = env_limit
            if verbose:
                print(
                    f"[torch_perf] Using thread limit from environment: {n_threads}",
                    file=sys.stderr
                )
        else:
            n_threads = 4  # Fixed default: 4 threads per experiment
            if verbose:
                print(
                    f"[torch_perf] Using default thread count: {n_threads} "
                    f"(available={n_avail})",
                    file=sys.stderr
                )

    # Environment variables can influence BLAS/OpenMP backends.
    _set_env_threads(n_threads, force=force_env)

    # Best-effort PyTorch configuration.
    try:
        import torch

        torch.set_num_threads(int(n_threads))
        if torch_interop_threads is not None and int(torch_interop_threads) > 0:
            try:
                torch.set_num_interop_threads(int(torch_interop_threads))
            except Exception:
                # Must be set before any parallel work starts.
                pass

        if verbose:
            try:
                cur_threads = torch.get_num_threads()
                cur_interop = torch.get_num_interop_threads()
                print(
                    f"[torch_perf] Configured: intra={cur_threads}, interop={cur_interop} "
                    f"(available={n_avail})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
    except Exception:
        # torch not installed or failed to import; ignore.
        pass

    return n_threads, (int(torch_interop_threads) if torch_interop_threads is not None else None)
