"""IO, locking, and validation utilities for the replicate cache."""

from __future__ import annotations

import os
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np


def atomic_savez(path: str, **arrays) -> None:
    """
    Atomic save of numpy arrays to compressed npz file.

    Writes to a temporary file first, then atomically replaces the target.
    This prevents corruption if the process crashes mid-write.

    Parameters
    ----------
    path : str
        Target path for the .npz file
    **arrays
        Arrays to save (passed to np.savez_compressed)
    """
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp.npz",
        dir=dir_
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)  # atomic on POSIX + Windows
    finally:
        # Clean up temp file if it still exists (e.g., if savez failed)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def validate_cache(cache_path: str, required_keys: List[str]) -> Tuple[bool, List[str]]:
    """Validate that a cache exists and contains required keys."""
    if not os.path.exists(cache_path):
        return False, list(required_keys)
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            existing = set(data.files)
        missing = [k for k in required_keys if k not in existing]
        return len(missing) == 0, missing
    except Exception:
        return False, list(required_keys)


def _acquire_build_lock(lock_path: str, *, timeout_s: float = 6 * 3600, poll_s: float = 2.0) -> int:
    """Acquire an inter-process lock via atomic file creation."""
    start = time.time()
    # Consider locks older than timeout stale (previously was 2 hours which could
    # break legitimate long-running processes before timeout)
    stale_s = timeout_s
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={os.getpid()}\nstart={time.time()}\n".encode("utf-8"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            # Best-effort stale lock cleanup
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_s:
                    os.remove(lock_path)
                    continue
            except Exception:
                pass

            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(poll_s)


def _release_build_lock(lock_path: str, fd: Optional[int]) -> None:
    """Release an inter-process lock."""
    try:
        if fd is not None:
            os.close(fd)
    except Exception:
        pass
    try:
        os.remove(lock_path)
    except Exception:
        pass
