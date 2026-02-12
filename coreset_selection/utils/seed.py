"""
Random seed utilities for reproducibility.

Contains:
- set_global_seed: Set seeds for numpy, torch, and random
- stable_hash_int: Deterministic string hashing for seed generation
"""

from __future__ import annotations

import random
import zlib
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (if available)
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    seed = int(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stable_hash_int(s: str, mod: int = 10000) -> int:
    """
    Generate a stable integer hash from a string.
    
    Uses CRC32 for deterministic cross-platform hashing.
    Useful for generating reproducible seeds from string identifiers.
    
    Parameters
    ----------
    s : str
        String to hash
    mod : int
        Modulus for the result (default 10000)
        
    Returns
    -------
    int
        Hash value in range [0, mod)
    """
    # Use CRC32 for deterministic hashing
    hash_val = zlib.crc32(s.encode('utf-8')) & 0xffffffff
    return hash_val % mod


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a numpy random generator.
    
    Parameters
    ----------
    seed : Optional[int]
        Random seed. If None, uses entropy from OS.
        
    Returns
    -------
    np.random.Generator
        Initialized random generator
    """
    return np.random.default_rng(seed)


def seed_sequence(base_seed: int, n_seeds: int) -> list:
    """
    Generate a sequence of independent seeds from a base seed.
    
    Uses numpy's SeedSequence for proper seed generation.
    
    Parameters
    ----------
    base_seed : int
        Base seed value
    n_seeds : int
        Number of seeds to generate
        
    Returns
    -------
    list
        List of integer seeds
    """
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_seeds)
    return [cs.generate_state(1)[0] for cs in child_seeds]
