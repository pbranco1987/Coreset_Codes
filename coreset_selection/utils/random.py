"""
Random utilities - backward compatibility module.

Re-exports from seed module for backward compatibility with existing imports.
"""

from .seed import (
    set_global_seed,
    stable_hash_int,
    get_rng,
    seed_sequence,
)

__all__ = [
    "set_global_seed",
    "stable_hash_int",
    "get_rng",
    "seed_sequence",
]
