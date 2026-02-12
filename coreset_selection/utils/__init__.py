"""
Utility functions for coreset selection.

This module provides:
- Math utilities (median squared distance)
- I/O utilities (directory creation)
- Seed management (reproducibility)
- Plotting utilities (matplotlib configuration)
"""

from .math import median_sq_dist

from .io import ensure_dir

from .seed import (
    set_global_seed,
    stable_hash_int,
    get_rng,
    seed_sequence,
)

from .plotting import (
    set_manuscript_style,
    get_method_style,
    get_objective_color,
    objective_label,
    method_label,
    figure_size,
    add_panel_label,
    truncate_colormap,
    PALETTE_METHODS,
    PALETTE_OBJECTIVES,
    MARKERS_METHODS,
)

__all__ = [
    # Math
    "median_sq_dist",
    # I/O
    "ensure_dir",
    # Seed
    "set_global_seed",
    "stable_hash_int",
    "get_rng",
    "seed_sequence",
    # Plotting
    "set_manuscript_style",
    "get_method_style",
    "get_objective_color",
    "objective_label",
    "method_label",
    "figure_size",
    "add_panel_label",
    "truncate_colormap",
    "PALETTE_METHODS",
    "PALETTE_OBJECTIVES",
    "MARKERS_METHODS",
]
