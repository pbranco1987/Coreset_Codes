"""
Geographic constraint handling module.

This module provides:
- GeoInfo: Geographic group information
- KL divergence utilities for quota computation
- GeographicConstraintProjector: Project solutions to satisfy quotas
"""

from .info import (
    GeoInfo,
    build_geo_info,
    merge_small_groups,
)

from .kl import (
    kl_pi_hat_from_counts,
    kl_weighted_from_subset,
    compute_constraint_violations,
    kl_optimal_integer_counts_bounded,
    min_achievable_geo_kl_bounded,
    proportional_allocation,
    compute_quota_path,
    save_quota_path,
)

from .projector import (
    GeographicConstraintProjector,
    project_to_exact_k_mask,
    build_feasible_quota_mask,
    compute_quota_violation,
)

# Shapefile / choropleth utilities (optional — requires geopandas)
try:
    from .shapefile import (
        load_brazil_municipalities,
        get_brazil_border,
        plot_choropleth_panel,
    )
    _HAS_SHAPEFILE = True
except ImportError:
    _HAS_SHAPEFILE = False

__all__ = [
    # Info
    "GeoInfo",
    "build_geo_info",
    "merge_small_groups",
    # KL
    "kl_pi_hat_from_counts",
    "kl_weighted_from_subset",
    "compute_constraint_violations",
    "kl_optimal_integer_counts_bounded",
    "min_achievable_geo_kl_bounded",
    "proportional_allocation",
    "compute_quota_path",
    "save_quota_path",
    # Projector
    "GeographicConstraintProjector",
    "project_to_exact_k_mask",
    "build_feasible_quota_mask",
    "compute_quota_violation",
    # Shapefile (conditional)
    "load_brazil_municipalities",
    "get_brazil_border",
    "plot_choropleth_panel",
]
