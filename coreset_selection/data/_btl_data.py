"""
Brazil Telecom Loader -- Data container class.

Split out from ``brazil_telecom_loader.py`` for maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class BrazilTelecomData:
    """
    Container for loaded Brazil telecom infrastructure data.

    Attributes
    ----------
    extra_targets : Dict[str, np.ndarray]
        Additional coverage targets beyond the primary 4G/5G area-coverage
        pair.  Keys follow the canonical names defined in
        ``config.constants.COVERAGE_TARGETS`` (e.g. ``"cov_households_4G"``).
        Each value is an (N,) array of percentage-point values.
    feature_types : List[str]
        Parallel list of feature type strings aligned with ``feature_names``.
        Each entry is one of ``"numeric"``, ``"ordinal"``, ``"categorical"``.
    category_maps : Dict[str, Dict]
        For categorical features, ``{col_name: {original_value: int_code}}``.
    """
    X: np.ndarray
    state_labels: np.ndarray
    state_indices: np.ndarray
    y_4G: np.ndarray
    y_5G: np.ndarray
    ibge_codes: np.ndarray
    population: np.ndarray
    coords: np.ndarray
    feature_names: List[str]
    municipality_names: np.ndarray
    extra_targets: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_types: List[str] = field(default_factory=list)
    category_maps: Dict[str, Dict] = field(default_factory=dict)
    raw_df: object = None  # Full pre-feature-selection DataFrame (for derived targets)
