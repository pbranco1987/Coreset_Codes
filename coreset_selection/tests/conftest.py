"""
Pytest configuration and fixtures for coreset_selection tests.
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """Simple synthetic dataset."""
    N, d = 100, 10
    X = rng.normal(size=(N, d))
    y = rng.normal(size=N)
    return X, y


@pytest.fixture
def simple_geo_info():
    """Simple GeoInfo with 3 groups."""
    from coreset_selection.geo.info import GeoInfo
    
    # 30 points, 3 groups of 10 each
    group_ids = np.array([0]*10 + [1]*10 + [2]*10)
    groups = ["A", "B", "C"]
    return GeoInfo.from_group_ids(group_ids, groups)
