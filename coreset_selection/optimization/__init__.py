"""
Optimization utilities.

This package includes:
- NSGA-II: Self-contained implementation for 2-3 objectives

Legacy pymoo-based operators are imported only if `pymoo` is available.
"""

from __future__ import annotations

from .selection import (
    crowding_distance,
    feasible_filter,
    select_knee,
    select_pareto_representatives,
)

from .nsga2_internal import fast_non_dominated_sort, nsga2_optimize

# Optional legacy imports (require pymoo)
_HAS_PYMOO = False
try:
    from .problem import CoresetMOOProblem  # noqa: F401
    from .sampling import QuotaBinarySampling, ExactKSampling  # noqa: F401
    from .repair import (  # noqa: F401
        QuotaAndCardinalityRepair,
        ExactKRepair,
        LeastHarmQuotaRepair,
        LeastHarmExactKRepair,
        RepairActivityTracker,
    )
    from .operators import UniformBinaryCrossover, QuotaSwapMutation  # noqa: F401

    _HAS_PYMOO = True
except Exception:
    # Keep package importable even when pymoo is not installed.
    pass

__all__ = [
    "crowding_distance",
    "feasible_filter",
    "select_knee",
    "select_pareto_representatives",
    "fast_non_dominated_sort",
    "nsga2_optimize",
    "_HAS_PYMOO",
]
