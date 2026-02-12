"""
Repair operators for NSGA-II coreset selection.

Implements Algorithm 2 from manuscript (§5.2):
- QuotaAndCardinalityRepair: geographic quota + exact-k repair
- ExactKRepair: exact-k only repair
- LeastHarmRepair: proxy-aware repair minimizing RFF-MMD increase

Per manuscript (lines 933-936):
"As a heuristic, we implement a 'least-harm' variant that greedily removes
(or adds) indices that minimally worsen (or most improve) a cheap proxy
objective such as the RFF-MMD surrogate."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
try:  # optional dependency; the manuscript uses our internal NSGA-II
    from pymoo.core.repair import Repair  # type: ignore
except Exception:  # pragma: no cover
    class Repair:  # minimal fallback matching pymoo's interface
        """Lightweight stand-in for :class:`pymoo.core.repair.Repair`.

        The repository ships an internal NSGA-II implementation and therefore
        does not require `pymoo` to run the manuscript experiments. However,
        we keep these classes for API completeness and for users who prefer
        a pymoo-based pipeline.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

        def _do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
            raise NotImplementedError

        def do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
            return self._do(problem, X, **kwargs)

from ..geo import GeographicConstraintProjector


@dataclass
class RepairActivityTracker:
    """
    Tracks repair operator activity for diagnostics (R6).

    Monitors how often repair is invoked and how much it changes individuals.
    Per manuscript Section 6.8 (repair-operator activity).

    Attributes
    ----------
    total_offspring : int
        Total number of offspring processed
    repaired_count : int
        Number of offspring that required repair
    hamming_distances : List[int]
        Hamming distances between pre- and post-repair masks
    """
    total_offspring: int = 0
    repaired_count: int = 0
    hamming_distances: List[int] = field(default_factory=list)

    def record(self, pre_repair: np.ndarray, post_repair: np.ndarray) -> None:
        """
        Record a repair event.

        Parameters
        ----------
        pre_repair : np.ndarray
            Boolean mask before repair (n_individuals, n_var)
        post_repair : np.ndarray
            Boolean mask after repair (n_individuals, n_var)
        """
        n = pre_repair.shape[0]
        self.total_offspring += n
        for i in range(n):
            hd = int(np.sum(pre_repair[i] != post_repair[i]))
            if hd > 0:
                self.repaired_count += 1
                self.hamming_distances.append(hd)

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for R6 diagnostics.

        Returns
        -------
        Dict[str, Any]
            Summary with total, repaired count, fraction, and hamming stats
        """
        if not self.hamming_distances:
            return {
                "total_offspring": self.total_offspring,
                "repaired_count": self.repaired_count,
                "repaired_fraction": 0.0,
                "hamming_mean": 0.0,
                "hamming_std": 0.0,
                "hamming_max": 0,
                "hamming_median": 0.0,
                "hamming_q25": 0.0,
                "hamming_q75": 0.0,
            }
        hd = np.array(self.hamming_distances)
        return {
            "total_offspring": self.total_offspring,
            "repaired_count": self.repaired_count,
            "repaired_fraction": self.repaired_count / max(1, self.total_offspring),
            "hamming_mean": float(np.mean(hd)),
            "hamming_std": float(np.std(hd)),
            "hamming_max": int(np.max(hd)),
            "hamming_median": float(np.median(hd)),
            "hamming_q25": float(np.percentile(hd, 25)),
            "hamming_q75": float(np.percentile(hd, 75)),
        }


class QuotaAndCardinalityRepair(Repair):
    """
    Repair operator ensuring both geographic quotas and exact cardinality.

    Projects each individual to satisfy:
    1. Exactly k total selected points
    2. Exactly c*_g points from each geographic group g

    Attributes
    ----------
    k : int
        Target subset size
    projector : GeographicConstraintProjector
        Projector defining the quota constraints
    rng : np.random.Generator
        Random number generator
    tracker : Optional[RepairActivityTracker]
        Activity tracker for diagnostics
    """

    def __init__(
        self,
        k: int,
        projector: GeographicConstraintProjector,
        seed: int,
        track_activity: bool = False
    ):
        """
        Initialize the repair operator.

        Parameters
        ----------
        k : int
            Target subset size
        projector : GeographicConstraintProjector
            Projector defining the quota constraints
        seed : int
            Random seed
        track_activity : bool
            Whether to track repair activity statistics
        """
        super().__init__()
        self.k = int(k)
        self.projector = projector
        self.rng = np.random.default_rng(int(seed))
        self.tracker: Optional[RepairActivityTracker] = (
            RepairActivityTracker() if track_activity else None
        )

    def _do(self, problem, X, **kwargs):
        """
        Repair population to satisfy quota constraints.

        Parameters
        ----------
        problem : Problem
            The optimization problem
        X : np.ndarray
            Population of boolean masks (n_individuals, n_var)

        Returns
        -------
        np.ndarray
            Repaired population
        """
        X = np.asarray(X, dtype=bool)
        pre_repair = X.copy() if self.tracker else None

        for i in range(X.shape[0]):
            X[i] = self.projector.project_to_quota_mask(X[i], self.k, self.rng)

        if self.tracker is not None:
            self.tracker.record(pre_repair, X)

        return X


class ExactKRepair(Repair):
    """
    Repair operator ensuring exact cardinality only.

    Projects each individual to have exactly k selected points,
    without geographic quota constraints.

    Attributes
    ----------
    k : int
        Target subset size
    rng : np.random.Generator
        Random number generator
    tracker : Optional[RepairActivityTracker]
        Activity tracker for diagnostics
    """

    def __init__(self, k: int, seed: int, track_activity: bool = False):
        """
        Initialize the repair operator.

        Parameters
        ----------
        k : int
            Target subset size
        seed : int
            Random seed
        track_activity : bool
            Whether to track repair activity statistics
        """
        super().__init__()
        self.k = int(k)
        self.rng = np.random.default_rng(int(seed))
        self.tracker: Optional[RepairActivityTracker] = (
            RepairActivityTracker() if track_activity else None
        )

    def _do(self, problem, X, **kwargs):
        """
        Repair population to have exact cardinality k.

        Parameters
        ----------
        problem : Problem
            The optimization problem
        X : np.ndarray
            Population of boolean masks (n_individuals, n_var)

        Returns
        -------
        np.ndarray
            Repaired population
        """
        X = np.asarray(X, dtype=bool)
        pre_repair = X.copy() if self.tracker else None

        for i in range(X.shape[0]):
            X[i] = _project_to_exact_k_mask(X[i], self.k, self.rng)

        if self.tracker is not None:
            self.tracker.record(pre_repair, X)

        return X


def _project_to_exact_k_mask(
    mask: np.ndarray,
    k: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Project a boolean mask to have exactly k True values.

    If |mask| > k, randomly remove excess.
    If |mask| < k, randomly add from unselected.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask
    k : int
        Target cardinality
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Projected mask with exactly k True values
    """
    mask = mask.copy()
    current = int(mask.sum())

    if current > k:
        # Remove excess
        selected = np.where(mask)[0]
        to_remove = rng.choice(selected, size=current - k, replace=False)
        mask[to_remove] = False
    elif current < k:
        # Add missing
        unselected = np.where(~mask)[0]
        to_add = rng.choice(unselected, size=k - current, replace=False)
        mask[to_add] = True

    return mask


# ============================================================================
# Least-Harm Repair (Proxy-Aware) — extracted to _repair_least_harm
# Re-exported here for backward compatibility.
# ============================================================================

from ._repair_least_harm import (
    _project_to_exact_k_least_harm,
    LeastHarmQuotaRepair,
    LeastHarmExactKRepair,
)
