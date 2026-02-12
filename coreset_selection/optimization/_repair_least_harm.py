"""
Least-harm repair operators for NSGA-II coreset selection.

Extracted from ``optimization.repair`` for modularity.

Manuscript Section 5.2 (lines 933-936):
"As a heuristic, we implement a 'least-harm' variant that greedily removes
(or adds) indices that minimally worsen (or most improve) a cheap proxy
objective such as the RFF-MMD surrogate."
"""

from __future__ import annotations

from typing import Any, Callable, Optional

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
from .repair import RepairActivityTracker


def _project_to_exact_k_least_harm(
    mask: np.ndarray,
    k: int,
    rng: np.random.Generator,
    proxy_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """
    Least-harm repair: project mask to exact-k minimizing proxy increase.

    Per manuscript (§5.2): greedily remove/add indices that minimally
    worsen (or most improve) a cheap proxy objective.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask
    k : int
        Target cardinality
    rng : np.random.Generator
        Random number generator (for tie-breaking)
    proxy_fn : Callable[[np.ndarray], float]
        Function that computes proxy objective given index array

    Returns
    -------
    np.ndarray
        Projected mask with exactly k True values
    """
    mask = mask.copy()
    current = int(mask.sum())

    if current == k:
        return mask

    if current > k:
        # Greedy removal: remove indices that cause minimal proxy increase
        excess = current - k
        selected = np.where(mask)[0].tolist()

        for _ in range(excess):
            if len(selected) <= k:
                break

            best_idx = None
            best_delta = np.inf

            # Try removing each selected index
            for idx in selected:
                test_selected = [i for i in selected if i != idx]
                delta = proxy_fn(np.array(test_selected)) - proxy_fn(np.array(selected))
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx

            if best_idx is not None:
                selected.remove(best_idx)
                mask[best_idx] = False

    elif current < k:
        # Greedy addition: add indices that cause maximal proxy improvement
        deficit = k - current
        selected = set(np.where(mask)[0].tolist())
        unselected = np.where(~mask)[0].tolist()

        for _ in range(deficit):
            if not unselected:
                break

            best_idx = None
            best_delta = np.inf  # Want most negative (improvement)

            # Try adding each unselected index
            current_arr = np.array(list(selected))
            current_val = proxy_fn(current_arr) if len(current_arr) > 0 else np.inf

            for idx in unselected:
                test_selected = list(selected) + [idx]
                new_val = proxy_fn(np.array(test_selected))
                delta = new_val - current_val
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx

            if best_idx is not None:
                selected.add(best_idx)
                unselected.remove(best_idx)
                mask[best_idx] = True

    return mask


class LeastHarmQuotaRepair(Repair):
    """
    Least-harm repair with geographic quota constraints.

    Per manuscript Section 5.2 (lines 933-936):
    Greedily removes/adds indices that minimally worsen (or most improve)
    a cheap proxy objective such as the RFF-MMD surrogate, while respecting
    geographic quota constraints.

    Attributes
    ----------
    k : int
        Target subset size
    projector : GeographicConstraintProjector
        Projector defining the quota constraints
    proxy_fn : Callable[[np.ndarray], float]
        Proxy objective function (e.g., RFF-MMD)
    rng : np.random.Generator
        Random number generator
    tracker : Optional[RepairActivityTracker]
        Activity tracker for diagnostics
    """

    def __init__(
        self,
        k: int,
        projector: GeographicConstraintProjector,
        proxy_fn: Callable[[np.ndarray], float],
        seed: int,
        track_activity: bool = False,
    ):
        """
        Initialize the least-harm repair operator.

        Parameters
        ----------
        k : int
            Target subset size
        projector : GeographicConstraintProjector
            Projector defining the quota constraints
        proxy_fn : Callable[[np.ndarray], float]
            Function computing proxy objective given index array
        seed : int
            Random seed
        track_activity : bool
            Whether to track repair activity statistics
        """
        super().__init__()
        self.k = int(k)
        self.projector = projector
        self.proxy_fn = proxy_fn
        self.rng = np.random.default_rng(int(seed))
        self.tracker: Optional[RepairActivityTracker] = (
            RepairActivityTracker() if track_activity else None
        )

    def _do(self, problem, X, **kwargs):
        """
        Repair population using least-harm strategy with quotas.

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
            X[i] = self._repair_single_least_harm(X[i])

        if self.tracker is not None:
            self.tracker.record(pre_repair, X)

        return X

    def _repair_single_least_harm(self, mask: np.ndarray) -> np.ndarray:
        """
        Repair a single mask using least-harm strategy per group.

        For each group, greedily add/remove to match quota while
        minimizing proxy objective increase.
        """
        mask = mask.copy()
        target = self.projector.target_counts(self.k)
        group_ids = self.projector.geo.group_ids
        G = self.projector.geo.G

        for g in range(G):
            c_target = int(target[g])
            idx_g = np.where(group_ids == g)[0]

            if len(idx_g) == 0:
                continue

            selected_g = idx_g[mask[idx_g]]
            unselected_g = idx_g[~mask[idx_g]]
            c_current = len(selected_g)

            if c_current > c_target:
                # Remove excess from group g using least-harm
                excess = c_current - c_target
                selected_list = selected_g.tolist()

                for _ in range(excess):
                    if len(selected_list) <= c_target:
                        break

                    best_idx = None
                    best_delta = np.inf
                    all_selected = np.where(mask)[0]

                    for idx in selected_list:
                        test_all = all_selected[all_selected != idx]
                        if len(test_all) == 0:
                            delta = 0
                        else:
                            delta = self.proxy_fn(test_all) - self.proxy_fn(all_selected)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = idx

                    if best_idx is not None:
                        selected_list.remove(best_idx)
                        mask[best_idx] = False

            elif c_current < c_target:
                # Add to group g using least-harm
                deficit = c_target - c_current
                unselected_list = unselected_g.tolist()

                for _ in range(deficit):
                    if not unselected_list:
                        break

                    best_idx = None
                    best_delta = np.inf
                    all_selected = np.where(mask)[0]
                    current_val = self.proxy_fn(all_selected) if len(all_selected) > 0 else np.inf

                    for idx in unselected_list:
                        test_all = np.append(all_selected, idx)
                        new_val = self.proxy_fn(test_all)
                        delta = new_val - current_val
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = idx

                    if best_idx is not None:
                        unselected_list.remove(best_idx)
                        mask[best_idx] = True

        return mask


class LeastHarmExactKRepair(Repair):
    """
    Least-harm repair for exact-k only (no quota constraints).

    Per manuscript Section 5.2: greedily removes/adds indices that
    minimally worsen (or most improve) a proxy objective.

    Attributes
    ----------
    k : int
        Target subset size
    proxy_fn : Callable[[np.ndarray], float]
        Proxy objective function (e.g., RFF-MMD)
    rng : np.random.Generator
        Random number generator
    tracker : Optional[RepairActivityTracker]
        Activity tracker for diagnostics
    """

    def __init__(
        self,
        k: int,
        proxy_fn: Callable[[np.ndarray], float],
        seed: int,
        track_activity: bool = False,
    ):
        """
        Initialize the least-harm repair operator.

        Parameters
        ----------
        k : int
            Target subset size
        proxy_fn : Callable[[np.ndarray], float]
            Function computing proxy objective given index array
        seed : int
            Random seed
        track_activity : bool
            Whether to track repair activity statistics
        """
        super().__init__()
        self.k = int(k)
        self.proxy_fn = proxy_fn
        self.rng = np.random.default_rng(int(seed))
        self.tracker: Optional[RepairActivityTracker] = (
            RepairActivityTracker() if track_activity else None
        )

    def _do(self, problem, X, **kwargs):
        """
        Repair population using least-harm strategy.

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
            X[i] = _project_to_exact_k_least_harm(
                X[i], self.k, self.rng, self.proxy_fn
            )

        if self.tracker is not None:
            self.tracker.record(pre_repair, X)

        return X
