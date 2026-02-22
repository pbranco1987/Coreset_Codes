"""
Geographic constraint projection and quota masks.

Contains:
- GeographicConstraintProjector: Projects solutions to satisfy quotas
- project_to_exact_k_mask: Project to exact cardinality
- build_feasible_quota_mask: Build a feasible starting solution
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .info import GeoInfo
from .kl import min_achievable_geo_kl_bounded, compute_quota_path


class GeographicConstraintProjector:
    """
    Projects solutions to satisfy geographic quota constraints.

    Computes KL-optimal quota allocation and provides methods
    to project arbitrary masks to satisfy these quotas.

    Attributes
    ----------
    geo : GeoInfo
        Geographic group information
    alpha_geo : float
        Dirichlet smoothing parameter
    min_one_per_group : bool
        Whether to require at least one per group
    _quota_cache : dict
        Cache of quota allocations for different k values
    _quota_path_cache : list or None
        Cached full quota path (computed lazily)
    """

    def __init__(
        self,
        geo: GeoInfo,
        alpha_geo: float,
        min_one_per_group: bool = True,
        bounds_eps: float = 0.0,
        weight_type: str = "muni",
    ):
        """
        Initialize the projector.

        Parameters
        ----------
        geo : GeoInfo
            Geographic group information
        alpha_geo : float
            Dirichlet smoothing parameter for KL computation
        min_one_per_group : bool
            Whether to require at least one sample per group
        weight_type : str
            Target distribution type: ``"muni"`` for municipality-share
            (π_g = n_g/N) or ``"pop"`` for population-share
            (π_g^{pop} = Σ pop_i / Σ pop).  Default ``"muni"`` preserves
            backward compatibility.
        """
        # `bounds_eps` is accepted for backward compatibility with earlier
        # versions of the codebase; the bounded feasibility checks are handled
        # by the quota computation itself (Algorithm 1) and by explicit
        # capacity checks.
        self.geo = geo
        self.alpha_geo = alpha_geo
        self.min_one_per_group = min_one_per_group
        self.bounds_eps = float(bounds_eps)
        self.weight_type = weight_type
        self._quota_cache = {}
        self._quota_path_cache: Optional[list] = None

    def project_to_exact_k_mask(
        self,
        mask: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Project a mask to satisfy exact-k (no quota constraints)."""
        return project_to_exact_k_mask(mask, k, rng)

    def target_counts(self, k: int) -> np.ndarray:
        """
        Get KL-optimal target counts for each group.
        
        Parameters
        ----------
        k : int
            Total coreset size
            
        Returns
        -------
        np.ndarray
            Target count per group, shape (G,)
        """
        if k not in self._quota_cache:
            pi = self.geo.get_target_distribution(self.weight_type)
            _, counts = min_achievable_geo_kl_bounded(
                pi=pi,
                group_sizes=self.geo.group_sizes,
                k=k,
                alpha_geo=self.alpha_geo,
                min_one_per_group=self.min_one_per_group,
            )
            self._quota_cache[k] = counts
        
        return self._quota_cache[k].copy()

    def project_to_quota_mask(
        self,
        mask: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Project a mask to satisfy quota constraints.
        
        Adjusts the selection to have exactly the target count
        for each geographic group.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean selection mask, shape (N,)
        k : int
            Target total count
        rng : np.random.Generator
            Random number generator
            
        Returns
        -------
        np.ndarray
            Projected mask satisfying quotas
        """
        mask = np.asarray(mask, dtype=bool).copy()
        target_counts = self.target_counts(k)
        
        # Adjust each group
        for g in range(self.geo.G):
            idx_g = self.geo.group_to_indices[g]
            target_g = int(target_counts[g])
            
            # Current selection in this group
            selected_in_g = idx_g[mask[idx_g]]
            current_g = len(selected_in_g)
            
            if current_g == target_g:
                continue
            elif current_g > target_g:
                # Remove excess
                excess = current_g - target_g
                to_remove = rng.choice(selected_in_g, size=excess, replace=False)
                mask[to_remove] = False
            else:
                # Add missing
                unselected_in_g = idx_g[~mask[idx_g]]
                needed = target_g - current_g
                
                if len(unselected_in_g) < needed:
                    raise RuntimeError(
                        f"Group {self.geo.groups[g]}: need {needed} more but only "
                        f"{len(unselected_in_g)} unselected available"
                    )
                
                to_add = rng.choice(unselected_in_g, size=needed, replace=False)
                mask[to_add] = True
        
        # Verify
        assert mask.sum() == k, f"Projection failed: got {mask.sum()}, expected {k}"
        
        return mask

    def check_quota_satisfied(self, mask: np.ndarray, k: int) -> bool:
        """
        Check if a mask satisfies quota constraints.

        Parameters
        ----------
        mask : np.ndarray
            Boolean selection mask
        k : int
            Expected total count

        Returns
        -------
        bool
            True if quotas are satisfied
        """
        if mask.sum() != k:
            return False

        target_counts = self.target_counts(k)

        for g in range(self.geo.G):
            idx_g = self.geo.group_to_indices[g]
            actual = mask[idx_g].sum()
            if actual != target_counts[g]:
                return False

        return True

    # ----- Phase 6 §6.2 additions -----

    def get_cstar(self, k: int) -> np.ndarray:
        """Return the KL-optimal quota vector c*(k) (alias for target_counts).

        This is the canonical interface name matching the manuscript notation.
        """
        return self.target_counts(k)

    def quota_path(
        self,
        k_grid: Optional[Sequence[int]] = None,
    ) -> list:
        """Compute the full quota path c*(k) and KL_min(k) for every k in *k_grid*.

        If *k_grid* is ``None``, defaults to the manuscript grid
        ``[50, 100, 200, 300, 400, 500]``.

        The path is computed incrementally (O(k_max log G)) and cached for
        subsequent calls with the same grid.

        Parameters
        ----------
        k_grid : sequence of int, optional
            Coreset sizes to evaluate.

        Returns
        -------
        list of dict
            One dict per k with keys ``k``, ``kl_min``, ``cstar``,
            ``geo_l1``, ``geo_maxdev``.
        """
        if k_grid is None:
            from ..config.constants import K_GRID
            k_grid = K_GRID

        # Use cache if available and grid matches
        if self._quota_path_cache is not None:
            cached_ks = {r["k"] for r in self._quota_path_cache}
            if set(k_grid).issubset(cached_ks):
                requested = set(k_grid)
                return [r for r in self._quota_path_cache if r["k"] in requested]

        pi = self.geo.get_target_distribution(self.weight_type)
        path = compute_quota_path(
            pi=pi,
            group_sizes=self.geo.group_sizes,
            k_grid=k_grid,
            alpha_geo=self.alpha_geo,
            min_one_per_group=self.min_one_per_group,
        )
        self._quota_path_cache = path

        # Also populate the per-k cache so target_counts(k) doesn't recompute
        for row in path:
            k_val = row["k"]
            self._quota_cache[k_val] = np.array(row["cstar"], dtype=int)

        return path

    def validate_capacity(self, k: int) -> dict:
        """Check whether k is feasible given group capacities.

        Returns
        -------
        dict
            ``feasible`` : bool,
            ``total_capacity`` : int,
            ``min_k`` : int  (sum of lower bounds),
            ``max_k`` : int  (sum of upper bounds),
            ``saturated_groups`` : list of str (groups at capacity in c*(k)).
        """
        G = self.geo.G
        pi = self.geo.get_target_distribution(self.weight_type)
        supported = pi > 0
        G_pi = int(np.sum(supported))

        lb = np.zeros(G, dtype=int)
        if self.min_one_per_group:
            lb[supported] = 1

        ub = self.geo.group_sizes.copy()
        min_k = int(lb.sum())
        max_k = int(ub.sum())
        feasible = min_k <= k <= max_k

        saturated: list = []
        if feasible:
            cstar = self.target_counts(k)
            for g in range(G):
                if cstar[g] >= ub[g]:
                    saturated.append(self.geo.groups[g])

        return {
            "feasible": feasible,
            "total_capacity": max_k,
            "min_k": min_k,
            "max_k": max_k,
            "saturated_groups": saturated,
        }

    def most_constrained_groups(
        self,
        k: int,
        top_n: int = 10,
    ) -> list:
        """Return the *top_n* most constrained groups at a given *k*.

        "Most constrained" is defined as highest ratio ``c*(k) / n_g``
        (fraction of available capacity consumed by the quota).

        Parameters
        ----------
        k : int
            Coreset size.
        top_n : int
            Number of groups to return.

        Returns
        -------
        list of dict
            Sorted by utilisation descending.  Keys:
            ``group``, ``cstar``, ``n_g``, ``utilisation``, ``pi_g``.
        """
        pi = self.geo.get_target_distribution(self.weight_type)
        cstar = self.target_counts(k)
        rows = []
        for g in range(self.geo.G):
            n_g = int(self.geo.group_sizes[g])
            c_g = int(cstar[g])
            util = c_g / max(n_g, 1)
            rows.append({
                "group": self.geo.groups[g],
                "cstar": c_g,
                "n_g": n_g,
                "utilisation": round(util, 4),
                "pi_g": round(float(pi[g]), 6),
            })
        rows.sort(key=lambda r: r["utilisation"], reverse=True)
        return rows[:top_n]


def project_to_exact_k_mask(
    mask: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Project a mask to have exactly k selected points.
    
    If too many: randomly remove excess.
    If too few: randomly add from unselected.
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean selection mask
    k : int
        Target count
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Projected mask with exactly k True values
    """
    mask = np.asarray(mask, dtype=bool).copy()
    current = int(mask.sum())
    
    if current == k:
        return mask
    elif current > k:
        # Remove excess
        selected = np.where(mask)[0]
        to_remove = rng.choice(selected, size=current - k, replace=False)
        mask[to_remove] = False
    else:
        # Add missing
        unselected = np.where(~mask)[0]
        to_add = rng.choice(unselected, size=k - current, replace=False)
        mask[to_add] = True
    
    return mask


def build_feasible_quota_mask(
    projector: GeographicConstraintProjector,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Build a feasible mask satisfying quota constraints.
    
    Randomly selects the target number of points from each group.
    
    Parameters
    ----------
    projector : GeographicConstraintProjector
        Projector with quota information
    k : int
        Total coreset size
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Boolean mask, shape (N,)
    """
    rng = np.random.default_rng(seed)
    geo = projector.geo
    target_counts = projector.target_counts(k)
    
    mask = np.zeros(geo.N, dtype=bool)
    
    for g in range(geo.G):
        count_g = int(target_counts[g])
        if count_g == 0:
            continue
        
        idx_g = geo.group_to_indices[g]
        
        if count_g > len(idx_g):
            raise RuntimeError(
                f"Group {geo.groups[g]}: need {count_g} but only {len(idx_g)} available"
            )
        
        selected = rng.choice(idx_g, size=count_g, replace=False)
        mask[selected] = True
    
    return mask


def compute_quota_violation(
    mask: np.ndarray,
    projector: GeographicConstraintProjector,
    k: int,
) -> int:
    """
    Compute total quota violation (sum of |actual - target|).
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean selection mask
    projector : GeographicConstraintProjector
        Projector with quota information
    k : int
        Target total count
        
    Returns
    -------
    int
        Total violation
    """
    target_counts = projector.target_counts(k)
    geo = projector.geo
    
    violation = 0
    for g in range(geo.G):
        idx_g = geo.group_to_indices[g]
        actual = mask[idx_g].sum()
        violation += abs(actual - target_counts[g])
    
    # Also count total deviation
    violation += abs(mask.sum() - k)
    
    return int(violation)
