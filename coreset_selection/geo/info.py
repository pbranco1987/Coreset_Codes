"""
Geographic group information.

Contains:
- GeoInfo: Dataclass holding geographic grouping information
- build_geo_info: Build GeoInfo from state labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class GeoInfo:
    """
    Geographic group information for constraint handling.

    Per manuscript Section IV-B, the constraint interface supports multiple
    weight vectors.  This dataclass stores:
      - pi (municipality-share, w_i≡1):  π_g = n_g / N
      - pi_pop (population-share, w_i=pop_i):  π_g^{(pop)} = Σ_{i∈I_g} pop_i / Σ pop

    Attributes
    ----------
    groups : List[str]
        Unique group names (e.g., state codes)
    group_ids : np.ndarray
        Group assignment for each point, shape (N,)
    group_to_indices : List[np.ndarray]
        Mapping from group index to point indices
    pi : np.ndarray
        Municipality-share (count-based) proportions, shape (G,)
    group_sizes : np.ndarray
        Number of points in each group (capacities n_g), shape (G,)
    population_weights : Optional[np.ndarray]
        Per-municipality population weights, shape (N,). None if unavailable.
    pi_pop : Optional[np.ndarray]
        Population-share target distribution, shape (G,). None if population
        weights are not available.
    """
    groups: List[str]
    group_ids: np.ndarray
    group_to_indices: List[np.ndarray]
    pi: np.ndarray
    group_sizes: np.ndarray
    population_weights: Optional[np.ndarray] = None
    pi_pop: Optional[np.ndarray] = None

    @property
    def G(self) -> int:
        """Number of geographic groups."""
        return len(self.groups)

    @property
    def N(self) -> int:
        """Total number of points."""
        return len(self.group_ids)

    def get_group_name(self, group_idx: int) -> str:
        """Get group name by index."""
        return self.groups[group_idx]

    def get_group_idx(self, group_name: str) -> int:
        """Get group index by name."""
        return self.groups.index(group_name)

    def get_target_distribution(self, weight_type: str = "muni") -> np.ndarray:
        """Return π^{(w)} for the specified weight type.

        Parameters
        ----------
        weight_type : str
            ``"muni"`` for municipality-share (w≡1) or ``"pop"`` for
            population-share (w=pop_i).

        Returns
        -------
        np.ndarray
            Target distribution, shape ``(G,)``.
        """
        if weight_type == "muni":
            return self.pi.copy()
        elif weight_type == "pop":
            if self.pi_pop is None:
                raise ValueError(
                    "Population-share target distribution (pi_pop) is not available. "
                    "Provide population_weights when building GeoInfo."
                )
            return self.pi_pop.copy()
        else:
            raise ValueError(f"Unknown weight_type: {weight_type!r}. Use 'muni' or 'pop'.")

    @classmethod
    def from_group_ids(
        cls, 
        group_ids: np.ndarray, 
        groups: List[str] = None,
        population_weights: Optional[np.ndarray] = None,
    ) -> "GeoInfo":
        """
        Create GeoInfo from group ID array.
        
        Parameters
        ----------
        group_ids : np.ndarray
            Integer group IDs for each point, shape (N,)
        groups : List[str], optional
            Group names. If None, uses string representations of integers.
        population_weights : np.ndarray, optional
            Per-municipality population weights, shape (N,). If provided,
            computes the population-share target distribution π_pop.
            
        Returns
        -------
        GeoInfo
            Populated geographic information
        """
        group_ids = np.asarray(group_ids, dtype=int)
        G = int(group_ids.max() + 1)
        N = len(group_ids)
        
        if groups is None:
            groups = [str(g) for g in range(G)]
        else:
            if len(groups) != G:
                raise ValueError(f"groups length {len(groups)} != expected {G}")
        
        # Build group-to-indices mapping
        group_to_indices = [np.where(group_ids == g)[0] for g in range(G)]
        
        # Compute group sizes (capacities n_g)
        group_sizes = np.array([len(idx) for idx in group_to_indices], dtype=int)
        
        # Municipality-share proportions: π_g = n_g / N
        pi = group_sizes / N
        
        # Population-share proportions: π_g^{(pop)} = Σ_{i∈I_g} pop_i / Σ pop
        pop_w = None
        pi_pop = None
        if population_weights is not None:
            pop_w = np.asarray(population_weights, dtype=np.float64)
            if len(pop_w) != N:
                raise ValueError(
                    f"population_weights length {len(pop_w)} != N={N}"
                )
            pop_w = np.where(np.isfinite(pop_w) & (pop_w > 0), pop_w, 0.0)
            W = float(pop_w.sum())
            if W > 0:
                Wg = np.array(
                    [float(pop_w[idx].sum()) if len(idx) > 0 else 0.0
                     for idx in group_to_indices],
                    dtype=np.float64,
                )
                pi_pop = Wg / W
            else:
                pop_w = None  # degenerate; fall back to None
        
        return cls(
            groups=list(groups),
            group_ids=group_ids,
            group_to_indices=group_to_indices,
            pi=pi,
            group_sizes=group_sizes,
            population_weights=pop_w,
            pi_pop=pi_pop,
        )


def build_geo_info(
    state_labels: np.ndarray,
    population_weights: Optional[np.ndarray] = None,
) -> GeoInfo:
    """
    Build GeoInfo from state/group labels.
    
    Parameters
    ----------
    state_labels : np.ndarray
        Array of group labels (strings or integers), shape (N,)
    population_weights : np.ndarray, optional
        Per-municipality population weights, shape (N,). If provided,
        computes population-share target distribution π_pop.
        
    Returns
    -------
    GeoInfo
        Populated geographic information
    """
    state_labels = np.asarray(state_labels)
    
    # Get unique groups (sorted for reproducibility)
    unique_groups = sorted(set(state_labels))
    groups = [str(g) for g in unique_groups]
    G = len(groups)
    
    # Create mapping from group name to index
    group_name_to_idx = {g: i for i, g in enumerate(groups)}
    
    # Assign group IDs
    group_ids = np.array([group_name_to_idx[str(s)] for s in state_labels], dtype=int)
    
    # Build group-to-indices mapping
    group_to_indices = [np.where(group_ids == g)[0] for g in range(G)]
    
    # Compute group sizes (capacities n_g)
    group_sizes = np.array([len(idx) for idx in group_to_indices], dtype=int)
    
    # Municipality-share proportions: π_g = n_g / N
    N = len(state_labels)
    pi = group_sizes / N
    
    # Population-share proportions
    pop_w = None
    pi_pop = None
    if population_weights is not None:
        pop_w = np.asarray(population_weights, dtype=np.float64)
        if len(pop_w) != N:
            raise ValueError(
                f"population_weights length {len(pop_w)} != N={N}"
            )
        pop_w = np.where(np.isfinite(pop_w) & (pop_w > 0), pop_w, 0.0)
        W = float(pop_w.sum())
        if W > 0:
            Wg = np.array(
                [float(pop_w[idx].sum()) if len(idx) > 0 else 0.0
                 for idx in group_to_indices],
                dtype=np.float64,
            )
            pi_pop = Wg / W
        else:
            pop_w = None
    
    return GeoInfo(
        groups=groups,
        group_ids=group_ids,
        group_to_indices=group_to_indices,
        pi=pi,
        group_sizes=group_sizes,
        population_weights=pop_w,
        pi_pop=pi_pop,
    )


def merge_small_groups(
    geo: GeoInfo,
    min_size: int,
    merge_name: str = "Other",
) -> GeoInfo:
    """
    Merge small groups into a single "Other" category.
    
    Parameters
    ----------
    geo : GeoInfo
        Original geographic info
    min_size : int
        Minimum group size to keep separate
    merge_name : str
        Name for merged group
        
    Returns
    -------
    GeoInfo
        New GeoInfo with small groups merged
    """
    # Identify groups to keep vs merge
    keep_mask = geo.group_sizes >= min_size
    
    if keep_mask.all():
        return geo  # No merging needed
    
    # Build new labels
    new_labels = []
    for i, g in enumerate(geo.group_ids):
        if keep_mask[g]:
            new_labels.append(geo.groups[g])
        else:
            new_labels.append(merge_name)
    
    return build_geo_info(np.array(new_labels))
