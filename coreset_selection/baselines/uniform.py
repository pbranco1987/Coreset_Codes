"""
Uniform random sampling baseline methods.

Contains:
- baseline_uniform: Simple uniform random sampling
- baseline_uniform_quota: Uniform sampling with geographic quotas
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import quota_sample


def baseline_uniform(
    n: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Simple uniform random sampling without replacement.
    
    Parameters
    ----------
    n : int
        Total number of points
    k : int
        Number of points to select
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
        
    Raises
    ------
    ValueError
        If k < 0 or k > n
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return np.array([], dtype=int)
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)


def baseline_uniform_quota(
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    seed: int,
    min_one_per_group: bool = True,
) -> np.ndarray:
    """
    Uniform random sampling with geographic quota constraints.
    
    Samples uniformly within each group according to the KL-optimal
    quota allocation.
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    alpha_geo : float
        Dirichlet smoothing parameter for KL computation
    seed : int
        Random seed
    min_one_per_group : bool
        Whether to ensure at least one sample per group
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    rng = np.random.default_rng(seed)
    
    # Build projector to get optimal quota counts
    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=min_one_per_group,
    )
    
    target_counts = projector.target_counts(k)
    
    # Sample uniformly within each group
    selected = quota_sample(
        indices_by_group=geo.group_to_indices,
        target_counts=target_counts,
        weights_by_group=None,  # Uniform
        rng=rng,
    )
    
    return selected


def baseline_uniform_stratified(
    geo: GeoInfo,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Stratified uniform sampling proportional to group sizes.
    
    Samples proportionally to each group's size in the population.
    This is a special case of quota sampling where the quotas are
    proportional to π_g.
    
    Parameters
    ----------
    geo : GeoInfo
        Geographic group information
    k : int
        Total number of points to select
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
    """
    rng = np.random.default_rng(seed)
    
    # Proportional allocation
    target_counts = _proportional_allocation(geo.pi, k, geo.group_sizes)
    
    # Sample uniformly within each group
    selected = quota_sample(
        indices_by_group=geo.group_to_indices,
        target_counts=target_counts,
        weights_by_group=None,
        rng=rng,
    )
    
    return selected


def _proportional_allocation(
    pi: np.ndarray,
    k: int,
    group_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute proportional allocation of k samples across groups.
    
    Uses largest remainder method for integer rounding.
    
    Parameters
    ----------
    pi : np.ndarray
        Target proportions
    k : int
        Total samples
    group_sizes : np.ndarray
        Available samples per group (upper bounds)
        
    Returns
    -------
    np.ndarray
        Integer counts summing to k
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()  # Normalize
    
    # Fractional allocation
    frac_counts = pi * k
    
    # Integer parts
    int_counts = np.floor(frac_counts).astype(int)
    
    # Remainders
    remainders = frac_counts - int_counts
    
    # Distribute remaining samples to groups with largest remainders
    remaining = k - int_counts.sum()
    
    if remaining > 0:
        # Sort by remainder descending
        order = np.argsort(-remainders)
        for i in range(remaining):
            g = order[i % len(order)]
            # Check capacity
            if int_counts[g] < group_sizes[g]:
                int_counts[g] += 1
            else:
                # Find next group with capacity
                for j in range(len(order)):
                    g2 = order[(i + j) % len(order)]
                    if int_counts[g2] < group_sizes[g2]:
                        int_counts[g2] += 1
                        break
    
    return int_counts


def baseline_uniform_population_weighted(
    n: int,
    k: int,
    population: np.ndarray,
    seed: int,
) -> np.ndarray:
    """
    Population-weighted random sampling.
    
    Each point is selected with probability proportional to its
    population weight.
    
    Parameters
    ----------
    n : int
        Total number of points
    k : int
        Number of points to select
    population : np.ndarray
        Population weights for each point
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Selected indices
        
    Raises
    ------
    ValueError
        If k < 0 or k > n
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return np.array([], dtype=int)
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    
    rng = np.random.default_rng(seed)
    
    population = np.asarray(population, dtype=np.float64)
    weights = population / population.sum()
    
    # Sample without replacement using Gumbel-max trick
    log_weights = np.log(np.maximum(weights, 1e-30))
    gumbel_noise = -np.log(-np.log(rng.uniform(size=n) + 1e-30) + 1e-30)
    perturbed = log_weights + gumbel_noise
    
    return np.argpartition(perturbed, -k)[-k:]
