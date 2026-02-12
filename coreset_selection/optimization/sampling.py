"""
Sampling operators for NSGA-II initialization.

Contains:
- QuotaBinarySampling: Generate feasible solutions satisfying quotas
- ExactKSampling: Generate solutions with exactly k selected points
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymoo.core.sampling import Sampling

from ..geo.projector import GeographicConstraintProjector, build_feasible_quota_mask


class QuotaBinarySampling(Sampling):
    """
    Sampling operator that generates quota-feasible binary masks.
    
    Each sampled solution satisfies the geographic quota constraints,
    with exactly the target count in each geographic group.
    
    Attributes
    ----------
    k : int
        Target coreset size
    projector : GeographicConstraintProjector
        Projector for quota constraints
    seed : int
        Base random seed
    """
    
    def __init__(
        self,
        k: int,
        projector: GeographicConstraintProjector,
        seed: int = 0,
    ):
        """
        Initialize the sampling operator.
        
        Parameters
        ----------
        k : int
            Target coreset size
        projector : GeographicConstraintProjector
            Projector with quota information
        seed : int
            Base random seed
        """
        super().__init__()
        self.k = k
        self.projector = projector
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate n_samples quota-feasible solutions.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        n_samples : int
            Number of solutions to generate
            
        Returns
        -------
        np.ndarray
            Population of binary masks, shape (n_samples, n_var)
        """
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=bool)
        
        for i in range(n_samples):
            # Each solution gets a different seed
            mask = build_feasible_quota_mask(
                self.projector,
                self.k,
                seed=self.seed + i,
            )
            X[i] = mask
        
        return X


class ExactKSampling(Sampling):
    """
    Sampling operator that generates solutions with exactly k points.
    
    Does not enforce geographic quotas, just exact cardinality.
    
    Attributes
    ----------
    k : int
        Target coreset size
    seed : int
        Base random seed
    """
    
    def __init__(
        self,
        k: int,
        seed: int = 0,
    ):
        """
        Initialize the sampling operator.
        
        Parameters
        ----------
        k : int
            Target coreset size
        seed : int
            Base random seed
        """
        super().__init__()
        self.k = k
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate n_samples solutions with exactly k selected points.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        n_samples : int
            Number of solutions to generate
            
        Returns
        -------
        np.ndarray
            Population of binary masks, shape (n_samples, n_var)
        """
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=bool)
        
        for i in range(n_samples):
            rng = np.random.default_rng(self.seed + i)
            
            # Randomly select k indices
            selected = rng.choice(n_var, size=self.k, replace=False)
            X[i, selected] = True
        
        return X


class WeightedSampling(Sampling):
    """
    Sampling operator using importance weights.
    
    Selects k points with probability proportional to weights.
    Useful for initializing with leverage scores or other importance measures.
    
    Attributes
    ----------
    k : int
        Target coreset size
    weights : np.ndarray
        Importance weights for each point
    seed : int
        Base random seed
    """
    
    def __init__(
        self,
        k: int,
        weights: np.ndarray,
        seed: int = 0,
    ):
        """
        Initialize the sampling operator.
        
        Parameters
        ----------
        k : int
            Target coreset size
        weights : np.ndarray
            Importance weights, shape (n_var,)
        seed : int
            Base random seed
        """
        super().__init__()
        self.k = k
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()  # Normalize
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate n_samples solutions using weighted sampling.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        n_samples : int
            Number of solutions to generate
            
        Returns
        -------
        np.ndarray
            Population of binary masks, shape (n_samples, n_var)
        """
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=bool)
        
        # Handle k==0 edge case
        if self.k == 0:
            return X  # All-false masks
        if self.k > n_var:
            raise ValueError(f"k={self.k} exceeds n_var={n_var}")
        
        for i in range(n_samples):
            rng = np.random.default_rng(self.seed + i)
            
            # Weighted sampling without replacement (Gumbel-max trick)
            gumbel = rng.gumbel(size=n_var)
            keys = np.log(self.weights + 1e-30) + gumbel
            selected = np.argpartition(keys, -self.k)[-self.k:]
            
            X[i, selected] = True
        
        return X


class MixedSampling(Sampling):
    """
    Sampling operator that mixes multiple sampling strategies.
    
    Useful for diverse initialization combining quota-feasible,
    random, and weighted samples.
    
    Attributes
    ----------
    samplers : list
        List of (sampler, fraction) tuples
    seed : int
        Random seed for reproducible shuffling
    """
    
    def __init__(
        self,
        samplers: list,
        seed: int = 0,
    ):
        """
        Initialize the mixed sampling operator.
        
        Parameters
        ----------
        samplers : list
            List of (Sampling, fraction) tuples. Fractions should sum to 1.
        seed : int
            Random seed for reproducible shuffling
        """
        super().__init__()
        self.samplers = samplers
        self.seed = seed
        
        # Normalize fractions
        total = sum(frac for _, frac in samplers)
        self.samplers = [(s, f / total) for s, f in samplers]

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate n_samples solutions using mixed strategies.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        n_samples : int
            Number of solutions to generate
            
        Returns
        -------
        np.ndarray
            Population of binary masks, shape (n_samples, n_var)
        """
        n_var = problem.n_var
        X_list = []
        
        remaining = n_samples
        for sampler, frac in self.samplers[:-1]:
            n_this = int(n_samples * frac)
            if n_this > 0:
                X_this = sampler._do(problem, n_this, **kwargs)
                X_list.append(X_this)
                remaining -= n_this
        
        # Last sampler gets the remainder
        if remaining > 0:
            X_last = self.samplers[-1][0]._do(problem, remaining, **kwargs)
            X_list.append(X_last)
        
        X = np.vstack(X_list)
        
        # Shuffle to mix strategies using local RNG for reproducibility
        rng = np.random.default_rng(kwargs.get("seed", self.seed))
        rng.shuffle(X, axis=0)
        
        return X
