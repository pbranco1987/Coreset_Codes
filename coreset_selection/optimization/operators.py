"""
Genetic operators for NSGA-II coreset selection.

Contains:
- UniformBinaryCrossover for binary mask crossover
- QuotaSwapMutation for within-group point swapping
"""

from __future__ import annotations

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation


class UniformBinaryCrossover(Crossover):
    """
    Uniform crossover for binary masks.
    
    Each bit is inherited from either parent with equal probability.
    The result is then repaired to satisfy constraints.
    
    Attributes
    ----------
    prob : float
        Crossover probability (default 0.9)
    rng : np.random.Generator
        Random number generator
    """
    
    def __init__(self, prob: float = 0.9, seed: int = 0):
        """
        Initialize the crossover operator.
        
        Parameters
        ----------
        prob : float
            Probability of performing crossover (vs. copying parents)
        seed : int
            Random seed
        """
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = float(prob)
        self.rng = np.random.default_rng(int(seed))

    def _do(self, problem, X, **kwargs):
        """
        Perform uniform crossover.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        X : np.ndarray
            Parent pairs, shape (n_matings, 2, n_var)
            
        Returns
        -------
        np.ndarray
            Offspring, shape (n_matings, 2, n_var)
        """
        n_matings, _, n_var = X.shape
        Y = np.zeros((n_matings, 2, n_var), dtype=bool)

        for i in range(n_matings):
            p1, p2 = X[i, 0], X[i, 1]
            
            if self.rng.random() < self.prob:
                # Uniform crossover: each bit from random parent
                swap_mask = self.rng.random(n_var) < 0.5
                c1 = np.where(swap_mask, p1, p2)
                c2 = np.where(swap_mask, p2, p1)
            else:
                # No crossover: copy parents
                c1, c2 = p1.copy(), p2.copy()

            Y[i, 0] = c1
            Y[i, 1] = c2

        return Y


class QuotaSwapMutation(Mutation):
    """
    Mutation operator that swaps points within geographic groups.
    
    For each selected point, with probability prob, swap it with
    an unselected point from the same geographic group. This preserves
    the quota constraints while exploring the solution space.
    
    Attributes
    ----------
    group_ids : np.ndarray
        Group assignment for each point
    prob : float
        Per-bit mutation probability (default 0.1)
    rng : np.random.Generator
        Random number generator
    _group_to_indices : List[np.ndarray]
        Precomputed mapping from group to point indices
    """
    
    def __init__(self, group_ids: np.ndarray, prob: float = 0.1, seed: int = 0):
        """
        Initialize the mutation operator.
        
        Parameters
        ----------
        group_ids : np.ndarray
            Group assignment for each point (length N)
        prob : float
            Per-bit mutation probability
        seed : int
            Random seed
        """
        super().__init__()
        self.group_ids = np.asarray(group_ids, dtype=int)
        self.prob = float(prob)
        self.rng = np.random.default_rng(int(seed))
        
        # Precompute group-to-indices mapping
        unique_groups = np.unique(self.group_ids)
        self._group_to_indices = [
            np.where(self.group_ids == g)[0] for g in range(unique_groups.max() + 1)
        ]

    def _do(self, problem, X, **kwargs):
        """
        Perform quota-preserving swap mutation.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        X : np.ndarray
            Population of boolean masks (n_individuals, n_var)
            
        Returns
        -------
        np.ndarray
            Mutated population
        """
        X = np.asarray(X, dtype=bool).copy()

        for i in range(X.shape[0]):
            mask = X[i]
            selected = np.where(mask)[0]
            
            for idx in selected:
                if self.rng.random() < self.prob:
                    g = self.group_ids[idx]
                    group_indices = self._group_to_indices[g]
                    
                    # Find unselected points in same group
                    unselected_in_group = group_indices[~mask[group_indices]]
                    
                    if unselected_in_group.size > 0:
                        # Swap: deselect current, select random from group
                        new_idx = self.rng.choice(unselected_in_group)
                        mask[idx] = False
                        mask[new_idx] = True

            X[i] = mask

        return X


class BitFlipMutation(Mutation):
    """
    Simple bit-flip mutation for binary masks.
    
    Each bit is flipped with probability prob. The result needs
    to be repaired afterward to satisfy cardinality constraints.
    
    Attributes
    ----------
    prob : float
        Per-bit flip probability (default 0.01)
    rng : np.random.Generator
        Random number generator
    """
    
    def __init__(self, prob: float = 0.01, seed: int = 0):
        """
        Initialize the mutation operator.
        
        Parameters
        ----------
        prob : float
            Per-bit flip probability
        seed : int
            Random seed
        """
        super().__init__()
        self.prob = float(prob)
        self.rng = np.random.default_rng(int(seed))

    def _do(self, problem, X, **kwargs):
        """
        Perform bit-flip mutation.
        
        Parameters
        ----------
        problem : Problem
            The optimization problem
        X : np.ndarray
            Population of boolean masks (n_individuals, n_var)
            
        Returns
        -------
        np.ndarray
            Mutated population
        """
        X = np.asarray(X, dtype=bool).copy()
        flip_mask = self.rng.random(X.shape) < self.prob
        X = np.logical_xor(X, flip_mask)
        return X
