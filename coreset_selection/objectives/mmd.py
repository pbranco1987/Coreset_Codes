"""
Maximum Mean Discrepancy via Random Fourier Features.

Contains:
- RFFMMD: RFF-based MMD² estimator for efficient subset evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RFFMMD:
    """
    Random Fourier Feature approximation for MMD².
    
    Uses RFF to approximate the RBF kernel, enabling O(m) MMD² computation
    for subsets instead of O(n²).
    
    MMD²(P, Q) ≈ ||μ_P - μ_Q||² where μ = E[φ(X)]
    
    Attributes
    ----------
    Phi : np.ndarray
        RFF feature matrix, shape (N, m)
    mean_full : np.ndarray
        Mean RFF embedding of full dataset, shape (m,)
    """
    Phi: np.ndarray
    mean_full: np.ndarray

    @staticmethod
    def build(
        X: np.ndarray,
        rff_dim: int,
        sigma_sq: float,
        seed: int = 0,
    ) -> "RFFMMD":
        """
        Build an RFFMMD estimator.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, shape (N, d)
        rff_dim : int
            Number of random features (output dim = rff_dim)
        sigma_sq : float
            RBF kernel bandwidth (variance parameter)
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        RFFMMD
            Initialized estimator
            
        Raises
        ------
        ValueError
            If sigma_sq <= 0 or not finite, or if rff_dim <= 0
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        m = int(rff_dim)
        
        # Validate parameters
        sigma_sq = float(sigma_sq)
        if not np.isfinite(sigma_sq) or sigma_sq <= 0:
            raise ValueError(f"sigma_sq must be positive and finite, got {sigma_sq}")
        if m <= 0:
            raise ValueError(f"rff_dim must be positive, got {m}")
        
        rng = np.random.default_rng(seed)
        
        # Sample frequencies from N(0, 1/σ²) for RBF kernel
        # k(x,y) = exp(-||x-y||²/(2σ²)) has Fourier transform ~ N(0, 1/σ²)
        freq_scale = 1.0 / np.sqrt(sigma_sq)
        W = rng.standard_normal((d, m)) * freq_scale
        b = rng.uniform(0, 2 * np.pi, size=m)
        
        # Compute RFF features per manuscript:
        #   φ(x) = sqrt(2/m) * cos(Wx + b)  in R^m
        projection = X @ W + b  # (N, m)
        Phi = np.sqrt(2.0 / m) * np.cos(projection)  # (N, m)
        
        # Mean embedding of full dataset
        mean_full = Phi.mean(axis=0)
        
        return RFFMMD(Phi=Phi.astype(np.float32), mean_full=mean_full.astype(np.float64))

    def mmd2_subset(self, idx: np.ndarray) -> float:
        """
        Compute MMD² between subset and full dataset.
        
        MMD²(S, X) ≈ ||mean(Φ[S]) - mean(Φ[X])||²
        
        Parameters
        ----------
        idx : np.ndarray
            Indices of the subset
            
        Returns
        -------
        float
            MMD² estimate
        """
        idx = np.asarray(idx, dtype=int)
        
        if idx.size == 0:
            return float(np.dot(self.mean_full, self.mean_full))
        
        # Mean embedding of subset
        mean_subset = self.Phi[idx].mean(axis=0)
        
        # MMD² = ||μ_S - μ_X||²
        diff = mean_subset - self.mean_full
        mmd2 = float(np.dot(diff, diff))
        
        return mmd2

    def mmd2_between_subsets(
        self, 
        idx1: np.ndarray, 
        idx2: np.ndarray
    ) -> float:
        """
        Compute MMD² between two subsets.
        
        Parameters
        ----------
        idx1 : np.ndarray
            Indices of first subset
        idx2 : np.ndarray
            Indices of second subset
            
        Returns
        -------
        float
            MMD² estimate
        """
        idx1 = np.asarray(idx1, dtype=int)
        idx2 = np.asarray(idx2, dtype=int)
        
        mean1 = self.Phi[idx1].mean(axis=0) if idx1.size > 0 else np.zeros_like(self.mean_full)
        mean2 = self.Phi[idx2].mean(axis=0) if idx2.size > 0 else np.zeros_like(self.mean_full)
        
        diff = mean1 - mean2
        return float(np.dot(diff, diff))

    def kernel_mean_embedding(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get kernel mean embedding for a subset.
        
        Parameters
        ----------
        idx : Optional[np.ndarray]
            Indices of subset. If None, returns full dataset embedding.
            
        Returns
        -------
        np.ndarray
            Mean embedding, shape (2m,)
        """
        if idx is None:
            return self.mean_full.copy()
        
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return np.zeros_like(self.mean_full)
        
        return self.Phi[idx].mean(axis=0)


def compute_rff_features(
    X: np.ndarray,
    rff_dim: int,
    sigma_sq: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute Random Fourier Features for RBF kernel.
    
    Standalone function for computing RFF without building full estimator.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (N, d)
    rff_dim : int
        Number of random features
    sigma_sq : float
        RBF kernel bandwidth
    seed : int
        Random seed
        
    Returns
    -------
    np.ndarray
        RFF features, shape (N, rff_dim)
        
    Raises
    ------
    ValueError
        If sigma_sq <= 0 or not finite, or if rff_dim <= 0
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    m = int(rff_dim)
    
    # Validate parameters
    sigma_sq = float(sigma_sq)
    if not np.isfinite(sigma_sq) or sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive and finite, got {sigma_sq}")
    if m <= 0:
        raise ValueError(f"rff_dim must be positive, got {m}")
    
    rng = np.random.default_rng(seed)
    
    freq_scale = 1.0 / np.sqrt(sigma_sq)
    W = rng.standard_normal((d, m)) * freq_scale
    b = rng.uniform(0, 2 * np.pi, size=m)
    
    projection = X @ W + b
    Phi = np.sqrt(2.0 / m) * np.cos(projection)
    
    return Phi


def mmd_from_rff(
    Phi: np.ndarray,
    idx: np.ndarray,
) -> float:
    """
    Compute MMD using precomputed RFF features.
    
    This is a convenience function that computes MMD² between a subset
    (specified by indices) and the full dataset using RFF embeddings.
    
    Parameters
    ----------
    Phi : np.ndarray
        RFF features for full dataset, shape (N, m)
    idx : np.ndarray
        Indices of the subset
        
    Returns
    -------
    float
        MMD² estimate
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    idx = np.asarray(idx, dtype=int)
    
    if idx.size == 0:
        mean_full = Phi.mean(axis=0)
        return float(np.dot(mean_full, mean_full))
    
    mean_full = Phi.mean(axis=0)
    mean_subset = Phi[idx].mean(axis=0)
    
    diff = mean_full - mean_subset
    return float(np.dot(diff, diff))


def mmd2_exact(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_sq: float,
) -> float:
    """
    Compute exact MMD² using full kernel matrices.
    
    Warning: O(n² + m²) complexity, use only for small datasets.
    
    Parameters
    ----------
    X : np.ndarray
        First sample, shape (n, d)
    Y : np.ndarray
        Second sample, shape (m, d)
    sigma_sq : float
        RBF kernel bandwidth
        
    Returns
    -------
    float
        Exact MMD²
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    def rbf_kernel(A, B):
        sq_dist = (
            np.sum(A ** 2, axis=1, keepdims=True)
            + np.sum(B ** 2, axis=1)
            - 2 * A @ B.T
        )
        return np.exp(-sq_dist / (2 * sigma_sq))
    
    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)
    
    n = len(X)
    m = len(Y)
    
    # MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd2 = (K_xx.sum() / (n * n) + K_yy.sum() / (m * m) - 2 * K_xy.sum() / (n * m))
    
    return float(mmd2)
