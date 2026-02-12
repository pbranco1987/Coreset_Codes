"""
PCA-related utilities for representation learning.

Contains:
- fit_pca: Fit PCA model to data
- pca_embed: Transform data using fitted PCA
- IncrementalPCAWrapper: Wrapper for large datasets
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


def fit_pca(
    X: np.ndarray,
    n_components: int = 16,
    whiten: bool = False,
    random_state: int = 0,
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA and return model with transformed data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data, shape (N, d)
    n_components : int
        Number of PCA components
    whiten : bool
        Whether to whiten the output
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[PCA, np.ndarray]
        (fitted_pca_model, transformed_data)
    """
    X = np.asarray(X, dtype=np.float64)
    
    # Limit components to min(n_samples, n_features)
    max_components = min(X.shape[0], X.shape[1])
    n_components = min(n_components, max_components)
    
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        random_state=random_state,
    )
    
    Z = pca.fit_transform(X)
    
    return pca, Z.astype(np.float32)


def pca_embed(
    pca: PCA,
    X: np.ndarray,
) -> np.ndarray:
    """
    Transform data using a fitted PCA model.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA model
    X : np.ndarray
        Data to transform
        
    Returns
    -------
    np.ndarray
        Transformed data
    """
    return pca.transform(X).astype(np.float32)


def explained_variance_ratio(pca: PCA) -> np.ndarray:
    """
    Get explained variance ratio for each component.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA model
        
    Returns
    -------
    np.ndarray
        Explained variance ratio per component
    """
    return pca.explained_variance_ratio_


def cumulative_explained_variance(pca: PCA) -> np.ndarray:
    """
    Get cumulative explained variance.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA model
        
    Returns
    -------
    np.ndarray
        Cumulative explained variance
    """
    return np.cumsum(pca.explained_variance_ratio_)


def components_for_variance(
    X: np.ndarray,
    target_variance: float = 0.95,
    random_state: int = 0,
) -> int:
    """
    Determine number of components needed for target variance.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    target_variance : float
        Target cumulative explained variance (0-1)
    random_state : int
        Random seed
        
    Returns
    -------
    int
        Number of components needed
    """
    X = np.asarray(X, dtype=np.float64)
    
    # Fit full PCA
    max_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=max_components, random_state=random_state)
    pca.fit(X)
    
    # Find number of components
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumvar, target_variance) + 1
    
    return int(min(n_components, max_components))


class IncrementalPCAWrapper:
    """
    Wrapper for incremental PCA on large datasets.
    
    Processes data in batches to avoid memory issues.
    
    Attributes
    ----------
    n_components : int
        Number of components
    batch_size : int
        Batch size for incremental fitting
    pca : IncrementalPCA
        Underlying model
    """
    
    def __init__(
        self,
        n_components: int = 16,
        batch_size: int = 1000,
        whiten: bool = False,
    ):
        """
        Initialize the wrapper.
        
        Parameters
        ----------
        n_components : int
            Number of PCA components
        batch_size : int
            Batch size for incremental fitting
        whiten : bool
            Whether to whiten output
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.whiten = whiten
        self.pca = IncrementalPCA(n_components=n_components, whiten=whiten)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "IncrementalPCAWrapper":
        """
        Fit the model incrementally.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        IncrementalPCAWrapper
            Self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            self.pca.partial_fit(X[start:end])
        
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted model.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        np.ndarray
            Transformed data
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Transformed data
        """
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Get explained variance ratio."""
        return self.pca.explained_variance_ratio_

    @property
    def components_(self) -> np.ndarray:
        """Get principal components."""
        return self.pca.components_
