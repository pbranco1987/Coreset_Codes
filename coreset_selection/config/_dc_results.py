"""
Result dataclasses.

Contains data structures for experiment outputs:
- Cached replicate assets (pre-computed representations)
- Results bundles (optimization outputs and metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._dc_experiment import ExperimentConfig


@dataclass
class ReplicateAssets:
    """
    Cached assets for a single replicate.

    Contains pre-computed representations and data splits
    to ensure reproducibility across runs.

    Attributes
    ----------
    X_raw : np.ndarray
        Raw feature matrix (N, d_raw)
    X_scaled : np.ndarray
        Standardized feature matrix (N, d_raw)
    Z_vae : Optional[np.ndarray]
        VAE latent embeddings (N, d_latent)
    Z_logvar : Optional[np.ndarray]
        VAE log-variance (N, d_latent)
    Z_pca : Optional[np.ndarray]
        PCA embeddings (N, d_pca)
    state_labels : np.ndarray
        Geographic group labels (N,)
    train_idx : np.ndarray
        Training set indices (representation-learning train split)
    val_idx : Optional[np.ndarray]
        Validation indices (representation-learning validation split)
    eval_idx : np.ndarray
        Evaluation set indices
    eval_train_idx : np.ndarray
        Training indices within eval set (for CV)
    eval_test_idx : np.ndarray
        Test indices within eval set (for CV)
    y : Optional[np.ndarray]
        Target values for regression evaluation
    metadata : Dict[str, Any]
        Additional metadata
    """
    X_raw: np.ndarray
    X_scaled: np.ndarray
    Z_vae: Optional[np.ndarray]
    Z_logvar: Optional[np.ndarray]
    Z_pca: Optional[np.ndarray]
    state_labels: np.ndarray
    train_idx: np.ndarray
    eval_idx: np.ndarray
    eval_train_idx: np.ndarray
    eval_test_idx: np.ndarray
    val_idx: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    population: Optional[np.ndarray] = None  # Per-municipality population weights for pop-share constraint
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 2: Type-aware feature metadata
    feature_types: List[str] = field(default_factory=list)
    category_maps: Dict[str, Dict] = field(default_factory=dict)
    target_type: str = "regression"  # "regression" or "classification"


@dataclass
class ResultsBundle:
    """
    Bundle of results from a single experiment run.

    Attributes
    ----------
    run_id : str
        Run identifier
    rep_id : int
        Replicate identifier
    k : int
        Coreset size
    pareto_F : Optional[np.ndarray]
        Pareto front objective values
    pareto_X : Optional[np.ndarray]
        Pareto front decision variables
    selected_indices : Dict[str, np.ndarray]
        Selected indices for each method/representative
    metrics : Dict[str, Dict[str, float]]
        Metrics for each method
    timing : Dict[str, float]
        Timing information
    config : Optional[ExperimentConfig]
        Configuration used
    """
    run_id: str
    rep_id: int
    k: int
    pareto_F: Optional[np.ndarray] = None
    pareto_X: Optional[np.ndarray] = None
    selected_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    config: Optional[ExperimentConfig] = None
