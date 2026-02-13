r"""Quality of Service (QoS / IQS) downstream evaluation tasks.

Consolidated from the ``complex_downstream_tasks/`` Jupyter notebooks into a
single, reusable module.  These tasks evaluate coreset quality by training
supervised models on a coreset subset and testing on the full dataset
(or a held-out split), measuring how well the coreset preserves the
predictive structure of the original data.

Five model families are implemented, each in both **pooled** and
**fixed-effects** (entity-demeaned) variants:

1. **OLS** — Ordinary Least Squares (``LinearRegression``)
2. **Ridge** — L2-regularized regression with time-series CV for alpha
3. **Lasso / Elastic Net** — L1/L2-regularized regression (ADL / ARX
   formulation with optional lagged features)
4. **PLS** — Partial Least Squares with panel-aware component tuning
5. **Constrained OLS** — Convex-combination regression where
   coefficients sum to 1 and are bounded (portfolio / composite-index
   interpretation)

An additional **Heuristic** baseline applies fixed, pre-defined indicator
weights (matching the domain-expert weighting of the ISG composite index).

All evaluators follow the same contract used by the existing
``evaluation.downstream_metrics`` module:

- Accept numpy arrays and coreset index arrays
- Return ``Dict[str, float]`` with prefixed metric keys
- Support optional per-state (geographic group) breakdown

References
----------
- Notebooks: ``elastic_net_with_lags_model.ipynb``,
  ``organized_constrained_linear_model.ipynb``,
  ``partial_least_squares_model.ipynb``,
  ``partial_least_regression_with_elastic_net.ipynb``,
  ``organized_linear_model_part2.ipynb``,
  ``simplified_model_1.ipynb``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._downstream_helpers import _rmse, _mae, _r2

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class QoSConfig:
    """Configuration for QoS downstream evaluation.

    Parameters
    ----------
    models : list of str
        Which model families to run.  Any subset of
        ``["ols", "ridge", "elastic_net", "pls", "constrained",
        "heuristic"]``.  Default: all.
    run_fixed_effects : bool
        Whether to also run fixed-effects (entity-demeaned) variants.
    n_lags : int
        Number of autoregressive lags for ADL/ARX design.  0 = no lags
        (contemporaneous only).
    use_contemporaneous : bool
        Whether to include contemporaneous features alongside lags.
    ridge_alphas : list of float
        Grid of alpha values for Ridge CV.
    elastic_net_alpha : float
        Overall regularization strength for Elastic Net.
    elastic_net_l1_ratio : float
        L1 / (L1+L2) mixing ratio for Elastic Net.
    pls_max_components : int or None
        Maximum PLS components (None = n_features).
    pls_cv_folds : int
        Number of CV folds for PLS component tuning.
    constrained_epsilon : float
        Minimum coefficient weight for Constrained OLS.
    constrained_max_weight : float
        Maximum coefficient weight for Constrained OLS.
    constrained_variance_strength : float
        Diversity penalty strength for Constrained OLS.
    heuristic_weights : dict or None
        Fixed indicator weights for the Heuristic baseline.
        If None, uses the default ISG weights.
    feature_names : list of str or None
        Names of the feature columns (IND1..IND8).
        If None, generic names are assigned.
    metric_prefix : str
        Prefix for all returned metric keys (default: ``"qos_"``).
    """

    models: List[str] = field(default_factory=lambda: [
        "ols", "ridge", "elastic_net", "pls", "constrained", "heuristic",
    ])
    run_fixed_effects: bool = True
    n_lags: int = 0
    use_contemporaneous: bool = True

    # Ridge
    ridge_alphas: List[float] = field(default_factory=lambda: [
        0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
    ])

    # Elastic Net
    elastic_net_alpha: float = 0.01
    elastic_net_l1_ratio: float = 0.5

    # PLS
    pls_max_components: Optional[int] = None
    pls_cv_folds: int = 5

    # Constrained OLS
    constrained_epsilon: float = 0.05
    constrained_max_weight: float = 0.50
    constrained_variance_strength: float = 0.01

    # Heuristic
    heuristic_weights: Optional[Dict[str, float]] = None

    # Feature names
    feature_names: Optional[List[str]] = None

    # Metric key prefix
    metric_prefix: str = "qos_"


# Default ISG heuristic weights (from domain experts).
_DEFAULT_HEURISTIC_WEIGHTS = {
    "IND1": 0.1, "IND2": 0.1, "IND3": 0.1, "IND4": 0.2,
    "IND5": 0.1, "IND6": 0.1, "IND7": 0.1, "IND8": 0.2,
}


# =====================================================================
# Panel-data utilities
# =====================================================================

class Demeaner:
    """Fixed-effects (within) transformation for panel data.

    Learns entity-specific means from training data, then subtracts them
    to remove time-invariant heterogeneity.  Can also re-inflate
    predictions by adding back the entity means.
    """

    def __init__(self) -> None:
        self.entity_means_: Optional[np.ndarray] = None   # (n_entities, d)
        self.global_mean_: Optional[np.ndarray] = None     # (d,)
        self._entity_ids: Optional[np.ndarray] = None      # unique sorted ids
        self._id_to_pos: Optional[Dict[int, int]] = None

    def fit(
        self,
        X: np.ndarray,
        entity_ids: np.ndarray,
    ) -> "Demeaner":
        """Learn per-entity means from training data.

        Parameters
        ----------
        X : (n, d)
            Feature matrix or target column(s).
        entity_ids : (n,)
            Integer entity identifiers (e.g. municipality codes).
        """
        X = np.asarray(X, dtype=np.float64)
        entity_ids = np.asarray(entity_ids)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.global_mean_ = np.nanmean(X, axis=0)

        unique_ids, inverse = np.unique(entity_ids, return_inverse=True)
        self._entity_ids = unique_ids
        self._id_to_pos = {int(eid): i for i, eid in enumerate(unique_ids)}

        n_entities = len(unique_ids)
        d = X.shape[1]

        # Vectorised per-entity mean: group-by via bincount
        counts = np.bincount(inverse, minlength=n_entities).astype(np.float64)
        counts[counts == 0] = 1.0  # avoid division by zero
        self.entity_means_ = np.zeros((n_entities, d), dtype=np.float64)
        for col in range(d):
            col_vals = np.where(np.isnan(X[:, col]), 0.0, X[:, col])
            col_valid = (~np.isnan(X[:, col])).astype(np.float64)
            sums = np.bincount(inverse, weights=col_vals, minlength=n_entities)
            valid_counts = np.bincount(inverse, weights=col_valid, minlength=n_entities)
            valid_counts[valid_counts == 0] = 1.0
            self.entity_means_[:, col] = sums / valid_counts

        # Replace any all-NaN entity means with global mean
        nan_rows = np.isnan(self.entity_means_).any(axis=1)
        if nan_rows.any():
            self.entity_means_[nan_rows] = self.global_mean_

        return self

    def _resolve_positions(self, entity_ids: np.ndarray) -> np.ndarray:
        """Map entity IDs to row positions in entity_means_.

        Returns an int array of length len(entity_ids).  Unseen IDs are
        mapped to -1 (handled by callers via the global mean).
        """
        # Fast path: try searchsorted against the sorted unique_ids
        positions = np.searchsorted(self._entity_ids, entity_ids)
        # Clamp out-of-bounds so the equality check below doesn't segfault
        positions = np.clip(positions, 0, len(self._entity_ids) - 1)
        found = self._entity_ids[positions] == entity_ids
        positions[~found] = -1
        return positions

    def transform(
        self,
        X: np.ndarray,
        entity_ids: np.ndarray,
    ) -> np.ndarray:
        """Subtract entity means.  Unseen entities use the global mean."""
        if self.entity_means_ is None:
            raise RuntimeError("Demeaner must be fit before transform.")

        X = np.asarray(X, dtype=np.float64)
        entity_ids = np.asarray(entity_ids)
        squeezed = X.ndim == 1
        if squeezed:
            X = X.reshape(-1, 1)

        positions = self._resolve_positions(entity_ids)

        # Build aligned means: known entities from entity_means_,
        # unknown (pos == -1) from global_mean_.
        known = positions >= 0
        means_aligned = np.tile(self.global_mean_, (X.shape[0], 1))
        if known.any():
            means_aligned[known] = self.entity_means_[positions[known]]

        result = X - means_aligned
        return result.ravel() if squeezed else result

    def reinflate(
        self,
        y_demeaned: np.ndarray,
        entity_ids: np.ndarray,
    ) -> np.ndarray:
        """Add entity means back to demeaned predictions."""
        if self.entity_means_ is None:
            raise RuntimeError("Demeaner must be fit before reinflate.")

        y_demeaned = np.asarray(y_demeaned, dtype=np.float64).ravel()
        entity_ids = np.asarray(entity_ids)

        positions = self._resolve_positions(entity_ids)
        known = positions >= 0

        global_y = float(self.global_mean_.ravel()[0])
        result = y_demeaned + global_y  # default: add global mean
        if known.any():
            # Overwrite known entities with their specific means
            result[known] = y_demeaned[known] + self.entity_means_[positions[known], 0]
        return result


def build_lagged_features(
    X: np.ndarray,
    y: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: np.ndarray,
    n_lags: int = 1,
    use_contemporaneous: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ADL (Autoregressive Distributed Lag) feature matrix.

    Adds lagged versions of X and y to create a richer feature set that
    captures temporal dynamics within each entity.

    Parameters
    ----------
    X : (n, d)
        Contemporaneous features.
    y : (n,)
        Target variable.
    entity_ids : (n,)
        Entity identifiers.
    time_ids : (n,)
        Time identifiers (must be sortable).
    n_lags : int
        Number of lags to include.
    use_contemporaneous : bool
        Whether to keep contemporaneous X alongside lags.

    Returns
    -------
    X_adl : (n', d')
        ADL feature matrix (rows with incomplete lags are dropped).
    y_adl : (n',)
        Corresponding target values.
    entity_ids_adl : (n',)
        Corresponding entity IDs.
    time_ids_adl : (n',)
        Corresponding time IDs.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    entity_ids = np.asarray(entity_ids)
    time_ids = np.asarray(time_ids)

    if n_lags <= 0:
        return X, y, entity_ids, time_ids

    n, d = X.shape

    # Sort by entity then time
    order = np.lexsort((time_ids, entity_ids))
    X = X[order]
    y = y[order]
    entity_ids = entity_ids[order]
    time_ids = time_ids[order]

    # Build lagged columns per entity
    all_x_lags = []  # list of (n, d) arrays per lag
    all_y_lags = []  # list of (n,) arrays per lag
    valid_mask = np.ones(n, dtype=bool)

    for lag in range(1, n_lags + 1):
        x_lag = np.full((n, d), np.nan)
        y_lag = np.full(n, np.nan)

        for eid in np.unique(entity_ids):
            emask = entity_ids == eid
            idx = np.where(emask)[0]
            if len(idx) <= lag:
                valid_mask[idx] = False
                continue
            # Shift within this entity
            x_lag[idx[lag:]] = X[idx[:-lag]]
            y_lag[idx[lag:]] = y[idx[:-lag]]
            # First `lag` rows for this entity are invalid
            valid_mask[idx[:lag]] = False

        all_x_lags.append(x_lag)
        all_y_lags.append(y_lag)

    # Assemble feature matrix
    parts = []
    if use_contemporaneous:
        parts.append(X)
    parts.extend(all_x_lags)
    parts.extend([yl.reshape(-1, 1) for yl in all_y_lags])

    X_adl_full = np.column_stack(parts)

    # Drop rows with incomplete lags
    X_adl = X_adl_full[valid_mask]
    y_adl = y[valid_mask]
    entity_ids_adl = entity_ids[valid_mask]
    time_ids_adl = time_ids[valid_mask]

    return X_adl, y_adl, entity_ids_adl, time_ids_adl


# =====================================================================
# Model implementations
# =====================================================================

def _fit_ols(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit OLS via normal equations.  Returns (coef, intercept)."""
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    return model.coef_.ravel(), float(model.intercept_)


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: List[float],
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, float, float]:
    """Fit Ridge with time-series CV for alpha selection.

    Returns (coef, intercept, best_alpha).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    best_alpha = alphas[0]
    best_mse = np.inf

    n_splits_safe = min(n_splits, max(2, X_train.shape[0] // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits_safe)

    for alpha in alphas:
        fold_mses = []
        for tr_idx, va_idx in tscv.split(X_train):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train[tr_idx], y_train[tr_idx])
            pred = model.predict(X_train[va_idx])
            fold_mses.append(mean_squared_error(y_train[va_idx], pred))
        avg_mse = float(np.mean(fold_mses))
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = alpha

    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    return model.coef_.ravel(), float(model.intercept_), best_alpha


def _fit_elastic_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Fit Elastic Net.  Returns (coef, intercept)."""
    from sklearn.linear_model import ElasticNet

    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=True, max_iter=10000,
    )
    model.fit(X_train, y_train)
    return model.coef_.ravel(), float(model.intercept_)


def _fit_pls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_components: Optional[int] = None,
    cv_folds: int = 5,
) -> Tuple[np.ndarray, float, int]:
    """Fit PLS with CV for component selection.

    Returns (coef, intercept, n_components).
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    max_comp = min(max_components or n_features, n_features, n_samples - 1)

    if max_comp < 1:
        logger.warning("PLS: Not enough features or samples for any components.")
        return np.zeros(n_features), 0.0, 0

    n_splits_safe = min(cv_folds, max(2, n_samples // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits_safe)

    best_k = 1
    best_mse = np.inf

    import warnings as _w

    for k in range(1, max_comp + 1):
        fold_mses = []
        for tr_idx, va_idx in tscv.split(X_train):
            if len(tr_idx) < k or len(va_idx) == 0:
                continue
            try:
                with _w.catch_warnings():
                    _w.filterwarnings("ignore", message="y residual is constant")
                    pls = PLSRegression(n_components=k, scale=True)
                    pls.fit(X_train[tr_idx], y_train[tr_idx])
                    pred = pls.predict(X_train[va_idx]).ravel()
                fold_mses.append(mean_squared_error(y_train[va_idx], pred))
            except Exception:
                fold_mses.append(np.nan)

        valid = [m for m in fold_mses if np.isfinite(m)]
        if valid:
            avg_mse = float(np.mean(valid))
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_k = k

    # Fit final model with optimal components
    with _w.catch_warnings():
        _w.filterwarnings("ignore", message="y residual is constant")
        pls = PLSRegression(n_components=best_k, scale=True)
        pls.fit(X_train, y_train)

    coef = pls.coef_.ravel()

    # Compute intercept: y_mean - coef @ X_mean  (since PLS centers internally)
    intercept = float(np.mean(y_train) - coef @ np.mean(X_train, axis=0))

    return coef, intercept, best_k


def _fit_constrained_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float = 0.05,
    max_weight: float = 0.50,
    variance_strength: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Fit constrained OLS where weights sum to 1 and are bounded.

    When D >> N (more features than samples), PCA is applied first to
    reduce dimensionality to min(N-1, 50) components.  The constrained
    weights are then fitted in the reduced space and projected back to
    the original feature space.

    Uses SLSQP optimization.  Returns (coef, intercept=0).
    """
    import scipy.optimize as sco
    from sklearn.decomposition import PCA

    n_samples, n_features = X_train.shape

    # When D >> N, reduce to a tractable number of components
    pca = None
    max_components = min(n_samples - 1, 50)
    if n_features > max_components:
        pca = PCA(n_components=max_components)
        X_reduced = pca.fit_transform(X_train)
        n_opt = max_components
    else:
        X_reduced = X_train
        n_opt = n_features

    # Feasibility check — silently relax bounds when needed
    if n_opt * epsilon > 1.0 + 1e-9:
        epsilon = max(1e-6, 0.5 / n_opt)
    if max_weight * n_opt < 1.0 - 1e-9:
        max_weight = min(1.0, 2.0 / n_opt)

    def objective(beta: np.ndarray) -> float:
        mse = float(np.mean((y_train - X_reduced @ beta) ** 2))
        diversity = -variance_strength * float(np.var(beta))
        return mse + diversity

    constraints = {"type": "eq", "fun": lambda beta: np.sum(beta) - 1.0}
    bounds = sco.Bounds(epsilon, max_weight)
    beta_init = np.full(n_opt, 1.0 / n_opt)

    result = sco.minimize(
        objective, beta_init, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        logger.debug("Constrained OLS: %s", result.message)

    # Project back to original feature space if PCA was used
    if pca is not None:
        coef_full = pca.inverse_transform(result.x.reshape(1, -1)).ravel()
    else:
        coef_full = result.x

    return coef_full, 0.0


def _predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Compute predictions: X @ coef + intercept."""
    return X @ coef + intercept


# =====================================================================
# Core evaluation function: evaluate one model on one (train, test) pair
# =====================================================================

def _evaluate_single_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    config: QoSConfig,
    state_labels_test: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Train one model on the coreset and evaluate on the test set.

    Parameters
    ----------
    X_train, y_train : training data (coreset)
    X_test, y_test : test data (full set or held-out evaluation split)
    model_name : one of "ols", "ridge", "elastic_net", "pls", "constrained"
    config : QoSConfig
    state_labels_test : optional per-point state labels for the test set
    prefix : string prefix for metric keys

    Returns
    -------
    Dict[str, float] with prefixed metric keys.
    """
    from ._downstream_helpers import (
        tail_absolute_errors,
        per_state_downstream_metrics,
        aggregate_group_metrics,
    )

    sfx = prefix
    out: Dict[str, Any] = {}

    try:
        if model_name == "ols":
            coef, intercept = _fit_ols(X_train, y_train)
        elif model_name == "ridge":
            coef, intercept, best_alpha = _fit_ridge(
                X_train, y_train, config.ridge_alphas,
            )
            out[f"{sfx}ridge_alpha"] = best_alpha
        elif model_name == "elastic_net":
            coef, intercept = _fit_elastic_net(
                X_train, y_train,
                alpha=config.elastic_net_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
            )
        elif model_name == "pls":
            coef, intercept, n_comp = _fit_pls(
                X_train, y_train,
                max_components=config.pls_max_components,
                cv_folds=config.pls_cv_folds,
            )
            out[f"{sfx}pls_n_components"] = n_comp
        elif model_name == "constrained":
            coef, intercept = _fit_constrained_ols(
                X_train, y_train,
                epsilon=config.constrained_epsilon,
                max_weight=config.constrained_max_weight,
                variance_strength=config.constrained_variance_strength,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.warning("QoS %s%s fit failed: %s", sfx, model_name, e)
        out[f"{sfx}rmse"] = np.nan
        out[f"{sfx}mae"] = np.nan
        out[f"{sfx}r2"] = np.nan
        return out

    y_pred = _predict(X_test, coef, intercept)

    # Core metrics
    out[f"{sfx}rmse"] = _rmse(y_test, y_pred)
    out[f"{sfx}mae"] = _mae(y_test, y_pred)
    out[f"{sfx}r2"] = _r2(y_test, y_pred)

    # Tail errors
    tails = tail_absolute_errors(y_test, y_pred)
    for k, v in tails.items():
        out[f"{sfx}{k}"] = v

    # Per-state breakdown
    if state_labels_test is not None:
        per_state = per_state_downstream_metrics(y_test, y_pred, state_labels_test)
        agg = aggregate_group_metrics(per_state)
        for k, v in agg.items():
            out[f"{sfx}{k}"] = v

    return out


# =====================================================================
# Fixed-effects wrapper
# =====================================================================

def _evaluate_fixed_effects_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    entity_ids_train: np.ndarray,
    entity_ids_test: np.ndarray,
    model_name: str,
    config: QoSConfig,
    state_labels_test: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Train a fixed-effects model and evaluate.

    Demeans both X and y using entity means learned from training data,
    fits the model on demeaned data, then re-inflates predictions for
    evaluation against the original (raw) test targets.
    """
    # Fit demeaners on training data
    demeaner_X = Demeaner()
    demeaner_y = Demeaner()
    demeaner_X.fit(X_train, entity_ids_train)
    demeaner_y.fit(y_train.reshape(-1, 1), entity_ids_train)

    # Transform
    X_train_dm = demeaner_X.transform(X_train, entity_ids_train)
    y_train_dm = demeaner_y.transform(y_train, entity_ids_train)
    X_test_dm = demeaner_X.transform(X_test, entity_ids_test)

    # Impute NaNs in demeaned data with column medians
    for col in range(X_train_dm.shape[1]):
        col_median = np.nanmedian(X_train_dm[:, col])
        nan_train = np.isnan(X_train_dm[:, col])
        nan_test = np.isnan(X_test_dm[:, col])
        if nan_train.any():
            X_train_dm[nan_train, col] = col_median
        if nan_test.any():
            X_test_dm[nan_test, col] = col_median

    nan_y = np.isnan(y_train_dm)
    if nan_y.any():
        y_train_dm[nan_y] = np.nanmedian(y_train_dm)

    # Run the model on demeaned data, but evaluate against raw y_test
    # (we re-inflate predictions).
    from ._downstream_helpers import (
        tail_absolute_errors,
        per_state_downstream_metrics,
        aggregate_group_metrics,
    )

    sfx = prefix
    out: Dict[str, Any] = {}

    try:
        if model_name == "ols":
            coef, intercept = _fit_ols(X_train_dm, y_train_dm)
        elif model_name == "ridge":
            coef, intercept, best_alpha = _fit_ridge(
                X_train_dm, y_train_dm, config.ridge_alphas,
            )
            out[f"{sfx}ridge_alpha"] = best_alpha
        elif model_name == "elastic_net":
            coef, intercept = _fit_elastic_net(
                X_train_dm, y_train_dm,
                alpha=config.elastic_net_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
            )
        elif model_name == "pls":
            coef, intercept, n_comp = _fit_pls(
                X_train_dm, y_train_dm,
                max_components=config.pls_max_components,
                cv_folds=config.pls_cv_folds,
            )
            out[f"{sfx}pls_n_components"] = n_comp
        elif model_name == "constrained":
            coef, intercept = _fit_constrained_ols(
                X_train_dm, y_train_dm,
                epsilon=config.constrained_epsilon,
                max_weight=config.constrained_max_weight,
                variance_strength=config.constrained_variance_strength,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.warning("QoS FE %s%s fit failed: %s", sfx, model_name, e)
        out[f"{sfx}rmse"] = np.nan
        out[f"{sfx}mae"] = np.nan
        out[f"{sfx}r2"] = np.nan
        return out

    # Predict on demeaned test data, then re-inflate
    y_pred_dm = _predict(X_test_dm, coef, intercept)
    y_pred = demeaner_y.reinflate(y_pred_dm, entity_ids_test)

    # Core metrics (against raw y_test)
    out[f"{sfx}rmse"] = _rmse(y_test, y_pred)
    out[f"{sfx}mae"] = _mae(y_test, y_pred)
    out[f"{sfx}r2"] = _r2(y_test, y_pred)

    # Tail errors
    tails = tail_absolute_errors(y_test, y_pred)
    for k, v in tails.items():
        out[f"{sfx}{k}"] = v

    # Per-state breakdown
    if state_labels_test is not None:
        per_state = per_state_downstream_metrics(y_test, y_pred, state_labels_test)
        agg = aggregate_group_metrics(per_state)
        for k, v in agg.items():
            out[f"{sfx}{k}"] = v

    return out


# =====================================================================
# Heuristic evaluation
# =====================================================================

def _evaluate_heuristic(
    *,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    state_labels_test: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Evaluate the fixed-weight heuristic baseline.

    The heuristic applies pre-defined indicator weights to produce
    predictions as a weighted sum of features (no training needed).
    """
    from ._downstream_helpers import (
        tail_absolute_errors,
        per_state_downstream_metrics,
        aggregate_group_metrics,
    )

    sfx = prefix
    out: Dict[str, Any] = {}

    n_features = X_test.shape[1]
    if len(weights) < n_features:
        # Pad with zeros for lagged features
        w = np.zeros(n_features)
        w[:len(weights)] = weights
    else:
        w = weights[:n_features]

    y_pred = X_test @ w

    out[f"{sfx}rmse"] = _rmse(y_test, y_pred)
    out[f"{sfx}mae"] = _mae(y_test, y_pred)
    out[f"{sfx}r2"] = _r2(y_test, y_pred)

    tails = tail_absolute_errors(y_test, y_pred)
    for k, v in tails.items():
        out[f"{sfx}{k}"] = v

    if state_labels_test is not None:
        per_state = per_state_downstream_metrics(y_test, y_pred, state_labels_test)
        agg = aggregate_group_metrics(per_state)
        for k, v in agg.items():
            out[f"{sfx}{k}"] = v

    return out


# =====================================================================
# Top-level evaluation dispatcher
# =====================================================================

def qos_coreset_evaluation(
    *,
    X_full: np.ndarray,
    y_full: np.ndarray,
    S_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    entity_ids: Optional[np.ndarray] = None,
    time_ids: Optional[np.ndarray] = None,
    state_labels: Optional[np.ndarray] = None,
    config: Optional[QoSConfig] = None,
) -> Dict[str, Any]:
    """Run all QoS downstream tasks comparing a coreset to the full dataset.

    This is the **main entry point** for QoS evaluation.  It trains each
    enabled model on the coreset (``X_full[S_idx]``) and evaluates on
    the held-out test set (``X_full[eval_test_idx]``).

    Parameters
    ----------
    X_full : (N, d) ndarray
        Full feature matrix for all municipalities.
    y_full : (N,) ndarray
        Full target vector (ISG / QoS score).
    S_idx : (k,) ndarray of int
        Coreset indices (subset of ``range(N)``).
    eval_test_idx : (n_test,) ndarray of int
        Held-out test indices for evaluation.
    entity_ids : (N,) ndarray, optional
        Municipality codes.  Required for fixed-effects models and
        lagged features.
    time_ids : (N,) ndarray, optional
        Time period identifiers.  Required for lagged features.
    state_labels : (N,) ndarray, optional
        Geographic group labels for per-state metrics.
    config : QoSConfig, optional
        Configuration.  If None, uses defaults (all models, no lags).

    Returns
    -------
    Dict[str, float]
        Combined metric dictionary with keys like
        ``"qos_ols_pooled_rmse"``, ``"qos_pls_fe_r2"``, etc.
    """
    if config is None:
        config = QoSConfig()

    X_full = np.asarray(X_full, dtype=np.float64)
    y_full = np.asarray(y_full, dtype=np.float64).ravel()
    S_idx = np.asarray(S_idx, dtype=int)
    eval_test_idx = np.asarray(eval_test_idx, dtype=int)

    pfx = config.metric_prefix  # "qos_"

    # Build training and test sets
    X_train = X_full[S_idx]
    y_train = y_full[S_idx]
    X_test = X_full[eval_test_idx]
    y_test = y_full[eval_test_idx]

    state_labels_test = (
        np.asarray(state_labels)[eval_test_idx]
        if state_labels is not None else None
    )

    # Entity / time IDs for the subsets
    eid_train = (
        np.asarray(entity_ids)[S_idx]
        if entity_ids is not None else None
    )
    eid_test = (
        np.asarray(entity_ids)[eval_test_idx]
        if entity_ids is not None else None
    )
    tid_train = (
        np.asarray(time_ids)[S_idx]
        if time_ids is not None else None
    )
    tid_test = (
        np.asarray(time_ids)[eval_test_idx]
        if time_ids is not None else None
    )

    # Impute NaNs in raw features with training medians
    col_medians = np.nanmedian(X_train, axis=0)
    for col in range(X_train.shape[1]):
        nan_tr = np.isnan(X_train[:, col])
        nan_te = np.isnan(X_test[:, col])
        if nan_tr.any():
            X_train[nan_tr, col] = col_medians[col]
        if nan_te.any():
            X_test[nan_te, col] = col_medians[col]

    nan_y = np.isnan(y_train)
    if nan_y.any():
        y_train[nan_y] = np.nanmedian(y_train)

    # Build ADL features if lags are requested
    X_train_adl, y_train_adl = X_train, y_train
    X_test_adl, y_test_adl = X_test, y_test
    eid_train_adl, eid_test_adl = eid_train, eid_test
    state_test_adl = state_labels_test

    has_panel_info = (entity_ids is not None and time_ids is not None)
    adl_active = config.n_lags > 0 and has_panel_info

    if adl_active:
        try:
            X_train_adl, y_train_adl, eid_train_adl, _ = build_lagged_features(
                X_train, y_train, eid_train, tid_train,
                n_lags=config.n_lags,
                use_contemporaneous=config.use_contemporaneous,
            )
            X_test_adl, y_test_adl, eid_test_adl, _ = build_lagged_features(
                X_test, y_test, eid_test, tid_test,
                n_lags=config.n_lags,
                use_contemporaneous=config.use_contemporaneous,
            )
            # Re-align state labels for ADL test set (some rows dropped)
            if state_labels_test is not None and len(y_test_adl) < len(y_test):
                # The build_lagged_features sorts by (entity, time) and drops
                # leading rows.  We need to re-derive the state labels.
                # Since build_lagged_features returns the entity_ids, we can
                # look up state labels.  However, state_labels are indexed
                # the same as entity_ids in the original data.  For simplicity,
                # we use the entity-to-state mapping.
                state_test_adl = state_labels_test  # best-effort; may be shorter
                if len(state_test_adl) != len(y_test_adl):
                    state_test_adl = None  # disable per-state for ADL
        except Exception as e:
            logger.warning("ADL feature construction failed: %s. "
                           "Falling back to contemporaneous.", e)
            adl_active = False
            X_train_adl, y_train_adl = X_train, y_train
            X_test_adl, y_test_adl = X_test, y_test
            eid_train_adl, eid_test_adl = eid_train, eid_test
            state_test_adl = state_labels_test

    combined: Dict[str, Any] = {}

    # Determine which feature sets to use for each model
    # ADL features include lags and are used for all except heuristic.
    # Even when n_lags=0, the "adl" variables just alias the original data.

    model_list = config.models

    for model_name in model_list:
        if model_name == "heuristic":
            # Heuristic uses only contemporaneous features (no lags)
            hw = config.heuristic_weights or _DEFAULT_HEURISTIC_WEIGHTS
            if config.feature_names:
                weights = np.array([
                    hw.get(fn, 0.0) for fn in config.feature_names
                ])
            else:
                weights = np.array(list(hw.values()))

            heur_metrics = _evaluate_heuristic(
                X_test=X_test,
                y_test=y_test,
                weights=weights,
                state_labels_test=state_labels_test,
                prefix=f"{pfx}heuristic_",
            )
            combined.update(heur_metrics)
            continue

        # --- Pooled variant ---
        pooled_metrics = _evaluate_single_model(
            X_train=X_train_adl,
            y_train=y_train_adl,
            X_test=X_test_adl,
            y_test=y_test_adl,
            model_name=model_name,
            config=config,
            state_labels_test=state_test_adl,
            prefix=f"{pfx}{model_name}_pooled_",
        )
        combined.update(pooled_metrics)

        # --- Fixed-effects variant ---
        if config.run_fixed_effects and has_panel_info:
            fe_metrics = _evaluate_fixed_effects_model(
                X_train=X_train_adl,
                y_train=y_train_adl,
                X_test=X_test_adl,
                y_test=y_test_adl,
                entity_ids_train=eid_train_adl,
                entity_ids_test=eid_test_adl,
                model_name=model_name,
                config=config,
                state_labels_test=state_test_adl,
                prefix=f"{pfx}{model_name}_fe_",
            )
            combined.update(fe_metrics)

    return combined


# =====================================================================
# Convenience: full-set reference evaluation
# =====================================================================

def qos_fullset_reference(
    *,
    X_full: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    entity_ids: Optional[np.ndarray] = None,
    time_ids: Optional[np.ndarray] = None,
    state_labels: Optional[np.ndarray] = None,
    config: Optional[QoSConfig] = None,
) -> Dict[str, Any]:
    """Compute QoS reference metrics using the full training set.

    Same as :func:`qos_coreset_evaluation` but uses ``train_idx`` as the
    training set instead of a coreset.  This provides the baseline
    "best achievable" performance that coresets are compared against.
    """
    return qos_coreset_evaluation(
        X_full=X_full,
        y_full=y_full,
        S_idx=train_idx,
        eval_test_idx=test_idx,
        entity_ids=entity_ids,
        time_ids=time_ids,
        state_labels=state_labels,
        config=config,
    )


# =====================================================================
# Summary table builder
# =====================================================================

def qos_summary_table(
    results: Dict[str, Dict[str, Any]],
    metric_prefix: str = "qos_",
) -> Dict[str, Dict[str, float]]:
    """Build a summary table of QoS metrics across multiple methods.

    Parameters
    ----------
    results : dict mapping method_name -> metric_dict
        Each metric_dict is as returned by :func:`qos_coreset_evaluation`.
    metric_prefix : str
        The prefix used in metric keys.

    Returns
    -------
    Dict mapping method_name -> summary dict with keys:
        "best_pooled_rmse", "best_fe_rmse", "best_overall_rmse",
        "best_pooled_r2", "best_fe_r2", "best_overall_r2",
        "n_models_evaluated"
    """
    summary: Dict[str, Dict[str, float]] = {}

    for method_name, metrics in results.items():
        pooled_rmses = {}
        fe_rmses = {}
        pooled_r2s = {}
        fe_r2s = {}

        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            if not key.startswith(metric_prefix):
                continue

            suffix_key = key[len(metric_prefix):]

            if suffix_key.endswith("_pooled_rmse"):
                model = suffix_key.replace("_pooled_rmse", "")
                pooled_rmses[model] = val
            elif suffix_key.endswith("_fe_rmse"):
                model = suffix_key.replace("_fe_rmse", "")
                fe_rmses[model] = val
            elif suffix_key.endswith("_pooled_r2"):
                model = suffix_key.replace("_pooled_r2", "")
                pooled_r2s[model] = val
            elif suffix_key.endswith("_fe_r2"):
                model = suffix_key.replace("_fe_r2", "")
                fe_r2s[model] = val

        all_rmses = {**pooled_rmses, **fe_rmses}
        all_r2s = {**pooled_r2s, **fe_r2s}

        valid_rmses = {k: v for k, v in all_rmses.items() if np.isfinite(v)}
        valid_r2s = {k: v for k, v in all_r2s.items() if np.isfinite(v)}

        row: Dict[str, float] = {
            "n_models_evaluated": float(len(valid_rmses)),
        }

        if pooled_rmses:
            vp = {k: v for k, v in pooled_rmses.items() if np.isfinite(v)}
            if vp:
                row["best_pooled_rmse"] = min(vp.values())
        if fe_rmses:
            vf = {k: v for k, v in fe_rmses.items() if np.isfinite(v)}
            if vf:
                row["best_fe_rmse"] = min(vf.values())
        if valid_rmses:
            row["best_overall_rmse"] = min(valid_rmses.values())
        if pooled_r2s:
            vp = {k: v for k, v in pooled_r2s.items() if np.isfinite(v)}
            if vp:
                row["best_pooled_r2"] = max(vp.values())
        if fe_r2s:
            vf = {k: v for k, v in fe_r2s.items() if np.isfinite(v)}
            if vf:
                row["best_fe_r2"] = max(vf.values())
        if valid_r2s:
            row["best_overall_r2"] = max(valid_r2s.values())

        summary[method_name] = row

    return summary
