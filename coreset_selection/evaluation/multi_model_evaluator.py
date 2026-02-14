"""
Multi-model downstream evaluation using Nyström features.

Evaluates coreset quality by training multiple downstream models (KNN, RF, LR,
GBT) on the Nyström feature matrix Phi derived from the landmark set S.  All
models share the same Phi — no kernel re-computation needed.

Regression models are evaluated on coverage + extra regression targets.
Classification models are evaluated on derived classification targets.

Output key convention: ``{model}_{metric}_{target}``
    e.g. ``rf_rmse_cov_area_4G``, ``knn_accuracy_concentrated_mobile_market``
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
)


# ── Model registries ────────────────────────────────────────────────────

def _regression_models(seed: int) -> Dict[str, object]:
    """Return a dict of ``{model_name: sklearn_estimator}`` for regression."""
    return {
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=1),
        "rf": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=1,
        ),
        "gbt": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=seed,
        ),
    }


def _classification_models(seed: int) -> Dict[str, object]:
    """Return a dict of ``{model_name: sklearn_estimator}`` for classification."""
    return {
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=1),
        "rf": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1,
        ),
        "lr": LogisticRegression(
            max_iter=1000, random_state=seed, n_jobs=1,
        ),
        "gbt": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=seed,
        ),
    }


# ── Single-target evaluation ────────────────────────────────────────────

def _evaluate_regression_target(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    seed: int,
) -> Dict[str, float]:
    """Train all regression models on one target and return metrics."""
    results: Dict[str, float] = {}
    models = _regression_models(seed)

    for model_name, estimator in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(Phi_train, y_train)
                y_pred = estimator.predict(Phi_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            results[f"{model_name}_rmse_{target_name}"] = rmse
            results[f"{model_name}_mae_{target_name}"] = mae
            results[f"{model_name}_r2_{target_name}"] = r2
        except Exception as e:
            # Log but don't crash — a single model failure shouldn't
            # abort the entire evaluation
            print(f"[multi_model] WARNING: {model_name} failed on "
                  f"regression target '{target_name}': {e}")

    return results


def _evaluate_classification_target(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    seed: int,
) -> Dict[str, float]:
    """Train all classification models on one target and return metrics."""
    results: Dict[str, float] = {}
    models = _classification_models(seed)

    # Check for degenerate targets (single class in train or test)
    n_classes_train = len(np.unique(y_train))
    n_classes_test = len(np.unique(y_test))
    if n_classes_train < 2:
        print(f"[multi_model] WARNING: classification target '{target_name}' "
              f"has only {n_classes_train} class(es) in train — skipping")
        return results

    for model_name, estimator in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(Phi_train, y_train)
                y_pred = estimator.predict(Phi_test)

            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            # macro_f1 handles multiclass via macro-averaging
            macro_f1 = float(f1_score(
                y_test, y_pred, average="macro", zero_division=0,
            ))

            results[f"{model_name}_accuracy_{target_name}"] = acc
            results[f"{model_name}_bal_accuracy_{target_name}"] = bal_acc
            results[f"{model_name}_macro_f1_{target_name}"] = macro_f1
        except Exception as e:
            print(f"[multi_model] WARNING: {model_name} failed on "
                  f"classification target '{target_name}': {e}")

    return results


# ── Main entry point ────────────────────────────────────────────────────

def evaluate_all_downstream_models(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    regression_targets: Dict[str, np.ndarray],
    classification_targets: Dict[str, np.ndarray],
    seed: int = 123,
) -> Dict[str, float]:
    """Evaluate all downstream models on all targets.

    Parameters
    ----------
    Phi_train : (n_train, m) ndarray
        Nyström features for evaluation train set.
    Phi_test : (n_test, m) ndarray
        Nyström features for evaluation test set.
    eval_train_idx : (n_train,) ndarray
        Global indices of E_train (used to slice target arrays).
    eval_test_idx : (n_test,) ndarray
        Global indices of E_test (used to slice target arrays).
    regression_targets : Dict[str, np.ndarray]
        Mapping ``{target_name: (N,) float64 array}`` for regression targets.
    classification_targets : Dict[str, np.ndarray]
        Mapping ``{target_name: (N,) int64 array}`` for classification targets.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Flat dict of metrics keyed as ``{model}_{metric}_{target}``.
    """
    results: Dict[str, float] = {}

    # Regression targets
    for tname, y_full in regression_targets.items():
        y_full = np.asarray(y_full, dtype=np.float64)
        # Check for valid target values
        y_tr = y_full[eval_train_idx]
        y_te = y_full[eval_test_idx]
        if not np.all(np.isfinite(y_tr)) or not np.all(np.isfinite(y_te)):
            # Skip targets with non-finite values in train/test
            n_bad = int(np.sum(~np.isfinite(y_tr))) + int(np.sum(~np.isfinite(y_te)))
            print(f"[multi_model] WARNING: regression target '{tname}' "
                  f"has {n_bad} non-finite values — skipping")
            continue
        r = _evaluate_regression_target(Phi_train, Phi_test, y_tr, y_te, tname, seed)
        results.update(r)

    # Classification targets
    for tname, y_full in classification_targets.items():
        y_full = np.asarray(y_full, dtype=np.int64)
        y_tr = y_full[eval_train_idx]
        y_te = y_full[eval_test_idx]
        r = _evaluate_classification_target(Phi_train, Phi_test, y_tr, y_te, tname, seed)
        results.update(r)

    return results
