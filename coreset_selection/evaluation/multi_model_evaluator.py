"""
Multi-model downstream evaluation using Nyström features.

Evaluates coreset quality by training multiple downstream models (KNN, RF, LR,
GBT) on the Nyström feature matrix Phi derived from the landmark set S.  All
models share the same Phi — no kernel re-computation needed.

Regression models are evaluated on coverage + extra regression targets.
Classification models are evaluated on derived classification targets.

Output key convention: ``{model}_{metric}_{target}``
    e.g. ``rf_rmse_cov_area_4G``, ``knn_accuracy_concentrated_mobile_market``

Performance notes:
- GBT uses 50 estimators (inherently sequential — cannot parallelize across trees).
- By default, targets are evaluated in parallel via joblib (4 workers).
  Set CORESET_EVAL_NJOBS=1 for sequential mode with full per-model verbose output.
- In parallel mode: a compact progress bar per batch is printed.
- In sequential mode: each model x target prints its own line with timing.
"""

from __future__ import annotations

import os
import time as _time
import warnings
from typing import Dict, List, Tuple

import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score,
)


# Number of parallel workers for target-level parallelism.
# Set CORESET_EVAL_NJOBS=1 for fully sequential + verbose per-model output.
_N_WORKERS = int(os.environ.get("CORESET_EVAL_NJOBS", "4"))


# ── Model registries ────────────────────────────────────────────────────

def _regression_models(seed: int) -> Dict[str, object]:
    """Return a dict of ``{model_name: sklearn_estimator}`` for regression."""
    return {
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=1),
        "rf": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=seed, n_jobs=1,
        ),
        "gbt": GradientBoostingRegressor(
            n_estimators=50, max_depth=5, random_state=seed,
        ),
        "ridge": Ridge(alpha=1.0),
        "svr": SVR(kernel="rbf", C=1.0),
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
            n_estimators=50, max_depth=5, random_state=seed,
        ),
        "svc": SVC(kernel="rbf", C=1.0, random_state=seed),
    }


# ── Single-target evaluation ───────────────────────────────────────────

def _evaluate_regression_target(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    seed: int,
    *,
    task_idx: int = 0,
    task_total: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train all regression models on one target and return metrics."""
    results: Dict[str, float] = {}
    models = _regression_models(seed)
    prefix = f"              reg {task_idx}/{task_total}" if task_total else "              reg"

    for model_name, estimator in models.items():
        if verbose:
            print(f"{prefix} {target_name} x {model_name}...", end=" ", flush=True)
        _t0 = _time.perf_counter()
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
            if verbose:
                _dt = _time.perf_counter() - _t0
                print(f"({_dt:.1f}s)", flush=True)
        except Exception as e:
            if verbose:
                _dt = _time.perf_counter() - _t0
                print(f"FAILED ({_dt:.1f}s): {e}", flush=True)

    return results


def _evaluate_classification_target(
    Phi_train: np.ndarray,
    Phi_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    seed: int,
    *,
    task_idx: int = 0,
    task_total: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train all classification models on one target and return metrics."""
    results: Dict[str, float] = {}
    models = _classification_models(seed)
    prefix = f"              cls {task_idx}/{task_total}" if task_total else "              cls"

    # Check for degenerate targets (single class in train or test)
    n_classes_train = len(np.unique(y_train))
    if n_classes_train < 2:
        if verbose:
            print(f"{prefix} {target_name} -- skipped (only {n_classes_train} class)", flush=True)
        return results

    for model_name, estimator in models.items():
        if verbose:
            print(f"{prefix} {target_name} x {model_name}...", end=" ", flush=True)
        _t0 = _time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(Phi_train, y_train)
                y_pred = estimator.predict(Phi_test)

            acc = float(accuracy_score(y_test, y_pred))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred))
            macro_f1 = float(f1_score(
                y_test, y_pred, average="macro", zero_division=0,
            ))

            results[f"{model_name}_accuracy_{target_name}"] = acc
            results[f"{model_name}_bal_accuracy_{target_name}"] = bal_acc
            results[f"{model_name}_macro_f1_{target_name}"] = macro_f1
            if verbose:
                _dt = _time.perf_counter() - _t0
                print(f"({_dt:.1f}s)", flush=True)
        except Exception as e:
            if verbose:
                _dt = _time.perf_counter() - _t0
                print(f"FAILED ({_dt:.1f}s): {e}", flush=True)

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
    # ── Prepare tasks ──
    reg_tasks: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for tname, y_full in regression_targets.items():
        y_full = np.asarray(y_full, dtype=np.float64)
        y_tr = y_full[eval_train_idx]
        y_te = y_full[eval_test_idx]
        if not np.all(np.isfinite(y_tr)) or not np.all(np.isfinite(y_te)):
            n_bad = int(np.sum(~np.isfinite(y_tr))) + int(np.sum(~np.isfinite(y_te)))
            print(f"[multi_model] WARNING: regression target '{tname}' "
                  f"has {n_bad} non-finite values -- skipping")
            continue
        reg_tasks.append((tname, y_tr, y_te))

    cls_tasks: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for tname, y_full in classification_targets.items():
        y_full = np.asarray(y_full, dtype=np.int64)
        y_tr = y_full[eval_train_idx]
        y_te = y_full[eval_test_idx]
        cls_tasks.append((tname, y_tr, y_te))

    n_reg = len(reg_tasks)
    n_cls = len(cls_tasks)
    total_tasks = n_reg + n_cls
    n_workers = min(_N_WORKERS, total_tasks) if total_tasks > 0 else 1

    # ── Parallel path (joblib multiprocessing — real CPU parallelism) ──
    if n_workers > 1 and total_tasks >= 3:
        from joblib import Parallel, delayed

        n_reg_models = len(_regression_models(seed))
        n_cls_models = len(_classification_models(seed))
        n_fits = n_reg_models * n_reg + n_cls_models * n_cls
        print(
            f"              parallel: {n_reg} reg + {n_cls} cls targets, "
            f"{n_fits} model fits, {n_workers} workers...",
            end=" ", flush=True,
        )
        _t0 = _time.perf_counter()

        def _run_reg(tname, y_tr, y_te):
            return _evaluate_regression_target(
                Phi_train, Phi_test, y_tr, y_te, tname, seed,
                verbose=False,
            )

        def _run_cls(tname, y_tr, y_te):
            return _evaluate_classification_target(
                Phi_train, Phi_test, y_tr, y_te, tname, seed,
                verbose=False,
            )

        jobs = (
            [delayed(_run_reg)(t, ytr, yte) for t, ytr, yte in reg_tasks]
            + [delayed(_run_cls)(t, ytr, yte) for t, ytr, yte in cls_tasks]
        )

        partial_results = Parallel(
            n_jobs=n_workers,
            backend="loky",
            prefer="processes",
        )(jobs)

        results: Dict[str, float] = {}
        for r in partial_results:
            results.update(r)

        _dt = _time.perf_counter() - _t0
        print(f"done ({_dt:.1f}s)", flush=True)
        return results

    # ── Sequential path (CORESET_EVAL_NJOBS=1 — full per-model verbose) ──
    results: Dict[str, float] = {}
    for i, (tname, y_tr, y_te) in enumerate(reg_tasks, 1):
        r = _evaluate_regression_target(
            Phi_train, Phi_test, y_tr, y_te, tname, seed,
            task_idx=i, task_total=n_reg, verbose=True,
        )
        results.update(r)

    for i, (tname, y_tr, y_te) in enumerate(cls_tasks, 1):
        r = _evaluate_classification_target(
            Phi_train, Phi_test, y_tr, y_te, tname, seed,
            task_idx=i, task_total=n_cls, verbose=True,
        )
        results.update(r)

    return results
