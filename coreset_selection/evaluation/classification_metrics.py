r"""Classification evaluation metrics for categorical targets (Phase 2).

When the downstream task target is categorical (or ordinal used as a
classification target), regression metrics (RMSE, R², MAE) are
inappropriate.  This module provides the classification-compatible
counterparts:

    - **Accuracy**: fraction of correctly predicted labels.
    - **Cohen's Kappa** (κ): chance-corrected agreement between
      predicted and true labels.  κ=1 is perfect, κ=0 is no better
      than chance, κ<0 is worse than chance.
    - **Precision, Recall, F1** (macro-averaged): computed per class
      and averaged with equal weight.
    - **Per-state classification metrics**: accuracy, κ, and macro-F1
      broken down by geographic group.
    - **Confusion-matrix summary**: provides counts for downstream
      analysis without matplotlib dependency.

All functions accept raw numpy arrays and are agnostic to the upstream
selection method, mirroring the API of ``downstream_metrics.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Target type detection
# =====================================================================

def infer_target_type(
    y: np.ndarray,
    *,
    cardinality_threshold: int = 50,
    float_fraction_threshold: float = 0.05,
) -> str:
    """Heuristically determine whether a target vector is regression or classification.

    Logic:
      1. If dtype is object/string → classification.
      2. If dtype is bool → classification.
      3. If all values are integers (even if stored as float) AND
         number of unique values ≤ ``cardinality_threshold``
         → classification.
      4. Otherwise → regression.

    Parameters
    ----------
    y : np.ndarray
        Target vector (1-D).
    cardinality_threshold : int
        Maximum number of unique values to still consider classification.
    float_fraction_threshold : float
        If the fraction of non-integer float values is below this threshold
        AND cardinality is low, treat as classification (handles noisy
        integer targets stored as floats).

    Returns
    -------
    str
        ``"classification"`` or ``"regression"``.
    """
    y = np.asarray(y).ravel()
    y_valid = y[np.isfinite(y)] if np.issubdtype(y.dtype, np.floating) else y[~(y == None)]  # noqa: E711

    if y_valid.size == 0:
        return "regression"

    # Object / string → classification
    if y.dtype.kind in ("U", "S", "O"):
        return "classification"

    # Boolean → classification
    if y.dtype.kind == "b":
        return "classification"

    n_unique = len(np.unique(y_valid))

    # Integer dtype with low cardinality → classification
    if np.issubdtype(y.dtype, np.integer):
        if n_unique <= cardinality_threshold:
            return "classification"
        return "regression"

    # Float dtype: check if values are effectively integers
    if np.issubdtype(y.dtype, np.floating):
        y_finite = y_valid[np.isfinite(y_valid)]
        if y_finite.size == 0:
            return "regression"
        is_int = np.allclose(y_finite, np.round(y_finite), atol=1e-9)
        if is_int and n_unique <= cardinality_threshold:
            return "classification"
        # Also handle case where almost all values are integer
        frac_non_int = np.mean(np.abs(y_finite - np.round(y_finite)) > 1e-9)
        if frac_non_int <= float_fraction_threshold and n_unique <= cardinality_threshold:
            return "classification"

    return "regression"


# =====================================================================
# Core classification metrics
# =====================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Compute Cohen's Kappa (κ) — chance-corrected agreement.

    .. math::
        \kappa = \frac{p_o - p_e}{1 - p_e}

    where :math:`p_o` is observed agreement (accuracy) and :math:`p_e`
    is expected agreement under independence.

    Returns 0.0 when :math:`p_e = 1` (degenerate case).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.size
    if n == 0:
        return 0.0

    classes = np.union1d(y_true, y_pred)
    p_o = float(np.mean(y_true == y_pred))

    # Expected agreement under independence
    p_e = 0.0
    for c in classes:
        p_true_c = float(np.mean(y_true == c))
        p_pred_c = float(np.mean(y_pred == c))
        p_e += p_true_c * p_pred_c

    if abs(1.0 - p_e) < 1e-15:
        return 0.0 if abs(p_o - 1.0) < 1e-15 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def _per_class_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[Dict[Any, float], Dict[Any, float], Dict[Any, float]]:
    """Compute per-class precision, recall, F1.

    Returns three dicts mapping class label → metric value.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.union1d(y_true, y_pred)

    prec_dict: Dict[Any, float] = {}
    rec_dict: Dict[Any, float] = {}
    f1_dict: Dict[Any, float] = {}

    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        prec_dict[c] = prec
        rec_dict[c] = rec
        f1_dict[c] = f1

    return prec_dict, rec_dict, f1_dict


def macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged precision (unweighted mean over classes)."""
    prec, _, _ = _per_class_precision_recall_f1(y_true, y_pred)
    return float(np.mean(list(prec.values()))) if prec else 0.0


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall (unweighted mean over classes)."""
    _, rec, _ = _per_class_precision_recall_f1(y_true, y_pred)
    return float(np.mean(list(rec.values()))) if rec else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score (unweighted mean over classes)."""
    _, _, f1 = _per_class_precision_recall_f1(y_true, y_pred)
    return float(np.mean(list(f1.values()))) if f1 else 0.0


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 score (class-frequency weighted mean)."""
    y_true = np.asarray(y_true).ravel()
    _, _, f1 = _per_class_precision_recall_f1(y_true, y_pred)
    if not f1:
        return 0.0
    classes = list(f1.keys())
    weights = np.array([np.sum(y_true == c) for c in classes], dtype=float)
    total = weights.sum()
    if total == 0:
        return 0.0
    return float(np.sum([f1[c] * w for c, w in zip(classes, weights)]) / total)


def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute a confusion matrix as a dictionary structure.

    Returns
    -------
    Dict with keys:
      - ``classes``: sorted list of unique class labels.
      - ``matrix``: 2-D list of counts ``matrix[i][j]`` = count of
        (true=classes[i], pred=classes[j]).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = sorted(np.union1d(y_true, y_pred).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        mat[cls_to_idx[yt]][cls_to_idx[yp]] += 1
    return {"classes": classes, "matrix": mat}


# =====================================================================
# Full classification evaluation (mirrors downstream_metrics API)
# =====================================================================

def full_classification_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_suffix: str = "",
) -> Dict[str, Any]:
    r"""One-call classification evaluation producing all metrics.

    Mirrors the API of ``downstream_metrics.full_downstream_evaluation``
    but uses classification-appropriate metrics.

    Returns
    -------
    Dict with keys like:
      accuracy_{suf}, cohens_kappa_{suf}, macro_precision_{suf},
      macro_recall_{suf}, macro_f1_{suf}, weighted_f1_{suf},
      macro_accuracy_{suf} (if state_labels given), worst_group_accuracy_{suf}, ...
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Round predictions to nearest integer for classification
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.round(y_pred).astype(y_true.dtype)

    sfx = target_suffix
    out: Dict[str, Any] = {}

    # Global metrics
    out[f"accuracy{sfx}"] = accuracy(y_true, y_pred)
    out[f"cohens_kappa{sfx}"] = cohens_kappa(y_true, y_pred)
    out[f"macro_precision{sfx}"] = macro_precision(y_true, y_pred)
    out[f"macro_recall{sfx}"] = macro_recall(y_true, y_pred)
    out[f"macro_f1{sfx}"] = macro_f1(y_true, y_pred)
    out[f"weighted_f1{sfx}"] = weighted_f1(y_true, y_pred)
    out[f"n_classes{sfx}"] = len(np.unique(y_true))

    # Per-state classification metrics
    if state_labels is not None:
        state_labels = np.asarray(state_labels)
        per_state: Dict[str, Dict[str, float]] = {}
        for g in np.unique(state_labels):
            mask = state_labels == g
            n_g = int(mask.sum())
            if n_g < 2:
                continue
            yt_g, yp_g = y_true[mask], y_pred[mask]
            per_state[str(g)] = {
                "accuracy": accuracy(yt_g, yp_g),
                "cohens_kappa": cohens_kappa(yt_g, yp_g),
                "macro_f1": macro_f1(yt_g, yp_g),
                "n": n_g,
            }

        if per_state:
            accs = [v["accuracy"] for v in per_state.values()]
            kappas = [v["cohens_kappa"] for v in per_state.values()]
            f1s = [v["macro_f1"] for v in per_state.values()]

            out[f"macro_accuracy{sfx}"] = float(np.mean(accs))
            out[f"worst_group_accuracy{sfx}"] = float(np.min(accs))
            out[f"best_group_accuracy{sfx}"] = float(np.max(accs))
            out[f"accuracy_dispersion{sfx}"] = float(np.std(accs))
            out[f"macro_cohens_kappa{sfx}"] = float(np.mean(kappas))
            out[f"worst_group_kappa{sfx}"] = float(np.min(kappas))
            out[f"macro_macro_f1{sfx}"] = float(np.mean(f1s))
            out[f"worst_group_f1{sfx}"] = float(np.min(f1s))
            out[f"n_groups_evaluated{sfx}"] = len(per_state)

            # Stash per-state detail for CSV export
            out[f"_per_state_detail{sfx}"] = per_state

    return out


# =====================================================================
# Multi-target classification wrapper
# =====================================================================

def multitarget_classification_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    r"""Evaluate all classification targets and return combined dict.

    Mirrors ``downstream_metrics.multitarget_downstream_evaluation``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    T = y_true.shape[1]
    if target_names is None:
        target_names = [f"_{i}" for i in range(T)] if T > 1 else [""]

    combined: Dict[str, Any] = {}
    macro_accs: List[float] = []
    worst_accs: List[float] = []

    for t in range(T):
        sfx = target_names[t]
        res = full_classification_evaluation(
            y_true[:, t], y_pred[:, t],
            state_labels=state_labels,
            target_suffix=sfx,
        )
        combined.update(res)
        if f"macro_accuracy{sfx}" in res:
            macro_accs.append(res[f"macro_accuracy{sfx}"])
        if f"worst_group_accuracy{sfx}" in res:
            worst_accs.append(res[f"worst_group_accuracy{sfx}"])

    if macro_accs:
        combined["mean_macro_accuracy"] = float(np.mean(macro_accs))
    if worst_accs:
        combined["mean_worst_group_accuracy"] = float(np.mean(worst_accs))

    return combined
