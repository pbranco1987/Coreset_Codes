r"""Target column detection and leakage prevention (Phase 4 — Milestone 4.3).

Per the manuscript (Section VII), target-defining columns **must** be excluded
from the feature matrix before PCA/VAE training.  This module provides:

1. ``TARGET_COLUMN_PATTERNS`` — regex-based rules for identifying columns that
   encode or derive from the prediction targets (coverage area 4G/5G).
2. ``detect_target_columns(feature_names)`` — returns a list of feature names
   that match the target-defining rules.
3. ``remove_target_columns(X, feature_names)`` — returns (X_clean, kept_names,
   removed_names) with target columns stripped.
4. ``validate_no_leakage(feature_names)`` — raises ``ValueError`` if any target
   column is still present (for use as a safety guard before representation
   learning).

These functions are consumed by ``data/cache.py::build_replicate_cache`` and
by the unit-test ``tests/test_preprocessing.py::TestTargetLeakagePrevention``.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Target column patterns
# ---------------------------------------------------------------------------
# Each pattern is a compiled regex that is matched against lower-cased feature
# names.  A match means the column *derives from or encodes* the prediction
# target and must be excluded from the covariate matrix.

TARGET_COLUMN_PATTERNS: List[re.Pattern] = [
    # Primary targets: 4G/5G area-coverage columns
    re.compile(r"^cov_area_4g$", re.IGNORECASE),
    re.compile(r"^cov_area_5g$", re.IGNORECASE),
    # Additional coverage indicators used as multi-target KRR targets
    re.compile(r"^cov_hh_(4g|5g|4g_5g|all)$", re.IGNORECASE),
    re.compile(r"^cov_res_(4g|5g|4g_5g|all)$", re.IGNORECASE),
    re.compile(r"^cov_area_(4g_5g|all)$", re.IGNORECASE),
    # Common raw-data variants that encode the same quantity
    re.compile(r"^y_4g$", re.IGNORECASE),
    re.compile(r"^y_5g$", re.IGNORECASE),
    re.compile(r"^target$", re.IGNORECASE),
    re.compile(r"^tx_cobertura", re.IGNORECASE),
    re.compile(r"^indicador_principal", re.IGNORECASE),
    # Generic catch-all for columns explicitly labeled as targets
    re.compile(r"^target_", re.IGNORECASE),
    # Coverage-related columns that leak target information
    re.compile(r"cobertura_area_(4g|5g)", re.IGNORECASE),
    # Extra regression targets (must not appear in features)
    re.compile(r"^velocidade_mediana_(mean|std|median)$", re.IGNORECASE),
    re.compile(r"^pct_limite_mean$", re.IGNORECASE),
    re.compile(r"^renda_media_(mean|std|median)$", re.IGNORECASE),
    re.compile(r"^hhi\s*(smp|scm)_\d{4}$", re.IGNORECASE),
    re.compile(r"^pct_fibra_backhaul$", re.IGNORECASE),
    re.compile(r"^pct_escolas_(internet|fibra)$", re.IGNORECASE),
    re.compile(r"^densidade_(banda\s*larga\s*fixa|telefonia\s*m[oó]vel)_\d{4}$", re.IGNORECASE),
    # Classification target source columns
    re.compile(r"^pct_agl_alta_velocidade$", re.IGNORECASE),
    re.compile(r"^pct_urbano$", re.IGNORECASE),
    re.compile(r"^pct_cat_(low|high)_renda_(low|high)_vel$", re.IGNORECASE),
    re.compile(r"^n_estacoes_smp$", re.IGNORECASE),
    re.compile(r"^rod_pct_cob_todas_4g$", re.IGNORECASE),
    # QoS / Satisfaction survey targets (ISG sub-components from Stage R).
    # qf_mean is used as the QoS evaluation target; ISG and QIC are
    # correlated sub-components (ISG = weighted composite of QF + QIC + QCR)
    # that would create indirect target leakage if left in X.
    re.compile(r"^(isg|qf|qic)(_\w+)?_mean$", re.IGNORECASE),
    re.compile(r"^n_respostas", re.IGNORECASE),
]


def detect_target_columns(feature_names: Sequence[str]) -> List[str]:
    """Return feature names that match target-defining patterns.

    Parameters
    ----------
    feature_names : Sequence[str]
        List of feature column names.

    Returns
    -------
    List[str]
        Subset of *feature_names* that are detected as target columns.
    """
    matched: List[str] = []
    for name in feature_names:
        for pat in TARGET_COLUMN_PATTERNS:
            if pat.search(name):
                matched.append(name)
                break
    return matched


def remove_target_columns(
    X: np.ndarray,
    feature_names: List[str],
    *,
    explicit_targets: Sequence[str] = (),
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Remove target-defining columns from the feature matrix.

    Columns are removed if they match *either*:

    1. The regex-based ``TARGET_COLUMN_PATTERNS`` (manuscript mode), **or**
    2. The ``explicit_targets`` list (Phase 4 — user-declared target columns).

    This allows you to point the pipeline at any target column(s) in the CSV
    without editing regex patterns.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, D)``.
    feature_names : List[str]
        Column names corresponding to the columns of *X*.
    explicit_targets : Sequence[str]
        Additional column names to treat as targets and remove from *X*.
        These are unioned with the regex-detected targets.

    Returns
    -------
    X_clean : np.ndarray
        Feature matrix with target columns removed, shape ``(N, D')``.
    kept_names : List[str]
        Feature names that were retained.
    removed_names : List[str]
        Feature names that were removed.

    Raises
    ------
    ValueError
        If ``len(feature_names) != X.shape[1]``.
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X.shape[1] ({X.shape[1]})"
        )

    # Union of regex-detected + explicitly declared targets
    target_names = set(detect_target_columns(feature_names))
    if explicit_targets:
        target_names |= set(explicit_targets)

    keep_mask = [name not in target_names for name in feature_names]
    kept_names = [name for name, keep in zip(feature_names, keep_mask) if keep]
    removed_names = [name for name in feature_names if name in target_names]

    if not any(keep_mask):
        raise ValueError("All columns detected as target columns — nothing left!")

    X_clean = X[:, keep_mask]
    return X_clean, kept_names, removed_names


def validate_no_leakage(feature_names: Sequence[str]) -> None:
    """Raise ``ValueError`` if any target column is detected.

    Intended as a pre-flight guard before PCA / VAE training.

    Parameters
    ----------
    feature_names : Sequence[str]
        Column names of the feature matrix that will be used for
        representation learning.

    Raises
    ------
    ValueError
        If one or more target columns are detected.
    """
    leaked = detect_target_columns(feature_names)
    if leaked:
        raise ValueError(
            f"Target leakage detected!  The following columns must be "
            f"removed before representation learning: {leaked}"
        )
