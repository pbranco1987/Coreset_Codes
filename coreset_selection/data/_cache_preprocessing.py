"""Preprocessing pipeline functions for the replicate cache."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def _stratified_split(
    indices: np.ndarray,
    groups: np.ndarray,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split helper with fallback for small groups.

    If any group has fewer than 2 samples, falls back to non-stratified
    splitting to avoid sklearn errors.
    """
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups)

    if test_frac <= 0.0:
        return indices.copy(), np.array([], dtype=int)
    if test_frac >= 1.0:
        return np.array([], dtype=int), indices.copy()

    # Check if stratification is possible
    unique_groups, counts = np.unique(groups, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        # Some groups have <2 samples - can't do full stratification
        # Strategy: stratify only groups with >=2 samples, handle small groups separately
        small_group_mask = np.isin(groups, unique_groups[counts < 2])
        large_group_mask = ~small_group_mask

        if large_group_mask.sum() == 0:
            # All groups are small - fall back to non-stratified
            print(f"[WARNING] All groups have <2 samples. Using non-stratified split.")
            tr, te = train_test_split(
                indices,
                test_size=test_frac,
                random_state=seed,
                stratify=None,
            )
            return np.sort(tr), np.sort(te)

        # Split large groups with stratification
        large_indices = indices[large_group_mask]
        large_groups = groups[large_group_mask]

        tr_large, te_large = train_test_split(
            large_indices,
            test_size=test_frac,
            random_state=seed,
            stratify=large_groups,
        )

        # Small groups: randomly assign to train or test (proportionally)
        small_indices = indices[small_group_mask]
        rng = np.random.default_rng(seed)

        # Assign each small-group sample to test with probability test_frac
        small_to_test = rng.random(len(small_indices)) < test_frac
        tr_small = small_indices[~small_to_test]
        te_small = small_indices[small_to_test]

        small_groups_list = unique_groups[counts < 2].tolist()
        print(f"[WARNING] Groups with <2 samples (handled separately): {small_groups_list}")

        tr = np.concatenate([tr_large, tr_small])
        te = np.concatenate([te_large, te_small])

        return np.sort(tr), np.sort(te)

    # Normal stratified split
    tr, te = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed,
        stratify=groups,
    )
    return np.sort(tr), np.sort(te)


def _impute_by_train_median(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Impute NaN/inf using per-column medians computed on the train split."""
    X = np.asarray(X, dtype=np.float64).copy()

    # Treat inf as missing for robustness
    X[~np.isfinite(X)] = np.nan

    # Column medians on train split
    med = np.nanmedian(X[train_idx], axis=0)

    # If a column is entirely NaN in train split, nanmedian returns NaN; replace by 0.
    med = np.where(np.isfinite(med), med, 0.0)

    nan_rows, nan_cols = np.where(np.isnan(X))
    if nan_rows.size > 0:
        X[nan_rows, nan_cols] = med[nan_cols]

    # Final safety
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _impute_by_train_mode(X: np.ndarray, train_idx: np.ndarray, col_idx: int) -> float:
    """Compute the mode of column ``col_idx`` on the training split.

    Returns the most frequent non-NaN value.  Ties are broken by choosing
    the smallest value (deterministic).

    Phase 2: Used for categorical imputation instead of median.
    """
    vals = X[train_idx, col_idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0  # missing code
    # Find mode: most frequent value; break ties with min
    unique, counts = np.unique(vals, return_counts=True)
    max_count = counts.max()
    modes = unique[counts == max_count]
    return float(modes[0])


def _impute_typeaware(
    X: np.ndarray,
    train_idx: np.ndarray,
    feature_types: list,
    *,
    categorical_strategy: str = "mode",
    ordinal_strategy: str = "median",
    return_stats: bool = False,
) -> "np.ndarray | Tuple[np.ndarray, Dict[int, float]]":
    """Phase 2/3: Type-aware imputation.

    - **numeric**: median on I_train (existing behaviour).
    - **categorical**: mode on I_train (or ``-1`` missing code).
    - **ordinal**: median on I_train (rounded to nearest int) or mode.

    Parameters
    ----------
    X : (N, D) float64, may contain NaN
    train_idx : indices of training split
    feature_types : list of strings aligned with columns of X
        Each is ``"numeric"``, ``"ordinal"``, or ``"categorical"``.
    categorical_strategy : "mode" or "missing_code"
    ordinal_strategy : "median" or "mode"
    return_stats : bool
        If True, also return a dict mapping column index -> fill value used,
        enabling downstream reproducibility (Phase 3).

    Returns
    -------
    X_imp : (N, D) float64 with NaNs filled.
    impute_stats : Dict[int, float]  (only if ``return_stats=True``)
        Mapping ``{column_index: imputation_value}`` for every column that
        had at least one NaN.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    X[~np.isfinite(X)] = np.nan

    d = X.shape[1]
    # Ensure we have a type for every column
    if len(feature_types) < d:
        feature_types = list(feature_types) + ["numeric"] * (d - len(feature_types))

    # Pre-compute train medians for numeric/ordinal
    med = np.nanmedian(X[train_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    impute_stats: Dict[int, float] = {}

    for j in range(d):
        nan_mask = np.isnan(X[:, j])
        if not np.any(nan_mask):
            continue

        ft = feature_types[j] if j < len(feature_types) else "numeric"

        if ft == "categorical":
            if categorical_strategy == "mode":
                fill = _impute_by_train_mode(X, train_idx, j)
            else:
                fill = -1.0  # missing code
        elif ft == "ordinal":
            if ordinal_strategy == "mode":
                fill = _impute_by_train_mode(X, train_idx, j)
            else:
                fill = float(np.round(med[j]))  # rounded median
        else:
            fill = med[j]

        X[nan_mask, j] = fill
        impute_stats[j] = fill

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if return_stats:
        return X, impute_stats
    return X


def _preprocess_fit_transform(
    *,
    X_unscaled: np.ndarray,
    train_idx: np.ndarray,
    feature_names: list,
    feature_types: Optional[List[str]] = None,
    categorical_impute_strategy: str = "mode",
    ordinal_impute_strategy: str = "median",
    log1p_categoricals: bool = False,
    log1p_ordinals: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Fit-and-apply preprocessing per manuscript Section 5.7 (Phase 2 type-aware).

    Steps:
      1) Add binary missingness indicators for each feature column with any missing
         entries (NaN or +/-inf) in the full dataset.
         - Phase 2: skip missingness indicators for categorical columns (they use
           a dedicated missing code instead).
      2) Impute missing entries:
         - numeric: median on I_train (unchanged).
         - categorical: mode on I_train (Phase 2).
         - ordinal: rounded median on I_train (Phase 2).
      3) Apply log(1+x) to heavy-tailed non-negative variables (decision based on
         I_train statistics only).
         - Phase 2: skip log1p for categorical and ordinal columns by default.
      4) Return the transformed (but unstandardized) matrix and bookkeeping.

    Notes
    -----
    Standardization is applied *after* this function using I_train statistics.
    Phase 2 adds ``feature_types`` to control per-column behaviour.
    """
    X0 = np.asarray(X_unscaled, dtype=np.float64).copy()
    if X0.ndim != 2:
        raise ValueError("X_unscaled must be 2D")

    n, d = X0.shape
    if len(feature_names) != d:
        feature_names = [f"x{j}" for j in range(d)]

    # Default: all numeric
    if feature_types is None or len(feature_types) != d:
        feature_types = ["numeric"] * d

    # Treat inf as missing
    miss_mask = ~np.isfinite(X0)
    X0[miss_mask] = np.nan

    # Missingness indicators for columns with any missing values
    # Phase 2: only add indicators for numeric/ordinal columns; categoricals
    # use a missing code (e.g. -1) directly.
    # Skip columns that are already missingness indicators (suffix _missing),
    # and columns that already have a pre-engineered _missing partner in the
    # feature set, to avoid redundant double-indicators.
    missing_cols = np.flatnonzero(np.any(miss_mask, axis=0))
    feature_name_set = set(feature_names)
    indicator_cols = []
    for j in missing_cols:
        ft = feature_types[j] if j < len(feature_types) else "numeric"
        if ft == "categorical":
            continue
        name = feature_names[j] if j < len(feature_names) else ""
        if name.endswith("_missing"):
            continue
        if f"{name}_missing" in feature_name_set:
            continue
        indicator_cols.append(j)

    if indicator_cols:
        miss_ind = miss_mask[:, indicator_cols].astype(np.float32)
        missing_feature_names = [f"{feature_names[j]}__missing" for j in indicator_cols]
    else:
        miss_ind = None
        missing_feature_names = []

    # Phase 2/3: Type-aware imputation (with stats for reproducibility)
    X_imp, impute_stats = _impute_typeaware(
        X0, train_idx, feature_types,
        categorical_strategy=categorical_impute_strategy,
        ordinal_strategy=ordinal_impute_strategy,
        return_stats=True,
    )

    # Identify heavy-tailed non-negative columns on I_train only
    # Phase 2: restrict to numeric columns (skip categorical and ordinal)
    log1p_eligible = set()
    for j in range(d):
        ft = feature_types[j] if j < len(feature_types) else "numeric"
        if ft == "numeric":
            log1p_eligible.add(j)
        elif ft == "ordinal" and log1p_ordinals:
            log1p_eligible.add(j)
        elif ft == "categorical" and log1p_categoricals:
            log1p_eligible.add(j)

    all_log1p_cols = _detect_log1p_cols(X_imp, train_idx)
    log1p_cols = [j for j in all_log1p_cols if j in log1p_eligible]

    if log1p_cols:
        cols = np.array(log1p_cols, dtype=int)
        X_imp[:, cols] = np.log1p(np.maximum(X_imp[:, cols], 0.0))
        log1p_feature_names = [feature_names[j] for j in cols]
    else:
        log1p_feature_names = []

    # Append missing indicators at the end
    if miss_ind is not None:
        X_out = np.concatenate([X_imp, miss_ind], axis=1)
        feature_names_out = list(feature_names) + missing_feature_names
    else:
        X_out = X_imp
        feature_names_out = list(feature_names)

    # Phase 3: Build extended feature_types_out that includes missingness indicators
    feature_types_out = list(feature_types) + ["numeric"] * len(missing_feature_names)

    # Phase 3: Extract per-type column lists for metadata
    categorical_columns_out = [
        feature_names[j] for j in range(d) if feature_types[j] == "categorical"
    ]
    ordinal_columns_out = [
        feature_names[j] for j in range(d) if feature_types[j] == "ordinal"
    ]
    numeric_columns_out = [
        feature_names[j] for j in range(d) if feature_types[j] == "numeric"
    ]

    # Phase 3: Build imputation statistics dict (column_name -> fill_value)
    impute_values = {
        feature_names[j]: float(v) for j, v in impute_stats.items()
    }

    meta = {
        "feature_names": feature_names_out,
        "missing_feature_names": missing_feature_names,
        "log1p_feature_names": log1p_feature_names,
        # Phase 2: original feature types (aligned with input columns)
        "feature_types": list(feature_types),
        # Phase 3: extended feature types (aligned with output columns,
        # including appended missingness indicators marked as "numeric")
        "feature_types_out": feature_types_out,
        # Phase 3: per-type column lists for downstream consumers
        "categorical_columns": categorical_columns_out,
        "ordinal_columns": ordinal_columns_out,
        "numeric_columns": numeric_columns_out,
        # Phase 3: imputation statistics for reproducibility
        # Maps column_name -> fill_value used during imputation
        "impute_values": impute_values,
    }
    return X_out, meta


def _detect_log1p_cols(X: np.ndarray, train_idx: np.ndarray) -> list:
    """Heuristic to detect heavy-tailed, non-negative columns for log1p.

    We follow the manuscript's intent rather than a fixed feature list: columns
    that are (i) non-negative on I_train and (ii) highly skewed / heavy-tailed
    are log-transformed.

    This vectorized implementation is ~100x faster than the column-by-column loop.
    """
    X = np.asarray(X, dtype=np.float64)
    Xt = X[train_idx].copy()
    d = Xt.shape[1]

    # Handle non-finite values
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)

    # Vectorized checks across all columns at once
    col_min = np.min(Xt, axis=0)
    col_max = np.max(Xt, axis=0)

    # Compute quantiles for all columns at once (major speedup)
    q50 = np.quantile(Xt, 0.50, axis=0)
    q90 = np.quantile(Xt, 0.90, axis=0)
    q99 = np.quantile(Xt, 0.99, axis=0)

    out = []
    for j in range(d):
        # Skip negative-valued columns
        if col_min[j] < 0:
            continue

        # Skip (near-)binary columns: check if only 0s and 1s
        # Fast check: if min >= 0, max <= 1, and only 2 unique values
        if col_max[j] <= 1.0:
            uniq_count = len(np.unique(Xt[:, j]))
            if uniq_count <= 2:
                continue

        # Heavy-tail proxy: large tail-to-median ratio
        denom = max(q50[j], 1e-6)
        ratio = (q99[j] + 1e-6) / denom

        # Require both skew and scale to reduce false positives
        if ratio >= 25.0 and q99[j] >= 10.0 and q99[j] > q90[j]:
            out.append(j)

    return out
