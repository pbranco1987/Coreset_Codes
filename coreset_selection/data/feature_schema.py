r"""Feature schema inference and typing (Phase 1 + Phase 2).

This module introduces a **feature schema layer** that classifies every column
in the input DataFrame into one of the following types:

    ``numeric``      — continuous float (standard scaling, log1p, median imputation)
    ``ordinal``      — ordered integer (median imputation, optional scaling, no log1p by default)
    ``categorical``  — unordered category (integer-encoded, mode imputation, no log1p, no scaling by default)
    ``ignore``       — excluded from the feature matrix entirely
    ``target``       — prediction target (excluded from features, used as ``y``)

**Design goals** (Phase 1 of the refactoring plan):

* Safe defaults: string/object → categorical; int with low cardinality →
  categorical; everything else numeric.
* Fully overridable via ``PreprocessingConfig`` fields (explicit column lists
  take priority over inference heuristics).
* Deterministic: output order and classification are reproducible given the
  same DataFrame and config.
* Emits a human-readable schema summary at load time.

**Phase 2 additions**:

* Categorical variables are integer-encoded (no one-hot), compatible with
  downstream classification metrics (accuracy, Cohen's Kappa, F1, etc.).
* Ordinal variables are treated as ordered integers and are *not*
  log-transformed.  Scaling is optional (controlled by ``scale_ordinals``).
* The preprocessing pipeline respects feature types for imputation
  (mode for categorical, rounded median for ordinal), log1p exclusion,
  and standardization masking.
* Target type (regression vs. classification) is auto-detected based on
  cardinality and dtype, enabling appropriate metric selection downstream.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature type enumeration
# ---------------------------------------------------------------------------

class FeatureType(str, Enum):
    """Semantic type of a single feature column."""

    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    IGNORE = "ignore"
    TARGET = "target"

    def __repr__(self) -> str:  # pragma: no cover
        return f"FeatureType.{self.name}"


# ---------------------------------------------------------------------------
# Feature schema container
# ---------------------------------------------------------------------------

@dataclass
class FeatureSchema:
    """Container holding the inferred (or overridden) schema for every column.

    Attributes
    ----------
    column_types : OrderedDict[str, FeatureType]
        Mapping from column name → feature type, preserving DataFrame column
        order.
    category_cardinalities : Dict[str, int]
        For each categorical column, the number of unique non-null categories
        observed at inference time.
    """

    column_types: OrderedDict[str, FeatureType] = field(default_factory=OrderedDict)
    category_cardinalities: Dict[str, int] = field(default_factory=dict)

    # ---- Convenience accessors ----

    def columns_of_type(self, ftype: FeatureType) -> List[str]:
        """Return column names with the given type, preserving order."""
        return [c for c, t in self.column_types.items() if t == ftype]

    @property
    def numeric_columns(self) -> List[str]:
        return self.columns_of_type(FeatureType.NUMERIC)

    @property
    def ordinal_columns(self) -> List[str]:
        return self.columns_of_type(FeatureType.ORDINAL)

    @property
    def categorical_columns(self) -> List[str]:
        return self.columns_of_type(FeatureType.CATEGORICAL)

    @property
    def ignore_columns(self) -> List[str]:
        return self.columns_of_type(FeatureType.IGNORE)

    @property
    def target_columns(self) -> List[str]:
        return self.columns_of_type(FeatureType.TARGET)

    @property
    def feature_columns(self) -> List[str]:
        """All columns that will be used as features (numeric + ordinal + categorical)."""
        return [
            c for c, t in self.column_types.items()
            if t in (FeatureType.NUMERIC, FeatureType.ORDINAL, FeatureType.CATEGORICAL)
        ]

    @property
    def feature_types(self) -> List[FeatureType]:
        """Parallel list of FeatureType aligned with :attr:`feature_columns`."""
        return [
            self.column_types[c] for c in self.feature_columns
        ]

    # ---- Summary / logging ----

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dict suitable for logging."""
        return {
            "numeric": len(self.numeric_columns),
            "ordinal": len(self.ordinal_columns),
            "categorical": len(self.categorical_columns),
            "ignore": len(self.ignore_columns),
            "target": len(self.target_columns),
            "total_features": len(self.feature_columns),
        }

    def print_summary(self, prefix: str = "  ") -> None:
        """Print a human-readable schema summary to stdout."""
        s = self.summary()
        print(
            f"{prefix}Feature schema: "
            f"numeric={s['numeric']}, "
            f"ordinal={s['ordinal']}, "
            f"categorical={s['categorical']} (encoded), "
            f"ignored={s['ignore']}, "
            f"target={s['target']}, "
            f"total_features={s['total_features']}"
        )
        if self.categorical_columns:
            cat_info = ", ".join(
                f"{c}({self.category_cardinalities.get(c, '?')})"
                for c in self.categorical_columns[:10]
            )
            suffix = "..." if len(self.categorical_columns) > 10 else ""
            print(f"{prefix}  Categorical cols (name(K)): {cat_info}{suffix}")
        if self.ordinal_columns:
            ord_info = ", ".join(self.ordinal_columns[:10])
            suffix = "..." if len(self.ordinal_columns) > 10 else ""
            print(f"{prefix}  Ordinal cols: {ord_info}{suffix}")
        if self.target_columns:
            print(f"{prefix}  Target cols: {', '.join(self.target_columns)}")


# ---------------------------------------------------------------------------
# Default columns that should always be ignored (identifiers / metadata)
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE_EXACT: Set[str] = {
    "codigo_ibge",
    "uf",
    "municipio",
}

_DEFAULT_IGNORE_LOWER: Set[str] = {s.lower() for s in _DEFAULT_IGNORE_EXACT}


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_schema(
    df: pd.DataFrame,
    *,
    # Explicit overrides (highest priority)
    categorical_columns: Sequence[str] = (),
    ordinal_columns: Sequence[str] = (),
    ignore_columns: Sequence[str] = (),
    target_columns: Sequence[str] = (),
    # Heuristic knobs
    treat_low_cardinality_int_as_categorical: bool = True,
    low_cardinality_threshold: int = 25,
    high_cardinality_drop_threshold: Optional[int] = None,
    treat_bool_as_categorical: bool = True,
    # Target-detection regex patterns (reuse existing infra)
    auto_detect_targets: bool = True,
) -> FeatureSchema:
    """Infer a :class:`FeatureSchema` for a DataFrame.

    **Priority order** (highest first):

    1. Explicit ``target_columns``
    2. Explicit ``ignore_columns``
    3. Explicit ``categorical_columns``
    4. Explicit ``ordinal_columns``
    5. Built-in identifier ignore list (``codigo_ibge``, ``uf``, ``municipio``)
    6. ``auto_detect_targets`` via ``target_columns.py`` regex patterns
    7. Heuristic inference from dtype + cardinality

    Parameters
    ----------
    df : pd.DataFrame
        The (possibly pre-merge) DataFrame whose columns to classify.
    categorical_columns : Sequence[str]
        Columns to force as categorical.
    ordinal_columns : Sequence[str]
        Columns to force as ordinal.
    ignore_columns : Sequence[str]
        Columns to force-ignore (excluded from features).
    target_columns : Sequence[str]
        Columns to mark as targets (excluded from features).
    treat_low_cardinality_int_as_categorical : bool
        If True, integer columns with ≤ ``low_cardinality_threshold`` unique
        values are auto-classified as categorical.
    low_cardinality_threshold : int
        Unique-value cutoff for the low-cardinality heuristic.
    high_cardinality_drop_threshold : Optional[int]
        If set, *inferred* categorical columns with more unique values than
        this threshold are dropped to ``ignore`` (safety valve against
        accidental one-hot explosion). Explicit overrides are never dropped.
    treat_bool_as_categorical : bool
        If True, boolean columns become categorical; otherwise numeric.
    auto_detect_targets : bool
        If True, apply the regex-based target detection from
        ``data.target_columns``.

    Returns
    -------
    FeatureSchema
        Fully resolved schema for every column in *df*.
    """
    # Build override sets (case-sensitive lookup; lowercase fallback)
    set_target = set(target_columns)
    set_ignore = set(ignore_columns)
    set_cat = set(categorical_columns)
    set_ord = set(ordinal_columns)

    # Auto-detect target columns via existing regex infrastructure
    auto_target_names: Set[str] = set()
    if auto_detect_targets:
        try:
            from .target_columns import detect_target_columns as _detect
            auto_target_names = set(_detect(list(df.columns)))
        except ImportError:
            pass

    schema = FeatureSchema()

    for col in df.columns:
        col_lower = col.lower()

        # --- 1. Explicit target override ---
        if col in set_target:
            schema.column_types[col] = FeatureType.TARGET
            continue

        # --- 2. Explicit ignore override ---
        if col in set_ignore:
            schema.column_types[col] = FeatureType.IGNORE
            continue

        # --- 3. Explicit categorical override ---
        if col in set_cat:
            schema.column_types[col] = FeatureType.CATEGORICAL
            n_unique = int(df[col].nunique(dropna=True))
            schema.category_cardinalities[col] = n_unique
            continue

        # --- 4. Explicit ordinal override ---
        if col in set_ord:
            schema.column_types[col] = FeatureType.ORDINAL
            continue

        # --- 5. Default ignore (identifiers) ---
        if col_lower in _DEFAULT_IGNORE_LOWER:
            schema.column_types[col] = FeatureType.IGNORE
            continue

        # --- 6. Auto-detected targets ---
        if col in auto_target_names:
            schema.column_types[col] = FeatureType.TARGET
            continue

        # --- 7. Columns starting with y_ are targets ---
        if col_lower.startswith("y_") or "y_cov_" in col_lower:
            schema.column_types[col] = FeatureType.TARGET
            continue

        # --- 8. Heuristic inference ---
        dtype = df[col].dtype

        # 8a. Object / string → categorical
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # Try numeric coercion first (some "object" columns are numeric strings)
            try:
                pd.to_numeric(df[col], errors="raise")
                # Succeeded — treat as numeric
                schema.column_types[col] = FeatureType.NUMERIC
                continue
            except (ValueError, TypeError):
                pass

            n_unique = int(df[col].nunique(dropna=True))
            if high_cardinality_drop_threshold is not None and n_unique > high_cardinality_drop_threshold:
                schema.column_types[col] = FeatureType.IGNORE
                logger.info(
                    "Column '%s' ignored: high-cardinality categorical (%d > %d)",
                    col, n_unique, high_cardinality_drop_threshold,
                )
            else:
                schema.column_types[col] = FeatureType.CATEGORICAL
                schema.category_cardinalities[col] = n_unique
            continue

        # 8b. Boolean → categorical or numeric
        if pd.api.types.is_bool_dtype(dtype):
            if treat_bool_as_categorical:
                schema.column_types[col] = FeatureType.CATEGORICAL
                schema.category_cardinalities[col] = 2
            else:
                schema.column_types[col] = FeatureType.NUMERIC
            continue

        # 8c. Integer with low cardinality → categorical (if enabled)
        if pd.api.types.is_integer_dtype(dtype) and treat_low_cardinality_int_as_categorical:
            n_unique = int(df[col].nunique(dropna=True))
            if n_unique <= low_cardinality_threshold:
                schema.column_types[col] = FeatureType.CATEGORICAL
                schema.category_cardinalities[col] = n_unique
                continue

        # 8d. Numeric (int or float) → numeric
        if pd.api.types.is_numeric_dtype(dtype):
            schema.column_types[col] = FeatureType.NUMERIC
            continue

        # 8e. Fallback: ignore anything we can't classify
        schema.column_types[col] = FeatureType.IGNORE
        logger.info("Column '%s' ignored: unrecognized dtype %s", col, dtype)

    return schema


# ---------------------------------------------------------------------------
# Utility: build schema from PreprocessingConfig + DataFrame
# ---------------------------------------------------------------------------

def build_schema_from_config(
    df: pd.DataFrame,
    preproc_cfg: "PreprocessingConfig",  # forward ref to avoid circular import
) -> FeatureSchema:
    """Convenience wrapper that feeds :class:`PreprocessingConfig` fields into
    :func:`infer_schema`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    preproc_cfg : PreprocessingConfig
        Configuration dataclass containing explicit column lists and heuristic
        knobs.

    Returns
    -------
    FeatureSchema
    """
    return infer_schema(
        df,
        categorical_columns=preproc_cfg.categorical_columns,
        ordinal_columns=preproc_cfg.ordinal_columns,
        ignore_columns=preproc_cfg.ignore_columns,
        target_columns=preproc_cfg.target_columns,
        treat_low_cardinality_int_as_categorical=preproc_cfg.treat_low_cardinality_int_as_categorical,
        low_cardinality_threshold=preproc_cfg.low_cardinality_threshold,
        high_cardinality_drop_threshold=preproc_cfg.high_cardinality_drop_threshold,
        treat_bool_as_categorical=preproc_cfg.treat_bool_as_categorical,
        auto_detect_targets=preproc_cfg.auto_detect_targets,
    )
