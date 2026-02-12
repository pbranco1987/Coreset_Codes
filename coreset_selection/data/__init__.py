"""
Data loading and preprocessing module.

This module provides:
- DataManager: Unified data loading and preprocessing
- BrazilTelecomDataLoader: Loader for Brazil telecom infrastructure data
- Loaders: Functions for loading various data sources
- Preprocessing: Column resolution, type conversion, imputation
- Cache: Replicate cache building and loading
"""

from .preprocessing import (
    resolve_column,
    _br_to_float,
    _to_int_id,
    parse_period_mm_yyyy,
    period_mm_tag,
    period_suffix_monYYYY,
    detect_numeric_columns,
    standardize_state_code,
    impute_missing_values,
)

from .loaders import (
    load_population_muni_csv,
    load_cobertura_features_and_targets,
    load_atendidos_features,
    load_setores_features,
    load_synthetic_data,
)

from .brazil_telecom_loader import (
    BrazilTelecomDataLoader,
    BrazilTelecomData,
    load_brazil_telecom_data,
    BRAZILIAN_STATES,
    STATE_TO_IDX,
)

from .manager import DataManager

from .cache import (
    build_replicate_cache,
    load_replicate_cache,
    prebuild_full_cache,
)

from .target_columns import (
    detect_target_columns,
    remove_target_columns,
    validate_no_leakage,
    TARGET_COLUMN_PATTERNS,
)

from .feature_schema import (
    FeatureType,
    FeatureSchema,
    infer_schema,
    build_schema_from_config,
)

from .split_persistence import (
    save_splits,
    load_splits,
    validate_splits,
)

__all__ = [
    # Preprocessing
    "resolve_column",
    "_br_to_float",
    "_to_int_id",
    "parse_period_mm_yyyy",
    "period_mm_tag",
    "period_suffix_monYYYY",
    "detect_numeric_columns",
    "standardize_state_code",
    "impute_missing_values",
    # Loaders
    "load_population_muni_csv",
    "load_cobertura_features_and_targets",
    "load_atendidos_features",
    "load_setores_features",
    "load_synthetic_data",
    # Brazil telecom
    "BrazilTelecomDataLoader",
    "BrazilTelecomData",
    "load_brazil_telecom_data",
    "BRAZILIAN_STATES",
    "STATE_TO_IDX",
    # Manager
    "DataManager",
    # Cache
    "build_replicate_cache",
    "load_replicate_cache",
    "prebuild_full_cache",
    # Target leakage (Phase 4.3)
    "detect_target_columns",
    "remove_target_columns",
    "validate_no_leakage",
    "TARGET_COLUMN_PATTERNS",
    # Feature schema (Phase 1)
    "FeatureType",
    "FeatureSchema",
    "infer_schema",
    "build_schema_from_config",
    # Split persistence (Phase 4.2)
    "save_splits",
    "load_splits",
    "validate_splits",
]
