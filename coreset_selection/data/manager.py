"""
DataManager for loading and merging multiple data sources.

Contains:
- DataManager: Unified interface for data loading and preprocessing

Supports two data modes:
1. Brazil telecom mode: Uses smp_main.csv with metadata
2. Legacy mode: Uses separate cobertura/atendidos/setores files
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .loaders import (
    load_population_muni_csv,
    load_cobertura_features_and_targets,
    load_atendidos_features,
    load_setores_features,
    load_synthetic_data,
)
from .preprocessing import (
    resolve_column,
    detect_numeric_columns,
    impute_missing_values,
    standardize_state_code,
)


class DataManager:
    """
    Unified data loading and preprocessing manager.
    
    Handles loading multiple data sources, merging, and preparing
    feature matrices for coreset selection.
    
    Attributes
    ----------
    cfg : FilesConfig
        File configuration
    seed : int
        Random seed
    df : Optional[pd.DataFrame]
        Merged DataFrame (after load())
    scaler : Optional[StandardScaler]
        Fitted scaler (after load())
    """
    
    def __init__(self, cfg, seed: int, preprocessing_cfg=None):
        """
        Initialize the DataManager.
        
        Parameters
        ----------
        cfg : FilesConfig
            File configuration with paths
        seed : int
            Random seed for reproducibility
        preprocessing_cfg : Optional[PreprocessingConfig]
            Feature typing / preprocessing configuration (Phase 1).
            If None, default heuristics are used.
        """
        self.cfg = cfg
        self.seed = seed
        self._preprocessing_cfg = preprocessing_cfg
        self.df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_cols: List[str] = []
        self._feature_types: List[str] = []
        self._category_maps: Dict = {}
        self._id_col = "CO_MUNICIPIO_IBGE"
        self._state_col = "UF"
        self._pop_col = "POPULACAO"

    def load(self, use_synthetic: bool = False) -> pd.DataFrame:
        """
        Load and merge all data sources.
        
        Parameters
        ----------
        use_synthetic : bool
            If True, use synthetic data for testing
            
        Returns
        -------
        pd.DataFrame
            Merged DataFrame with all features
        """
        if use_synthetic:
            return self._load_synthetic()
        
        # Check if Brazil telecom mode is enabled
        if getattr(self.cfg, 'use_brazil_telecom', False):
            return self._load_brazil_telecom()
        
        return self._load_real()

    def _load_synthetic(self) -> pd.DataFrame:
        """Load synthetic data for testing."""
        df, state_labels, y = load_synthetic_data(
            n_samples=5000,
            n_features=20,
            n_states=27,
            seed=self.seed,
        )
        
        df[self._id_col] = np.arange(len(df))
        df[self._state_col] = state_labels
        df[self._pop_col] = np.random.default_rng(self.seed).integers(1000, 100000, len(df))
        df["target"] = y
        
        self._feature_cols = [c for c in df.columns if c.startswith("feature_")]
        self.df = df
        
        return df

    def _load_brazil_telecom(self) -> pd.DataFrame:
        """Load Brazil telecom data from processed CSV files."""
        from .brazil_telecom_loader import BrazilTelecomDataLoader
        
        cfg = self.cfg
        
        # Resolve preprocessing config (may be on ExperimentConfig parent)
        preproc_cfg = getattr(self, '_preprocessing_cfg', None)
        
        # Create loader with file paths and preprocessing config
        loader = BrazilTelecomDataLoader(
            data_dir=cfg.data_dir,
            main_file=cfg.main_data_file,
            metadata_file=cfg.metadata_file,
            population_file=cfg.population_file,
            preprocessing_cfg=preproc_cfg,
        )
        
        # Load data without standardization and without imputation.
        # Imputation + missingness indicators are handled in the replicate
        # cache builder using I_train statistics per manuscript Section 5.7.
        data = loader.load(standardize=False, impute_missing=False)
        
        # Build DataFrame
        df = pd.DataFrame(data.X, columns=data.feature_names)
        df[self._id_col] = data.ibge_codes
        df[self._state_col] = data.state_labels
        df[self._pop_col] = data.population
        df['y_4G'] = data.y_4G
        df['y_5G'] = data.y_5G
        df['longitude'] = data.coords[:, 0]
        df['latitude'] = data.coords[:, 1]
        df['municipio'] = data.municipality_names
        
        self._feature_cols = list(data.feature_names)
        self._feature_types = list(data.feature_types)
        self._category_maps = dict(data.category_maps)
        self.df = df
        self._raw_df = data.raw_df  # Full DataFrame before feature/target separation
        self._y_4G = data.y_4G
        self._y_5G = data.y_5G
        self._extra_targets: Dict[str, np.ndarray] = dict(data.extra_targets)
        
        print(f"[DataManager] Loaded Brazil telecom data: {len(df)} samples, {len(self._feature_cols)} features")
        if self._extra_targets:
            print(f"  Extra coverage targets: {list(self._extra_targets.keys())}")
        
        return df

    def _load_real(self) -> pd.DataFrame:
        """Load real data from configured paths."""
        cfg = self.cfg
        
        # Load population data (base)
        if os.path.exists(cfg.population_csv):
            df_pop = load_population_muni_csv(cfg.population_csv)
        else:
            raise FileNotFoundError(f"Population CSV not found: {cfg.population_csv}")
        
        # Start with population as base
        df = df_pop.copy()
        all_feature_cols = []
        
        # Load and merge cobertura features
        if cfg.cobertura_zip and os.path.exists(cfg.cobertura_zip):
            df_cob, df_targets, _ = load_cobertura_features_and_targets(cfg.cobertura_zip)
            feature_cols = [c for c in df_cob.columns if c != self._id_col]
            all_feature_cols.extend(feature_cols)
            df = df.merge(df_cob, on=self._id_col, how="left")
            
            # Also merge targets
            target_cols = [c for c in df_targets.columns if c != self._id_col]
            df = df.merge(df_targets, on=self._id_col, how="left")
        
        # Load and merge atendidos features
        if cfg.atendidos_zip and os.path.exists(cfg.atendidos_zip):
            df_ate, _ = load_atendidos_features(cfg.atendidos_zip)
            feature_cols = [c for c in df_ate.columns if c != self._id_col]
            all_feature_cols.extend(feature_cols)
            df = df.merge(df_ate, on=self._id_col, how="left")
        
        # Load and merge setores features
        if cfg.setores_parquet and os.path.exists(cfg.setores_parquet):
            df_set = load_setores_features(cfg.setores_parquet)
            feature_cols = [c for c in df_set.columns if c != self._id_col]
            all_feature_cols.extend(feature_cols)
            df = df.merge(df_set, on=self._id_col, how="left")
        
        self._feature_cols = list(set(all_feature_cols))
        
        # Impute missing values
        df = impute_missing_values(df, self._feature_cols, strategy="median")
        
        # Drop rows with missing state
        df = df.dropna(subset=[self._state_col])
        
        self.df = df
        return df

    def X_numeric_unscaled(self) -> np.ndarray:
        """
        Get unscaled numeric feature matrix.
        
        Returns
        -------
        np.ndarray
            Feature matrix (N, d)
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        return self.df[self._feature_cols].values.astype(np.float32)

    def feature_names(self) -> List[str]:
        """Return the numeric feature column names used for X."""
        return list(self._feature_cols)

    def X_raw(self) -> np.ndarray:
        """
        Get standardized feature matrix.
        
        Returns
        -------
        np.ndarray
            Standardized feature matrix (N, d)
        """
        X_unscaled = self.X_numeric_unscaled()
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_unscaled)
        else:
            X_scaled = self.scaler.transform(X_unscaled)
        
        return X_scaled.astype(np.float32)

    def state_labels(self) -> np.ndarray:
        """
        Get state/geographic group labels.
        
        Returns
        -------
        np.ndarray
            String array of state codes (N,)
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        return self.df[self._state_col].values

    def population(self) -> Optional[np.ndarray]:
        """
        Get population values.
        
        Returns
        -------
        Optional[np.ndarray]
            Population array (N,) or None if not available
        """
        if self.df is None:
            return None
        
        if self._pop_col not in self.df.columns:
            return None
        
        return self.df[self._pop_col].values.astype(np.float64)

    def targets(self) -> Optional[np.ndarray]:
        """
        Get target values for evaluation.

        **Phase 4**: If ``preprocessing_cfg.target_columns`` is set, those
        columns are returned as the target array.  This lets you point the
        pipeline at *any* column(s) in the CSV without editing code.

        Falls back to:
        - Brazil telecom mode: stacked ``[y_4G, y_5G]``.
        - Legacy mode: first match from ``["target", "y", ...]``.

        Returns
        -------
        Optional[np.ndarray]
            Target array ``(N,)`` or ``(N, T)`` or ``None``.
        """
        if self.df is None:
            return None

        # Phase 4: Explicit target columns from config take priority
        explicit = (
            getattr(self._preprocessing_cfg, "target_columns", [])
            if self._preprocessing_cfg else []
        )
        if explicit:
            present = [c for c in explicit if c in self.df.columns]
            if present:
                vals = self.df[present].apply(
                    pd.to_numeric, errors="coerce"
                ).values.astype(np.float64)
                return vals if vals.shape[1] > 1 else vals.ravel()

        # Check for Brazil telecom multi-target data
        if hasattr(self, '_y_4G') and hasattr(self, '_y_5G'):
            return np.column_stack([self._y_4G, self._y_5G])

        # Look for target column
        target_candidates = ["target", "y", "TX_COBERTURA", "INDICADOR_PRINCIPAL"]
        target_col = resolve_column(self.df, target_candidates, required=False)

        if target_col is None:
            return None

        return self.df[target_col].values.astype(np.float64)

    def targets_4G(self) -> Optional[np.ndarray]:
        """Get 4G target values."""
        if hasattr(self, '_y_4G'):
            return self._y_4G
        return None
    
    def targets_5G(self) -> Optional[np.ndarray]:
        """Get 5G target values."""
        if hasattr(self, '_y_5G'):
            return self._y_5G
        return None

    def targets_all_dict(self) -> Dict[str, np.ndarray]:
        """Return all available coverage targets as ``{name: (N,) array}``.

        Always includes at least ``cov_area_4G`` and ``cov_area_5G`` (the
        primary pair).  Additional targets discovered at load time are
        appended.  This is the entry point for multi-target KRR evaluation
        (manuscript Table IV).
        """
        out: Dict[str, np.ndarray] = {}
        if hasattr(self, '_y_4G') and self._y_4G is not None:
            out["cov_area_4G"] = self._y_4G
        if hasattr(self, '_y_5G') and self._y_5G is not None:
            out["cov_area_5G"] = self._y_5G
        if hasattr(self, '_extra_targets'):
            out.update(self._extra_targets)
        return out

    def targets_multi(self) -> Optional[np.ndarray]:
        """Return all targets stacked as ``(N, T)`` array.

        Column order matches the canonical order in
        ``config.constants.COVERAGE_TARGET_NAMES``.  Targets not available
        in the dataset are omitted (so T ≤ 10).
        """
        d = self.targets_all_dict()
        if not d:
            return None
        from ..config.constants import COVERAGE_TARGET_NAMES
        ordered = [d[k] for k in COVERAGE_TARGET_NAMES if k in d]
        if not ordered:
            return None
        return np.column_stack(ordered)

    def targets_multi_names(self) -> list:
        """Return the ordered list of target names present in the dataset."""
        d = self.targets_all_dict()
        from ..config.constants import COVERAGE_TARGET_NAMES
        return [k for k in COVERAGE_TARGET_NAMES if k in d]

    def latlon(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get latitude and longitude coordinates.
        
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (latitudes, longitudes) or (None, None)
        """
        if self.df is None:
            return None, None
        
        lat_col = resolve_column(self.df, ["LATITUDE", "latitude", "lat", "LAT"], required=False)
        lon_col = resolve_column(self.df, ["LONGITUDE", "longitude", "lon", "LON", "lng"], required=False)
        
        if lat_col is None or lon_col is None:
            return None, None
        
        return self.df[lat_col].values, self.df[lon_col].values

    def feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self._feature_cols.copy()

    def feature_types_list(self) -> List[str]:
        """Get list of feature type strings aligned with feature_names().

        Each entry is one of ``"numeric"``, ``"ordinal"``, ``"categorical"``.
        Returns an empty list if type information is not available (e.g.
        legacy/synthetic mode).
        """
        return list(self._feature_types)

    def category_maps(self) -> Dict:
        """Return ``{col: {original_value: int_code}}`` for categorical features.

        Returns an empty dict if no categoricals were encoded.
        """
        return dict(self._category_maps)

    def n_samples(self) -> int:
        """Get number of samples."""
        if self.df is None:
            return 0
        return len(self.df)

    def n_features(self) -> int:
        """Get number of features."""
        return len(self._feature_cols)

    def summary(self) -> Dict:
        """Get data summary statistics."""
        if self.df is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "n_samples": self.n_samples(),
            "n_features": self.n_features(),
            "n_states": len(self.df[self._state_col].unique()),
            "feature_names": self.feature_names()[:10],  # First 10
        }
