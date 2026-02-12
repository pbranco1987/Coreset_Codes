"""
External data loaders for infrastructure and population data.

Contains:
- load_population_muni_csv: Load municipality population data
- load_cobertura_features_and_targets: Load coverage features
- load_atendidos_features: Load service attendance data
- load_setores_features: Load census sector features
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import (
    resolve_column,
    _br_to_float,
    _to_int_id,
    period_mm_tag,
    period_suffix_monYYYY,
    standardize_state_code,
)


def load_population_muni_csv(
    path: str,
    id_col: str = "CO_MUNICIPIO_IBGE",
    pop_col: str = "POPULACAO",
    state_col: str = "UF",
) -> pd.DataFrame:
    """
    Load municipality population data from CSV.
    
    Parameters
    ----------
    path : str
        Path to CSV file
    id_col : str
        Column name for municipality ID
    pop_col : str
        Column name for population
    state_col : str
        Column name for state code
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns [id_col, pop_col, state_col]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Population file not found: {path}")
    
    df = pd.read_csv(path, dtype=str)
    
    # Resolve column names
    id_col_actual = resolve_column(
        df, [id_col, "CO_MUNICIPIO", "IBGE", "cod_ibge", "codigo_ibge"]
    )
    pop_col_actual = resolve_column(
        df, [pop_col, "POP", "POPULACAO_ESTIMADA", "pop_estimada"]
    )
    state_col_actual = resolve_column(
        df, [state_col, "SIGLA_UF", "uf", "estado"]
    )
    
    # Convert types
    df[id_col_actual] = df[id_col_actual].apply(_to_int_id)
    df[pop_col_actual] = df[pop_col_actual].apply(_br_to_float)
    df[state_col_actual] = df[state_col_actual].apply(standardize_state_code)
    
    # Rename to standard names
    df = df.rename(columns={
        id_col_actual: id_col,
        pop_col_actual: pop_col,
        state_col_actual: state_col,
    })
    
    return df[[id_col, pop_col, state_col]]


def load_cobertura_features_and_targets(
    zip_path: str,
    period: str = "dez/2023",
    id_col: str = "CO_MUNICIPIO_IBGE",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load coverage features and targets from ZIP file.
    
    Parameters
    ----------
    zip_path : str
        Path to ZIP file containing coverage data
    period : str
        Period to load (e.g., "dez/2023")
    id_col : str
        Column name for municipality ID
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (features_df, targets_df, metadata_df)
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Cobertura ZIP not found: {zip_path}")
    
    period_tag = period_mm_tag(period)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the appropriate file
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {zip_path}")
        
        # Try to find file matching period
        target_file = None
        for f in csv_files:
            if period_tag in f or period_suffix_monYYYY(period) in f.lower():
                target_file = f
                break
        
        if target_file is None:
            target_file = csv_files[0]  # Use first file as fallback
        
        with zf.open(target_file) as f:
            df = pd.read_csv(f, dtype=str)
    
    # Resolve ID column
    id_col_actual = resolve_column(
        df, [id_col, "CO_MUNICIPIO", "IBGE", "cod_ibge"]
    )
    df[id_col_actual] = df[id_col_actual].apply(_to_int_id)
    
    # Separate features, targets, and metadata
    feature_cols = [c for c in df.columns if c.startswith(('QT_', 'VL_', 'NU_', 'PC_'))]
    target_cols = [c for c in df.columns if c.startswith(('TX_', 'META_', 'INDICADOR_'))]
    meta_cols = [id_col_actual] + [c for c in df.columns if c not in feature_cols + target_cols]
    
    # Convert numeric columns
    for col in feature_cols + target_cols:
        df[col] = df[col].apply(_br_to_float)
    
    # Split into separate DataFrames
    df_features = df[[id_col_actual] + feature_cols].rename(columns={id_col_actual: id_col})
    df_targets = df[[id_col_actual] + target_cols].rename(columns={id_col_actual: id_col})
    df_meta = df[meta_cols].rename(columns={id_col_actual: id_col})
    
    return df_features, df_targets, df_meta


def load_atendidos_features(
    zip_path: str,
    period: str = "dez/2023",
    id_col: str = "CO_MUNICIPIO_IBGE",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load service attendance features from ZIP file.
    
    Parameters
    ----------
    zip_path : str
        Path to ZIP file
    period : str
        Period to load
    id_col : str
        ID column name
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, metadata_df)
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Atendidos ZIP not found: {zip_path}")
    
    period_tag = period_mm_tag(period)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {zip_path}")
        
        # Find matching file
        target_file = csv_files[0]
        for f in csv_files:
            if period_tag in f:
                target_file = f
                break
        
        with zf.open(target_file) as f:
            df = pd.read_csv(f, dtype=str)
    
    # Resolve ID column
    id_col_actual = resolve_column(
        df, [id_col, "CO_MUNICIPIO", "IBGE", "cod_ibge"]
    )
    df[id_col_actual] = df[id_col_actual].apply(_to_int_id)
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c.startswith(('QT_', 'VL_', 'NU_', 'MEDIA_', 'TOTAL_'))]
    meta_cols = [id_col_actual] + [c for c in df.columns if c not in feature_cols]
    
    # Convert numeric
    for col in feature_cols:
        df[col] = df[col].apply(_br_to_float)
    
    df_features = df[[id_col_actual] + feature_cols].rename(columns={id_col_actual: id_col})
    df_meta = df[meta_cols].rename(columns={id_col_actual: id_col})
    
    return df_features, df_meta


def load_setores_features(
    parquet_path: str,
    id_col: str = "CO_MUNICIPIO_IBGE",
    aggregate_to_muni: bool = True,
) -> pd.DataFrame:
    """
    Load census sector features from Parquet file.
    
    Parameters
    ----------
    parquet_path : str
        Path to Parquet file
    id_col : str
        Municipality ID column
    aggregate_to_muni : bool
        If True, aggregate sector-level data to municipality level
        
    Returns
    -------
    pd.DataFrame
        Features DataFrame
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Setores Parquet not found: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Resolve ID column
    id_col_actual = resolve_column(
        df, [id_col, "CO_MUNICIPIO", "CD_MUN", "cod_muni"], required=False
    )
    
    if id_col_actual is None:
        # Try to extract from sector code
        setor_col = resolve_column(df, ["CD_SETOR", "cod_setor", "SETOR"])
        if setor_col:
            # Municipality code is first 7 digits of sector code
            df[id_col] = df[setor_col].astype(str).str[:7].astype(int)
            id_col_actual = id_col
    
    # Identify numeric feature columns
    exclude_cols = {id_col_actual, 'CD_SETOR', 'cod_setor', 'SETOR', 'UF', 'NOME_MUN'}
    feature_cols = [c for c in df.columns if c not in exclude_cols 
                    and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    
    if aggregate_to_muni:
        # Aggregate to municipality level
        agg_dict = {col: 'sum' for col in feature_cols}
        df_agg = df.groupby(id_col_actual).agg(agg_dict).reset_index()
        df_agg = df_agg.rename(columns={id_col_actual: id_col})
        return df_agg
    
    return df.rename(columns={id_col_actual: id_col})


def load_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 20,
    n_states: int = 27,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of numeric features
    n_states : int
        Number of geographic groups (states)
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        (features_df, state_labels, targets)
    """
    rng = np.random.default_rng(seed)
    
    # Generate features with some structure
    X = rng.standard_normal((n_samples, n_features))
    
    # Add some correlations
    for i in range(1, n_features):
        X[:, i] = 0.5 * X[:, i-1] + 0.5 * X[:, i]
    
    # Generate state assignments (unequal sizes)
    state_probs = rng.dirichlet(np.ones(n_states) * 2)
    state_labels = rng.choice(n_states, size=n_samples, p=state_probs)
    
    # Add state-specific effects
    state_effects = rng.standard_normal((n_states, n_features)) * 0.5
    for i, s in enumerate(state_labels):
        X[i] += state_effects[s]
    
    # Generate target
    true_weights = rng.standard_normal(n_features)
    y = X @ true_weights + rng.standard_normal(n_samples) * 0.5
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["state"] = [f"S{s:02d}" for s in state_labels]
    
    state_labels_arr = np.array([f"S{s:02d}" for s in state_labels])
    
    return df, state_labels_arr, y
