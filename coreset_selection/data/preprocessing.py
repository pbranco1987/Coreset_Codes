"""
Data preprocessing utilities.

Contains:
- Column name resolution
- Brazilian number formatting conversion
- Data type conversion helpers
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..config.constants_common import _MONTH_ABBR


def resolve_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:
    """
    Find the first matching column name from a list of candidates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search
    candidates : List[str]
        List of candidate column names (in priority order)
    required : bool
        If True, raise error if no match found
        
    Returns
    -------
    Optional[str]
        Matched column name or None
        
    Raises
    ------
    KeyError
        If required=True and no match found
    """
    for col in candidates:
        if col in df.columns:
            return col
    
    if required:
        raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")
    
    return None


def _br_to_float(s: Union[str, float, int]) -> float:
    """
    Convert Brazilian-formatted number to float.
    
    Brazilian format: "1.234,56" (period for thousands, comma for decimal)
    
    Parameters
    ----------
    s : Union[str, float, int]
        Input value
        
    Returns
    -------
    float
        Converted value
    """
    if pd.isna(s):
        return np.nan
    
    if isinstance(s, (int, float)):
        return float(s)
    
    s = str(s).strip()
    
    if not s or s.lower() in ('', 'nan', 'null', '-'):
        return np.nan
    
    # Remove thousands separator (.) and convert decimal separator (,) to (.)
    s = s.replace('.', '').replace(',', '.')
    
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_int_id(s: Union[str, int, float]) -> int:
    """
    Convert value to integer ID.
    
    Parameters
    ----------
    s : Union[str, int, float]
        Input value
        
    Returns
    -------
    int
        Integer ID
    """
    if pd.isna(s):
        return -1
    
    if isinstance(s, float):
        return int(s)
    
    if isinstance(s, int):
        return s
    
    s = str(s).strip()
    
    # Remove any non-numeric characters
    s = re.sub(r'[^\d]', '', s)
    
    if not s:
        return -1
    
    return int(s)


def parse_period_mm_yyyy(period: str) -> tuple:
    """
    Parse period string like "jan/2020" to (month, year).
    
    Parameters
    ----------
    period : str
        Period string
        
    Returns
    -------
    tuple
        (month_str, year_str) e.g., ("01", "2020")
    """
    period = str(period).strip().lower()
    
    # Try format: "jan/2020" or "jan-2020" or "jan 2020"
    match = re.match(r'([a-z]{3})[\s/\-](\d{4})', period)
    if match:
        month_abbr, year = match.groups()
        month = _MONTH_ABBR.get(month_abbr, "01")
        return month, year
    
    # Try format: "01/2020" or "1/2020"
    match = re.match(r'(\d{1,2})[\s/\-](\d{4})', period)
    if match:
        month, year = match.groups()
        return month.zfill(2), year
    
    # Try format: "202001" or "2020-01"
    match = re.match(r'(\d{4})[\-]?(\d{2})', period)
    if match:
        year, month = match.groups()
        return month, year
    
    return "01", "2020"


def period_mm_tag(period: str) -> str:
    """
    Convert period string to YYYYMM format.
    
    Parameters
    ----------
    period : str
        Period string
        
    Returns
    -------
    str
        YYYYMM string
    """
    month, year = parse_period_mm_yyyy(period)
    return f"{year}{month}"


def period_suffix_monYYYY(period: str) -> str:
    """
    Convert period string to _monYYYY suffix format.
    
    Parameters
    ----------
    period : str
        Period string
        
    Returns
    -------
    str
        Suffix like "_jan2020"
    """
    period = str(period).strip().lower()
    
    match = re.match(r'([a-z]{3})[\s/\-](\d{4})', period)
    if match:
        month_abbr, year = match.groups()
        return f"_{month_abbr}{year}"
    
    return "_unknown"


def detect_numeric_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Detect numeric columns suitable for features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    exclude_cols : Optional[List[str]]
        Columns to exclude
        
    Returns
    -------
    List[str]
        List of numeric column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude_set = set(exclude_cols)
    
    numeric_cols = []
    for col in df.columns:
        if col in exclude_set:
            continue
        
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            numeric_cols.append(col)
    
    return numeric_cols


def standardize_state_code(s: Union[str, int]) -> str:
    """
    Standardize Brazilian state code (UF).
    
    Parameters
    ----------
    s : Union[str, int]
        State code or name
        
    Returns
    -------
    str
        Two-letter uppercase state code
    """
    s = str(s).strip().upper()
    
    # Already a 2-letter code
    if len(s) == 2:
        return s
    
    # Common abbreviations
    state_map = {
        "SAO PAULO": "SP",
        "RIO DE JANEIRO": "RJ",
        "MINAS GERAIS": "MG",
        "BAHIA": "BA",
        "RIO GRANDE DO SUL": "RS",
        "PARANA": "PR",
        "PERNAMBUCO": "PE",
        "CEARA": "CE",
        "PARA": "PA",
        "SANTA CATARINA": "SC",
        "MARANHAO": "MA",
        "GOIAS": "GO",
        "AMAZONAS": "AM",
        "PARAIBA": "PB",
        "ESPIRITO SANTO": "ES",
        "RIO GRANDE DO NORTE": "RN",
        "ALAGOAS": "AL",
        "PIAUI": "PI",
        "MATO GROSSO": "MT",
        "DISTRITO FEDERAL": "DF",
        "MATO GROSSO DO SUL": "MS",
        "SERGIPE": "SE",
        "RONDONIA": "RO",
        "TOCANTINS": "TO",
        "ACRE": "AC",
        "AMAPA": "AP",
        "RORAIMA": "RR",
    }
    
    return state_map.get(s, s[:2])


def impute_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numeric_cols : List[str]
        Columns to impute
    strategy : str
        Imputation strategy ('median', 'mean', 'zero')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values
    """
    df = df.copy()
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        if strategy == "median":
            fill_value = df[col].median()
        elif strategy == "mean":
            fill_value = df[col].mean()
        elif strategy == "zero":
            fill_value = 0.0
        else:
            fill_value = 0.0
        
        df[col] = df[col].fillna(fill_value)
    
    return df
