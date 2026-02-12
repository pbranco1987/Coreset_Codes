"""
Derived target construction for downstream evaluation.

Extracts extra regression targets (continuous columns beyond coverage) and
derives classification targets (binary + multiclass) from the raw DataFrame.

Called at cache-build time so that targets are persisted in the replicate
cache and reused identically across all experiment configurations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ── Column-name resolution ──────────────────────────────────────────────

def _find_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find a column in *df* by exact match, then by case-insensitive match.

    Returns the actual column name or ``None``.
    """
    if name in df.columns:
        return name
    # Case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(name.lower())


def _get_column(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    """Return a column as a 1-D float64 array, or ``None`` if not found."""
    col = _find_column(df, name)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


# ── Extra regression targets ────────────────────────────────────────────

# Mapping: canonical target name → column name(s) to search for in the CSV.
_EXTRA_REG_COLUMN_MAP: Dict[str, List[str]] = {
    "velocidade_mediana_mean":         ["velocidade_mediana_mean"],
    "velocidade_mediana_std":          ["velocidade_mediana_std"],
    "pct_limite_mean":                 ["pct_limite_mean"],
    "renda_media_mean":                ["renda_media_mean"],
    "renda_media_std":                 ["renda_media_std"],
    "HHI SMP_2024":                    ["HHI SMP_2024", "HHI_SMP_2024"],
    "HHI SCM_2024":                    ["HHI SCM_2024", "HHI_SCM_2024"],
    "pct_fibra_backhaul":              ["pct_fibra_backhaul"],
    "pct_escolas_internet":            ["pct_escolas_internet"],
    "pct_escolas_fibra":               ["pct_escolas_fibra"],
    "Densidade_Banda Larga Fixa_2025": [
        "Densidade_Banda Larga Fixa_2025",
        "Densidade_Banda_Larga_Fixa_2025",
    ],
    "Densidade_Telefonia Móvel_2025":  [
        "Densidade_Telefonia Móvel_2025",
        "Densidade_Telefonia_Móvel_2025",
        "Densidade_Telefonia Movel_2025",
    ],
}


def extract_extra_regression_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Extract extra continuous regression target columns from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (unprocessed) municipality-level DataFrame.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping ``{canonical_name: (N,) float64 array}``.
        Columns not found in *df* are skipped with a warning.
    """
    out: Dict[str, np.ndarray] = {}
    for canonical, candidates in _EXTRA_REG_COLUMN_MAP.items():
        arr = None
        for cand in candidates:
            arr = _get_column(df, cand)
            if arr is not None:
                break
        if arr is None:
            print(f"[derived_targets] WARNING: extra regression target "
                  f"'{canonical}' not found (tried {candidates})")
            continue
        # Replace remaining NaN with 0 (these are targets, not features)
        arr = np.nan_to_num(arr, nan=0.0)
        out[canonical] = arr
    return out


# ── Classification targets ──────────────────────────────────────────────

def _tercile_bin(arr: np.ndarray) -> np.ndarray:
    """Bin a continuous array into 3 classes (0, 1, 2) by tercile thresholds."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.int64)
    t1 = np.percentile(valid, 33.33)
    t2 = np.percentile(valid, 66.67)
    out = np.zeros(len(arr), dtype=np.int64)
    out[arr > t1] = 1
    out[arr > t2] = 2
    return out


def _quartile_bin(arr: np.ndarray) -> np.ndarray:
    """Bin a continuous array into 4 classes (0–3) by quartile thresholds."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.int64)
    q25 = np.percentile(valid, 25)
    q50 = np.percentile(valid, 50)
    q75 = np.percentile(valid, 75)
    out = np.zeros(len(arr), dtype=np.int64)
    out[arr > q25] = 1
    out[arr > q50] = 2
    out[arr > q75] = 3
    return out


def _quintile_bin(arr: np.ndarray) -> np.ndarray:
    """Bin a continuous array into 5 classes (0–4) by quintile thresholds."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.int64)
    q20 = np.percentile(valid, 20)
    q40 = np.percentile(valid, 40)
    q60 = np.percentile(valid, 60)
    q80 = np.percentile(valid, 80)
    out = np.zeros(len(arr), dtype=np.int64)
    out[arr > q20] = 1
    out[arr > q40] = 2
    out[arr > q60] = 3
    out[arr > q80] = 4
    return out


def derive_classification_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Derive classification targets from raw DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        The raw municipality-level DataFrame.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping ``{target_name: (N,) int64 array}``.
        Targets whose source columns are absent are skipped with a warning.
    """
    N = len(df)
    out: Dict[str, np.ndarray] = {}

    # ── Binary targets ──────────────────────────────────────────────

    # has_5g: any 5G area coverage > 0
    col_5g = _find_column(df, "cov_pct_area_coberta__tec_5g__op_todas__2025_03")
    if col_5g is None:
        # Try shorter variant
        for c in df.columns:
            if "cov_pct_area_coberta" in c and "5g" in c.lower() and "op_todas" in c:
                col_5g = c
                break
    if col_5g is not None:
        vals = pd.to_numeric(df[col_5g], errors="coerce").fillna(0).to_numpy()
        out["has_5g"] = (vals > 0).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'has_5g' — "
              "no 5G coverage column found")

    # has_fiber_backhaul: pct_fibra_backhaul > 0
    arr = _get_column(df, "pct_fibra_backhaul")
    if arr is not None:
        out["has_fiber_backhaul"] = (arr > 0).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'has_fiber_backhaul'")

    # has_high_speed_internet: pct_agl_alta_velocidade > 50
    arr = _get_column(df, "pct_agl_alta_velocidade")
    if arr is not None:
        out["has_high_speed_internet"] = (np.nan_to_num(arr, nan=0.0) > 50).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'has_high_speed_internet'")

    # ── 3-class targets (tercile-binned) ────────────────────────────

    # urbanization_level: from pct_urbano
    arr = _get_column(df, "pct_urbano")
    if arr is not None:
        out["urbanization_level"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'urbanization_level'")

    # broadband_speed_tier: from velocidade_mediana_mean
    arr = _get_column(df, "velocidade_mediana_mean")
    if arr is not None:
        out["broadband_speed_tier"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'broadband_speed_tier'")

    # income_tier: from renda_media_mean
    arr = _get_column(df, "renda_media_mean")
    if arr is not None:
        out["income_tier"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'income_tier'")

    # ── 4-class targets ─────────────────────────────────────────────

    # income_speed_class: dominant quadrant from pct_cat_* columns
    c_ll = _get_column(df, "pct_cat_low_renda_low_vel")
    c_lh = _get_column(df, "pct_cat_low_renda_high_vel")
    c_hl = _get_column(df, "pct_cat_high_renda_low_vel")
    c_hh = _get_column(df, "pct_cat_high_renda_high_vel")
    if all(x is not None for x in [c_ll, c_lh, c_hl, c_hh]):
        quadrant = np.stack([
            np.nan_to_num(c_ll, nan=0.0),
            np.nan_to_num(c_lh, nan=0.0),
            np.nan_to_num(c_hl, nan=0.0),
            np.nan_to_num(c_hh, nan=0.0),
        ], axis=1)  # (N, 4)
        out["income_speed_class"] = np.argmax(quadrant, axis=1).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'income_speed_class' — "
              "missing pct_cat_* columns")

    # mobile_penetration_tier: from Densidade_Telefonia Móvel_2025 (quartiles)
    arr = None
    for cand in ["Densidade_Telefonia Móvel_2025",
                 "Densidade_Telefonia_Móvel_2025",
                 "Densidade_Telefonia Movel_2025"]:
        arr = _get_column(df, cand)
        if arr is not None:
            break
    if arr is not None:
        out["mobile_penetration_tier"] = _quartile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'mobile_penetration_tier'")

    # ── 5-class targets (quintile-binned) ───────────────────────────

    # infra_density_tier: from n_estacoes_smp
    arr = _get_column(df, "n_estacoes_smp")
    if arr is not None:
        out["infra_density_tier"] = _quintile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'infra_density_tier'")

    # road_coverage_4g_tier: from rod_pct_cob_todas_4g
    arr = _get_column(df, "rod_pct_cob_todas_4g")
    if arr is not None:
        out["road_coverage_4g_tier"] = _quintile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'road_coverage_4g_tier'")

    return out
