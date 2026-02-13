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


def _validate_target(
    name: str,
    labels: np.ndarray,
    min_class_frac: float = 0.05,
) -> bool:
    """Return True if every class has at least *min_class_frac* of the samples.

    A target where the smallest class has <5 % of the data will almost
    certainly produce single-class training sets when sub-sampling a
    coreset of size k ≪ N, making downstream classification meaningless.
    """
    N = len(labels)
    if N == 0:
        return False
    _, counts = np.unique(labels, return_counts=True)
    smallest_frac = counts.min() / N
    if smallest_frac < min_class_frac:
        print(
            f"[derived_targets] DROPPED '{name}': smallest class has "
            f"{smallest_frac:.1%} of data (need ≥{min_class_frac:.0%}). "
            f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )
        return False
    return True


def derive_classification_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Derive classification targets from raw DataFrame columns.

    Every candidate target is validated for class balance before inclusion:
    any class with <5 % of the samples is rejected, because coreset
    sub-sampling (k ≪ N) would almost certainly produce single-class
    training sets and make the downstream evaluation meaningless.

    Parameters
    ----------
    df : pd.DataFrame
        The raw municipality-level DataFrame.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping ``{target_name: (N,) int64 array}``.
        Targets whose source columns are absent or that fail class-balance
        validation are skipped with a printed message.
    """
    N = len(df)
    candidates: Dict[str, np.ndarray] = {}

    # ── Binary targets ──────────────────────────────────────────────
    # All binary targets use median-split to guarantee ~50/50 balance,
    # unless a domain-specific threshold is more meaningful AND safe.

    # concentrated_mobile_market: HHI SMP ≥ 0.25 (Anatel regulatory threshold)
    # HHI < 0.25 → competitive (0), HHI ≥ 0.25 → concentrated (1)
    arr_hhi = _get_column(df, "HHI SMP_2024")
    if arr_hhi is None:
        arr_hhi = _get_column(df, "HHI_SMP_2024")
    if arr_hhi is not None:
        arr_hhi = np.nan_to_num(arr_hhi, nan=0.0)
        candidates["concentrated_mobile_market"] = (arr_hhi >= 0.25).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'concentrated_mobile_market' — "
              "no HHI SMP column found")

    # has_fiber_backhaul: median-split on pct_fibra_backhaul
    # (plain >0 threshold was potentially skewed)
    arr = _get_column(df, "pct_fibra_backhaul")
    if arr is not None:
        arr = np.nan_to_num(arr, nan=0.0)
        median_val = np.median(arr[arr > 0]) if np.any(arr > 0) else 0.0
        candidates["high_fiber_backhaul"] = (arr >= median_val).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'high_fiber_backhaul'")

    # high_speed_broadband: median-split on pct_agl_alta_velocidade
    # (the old >50 threshold was arbitrary and potentially imbalanced)
    arr = _get_column(df, "pct_agl_alta_velocidade")
    if arr is not None:
        arr = np.nan_to_num(arr, nan=0.0)
        median_val = float(np.median(arr))
        candidates["high_speed_broadband"] = (arr > median_val).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'high_speed_broadband'")

    # ── 3-class targets (tercile-binned) ────────────────────────────

    # urbanization_level: from pct_urbano
    arr = _get_column(df, "pct_urbano")
    if arr is not None:
        candidates["urbanization_level"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'urbanization_level'")

    # broadband_speed_tier: from velocidade_mediana_mean
    arr = _get_column(df, "velocidade_mediana_mean")
    if arr is not None:
        candidates["broadband_speed_tier"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'broadband_speed_tier'")

    # income_tier: from renda_media_mean
    arr = _get_column(df, "renda_media_mean")
    if arr is not None:
        candidates["income_tier"] = _tercile_bin(np.nan_to_num(arr, nan=0.0))
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
        candidates["income_speed_class"] = np.argmax(quadrant, axis=1).astype(np.int64)
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
        candidates["mobile_penetration_tier"] = _quartile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'mobile_penetration_tier'")

    # ── 5-class targets (quintile-binned) ───────────────────────────

    # infra_density_tier: from n_estacoes_smp
    arr = _get_column(df, "n_estacoes_smp")
    if arr is not None:
        candidates["infra_density_tier"] = _quintile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'infra_density_tier'")

    # road_coverage_4g_tier: from rod_pct_cob_todas_4g
    arr = _get_column(df, "rod_pct_cob_todas_4g")
    if arr is not None:
        candidates["road_coverage_4g_tier"] = _quintile_bin(np.nan_to_num(arr, nan=0.0))
    else:
        print("[derived_targets] WARNING: cannot derive 'road_coverage_4g_tier'")

    # ── Validate all candidates ─────────────────────────────────────
    # Drop any target where the smallest class has <5% of samples.
    out: Dict[str, np.ndarray] = {}
    for name, labels in candidates.items():
        if _validate_target(name, labels):
            out[name] = labels

    n_dropped = len(candidates) - len(out)
    print(f"[derived_targets] {len(out)} classification targets accepted"
          f"{f', {n_dropped} dropped due to class imbalance' if n_dropped else ''}")

    return out
