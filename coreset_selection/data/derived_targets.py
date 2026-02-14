"""
Derived target construction for downstream evaluation.

Extracts extra regression targets (continuous columns beyond coverage) and
derives classification targets (binary + multiclass) from the raw DataFrame.

Called at cache-build time so that targets are persisted in the replicate
cache and reused identically across all experiment configurations.

Two tiers of classification targets are produced:

- **Strict tier** (>=5 % minimum class fraction): 10 targets that are safe
  for downstream evaluation even with small coresets.
- **Relaxed tier** (>=2 % minimum class fraction): 5 additional targets that
  capture finer-grained structure.  Each relaxed-tier target has a built-in
  *failsafe*: if its primary derivation fails the >=2 % check, an alternative
  definition is substituted automatically.

Rich JSON metadata is produced for every target, recording whether it is
an engineered or naturally-derived target, how it was constructed, and
detailed class-distribution statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd



class _AttrDict(dict):
    """A dict subclass that allows setting arbitrary attributes."""
    pass


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

# Mapping: canonical target name -> column name(s) to search for in the CSV.
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
    "Densidade_Telefonia Movel_2025":  [
        "Densidade_Telefonia M\u00f3vel_2025",
        "Densidade_Telefonia Movel_2025",
        "Densidade_Telefonia_M\u00f3vel_2025",
        "Densidade_Telefonia_Movel_2025",
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
    """Bin a continuous array into 4 classes (0-3) by quartile thresholds."""
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
    """Bin a continuous array into 5 classes (0-4) by quintile thresholds."""
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


def _extreme_4class_bin(arr: np.ndarray) -> np.ndarray:
    """Bin into 4 classes with small extreme tails (~3 % each).

    Classes:
        0 -- extreme-low  (<= p3)
        1 -- low-mid      (p3 - p50)
        2 -- mid-high     (p50 - p97)
        3 -- extreme-high (>= p97)

    The extreme classes deliberately contain ~3 % of data each, placing
    the target in the 2-5 % zone.  The failsafe for these is always a
    standard tercile bin (3 balanced classes).
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(len(arr), dtype=np.int64)
    p3 = np.percentile(valid, 3)
    p50 = np.percentile(valid, 50)
    p97 = np.percentile(valid, 97)
    out = np.ones(len(arr), dtype=np.int64)      # default: class 1
    out[arr <= p3] = 0                            # extreme low
    out[(arr > p50) & (arr < p97)] = 2            # mid-high
    out[arr >= p97] = 3                           # extreme high
    return out


def _validate_target(
    name: str,
    labels: np.ndarray,
    min_class_frac: float = 0.05,
    *,
    quiet: bool = False,
) -> bool:
    """Return True if every class has at least *min_class_frac* of the samples.

    A target where the smallest class has <5 % of the data will almost
    certainly produce single-class training sets when sub-sampling a
    coreset of size k << N, making downstream classification meaningless.
    """
    N = len(labels)
    if N == 0:
        return False
    _, counts = np.unique(labels, return_counts=True)
    smallest_frac = counts.min() / N
    if smallest_frac < min_class_frac:
        if not quiet:
            print(
                f"[derived_targets] DROPPED '{name}': smallest class has "
                f"{smallest_frac:.1%} of data (need >={min_class_frac:.0%}). "
                f"Class distribution: "
                f"{dict(zip(*np.unique(labels, return_counts=True)))}"
            )
        return False
    return True


def _smallest_class_frac(labels: np.ndarray) -> float:
    """Return the fraction of data in the smallest class."""
    N = len(labels)
    if N == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.min() / N)


def _class_distribution(labels: np.ndarray) -> Dict[str, Any]:
    """Compute detailed class distribution statistics."""
    N = len(labels)
    if N == 0:
        return {"n_samples": 0, "n_classes": 0, "classes": {}}
    uniq, counts = np.unique(labels, return_counts=True)
    classes = {}
    for u, c in zip(uniq, counts):
        classes[str(int(u))] = {
            "count": int(c),
            "fraction": round(float(c / N), 6),
            "percentage": round(float(c / N * 100), 2),
        }
    return {
        "n_samples": int(N),
        "n_classes": int(len(uniq)),
        "min_class_fraction": round(float(counts.min() / N), 6),
        "max_class_fraction": round(float(counts.max() / N), 6),
        "class_imbalance_ratio": round(
            float(counts.max() / counts.min()), 2,
        ) if counts.min() > 0 else float("inf"),
        "classes": classes,
    }


# ── Target metadata registry ────────────────────────────────────────────
# Each entry fully describes one classification target: what it measures,
# how it was derived, whether it is engineered, and the class semantics.

_STRICT_TARGET_METADATA: Dict[str, Dict[str, Any]] = {
    "concentrated_mobile_market": {
        "description": (
            "Binary indicator of mobile market concentration based on "
            "the Herfindahl-Hirschman Index (HHI) for the SMP market."
        ),
        "source_columns": ["HHI SMP_2024"],
        "task_type": "classification",
        "n_classes_expected": 2,
        "class_semantics": {
            "0": "Competitive market (HHI < 0.25)",
            "1": "Concentrated market (HHI >= 0.25)",
        },
        "derivation_method": "domain_threshold",
        "derivation_details": {
            "operation": "binary threshold",
            "threshold": 0.25,
            "threshold_source": "Anatel regulatory threshold for market concentration",
            "rule": "HHI_SMP_2024 >= 0.25 -> class 1 (concentrated)",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "high_fiber_backhaul": {
        "description": (
            "Binary indicator of high fiber backhaul penetration, "
            "determined by median-split on non-zero pct_fibra_backhaul."
        ),
        "source_columns": ["pct_fibra_backhaul"],
        "task_type": "classification",
        "n_classes_expected": 2,
        "class_semantics": {
            "0": "Below-median fiber backhaul penetration",
            "1": "Above-median fiber backhaul penetration",
        },
        "derivation_method": "median_split",
        "derivation_details": {
            "operation": "binary threshold at median of non-zero values",
            "rule": "pct_fibra_backhaul >= median(nonzero) -> class 1",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "high_speed_broadband": {
        "description": (
            "Binary indicator of high-speed broadband availability, "
            "median-split on pct_agl_alta_velocidade."
        ),
        "source_columns": ["pct_agl_alta_velocidade"],
        "task_type": "classification",
        "n_classes_expected": 2,
        "class_semantics": {
            "0": "Below-median high-speed broadband share",
            "1": "Above-median high-speed broadband share",
        },
        "derivation_method": "median_split",
        "derivation_details": {
            "operation": "binary threshold at overall median",
            "rule": "pct_agl_alta_velocidade > median -> class 1",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "has_5g_coverage": {
        "description": (
            "Binary indicator of 5G operator presence in the municipality."
        ),
        "source_columns": ["att09_any_present_5G"],
        "task_type": "classification",
        "n_classes_expected": 2,
        "class_semantics": {
            "0": "No 5G operator present",
            "1": "At least one 5G operator present",
        },
        "derivation_method": "direct_column",
        "derivation_details": {
            "operation": "direct use of binary column (already 0/1)",
            "rule": "att09_any_present_5G == 1 -> class 1",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "urbanization_level": {
        "description": (
            "Three-class urbanization level from tercile binning of "
            "pct_urbano (share of urban population)."
        ),
        "source_columns": ["pct_urbano"],
        "task_type": "classification",
        "n_classes_expected": 3,
        "class_semantics": {
            "0": "Low urbanization (bottom tercile)",
            "1": "Medium urbanization (middle tercile)",
            "2": "High urbanization (top tercile)",
        },
        "derivation_method": "tercile_bin",
        "derivation_details": {
            "operation": "percentile-based tercile binning",
            "thresholds": "p33.33, p66.67",
            "rule": "<=p33 -> 0, p33-p67 -> 1, >p67 -> 2",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "broadband_speed_tier": {
        "description": (
            "Three-class broadband speed tier from tercile binning of "
            "velocidade_mediana_mean (median download speed)."
        ),
        "source_columns": ["velocidade_mediana_mean"],
        "task_type": "classification",
        "n_classes_expected": 3,
        "class_semantics": {
            "0": "Low speed (bottom tercile)",
            "1": "Medium speed (middle tercile)",
            "2": "High speed (top tercile)",
        },
        "derivation_method": "tercile_bin",
        "derivation_details": {
            "operation": "percentile-based tercile binning",
            "thresholds": "p33.33, p66.67",
            "rule": "<=p33 -> 0, p33-p67 -> 1, >p67 -> 2",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "income_tier": {
        "description": (
            "Three-class income tier from tercile binning of "
            "renda_media_mean (average household income)."
        ),
        "source_columns": ["renda_media_mean"],
        "task_type": "classification",
        "n_classes_expected": 3,
        "class_semantics": {
            "0": "Low income (bottom tercile)",
            "1": "Medium income (middle tercile)",
            "2": "High income (top tercile)",
        },
        "derivation_method": "tercile_bin",
        "derivation_details": {
            "operation": "percentile-based tercile binning",
            "thresholds": "p33.33, p66.67",
            "rule": "<=p33 -> 0, p33-p67 -> 1, >p67 -> 2",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "mobile_penetration_tier": {
        "description": (
            "Four-class mobile penetration tier from quartile binning of "
            "Densidade_Telefonia_Movel_2025 (mobile teledensity)."
        ),
        "source_columns": [
            "Densidade_Telefonia Movel_2025",
            "Densidade_Telefonia_Movel_2025",
        ],
        "task_type": "classification",
        "n_classes_expected": 4,
        "class_semantics": {
            "0": "Very low penetration (Q1)",
            "1": "Low penetration (Q2)",
            "2": "Medium penetration (Q3)",
            "3": "High penetration (Q4)",
        },
        "derivation_method": "quartile_bin",
        "derivation_details": {
            "operation": "percentile-based quartile binning",
            "thresholds": "p25, p50, p75",
            "rule": "<=p25 -> 0, p25-p50 -> 1, p50-p75 -> 2, >p75 -> 3",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "infra_density_tier": {
        "description": (
            "Five-class infrastructure density tier from quintile binning "
            "of n_estacoes_smp (number of SMP base stations)."
        ),
        "source_columns": ["n_estacoes_smp"],
        "task_type": "classification",
        "n_classes_expected": 5,
        "class_semantics": {
            "0": "Very low density (Q1)",
            "1": "Low density (Q2)",
            "2": "Medium density (Q3)",
            "3": "High density (Q4)",
            "4": "Very high density (Q5)",
        },
        "derivation_method": "quintile_bin",
        "derivation_details": {
            "operation": "percentile-based quintile binning",
            "thresholds": "p20, p40, p60, p80",
            "rule": "<=p20 -> 0, ... , >p80 -> 4",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
    "road_coverage_4g_tier": {
        "description": (
            "Five-class highway 4G coverage tier from quintile binning "
            "of rod_pct_cob_todas_4g (% of roads covered by all 4G ops)."
        ),
        "source_columns": ["rod_pct_cob_todas_4g"],
        "task_type": "classification",
        "n_classes_expected": 5,
        "class_semantics": {
            "0": "Very low road 4G coverage (Q1)",
            "1": "Low road 4G coverage (Q2)",
            "2": "Medium road 4G coverage (Q3)",
            "3": "High road 4G coverage (Q4)",
            "4": "Very high road 4G coverage (Q5)",
        },
        "derivation_method": "quintile_bin",
        "derivation_details": {
            "operation": "percentile-based quintile binning",
            "thresholds": "p20, p40, p60, p80",
            "rule": "<=p20 -> 0, ... , >p80 -> 4",
        },
        "is_engineered": False,
        "engineering_details": None,
        "tier": "strict",
        "tier_threshold": 0.05,
    },
}


_RELAXED_TARGET_METADATA: Dict[str, Dict[str, Any]] = {
    "income_speed_class": {
        "description": (
            "Income-speed quadrant dominance. Primary: 4-class argmax "
            "over pct_cat_{low,high}_renda_{low,high}_vel columns. "
            "Falls back to binary low-vs-high income dominance if "
            "primary fails >=2% validation."
        ),
        "source_columns": [
            "pct_cat_low_renda_low_vel",
            "pct_cat_low_renda_high_vel",
            "pct_cat_high_renda_low_vel",
            "pct_cat_high_renda_high_vel",
        ],
        "task_type": "classification",
        "n_classes_expected": "4 (primary) or 2 (failsafe)",
        "class_semantics_primary": {
            "0": "Dominant: low income, low speed",
            "1": "Dominant: low income, high speed",
            "2": "Dominant: high income, low speed",
            "3": "Dominant: high income, high speed",
        },
        "class_semantics_failsafe": {
            "0": "Low-income dominant (sum of low-income quadrants > high)",
            "1": "High-income dominant (sum of high-income quadrants > low)",
        },
        "derivation_method": "argmax_over_columns",
        "derivation_details": {
            "operation": (
                "Stack 4 percentage columns into (N,4) matrix, take "
                "argmax per row. The rare high-income-low-speed "
                "quadrant (~0.9%) causes the 4-class version to fail "
                "even >=2%."
            ),
            "primary_rule": "argmax([low_low, low_high, high_low, high_high])",
            "failsafe_rule": (
                "Binary: (high_low + high_high) > (low_low + low_high) -> 1"
            ),
        },
        "is_engineered": False,
        "engineering_details": None,
        "has_failsafe": True,
        "failsafe_description": (
            "Binary collapse: low-income-dominant (0) vs "
            "high-income-dominant (1). Guaranteed balanced."
        ),
        "tier": "relaxed",
        "tier_threshold": 0.02,
    },
    "urban_rural_extremes": {
        "description": (
            "Four-class urbanization with extreme tails. Uses asymmetric "
            "p3/p97 bins to isolate the ~3% of municipalities at each "
            "extreme of the urbanization spectrum."
        ),
        "source_columns": ["pct_urbano"],
        "task_type": "classification",
        "n_classes_expected": 4,
        "class_semantics": {
            "0": "Extreme rural (bottom 3%, <= p3)",
            "1": "Rural to moderate (p3 - p50)",
            "2": "Moderate to urban (p50 - p97)",
            "3": "Extreme urban (top 3%, >= p97)",
        },
        "derivation_method": "extreme_4class_bin",
        "derivation_details": {
            "operation": (
                "Asymmetric 4-class binning using p3, p50, p97 "
                "percentiles. Extreme classes contain ~3% of data "
                "each, deliberately placing the target in the 2-5% "
                "minimum-class-fraction zone."
            ),
            "thresholds": "p3, p50, p97",
            "rule": "<=p3 -> 0, p3-p50 -> 1, p50-p97 -> 2, >=p97 -> 3",
        },
        "is_engineered": True,
        "engineering_details": {
            "reason": (
                "Standard tercile binning produces ~33% per class "
                "(passes strict >=5%). To create a target in the "
                "relaxed 2-5% zone, we use extreme-tail bins at p3 "
                "and p97. This captures genuinely rare fully-rural "
                "and fully-urban municipalities."
            ),
            "technique": "asymmetric_extreme_tail_binning",
            "target_zone": "2-5% minimum class fraction",
            "expected_min_class_frac": "~3%",
        },
        "has_failsafe": True,
        "failsafe_description": (
            "Standard tercile bin on pct_urbano (3 balanced classes, "
            "~33% each). Applied if the extreme-4class version fails "
            "the >=2% check."
        ),
        "tier": "relaxed",
        "tier_threshold": 0.02,
    },
    "income_extremes": {
        "description": (
            "Four-class income distribution with extreme tails. Uses "
            "asymmetric p3/p97 bins to isolate the ~3% of municipalities "
            "at each extreme of the income spectrum."
        ),
        "source_columns": ["renda_media_mean"],
        "task_type": "classification",
        "n_classes_expected": 4,
        "class_semantics": {
            "0": "Extreme low income (bottom 3%, <= p3)",
            "1": "Low to moderate income (p3 - p50)",
            "2": "Moderate to high income (p50 - p97)",
            "3": "Extreme high income (top 3%, >= p97)",
        },
        "derivation_method": "extreme_4class_bin",
        "derivation_details": {
            "operation": (
                "Asymmetric 4-class binning using p3, p50, p97 "
                "percentiles on renda_media_mean."
            ),
            "thresholds": "p3, p50, p97",
            "rule": "<=p3 -> 0, p3-p50 -> 1, p50-p97 -> 2, >=p97 -> 3",
        },
        "is_engineered": True,
        "engineering_details": {
            "reason": (
                "Standard tercile binning on income produces ~33% per "
                "class. Extreme-tail bins capture the rare very-low and "
                "very-high income municipalities relevant for digital "
                "divide analysis."
            ),
            "technique": "asymmetric_extreme_tail_binning",
            "target_zone": "2-5% minimum class fraction",
            "expected_min_class_frac": "~3%",
        },
        "has_failsafe": True,
        "failsafe_description": (
            "Standard tercile bin on renda_media_mean (3 balanced "
            "classes). Applied if extreme-4class fails >=2%."
        ),
        "tier": "relaxed",
        "tier_threshold": 0.02,
    },
    "speed_extremes": {
        "description": (
            "Four-class broadband speed distribution with extreme tails. "
            "Uses asymmetric p3/p97 bins to isolate the ~3% of "
            "municipalities with extremely slow or fast broadband."
        ),
        "source_columns": ["velocidade_mediana_mean"],
        "task_type": "classification",
        "n_classes_expected": 4,
        "class_semantics": {
            "0": "Extreme low speed (bottom 3%, <= p3)",
            "1": "Low to moderate speed (p3 - p50)",
            "2": "Moderate to high speed (p50 - p97)",
            "3": "Extreme high speed (top 3%, >= p97)",
        },
        "derivation_method": "extreme_4class_bin",
        "derivation_details": {
            "operation": (
                "Asymmetric 4-class binning using p3, p50, p97 "
                "percentiles on velocidade_mediana_mean."
            ),
            "thresholds": "p3, p50, p97",
            "rule": "<=p3 -> 0, p3-p50 -> 1, p50-p97 -> 2, >=p97 -> 3",
        },
        "is_engineered": True,
        "engineering_details": {
            "reason": (
                "Standard tercile binning on speed produces ~33% per "
                "class. Extreme-tail bins capture municipalities with "
                "critically underserved or exceptionally fast broadband."
            ),
            "technique": "asymmetric_extreme_tail_binning",
            "target_zone": "2-5% minimum class fraction",
            "expected_min_class_frac": "~3%",
        },
        "has_failsafe": True,
        "failsafe_description": (
            "Standard tercile bin on velocidade_mediana_mean (3 balanced "
            "classes). Applied if extreme-4class fails >=2%."
        ),
        "tier": "relaxed",
        "tier_threshold": 0.02,
    },
    "pop_5g_digital_divide": {
        "description": (
            "Four-class digital divide indicator from cross-tabulation "
            "of population size (median-split) and 5G operator presence. "
            "The rare 'small city with 5G' class (~2.5%) captures an "
            "important digital-divide signal."
        ),
        "source_columns": ["populacao_2025", "att09_any_present_5G"],
        "task_type": "classification",
        "n_classes_expected": 4,
        "class_semantics": {
            "0": "Small population, no 5G",
            "1": "Small population, has 5G (rare)",
            "2": "Large population, no 5G",
            "3": "Large population, has 5G",
        },
        "derivation_method": "cross_tabulation",
        "derivation_details": {
            "operation": (
                "Cross-tabulate population (median-split: small=0, "
                "large=1) with 5G presence (0/1). Label = "
                "is_large*2 + has_5g."
            ),
            "variable_1": "populacao_2025 (median-split -> small/large)",
            "variable_2": "att09_any_present_5G (binary 0/1)",
            "rule": "0=small+no5g, 1=small+5g, 2=large+no5g, 3=large+5g",
        },
        "is_engineered": True,
        "engineering_details": {
            "reason": (
                "5G deployment is concentrated in larger cities. The "
                "cross-tabulation creates a naturally rare class "
                "(small municipalities that nevertheless have 5G, "
                "~2.5%) without artificial threshold manipulation."
            ),
            "technique": "cross_tabulation_of_two_variables",
            "target_zone": "2-5% minimum class fraction",
            "expected_min_class_frac": "~2.5%",
        },
        "has_failsafe": True,
        "failsafe_description": (
            "Binary has_5g (0 vs 1). The ~22% minority class is well "
            "above >=2%. Applied if the 4-class cross-tab fails >=2%."
        ),
        "tier": "relaxed",
        "tier_threshold": 0.02,
    },
}


# ── Strict-tier candidates (target: 10 that pass >=5 %) ─────────────────

def _build_strict_candidates(
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Build the strict-tier classification candidates.

    These are the 10+ candidates expected to comfortably pass the >=5 %
    minimum-class-fraction validation.
    """
    candidates: Dict[str, np.ndarray] = {}

    # ── Binary targets ──────────────────────────────────────────────

    # concentrated_mobile_market: HHI SMP >= 0.25 (Anatel regulatory threshold)
    arr_hhi = _get_column(df, "HHI SMP_2024")
    if arr_hhi is None:
        arr_hhi = _get_column(df, "HHI_SMP_2024")
    if arr_hhi is not None:
        arr_hhi = np.nan_to_num(arr_hhi, nan=0.0)
        candidates["concentrated_mobile_market"] = (
            (arr_hhi >= 0.25).astype(np.int64)
        )
    else:
        print("[derived_targets] WARNING: cannot derive "
              "'concentrated_mobile_market' -- no HHI SMP column found")

    # high_fiber_backhaul: median-split on pct_fibra_backhaul
    arr = _get_column(df, "pct_fibra_backhaul")
    if arr is not None:
        arr = np.nan_to_num(arr, nan=0.0)
        median_val = np.median(arr[arr > 0]) if np.any(arr > 0) else 0.0
        candidates["high_fiber_backhaul"] = (arr >= median_val).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'high_fiber_backhaul'")

    # high_speed_broadband: median-split on pct_agl_alta_velocidade
    arr = _get_column(df, "pct_agl_alta_velocidade")
    if arr is not None:
        arr = np.nan_to_num(arr, nan=0.0)
        median_val = float(np.median(arr))
        candidates["high_speed_broadband"] = (arr > median_val).astype(np.int64)
    else:
        print("[derived_targets] WARNING: cannot derive 'high_speed_broadband'")

    # has_5g_coverage: binary -- any 5G operator present in the municipality
    arr = _get_column(df, "att09_any_present_5G")
    if arr is not None:
        candidates["has_5g_coverage"] = (
            np.nan_to_num(arr, nan=0.0).astype(np.int64)
        )
    else:
        print("[derived_targets] WARNING: cannot derive 'has_5g_coverage'")

    # ── 3-class targets (tercile-binned) ────────────────────────────

    # urbanization_level: from pct_urbano
    arr = _get_column(df, "pct_urbano")
    if arr is not None:
        candidates["urbanization_level"] = _tercile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive 'urbanization_level'")

    # broadband_speed_tier: from velocidade_mediana_mean
    arr = _get_column(df, "velocidade_mediana_mean")
    if arr is not None:
        candidates["broadband_speed_tier"] = _tercile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive 'broadband_speed_tier'")

    # income_tier: from renda_media_mean
    arr = _get_column(df, "renda_media_mean")
    if arr is not None:
        candidates["income_tier"] = _tercile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive 'income_tier'")

    # ── 4-class targets ─────────────────────────────────────────────

    # mobile_penetration_tier: from Densidade_Telefonia_Movel_2025 (quartiles)
    arr = None
    for cand in ["Densidade_Telefonia M\u00f3vel_2025",
                 "Densidade_Telefonia Movel_2025",
                 "Densidade_Telefonia_M\u00f3vel_2025",
                 "Densidade_Telefonia_Movel_2025"]:
        arr = _get_column(df, cand)
        if arr is not None:
            break
    if arr is not None:
        candidates["mobile_penetration_tier"] = _quartile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive "
              "'mobile_penetration_tier'")

    # ── 5-class targets (quintile-binned) ───────────────────────────

    # infra_density_tier: from n_estacoes_smp
    arr = _get_column(df, "n_estacoes_smp")
    if arr is not None:
        candidates["infra_density_tier"] = _quintile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive 'infra_density_tier'")

    # road_coverage_4g_tier: from rod_pct_cob_todas_4g
    arr = _get_column(df, "rod_pct_cob_todas_4g")
    if arr is not None:
        candidates["road_coverage_4g_tier"] = _quintile_bin(
            np.nan_to_num(arr, nan=0.0)
        )
    else:
        print("[derived_targets] WARNING: cannot derive "
              "'road_coverage_4g_tier'")

    return candidates


# ── Relaxed-tier candidates (target: 5 that pass >=2 % but not >=5 %) ──
#
# Each entry is: (primary_labels, failsafe_labels)
# If the primary fails the >=2 % check, the failsafe is used instead.

def _build_relaxed_candidates(
    df: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Build relaxed-tier classification candidates with failsafes.

    Returns a dict of ``{name: (primary_labels, failsafe_labels)}``
    where *failsafe_labels* is a coarser alternative that is virtually
    guaranteed to pass the >=2 % threshold.

    Targets in this tier deliberately have small-but-meaningful extreme
    classes (~2-5 %) to test coreset methods under moderate imbalance.
    """
    cands: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    # ── 1. income_speed_class (4-class argmax -> failsafe: binary) ──
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
        primary = np.argmax(quadrant, axis=1).astype(np.int64)
        low_income_share = quadrant[:, 0] + quadrant[:, 1]
        high_income_share = quadrant[:, 2] + quadrant[:, 3]
        failsafe = (high_income_share > low_income_share).astype(np.int64)
        cands["income_speed_class"] = (primary, failsafe)
    else:
        print("[derived_targets] WARNING: cannot derive 'income_speed_class' "
              "-- missing pct_cat_* columns")

    # ── 2. urban_rural_extremes (extreme-4class on pct_urbano) ──────
    arr = _get_column(df, "pct_urbano")
    if arr is not None:
        arr_clean = np.nan_to_num(arr, nan=0.0)
        primary = _extreme_4class_bin(arr_clean)
        failsafe = _tercile_bin(arr_clean)
        cands["urban_rural_extremes"] = (primary, failsafe)
    else:
        print("[derived_targets] WARNING: cannot derive "
              "'urban_rural_extremes'")

    # ── 3. income_extremes (extreme-4class on renda_media_mean) ─────
    arr = _get_column(df, "renda_media_mean")
    if arr is not None:
        arr_clean = np.nan_to_num(arr, nan=0.0)
        primary = _extreme_4class_bin(arr_clean)
        failsafe = _tercile_bin(arr_clean)
        cands["income_extremes"] = (primary, failsafe)
    else:
        print("[derived_targets] WARNING: cannot derive 'income_extremes'")

    # ── 4. speed_extremes (extreme-4class on velocidade_mediana) ────
    arr = _get_column(df, "velocidade_mediana_mean")
    if arr is not None:
        arr_clean = np.nan_to_num(arr, nan=0.0)
        primary = _extreme_4class_bin(arr_clean)
        failsafe = _tercile_bin(arr_clean)
        cands["speed_extremes"] = (primary, failsafe)
    else:
        print("[derived_targets] WARNING: cannot derive 'speed_extremes'")

    # ── 5. pop_5g_digital_divide (cross-tab: pop x 5G) ─────────────
    arr_pop = _get_column(df, "populacao_2025")
    arr_5g = _get_column(df, "att09_any_present_5G")
    if arr_pop is not None and arr_5g is not None:
        pop = np.nan_to_num(arr_pop, nan=0.0)
        g5 = np.nan_to_num(arr_5g, nan=0.0).astype(np.int64)
        is_large = (pop > np.median(pop)).astype(np.int64)
        primary = (is_large * 2 + g5).astype(np.int64)
        failsafe = g5.copy()
        cands["pop_5g_digital_divide"] = (primary, failsafe)
    else:
        print("[derived_targets] WARNING: cannot derive "
              "'pop_5g_digital_divide'")

    return cands


# ── Main entry point ────────────────────────────────────────────────────

_STRICT_THRESHOLD = 0.05   # >=5 % per class
_RELAXED_THRESHOLD = 0.02  # >=2 % per class


def derive_classification_targets(
    df: pd.DataFrame,
    *,
    metadata_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Derive classification targets from raw DataFrame columns.

    Two tiers of targets are produced:

    - **Strict tier** (>=5 % minimum class fraction): up to 10 targets
      that are safe for downstream evaluation even with small coresets.
    - **Relaxed tier** (>=2 % minimum class fraction): up to 5 additional
      targets with failsafe alternatives.

    Rich JSON metadata is always generated.  If *metadata_path* is given,
    the JSON is written to that file; otherwise it is returned only via
    the second element of the tuple.

    Parameters
    ----------
    df : pd.DataFrame
        The raw municipality-level DataFrame.
    metadata_path : str, optional
        If provided, write the full target metadata JSON to this path.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping ``{target_name: (N,) int64 array}`` for all accepted
        targets from both tiers.
    """
    all_metadata: Dict[str, Any] = {}

    # ── Strict tier ──────────────────────────────────────────────────
    strict_candidates = _build_strict_candidates(df)

    strict_out: Dict[str, np.ndarray] = {}
    for name, labels in strict_candidates.items():
        if _validate_target(name, labels, _STRICT_THRESHOLD):
            strict_out[name] = labels

    n_strict_dropped = len(strict_candidates) - len(strict_out)
    print(
        f"[derived_targets] STRICT tier: {len(strict_out)} accepted "
        f"(>={_STRICT_THRESHOLD:.0%})"
        f"{f', {n_strict_dropped} dropped' if n_strict_dropped else ''}"
    )

    # Build metadata for strict targets
    for name, labels in strict_out.items():
        meta = dict(_STRICT_TARGET_METADATA.get(name, {}))
        meta["status"] = "accepted"
        meta["used_failsafe"] = False
        meta["class_distribution"] = _class_distribution(labels)
        all_metadata[name] = meta

    # Also record dropped strict candidates
    for name in strict_candidates:
        if name not in strict_out:
            labels = strict_candidates[name]
            meta = dict(_STRICT_TARGET_METADATA.get(name, {}))
            meta["status"] = "dropped"
            meta["drop_reason"] = (
                f"Smallest class has {_smallest_class_frac(labels):.1%} "
                f"of data (need >={_STRICT_THRESHOLD:.0%})"
            )
            meta["used_failsafe"] = False
            meta["class_distribution"] = _class_distribution(labels)
            all_metadata[f"DROPPED_strict_{name}"] = meta

    # ── Relaxed tier ─────────────────────────────────────────────────
    relaxed_candidates = _build_relaxed_candidates(df)

    relaxed_out: Dict[str, np.ndarray] = {}
    for name, (primary, failsafe) in relaxed_candidates.items():
        used_failsafe = False
        # Try primary first
        if _validate_target(
            name, primary, _RELAXED_THRESHOLD, quiet=True,
        ):
            relaxed_out[name] = primary
            tag = "primary"
        elif failsafe is not None and _validate_target(
            f"{name} [failsafe]", failsafe, _RELAXED_THRESHOLD, quiet=True,
        ):
            relaxed_out[name] = failsafe
            tag = "failsafe"
            used_failsafe = True
        else:
            # Both primary and failsafe failed
            prim_frac = _smallest_class_frac(primary)
            fs_frac = (
                _smallest_class_frac(failsafe) if failsafe is not None
                else 0.0
            )
            print(
                f"[derived_targets] DROPPED '{name}' (relaxed tier): "
                f"primary min_frac={prim_frac:.1%}, "
                f"failsafe min_frac={fs_frac:.1%} "
                f"(need >={_RELAXED_THRESHOLD:.0%})"
            )
            # Record dropped in metadata
            meta = dict(_RELAXED_TARGET_METADATA.get(name, {}))
            meta["status"] = "dropped"
            meta["drop_reason"] = (
                f"Primary min_frac={prim_frac:.1%}, "
                f"failsafe min_frac={fs_frac:.1%}"
            )
            meta["used_failsafe"] = False
            meta["class_distribution_primary"] = _class_distribution(primary)
            if failsafe is not None:
                meta["class_distribution_failsafe"] = (
                    _class_distribution(failsafe)
                )
            all_metadata[f"DROPPED_relaxed_{name}"] = meta
            continue

        # Check if it also passes strict
        passes_strict = _validate_target(
            name, relaxed_out[name], _STRICT_THRESHOLD, quiet=True,
        )
        tier_label = (
            f"relaxed-{tag} (also passes strict)"
            if passes_strict
            else f"relaxed-{tag}"
        )
        print(f"[derived_targets]   '{name}': accepted as {tier_label}, "
              f"min_frac={_smallest_class_frac(relaxed_out[name]):.1%}")

        # Record in metadata
        meta = dict(_RELAXED_TARGET_METADATA.get(name, {}))
        meta["status"] = "accepted"
        meta["used_failsafe"] = used_failsafe
        meta["accepted_variant"] = tag
        meta["also_passes_strict"] = passes_strict
        meta["class_distribution"] = _class_distribution(relaxed_out[name])
        # Also include the distribution of the variant NOT used
        if used_failsafe:
            meta["class_distribution_primary_rejected"] = (
                _class_distribution(primary)
            )
        elif failsafe is not None:
            meta["class_distribution_failsafe_available"] = (
                _class_distribution(failsafe)
            )
        all_metadata[name] = meta

    n_relaxed_dropped = len(relaxed_candidates) - len(relaxed_out)
    print(
        f"[derived_targets] RELAXED tier: {len(relaxed_out)} accepted "
        f"(>={_RELAXED_THRESHOLD:.0%})"
        f"{f', {n_relaxed_dropped} dropped' if n_relaxed_dropped else ''}"
    )

    # ── Merge both tiers ─────────────────────────────────────────────
    out: Dict[str, np.ndarray] = _AttrDict()
    out.update(strict_out)
    out.update(relaxed_out)

    print(
        f"[derived_targets] TOTAL: {len(out)} classification targets "
        f"({len(strict_out)} strict + {len(relaxed_out)} relaxed)"
    )

    # ── Build and save the summary JSON ─────────────────────────────
    summary: Dict[str, Any] = {
        "_schema_version": "1.0",
        "_description": (
            "Metadata for all classification targets used in downstream "
            "evaluation. Each entry documents how the target was derived, "
            "whether it is engineered or naturally selected, its tier "
            "(strict >=5% or relaxed >=2%), and full class distributions."
        ),
        "n_samples": int(len(df)),
        "strict_threshold": _STRICT_THRESHOLD,
        "relaxed_threshold": _RELAXED_THRESHOLD,
        "n_strict_accepted": len(strict_out),
        "n_relaxed_accepted": len(relaxed_out),
        "n_total_accepted": len(out),
        "strict_target_names": sorted(strict_out.keys()),
        "relaxed_target_names": sorted(relaxed_out.keys()),
        "targets": all_metadata,
    }

    if metadata_path is not None:
        p = Path(metadata_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[derived_targets] Metadata written to {p}")

    # Attach metadata to the output dict as a hidden attribute so
    # callers can access it without changing function signatures
    out._metadata = summary  # type: ignore[attr-defined]

    return out
