"""
Bootstrap target-variable evaluation for coreset quality assessment.

Implements the *variable cross-validation* approach: for each of B bootstrap
draws, n_reg regression + n_cls classification targets are randomly selected
from large pools of eligible columns, then excluded from the feature matrix
before computing Nystrom features and training downstream models.  This
prevents data leakage and avoids arbitrary target selection.

Key components:

1. ``build_target_pools`` — constructs strictly-typed regression (continuous)
   and classification (categorical) target pools from the raw CSV.
2. ``VARIABLE_GROUPS`` — semantic leakage groups for related-variable exclusion.
3. ``get_related_columns`` — given a target column, returns all columns that
   must also be excluded from features to prevent indirect leakage.
4. ``generate_bootstrap_samples`` — B reproducible draws of n_reg reg + n_cls cls.
5. ``build_bootstrap_features`` — removes targets + related columns from X.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Target pool construction
# ============================================================================

# Columns to always exclude from being targets (identifiers, non-substantive)
_EXCLUDE_PREFIXES = (
    "cd_ibge", "codibge", "cod_ibge", "municipio", "uf", "sigla_uf",
    "estado", "regiao", "nome_municipio", "latitude", "longitude",
    "populacao", "pop_", "area_km2", "pib",
)

# Frequency-band column marker (newline-containing names from CSV)
_FREQ_BAND_PATTERN = re.compile(r"Frequ[eê]ncia", re.IGNORECASE)


def _is_substantive_column(col: str) -> bool:
    """Check if a column is substantive (not _missing, not freq-band, not ID)."""
    if col.endswith("_missing"):
        return False
    if _FREQ_BAND_PATTERN.search(col):
        return False
    col_lower = col.lower()
    for prefix in _EXCLUDE_PREFIXES:
        if col_lower.startswith(prefix):
            return False
    return True


def build_target_pools(
    raw_csv_path: str,
    *,
    nan_threshold: float = 0.05,
    min_unique_continuous: int = 25,
    min_class_fraction: float = 0.05,
    max_unique_categorical: int = 10,
    var_threshold: float = 1e-10,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Build STRICTLY TYPED regression + classification target pools.

    Parameters
    ----------
    raw_csv_path : str
        Path to ``data/smp_main.csv``.
    nan_threshold : float
        Maximum fraction of NaN allowed (default 0.05 = 5%).
    min_unique_continuous : int
        Minimum unique values for a column to be considered continuous (>25).
    min_class_fraction : float
        Minimum fraction of samples in any class (categorical pool).
    max_unique_categorical : int
        Maximum unique values for natural categorical columns.
    var_threshold : float
        Minimum variance for continuous columns.

    Returns
    -------
    reg_pool : Dict[str, np.ndarray]
        ``{col_name: (N,) float64}`` — continuous regression targets.
    cls_pool : Dict[str, np.ndarray]
        ``{col_name: (N,) int64}`` — categorical classification targets.
    all_substantive_cols : List[str]
        Ordered list of all substantive column names (for feature matrix).
    """
    print("[bootstrap] Loading raw CSV...", flush=True)
    df = pd.read_csv(raw_csv_path)
    N = len(df)
    print(f"[bootstrap] Loaded {N} rows, {len(df.columns)} columns", flush=True)

    # Filter to substantive columns
    substantive_cols = [c for c in df.columns if _is_substantive_column(c)]
    print(f"[bootstrap] {len(substantive_cols)} substantive columns", flush=True)

    reg_pool: Dict[str, np.ndarray] = {}
    cls_pool: Dict[str, np.ndarray] = {}

    # ── Build regression pool (continuous ONLY) ──
    for col in substantive_cols:
        series = df[col]
        pct_nan = series.isna().mean()
        if pct_nan >= nan_threshold:
            continue

        # Must be numeric
        if not pd.api.types.is_numeric_dtype(series):
            continue

        vals = series.dropna()
        n_unique = vals.nunique()
        if n_unique < min_unique_continuous:
            continue

        variance = vals.var()
        if variance < var_threshold:
            continue

        # Impute remaining NaN with median
        imputed = series.fillna(series.median()).values.astype(np.float64)
        assert np.all(np.isfinite(imputed)), f"Non-finite after impute: {col}"
        reg_pool[col] = imputed

    print(f"[bootstrap] Regression pool: {len(reg_pool)} continuous columns", flush=True)

    # ── Build classification pool ──
    # Part A: natural categorical columns
    n_natural = 0
    for col in substantive_cols:
        series = df[col]
        pct_nan = series.isna().mean()
        if pct_nan >= nan_threshold:
            continue

        if not pd.api.types.is_numeric_dtype(series):
            continue

        vals = series.dropna()
        n_unique = vals.nunique()
        if n_unique < 2 or n_unique > max_unique_categorical:
            continue

        # Check min class fraction
        value_counts = vals.value_counts(normalize=True)
        if value_counts.min() < min_class_fraction:
            continue

        # Already qualifies as a natural categorical — skip if also in reg_pool
        if col in reg_pool:
            continue

        # Impute with mode, convert to int labels
        imputed = series.fillna(series.mode().iloc[0]).values.astype(np.int64)
        cls_pool[col] = imputed
        n_natural += 1

    print(f"[bootstrap] Natural categorical: {n_natural} columns", flush=True)

    # Part B: tercile-binned versions of continuous columns
    n_tercile = 0
    for col, values in reg_pool.items():
        tercile_name = f"{col}__tercile"

        try:
            # Use pd.qcut for balanced terciles
            labels = pd.qcut(values, q=3, labels=[0, 1, 2], duplicates="drop")
            labels = labels.astype(np.int64)
        except (ValueError, TypeError):
            continue

        # Verify min class fraction
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            continue
        fractions = counts / len(labels)
        if fractions.min() < min_class_fraction:
            continue

        cls_pool[tercile_name] = labels
        n_tercile += 1

    print(f"[bootstrap] Tercile-binned: {n_tercile} columns", flush=True)
    print(f"[bootstrap] Classification pool total: {len(cls_pool)} columns", flush=True)

    # Validate no overlap between pool keys
    overlap = set(reg_pool.keys()) & set(cls_pool.keys())
    assert len(overlap) == 0, f"Overlap between reg and cls pools: {overlap}"

    return reg_pool, cls_pool, substantive_cols


# ============================================================================
# Variable leakage groups
# ============================================================================

# Each group maps a descriptive name to a compiled regex.  When a target
# matches any group, ALL columns matching that group's regex are excluded
# from features for that bootstrap draw.

def _build_variable_groups(
    all_cols: List[str],
) -> Dict[str, List[str]]:
    """Pre-compute variable groups from the actual column list.

    Returns a dict mapping each column name to the list of related columns
    (its leakage group).
    """
    # Define group patterns
    GROUP_PATTERNS: List[Tuple[str, re.Pattern]] = [
        # Density per service type (across years)
        ("density_blf", re.compile(
            r"^Densidade_Banda Larga Fixa_\d{4}$", re.IGNORECASE)),
        ("density_movel", re.compile(
            r"^Densidade_Telefonia M.vel_\d{4}$", re.IGNORECASE)),
        ("density_fixa", re.compile(
            r"^Densidade_Telefonia Fixa_\d{4}$", re.IGNORECASE)),
        ("density_tv", re.compile(
            r"^Densidade_TV por Assinatura_\d{4}$", re.IGNORECASE)),
        ("density_scm", re.compile(
            r"^Densidade_SCM_\d{4}$", re.IGNORECASE)),
        # Access per service type (across years)
        ("access_blf", re.compile(
            r"^Acessos_Banda Larga Fixa_\d{4}$", re.IGNORECASE)),
        ("access_movel", re.compile(
            r"^Acessos_Telefonia M.vel_\d{4}$", re.IGNORECASE)),
        ("access_fixa", re.compile(
            r"^Acessos_Telefonia Fixa_\d{4}$", re.IGNORECASE)),
        ("access_tv", re.compile(
            r"^Acessos_TV por Assinatura_\d{4}$", re.IGNORECASE)),
        ("access_scm", re.compile(
            r"^Acessos_SCM_\d{4}$", re.IGNORECASE)),
        # HHI per market (across years)
        ("hhi_smp", re.compile(r"^HHI\s*SMP_\d{4}$", re.IGNORECASE)),
        ("hhi_scm", re.compile(r"^HHI\s*SCM_\d{4}$", re.IGNORECASE)),
        # Speed aggregations
        ("speed_agg", re.compile(
            r"^velocidade_mediana_(mean|std|median|agl_mean)$", re.IGNORECASE)),
        # Income aggregations
        ("income_agg", re.compile(
            r"^renda_media_(mean|std|median)$", re.IGNORECASE)),
        # Base station counts (all technology variants + percentages)
        ("station_counts", re.compile(
            r"^(n_estacoes_(blf|ativas_blf|tf|ativas_tf|geral|ativas_geral|"
            r"smp|4g|5g|2g|3g)|pct_(4g|5g)_smp|n_entidades_smp|"
            r"n_tecnologias_smp)$", re.IGNORECASE)),
        # Road coverage (all operator × technology combinations)
        ("road_coverage", re.compile(r"^rod_pct_cob_", re.IGNORECASE)),
        # Coverage area (metric × tech × operator × date)
        ("coverage_area", re.compile(r"^cov_pct_", re.IGNORECASE)),
        # School connectivity
        ("school_conn", re.compile(
            r"^pct_escolas_(internet|fibra|conect_adequada)$", re.IGNORECASE)),
        # Backhaul infrastructure
        ("backhaul", re.compile(
            r"^(pct_fibra_backhaul|backhaul_fibra_\d{4}|backhaul_ano_fibra|"
            r"n_prestadoras_backhaul)$", re.IGNORECASE)),
        # Urbanization complement
        ("urbanization", re.compile(r"^pct_(urbano|rural)$", re.IGNORECASE)),
        # QoS survey (ISG and components)
        ("qos_survey", re.compile(
            r"^(isg|qf|qic|qcr)(_\w+)?_mean$", re.IGNORECASE)),
        ("qos_responses", re.compile(
            r"^n_respostas(_scm|_movel)?$", re.IGNORECASE)),
        # Category composites
        ("cat_composites", re.compile(
            r"^pct_cat_(low|high)_renda_(low|high)_vel$", re.IGNORECASE)),
        # Fiber indicators (IBC)
        ("ibc_fiber", re.compile(
            r"^ibc_indicador_fibra_", re.IGNORECASE)),
        # Operator presence (att09)
        ("att09_presence", re.compile(r"^att09_", re.IGNORECASE)),
        # Active station percentages
        ("pct_ativas", re.compile(r"^pct_ativas_(blf|tf|geral)$", re.IGNORECASE)),
        # Coverage indicators (primary targets in original pipeline)
        ("cov_area", re.compile(
            r"^cov_(area|hh|res)_(4g|5g|4g_5g|all)$", re.IGNORECASE)),
    ]

    # Cross-family links: when any column in group A is a target,
    # group B columns are also excluded, and vice versa.
    CROSS_LINKS = [
        ("density_blf", "access_blf"),
        ("density_movel", "access_movel"),
        ("density_fixa", "access_fixa"),
        ("density_tv", "access_tv"),
        ("density_scm", "access_scm"),
    ]

    # Build group membership: column -> set of group names
    col_to_groups: Dict[str, Set[str]] = {}
    group_to_cols: Dict[str, List[str]] = {}

    for gname, pattern in GROUP_PATTERNS:
        members = [c for c in all_cols if pattern.search(c)]
        group_to_cols[gname] = members
        for c in members:
            col_to_groups.setdefault(c, set()).add(gname)

    # Expand cross-links into group_to_cols
    cross_link_map: Dict[str, Set[str]] = {}
    for g1, g2 in CROSS_LINKS:
        cross_link_map.setdefault(g1, set()).add(g2)
        cross_link_map.setdefault(g2, set()).add(g1)

    # Build the final related-columns lookup: column -> all related columns
    col_to_related: Dict[str, List[str]] = {}
    for col in all_cols:
        groups = col_to_groups.get(col, set())
        if not groups:
            continue

        related: Set[str] = set()
        # Add all members of the same group(s)
        expanded_groups = set(groups)
        for g in groups:
            if g in cross_link_map:
                expanded_groups |= cross_link_map[g]

        for g in expanded_groups:
            for member in group_to_cols.get(g, []):
                if member != col:
                    related.add(member)

        # Also add tercile-binned versions of related columns
        tercile_related = set()
        for r in related:
            tercile_related.add(f"{r}__tercile")
        related |= tercile_related

        # And tercile of self
        related.add(f"{col}__tercile")

        col_to_related[col] = sorted(related)

    return col_to_related


# Module-level cache for variable groups (built lazily)
_VARIABLE_GROUPS_CACHE: Optional[Dict[str, List[str]]] = None
_VARIABLE_GROUPS_COLS: Optional[List[str]] = None


def get_related_columns(
    target: str,
    all_cols: List[str],
) -> List[str]:
    """Return columns closely related to ``target`` that must also be excluded.

    Uses pre-computed variable groups based on semantic column naming patterns.
    Cross-family links (e.g., Density <-> Access for the same service type)
    are automatically included.

    Parameters
    ----------
    target : str
        The target column name.
    all_cols : List[str]
        All column names in the dataset (for group computation).

    Returns
    -------
    List[str]
        Related columns to exclude from features.
    """
    global _VARIABLE_GROUPS_CACHE, _VARIABLE_GROUPS_COLS

    # Build cache if needed or if columns changed
    if _VARIABLE_GROUPS_CACHE is None or _VARIABLE_GROUPS_COLS != all_cols:
        _VARIABLE_GROUPS_CACHE = _build_variable_groups(all_cols)
        _VARIABLE_GROUPS_COLS = list(all_cols)

    # Handle tercile-binned targets: strip __tercile suffix for lookup
    base_target = target
    if target.endswith("__tercile"):
        base_target = target[:-len("__tercile")]

    related = list(_VARIABLE_GROUPS_CACHE.get(base_target, []))

    # Also add the base column itself if target is a tercile version
    if target.endswith("__tercile") and base_target not in related:
        related.append(base_target)

    return related


# ============================================================================
# Bootstrap sampling
# ============================================================================

@dataclass
class BootstrapSample:
    """A single bootstrap draw of target variables."""
    boot_id: int
    reg_targets: List[str]     # n_reg regression target names
    cls_targets: List[str]     # n_cls classification target names
    excluded_cols: List[str]   # All columns excluded from features


def generate_bootstrap_samples(
    n_bootstrap: int,
    n_reg: int,
    n_cls: int,
    reg_pool: Dict[str, np.ndarray],
    cls_pool: Dict[str, np.ndarray],
    all_feature_cols: List[str],
    *,
    seed: int = 42,
) -> List[BootstrapSample]:
    """Generate B reproducible target subsets (n_reg reg + n_cls cls each).

    TYPE ENFORCEMENT:
    - Draws n_reg targets ONLY from reg_pool (continuous).
    - Draws n_cls targets ONLY from cls_pool (categorical).
    - Validates type after each draw.
    - Enforces non-overlap: if a continuous column is drawn for regression,
      its tercile-binned version is excluded from the classification draw.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations (B).
    n_reg : int
        Number of regression targets per draw.
    n_cls : int
        Number of classification targets per draw.
    reg_pool : Dict[str, np.ndarray]
        Regression pool: ``{col_name: (N,) float64}``.
    cls_pool : Dict[str, np.ndarray]
        Classification pool: ``{col_name: (N,) int64}``.
    all_feature_cols : List[str]
        All substantive column names (for related-variable exclusion).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[BootstrapSample]
        B bootstrap samples with targets and exclusion sets.
    """
    rng = np.random.default_rng(seed)
    reg_names = sorted(reg_pool.keys())
    cls_names = sorted(cls_pool.keys())

    assert len(reg_names) >= n_reg, (
        f"Not enough regression candidates: {len(reg_names)} < {n_reg}"
    )
    assert len(cls_names) >= n_cls, (
        f"Not enough classification candidates: {len(cls_names)} < {n_cls}"
    )

    samples: List[BootstrapSample] = []

    for b in range(n_bootstrap):
        # Draw regression targets
        reg_idx = rng.choice(len(reg_names), size=n_reg, replace=False)
        selected_reg = [reg_names[i] for i in reg_idx]

        # Build set of cls names to exclude (non-overlap rule)
        cls_exclude = set()
        for r in selected_reg:
            # If this continuous column has a tercile version, exclude it
            tercile_name = f"{r}__tercile"
            cls_exclude.add(tercile_name)

        # Eligible classification targets
        eligible_cls = [c for c in cls_names if c not in cls_exclude]
        assert len(eligible_cls) >= n_cls, (
            f"Bootstrap {b}: not enough classification candidates after "
            f"non-overlap exclusion: {len(eligible_cls)} < {n_cls}"
        )

        cls_idx = rng.choice(len(eligible_cls), size=n_cls, replace=False)
        selected_cls = [eligible_cls[i] for i in cls_idx]

        # ── Type enforcement assertions ──
        for t in selected_reg:
            vals = reg_pool[t]
            n_unique = len(np.unique(vals))
            assert n_unique > 10, (
                f"Regression target '{t}' has only {n_unique} unique values"
            )
            assert vals.dtype == np.float64, (
                f"Regression target '{t}' has dtype {vals.dtype}, expected float64"
            )

        for t in selected_cls:
            vals = cls_pool[t]
            n_unique = len(np.unique(vals))
            assert n_unique <= 20, (
                f"Classification target '{t}' has {n_unique} unique values (>20)"
            )

        # ── Build exclusion set: targets + related columns ──
        all_targets = selected_reg + selected_cls
        exclusion_set: Set[str] = set(all_targets)

        for t in all_targets:
            related = get_related_columns(t, all_feature_cols)
            exclusion_set.update(related)

        samples.append(BootstrapSample(
            boot_id=b,
            reg_targets=selected_reg,
            cls_targets=selected_cls,
            excluded_cols=sorted(exclusion_set),
        ))

    return samples


# ============================================================================
# Feature matrix construction per bootstrap draw
# ============================================================================

def build_bootstrap_features(
    X_full: np.ndarray,
    all_col_names: List[str],
    excluded_cols: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Remove targets + related columns from the full feature matrix.

    Parameters
    ----------
    X_full : np.ndarray
        Full standardized feature matrix (N, D_full).
    all_col_names : List[str]
        Column names corresponding to columns of X_full.
    excluded_cols : Sequence[str]
        Column names to exclude (targets + related columns).

    Returns
    -------
    X_eval : np.ndarray
        Feature matrix with excluded columns removed, (N, D').
    kept_names : List[str]
        Column names that were retained.
    """
    assert len(all_col_names) == X_full.shape[1], (
        f"Column count mismatch: {len(all_col_names)} vs {X_full.shape[1]}"
    )

    exclude_set = set(excluded_cols)
    keep_mask = [name not in exclude_set for name in all_col_names]
    kept_names = [name for name, keep in zip(all_col_names, keep_mask) if keep]

    n_excluded = sum(1 for k in keep_mask if not k)
    if n_excluded == 0:
        return X_full, list(all_col_names)

    X_eval = X_full[:, keep_mask]
    return X_eval, kept_names
