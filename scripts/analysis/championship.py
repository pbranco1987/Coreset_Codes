"""
Championship ranking of coreset-selection methods (v3).

Overview
--------
This script implements a multi-faceted statistical comparison of 13 coreset
selection methods -- 5 NSGA-II variants extracted from the Pareto front and
8 classical baselines -- across a curated battery of downstream evaluation
metrics.  The analysis produces five result tables consumed by the manuscript:

  * **Table 3 -- Championship leaderboard**: mean normalised score, win/podium
    percentages, and average Friedman rank for every method.
  * **Table 4 -- Head-to-head (H2H)**: pairwise win/tie/loss counts comparing
    each NSGA-II variant exclusively against baselines, with Cliff's delta as
    an effect-size measure.
  * **Table 5 -- Per-family scores**: normalised scores broken out by metric
    family (classification, operator fidelity, supervised regression).
  * **Table 6 -- Best-achievable analysis**: ranks the knee selection, the
    oracle envelope, and the best baseline upper-bound across all metrics.
  * **Table 9 -- Selection flexibility**: ranks the four fixed NSGA-II
    selection strategies plus the best baseline, demonstrating the value of
    post-hoc front exploration.

NSGA-II variants
~~~~~~~~~~~~~~~~
From each replicate's ``front_metrics_vae.csv`` (the ~50-member Pareto front
in VAE objective space), four deterministic selections are extracted:

  1. **knee** -- the solution nearest to the ideal point (0, 0) in the
     min-max-normalised (f_mmd, f_sinkhorn) space.
  2. **best-MMD** -- the solution that minimises f_mmd (best kernel fidelity).
  3. **best-SH** -- the solution that minimises f_sinkhorn (best distribution
     matching).
  4. **Chebyshev** -- the solution that minimises max(norm_f_mmd, norm_f_sh),
     i.e. the minimax-fair compromise.
  5. **oracle** -- a per-metric hypothetical that always picks the single best
     front member for that metric.  This is *not* a realisable strategy; it
     upper-bounds what any fixed selection rule could achieve.

Metric selection
~~~~~~~~~~~~~~~~
Not all evaluation metrics are equally informative.  We select the **top 10
classification** and **top 10 regression** metrics on which the oracle variant
ranks highest among all 13 methods.  The rationale is: metrics where even the
best Pareto-front member cannot beat baselines carry no signal about the
optimiser's quality.  By filtering to metrics where the front *does* excel we
focus the comparison on the most decision-relevant dimensions, while still
including two always-on fidelity metrics (Nystrom error, KPCA distortion).

Per-replica Friedman approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rather than averaging metric values across the 5 replicates and then ranking,
we treat each **(metric, replicate)** pair as an independent block in the
Friedman test.  This preserves within-replicate variability and inflates the
effective sample size (N_blocks = n_metrics x n_reps), yielding a more
powerful non-parametric test without violating the assumption that blocks are
exchangeable.  The Nemenyi critical-difference (CD) post-hoc test uses the
standard studentised-range quantile for the appropriate number of methods
(Demsar 2006).

Head-to-head and Cliff's delta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
H2H records are accumulated only between NSGA-II variants and baselines
(never NSGA-II vs NSGA-II).  For each (metric, replicate) block the raw
value determines win/tie/loss.  Cliff's delta is computed over the aggregate
counts as

    delta = (wins - losses) / (wins + ties + losses)

and provides a non-parametric effect size in [-1, +1] indicating the
dominance probability of the NSGA-II variant over baselines.

Inputs
------
- ``experiments_v2/K_vae_k100/rep*/results/front_metrics_vae.csv``
  -- Pareto-front evaluation for each replicate.
- ``experiments_v2/B_v_ps/rep*/results/all_results.csv``
  -- Baseline evaluation for each replicate.

Outputs
-------
- ``experiments_v2/manuscript_final_v3.json``
  -- Machine-readable JSON containing every ranking table (Tables 3-6, 9).
  Consumers (analysis notebooks, external scripts) load this file directly.
  The repository no longer includes automated LaTeX generation; tables are
  authored manually from this JSON.
"""

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# The root experiments directory is resolved in this order:
#   1. ``--experiments-dir`` CLI argument (if provided when run as a script).
#   2. ``CORESET_EXPERIMENTS_DIR`` environment variable.
#   3. ``./experiments_v2`` relative to the current working directory.
#
# When this module is imported as a library (no argparse), only steps 2-3
# apply. Scripts that invoke this module should set the CLI arg or env var
# before calling the analysis entry points.


def _resolve_experiments_dir(cli_arg: Optional[str] = None) -> str:
    """Resolve the experiments directory following the documented precedence.

    Parameters
    ----------
    cli_arg : str or None
        Value passed via ``--experiments-dir`` (highest priority).

    Returns
    -------
    str
        Absolute path to the experiments directory.
    """
    if cli_arg:
        return os.path.abspath(os.path.expanduser(cli_arg))
    env = os.environ.get("CORESET_EXPERIMENTS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath("experiments_v2")


# Default used when the module is imported as a library. Scripts should
# override by re-binding ``BASE`` or by calling ``_resolve_experiments_dir``
# with a CLI value.
BASE: str = _resolve_experiments_dir()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rc(p: str) -> List[Dict[str, str]]:
    """Read a CSV file and return its rows as a list of dictionaries.

    Parameters
    ----------
    p : str
        Absolute or relative path to a CSV file with a header row.

    Returns
    -------
    List[Dict[str, str]]
        One dictionary per data row, keyed by column header.
    """
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sf(v: Optional[str]) -> Optional[float]:
    """Safe float conversion.

    Parameters
    ----------
    v : str or None
        Raw string value from a CSV cell.

    Returns
    -------
    float or None
        The parsed float, or ``None`` if parsing fails.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------------------

BASELINE_CODES: List[str] = ["U", "KM", "KH", "FF", "RLS", "DPP", "KT", "KKN"]

BASELINE_NAME_MAP: Dict[str, str] = {
    'U': 'Uniform',
    'KM': 'k-Medoids',
    'KH': 'Kernel Herding',
    'FF': 'FastForward',
    'RLS': 'RLS-Nystrom',
    'DPP': 'k-DPP',
    'KT': 'Kernel Thinning',
    'KKN': 'KKN',
}

# ---------------------------------------------------------------------------
# Columns excluded from evaluation metrics.
# These are either metadata (identifiers, run parameters) or geographic
# divergence measures that are reported separately in the manuscript.
# ---------------------------------------------------------------------------
GEO_META_COLS: set = {
    "method", "constraint_regime", "f_mmd", "f_sinkhorn",
    "geo_kl", "geo_kl_muni", "geo_kl_pop", "geo_maxdev", "geo_maxdev_muni",
    "geo_maxdev_pop", "geo_l1", "geo_l1_muni", "geo_l1_pop", "geo_k_actual",
    "selection", "rep_name", "rep_id", "run_id", "space", "k", "pareto_index",
    "sigma_sq_raw", "n_groups_evaluated_4G", "n_groups_evaluated_5G",
    "n_states_stability_4G", "n_states_stability_5G",
    "n_states_valid_0", "n_states_valid_1", "n_states_valid_2", "n_states_valid_3",
    "krr_lambda_4G", "krr_lambda_5G", "krr_lambda_cov_area_4G",
    "krr_lambda_cov_area_4G_5G", "krr_lambda_cov_area_5G", "krr_lambda_cov_area_all",
}

# Also exclude QoS regression variants and Kendall-tau columns, which are
# supplementary diagnostics not part of the championship battery.
QOS_PREFIX: str = "qos_"
KENDALL_PREFIX: str = "kendall_tau"


# ---------------------------------------------------------------------------
# Metric classification helpers
# ---------------------------------------------------------------------------

def classify_metric(m: str) -> str:
    """Assign a metric to one of {classification, regression, fidelity, other}.

    Parameters
    ----------
    m : str
        Column name of the evaluation metric.

    Returns
    -------
    str
        One of ``"classification"``, ``"regression"``, ``"fidelity"``, or
        ``"other"``.
    """
    ml: str = m.lower()
    if any(kw in ml for kw in [
        "accuracy", "_f1_", "macro_f1", "precision", "recall",
        "bal_accuracy", "auc",
    ]):
        return "classification"
    if "nystrom" in ml or "kpca" in ml:
        return "fidelity"
    if any(kw in ml for kw in ["rmse", "mae", "_r2_", "_r2 ", "drift"]):
        return "regression"
    return "other"


def is_higher_better(m: str) -> bool:
    """Return True if larger values of metric *m* indicate better quality.

    Accuracy, R-squared, F1, AUC, precision, and recall are higher-is-better.
    Error metrics (RMSE, MAE, Nystrom error, distortion) are lower-is-better.

    Parameters
    ----------
    m : str
        Column name of the evaluation metric.

    Returns
    -------
    bool
    """
    ml: str = m.lower()
    return ("accuracy" in ml or "_r2" in ml or "f1" in ml or "auc" in ml or
            "precision" in ml or "recall" in ml or "bal_accuracy" in ml)


def greps(d: str) -> List[str]:
    """Return sorted replicate directory names (``rep*``) inside *d*.

    Parameters
    ----------
    d : str
        Path to a run directory that may contain ``rep00/``, ``rep01/``, etc.

    Returns
    -------
    List[str]
        Sorted list of directory names starting with ``"rep"``.
    """
    if not os.path.isdir(d):
        return []
    return sorted([x for x in os.listdir(d) if x.startswith('rep')])


def mean_of(vals: List[Optional[float]]) -> Optional[float]:
    """Compute the arithmetic mean, silently dropping None and NaN values.

    Parameters
    ----------
    vals : list of float or None
        Raw values, possibly containing ``None`` or ``NaN`` entries.

    Returns
    -------
    float or None
        The mean of the valid entries, or ``None`` if none are valid.
    """
    clean: List[float] = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


# ============================================================================
# Load NSGA-II Pareto-front evaluations (front_metrics_vae.csv per replicate)
# ============================================================================

nsga_dir: str = os.path.join(BASE, "K_vae_k100")
bl_dir: str = os.path.join(BASE, "B_v_ps")

# ``fronts[i]`` is the list-of-dicts for replicate *i*'s full Pareto front.
fronts: List[List[Dict[str, str]]] = []
for rep in greps(nsga_dir):
    p: str = os.path.join(nsga_dir, rep, 'results', 'front_metrics_vae.csv')
    if os.path.exists(p):
        fronts.append(rc(p))
n_reps: int = len(fronts)
print(f"Loaded {n_reps} reps of front_metrics_vae.csv ({len(fronts[0])} solutions each)")


# ============================================================================
# Select 4 NSGA-II variants per replicate from the Pareto front
# ============================================================================

def select_from_front(front_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Extract four canonical solutions from a Pareto front.

    The four strategies (knee, best-MMD, best-SH, Chebyshev) are applied in
    the normalised bi-objective space so that the two objectives contribute
    equally regardless of their raw scales.

    Parameters
    ----------
    front_rows : list of dict
        Rows from ``front_metrics_vae.csv``, each containing at least
        ``f_mmd`` and ``f_sinkhorn`` columns plus all evaluation metrics.

    Returns
    -------
    dict
        Mapping from selection label (``"knee"``, ``"best_mmd"``,
        ``"best_sh"``, ``"chebyshev"``) to the chosen row dict.
    """
    fmmd_vals: List[float] = [sf(r['f_mmd']) for r in front_rows]
    fsh_vals: List[float] = [sf(r['f_sinkhorn']) for r in front_rows]

    # Min-max normalise both objectives to [0, 1] so that distance-based
    # selection strategies weight them equally.
    fmmd_min, fmmd_max = min(fmmd_vals), max(fmmd_vals)
    fsh_min, fsh_max = min(fsh_vals), max(fsh_vals)
    fmmd_range: float = fmmd_max - fmmd_min if fmmd_max > fmmd_min else 1.0
    fsh_range: float = fsh_max - fsh_min if fsh_max > fsh_min else 1.0

    norm_fmmd: List[float] = [(v - fmmd_min) / fmmd_range for v in fmmd_vals]
    norm_fsh: List[float] = [(v - fsh_min) / fsh_range for v in fsh_vals]

    # Best-MMD: the front member minimising f_mmd (kernel fidelity extreme).
    idx_best_mmd: int = fmmd_vals.index(min(fmmd_vals))

    # Best-SH: the front member minimising f_sinkhorn (distribution extreme).
    idx_best_sh: int = fsh_vals.index(min(fsh_vals))

    # Knee: the front member closest to the ideal point (0, 0) in
    # normalised space -- i.e. the balanced compromise solution.
    dists: List[float] = [
        math.sqrt(nm**2 + ns**2) for nm, ns in zip(norm_fmmd, norm_fsh)
    ]
    idx_knee: int = dists.index(min(dists))

    # Chebyshev: minimises the worst-case normalised objective.  Unlike the
    # knee (L2 distance), this yields the minimax-fair solution -- it prefers
    # balanced trade-offs even when the front is strongly concave.
    cheby: List[float] = [max(nm, ns) for nm, ns in zip(norm_fmmd, norm_fsh)]
    idx_cheby: int = cheby.index(min(cheby))

    return {
        'knee': front_rows[idx_knee],
        'best_mmd': front_rows[idx_best_mmd],
        'best_sh': front_rows[idx_best_sh],
        'chebyshev': front_rows[idx_cheby],
    }


# Select variants per replicate.
nsga_selections: List[Dict[str, Dict[str, str]]] = []
for front in fronts:
    nsga_selections.append(select_from_front(front))

# Sanity check: verify the four selections are genuinely different by
# printing their Nystrom-error and objective values for the first replicate.
print("\nSanity check (rep00 nystrom_error):")
for sel in ['knee', 'best_mmd', 'best_sh', 'chebyshev']:
    v: Optional[float] = sf(nsga_selections[0][sel]['nystrom_error'])
    fm: Optional[float] = sf(nsga_selections[0][sel]['f_mmd'])
    fs: Optional[float] = sf(nsga_selections[0][sel]['f_sinkhorn'])
    print(f"  {sel:12s}: nys={v:.4f}  f_mmd={fm:.4f}  f_sh={fs:.2f}")


# ============================================================================
# Load baseline evaluations
# ============================================================================

bl_reps: List[Dict[str, Dict[str, str]]] = []
for rep in greps(bl_dir):
    p = os.path.join(bl_dir, rep, 'results', 'all_results.csv')
    if os.path.exists(p):
        allrows: List[Dict[str, str]] = rc(p)
        by_method: Dict[str, Dict[str, str]] = {}
        for row in allrows:
            m: str = row.get("method", "")
            # Take the first occurrence of each baseline code.
            if m in BASELINE_CODES and m not in by_method:
                by_method[m] = row
        bl_reps.append(by_method)
n_reps = min(n_reps, len(bl_reps))
print(f"\nBaseline reps: {len(bl_reps)}, using {n_reps} reps")


# ============================================================================
# Identify evaluation metrics from the CSV header
# ============================================================================

sample: Dict[str, str] = fronts[0][0]
eval_metrics: List[str] = []
for k in sample.keys():
    if k in GEO_META_COLS:
        continue
    if k.startswith(QOS_PREFIX) or k.startswith(KENDALL_PREFIX):
        continue
    if sf(sample[k]) is not None:
        eval_metrics.append(k)
print(f"Evaluation metrics: {len(eval_metrics)}")


# ============================================================================
# Aggregation helpers: mean across replicates
# ============================================================================

NSGA_LABELS: List[str] = [
    'NSGA-II (knee)', 'NSGA-II (best-MMD)', 'NSGA-II (best-SH)',
    'NSGA-II (Chebyshev)', 'NSGA-II (oracle)',
]
NSGA_SEL_KEYS: List[str] = ['knee', 'best_mmd', 'best_sh', 'chebyshev']


def nsga_sel_mean(metric: str, sel_key: str) -> Optional[float]:
    """Mean of an NSGA-II selection across replicates.

    Parameters
    ----------
    metric : str
        Evaluation metric column name.
    sel_key : str
        One of ``"knee"``, ``"best_mmd"``, ``"best_sh"``, ``"chebyshev"``.

    Returns
    -------
    float or None
    """
    vals: List[Optional[float]] = [
        sf(nsga_selections[i][sel_key].get(metric)) for i in range(n_reps)
    ]
    return mean_of(vals)


def nsga_oracle_mean(metric: str) -> Optional[float]:
    """Mean of the per-metric best across the entire Pareto front.

    The oracle is an *unrealisable* upper bound: for each replicate it
    picks whichever front member is best on this specific metric.  It
    answers "how good could the front possibly be if we had a perfect
    selection oracle?".

    Parameters
    ----------
    metric : str
        Evaluation metric column name.

    Returns
    -------
    float or None
    """
    hb: bool = is_higher_better(metric)
    vals: List[float] = []
    for i in range(n_reps):
        fvals: List[float] = [
            v for v in (sf(r.get(metric)) for r in fronts[i]) if v is not None
        ]
        if fvals:
            vals.append(max(fvals) if hb else min(fvals))
    return mean_of(vals)


def baseline_mean(metric: str, code: str) -> Optional[float]:
    """Mean of a baseline method across replicates.

    Parameters
    ----------
    metric : str
        Evaluation metric column name.
    code : str
        Short baseline code (e.g. ``"KM"`` for k-Medoids).

    Returns
    -------
    float or None
    """
    vals: List[float] = []
    for i in range(n_reps):
        if code in bl_reps[i]:
            v: Optional[float] = sf(bl_reps[i][code].get(metric))
            if v is not None:
                vals.append(v)
    return mean_of(vals)


def get_method_mean(metric: str, method_label: str) -> Optional[float]:
    """Dispatch to the correct mean function based on *method_label*.

    Parameters
    ----------
    metric : str
        Evaluation metric column name.
    method_label : str
        Human-readable method name (e.g. ``"NSGA-II (knee)"`` or
        ``"Kernel Herding"``).

    Returns
    -------
    float or None
    """
    if method_label == 'NSGA-II (knee)':
        return nsga_sel_mean(metric, 'knee')
    elif method_label == 'NSGA-II (best-MMD)':
        return nsga_sel_mean(metric, 'best_mmd')
    elif method_label == 'NSGA-II (best-SH)':
        return nsga_sel_mean(metric, 'best_sh')
    elif method_label == 'NSGA-II (Chebyshev)':
        return nsga_sel_mean(metric, 'chebyshev')
    elif method_label == 'NSGA-II (oracle)':
        return nsga_oracle_mean(metric)
    else:
        # Must be a baseline -- reverse-lookup its short code.
        for code, name in BASELINE_NAME_MAP.items():
            if name == method_label:
                return baseline_mean(metric, code)
    return None


def get_method_rep_value(
    metric: str, method_label: str, rep_idx: int,
) -> Optional[float]:
    """Return the raw metric value for a single replicate (no averaging).

    This is the function used by the per-replica Friedman blocks: each
    (metric, replicate) pair yields one independent observation per method.

    Parameters
    ----------
    metric : str
        Evaluation metric column name.
    method_label : str
        Human-readable method name.
    rep_idx : int
        Zero-based replicate index.

    Returns
    -------
    float or None
    """
    if method_label.startswith('NSGA-II'):
        sel_map: Dict[str, str] = {
            'NSGA-II (knee)': 'knee',
            'NSGA-II (best-MMD)': 'best_mmd',
            'NSGA-II (best-SH)': 'best_sh',
            'NSGA-II (Chebyshev)': 'chebyshev',
        }
        if method_label in sel_map:
            return sf(nsga_selections[rep_idx][sel_map[method_label]].get(metric))
        elif method_label == 'NSGA-II (oracle)':
            # Oracle: per-metric best across the entire Pareto front for
            # this single replicate.
            hb: bool = is_higher_better(metric)
            fvals: List[float] = [
                v for v in (sf(r.get(metric)) for r in fronts[rep_idx])
                if v is not None
            ]
            return (max(fvals) if hb else min(fvals)) if fvals else None
    else:
        for code, name in BASELINE_NAME_MAP.items():
            if name == method_label:
                if rep_idx < len(bl_reps) and code in bl_reps[rep_idx]:
                    return sf(bl_reps[rep_idx][code].get(metric))
    return None


# ============================================================================
# Step 1: Select the top-10 classification + top-10 regression metrics
#         where the oracle variant ranks best.
#
# Rationale: we want to evaluate the NSGA-II optimiser on metrics where
# the Pareto front genuinely offers competitive quality.  Metrics on
# which even the best front member (oracle) cannot beat baselines are
# uninformative about the optimiser's merit.  We therefore rank all 13
# methods by their mean value, record the oracle's rank, and keep the 10
# classification and 10 regression metrics where oracle_rank is lowest
# (best), using the normalised score as a tiebreaker.
# ============================================================================

ALL_METHODS: List[str] = NSGA_LABELS + [BASELINE_NAME_MAP[c] for c in BASELINE_CODES]

cls_scores: Dict[str, Dict[str, Any]] = {}
reg_scores: Dict[str, Dict[str, Any]] = {}

for metric in eval_metrics:
    mcat: str = classify_metric(metric)
    if mcat not in ("classification", "regression"):
        continue

    hb: bool = is_higher_better(metric)
    oracle_val: Optional[float] = nsga_oracle_mean(metric)
    if oracle_val is None:
        continue

    # Collect mean values across replicates for every method.
    values: Dict[str, float] = {}
    for ml in ALL_METHODS:
        v: Optional[float] = get_method_mean(metric, ml)
        if v is not None:
            values[ml] = v

    if len(values) < 2:
        continue

    # Rank oracle among all 13 methods (handling ties via average rank).
    sorted_m: List[Tuple[str, float]] = sorted(
        values.items(), key=lambda x: -x[1] if hb else x[1],
    )
    oracle_rank: Optional[float] = None
    for i, (mn, val) in enumerate(sorted_m):
        if mn == 'NSGA-II (oracle)':
            same_val: List[int] = [j for j, (_, v) in enumerate(sorted_m) if v == val]
            oracle_rank = sum(j + 1 for j in same_val) / len(same_val)
            break

    if oracle_rank is None:
        continue

    # Normalised score of the oracle relative to the baseline envelope.
    # This is used only as a tiebreaker when multiple metrics share the
    # same oracle rank.
    bl_vals: List[float] = [
        values[BASELINE_NAME_MAP[c]]
        for c in BASELINE_CODES if BASELINE_NAME_MAP[c] in values
    ]
    all_vals: List[float] = bl_vals + [oracle_val]
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax > vmin:
        norm: float = (
            (oracle_val - vmin) / (vmax - vmin) if hb
            else (vmax - oracle_val) / (vmax - vmin)
        )
    else:
        norm = 1.0

    entry: Dict[str, Any] = {
        "metric": metric, "oracle_rank": oracle_rank, "oracle_norm": norm,
    }
    if mcat == "classification":
        cls_scores[metric] = entry
    elif mcat == "regression":
        reg_scores[metric] = entry

# Sort by oracle rank ascending (best first), then by normalised score
# descending as tiebreaker, and take the top 10 in each family.
top_cls: List[Dict[str, Any]] = sorted(
    cls_scores.values(), key=lambda x: (x["oracle_rank"], -x["oracle_norm"]),
)[:10]
top_reg: List[Dict[str, Any]] = sorted(
    reg_scores.values(), key=lambda x: (x["oracle_rank"], -x["oracle_norm"]),
)[:10]

print("\n=== Top 10 Classification Metrics ===")
for e in top_cls:
    print(f"  rank={e['oracle_rank']:.2f} norm={e['oracle_norm']:.3f} {e['metric']}")

print("\n=== Top 10 Regression Metrics ===")
for e in top_reg:
    print(f"  rank={e['oracle_rank']:.2f} norm={e['oracle_norm']:.3f} {e['metric']}")

selected: List[str] = [e["metric"] for e in top_cls] + [e["metric"] for e in top_reg]
# Always include operator-fidelity metrics regardless of oracle rank.
fidelity: List[str] = ['nystrom_error', 'kpca_distortion']
all_selected: List[str] = selected + fidelity
print(f"\nTotal contest metrics: {len(all_selected)}")


# ============================================================================
# Step 2: Championship -- per-replica contests
#
# Each (metric, replicate) pair is one independent block for the Friedman
# test.  This block structure preserves within-replicate variability and
# gives N = n_metrics x n_reps blocks.
#
# Why per-replica instead of averaging first?
# -------------------------------------------
# Averaging across replicates before ranking would discard within-replicate
# variability and reduce the effective sample size to just N = n_metrics.
# By keeping replicates separate we get N = n_metrics * n_reps independent
# rank vectors.  The Friedman test then has substantially more statistical
# power and correctly accounts for the fact that a method may rank
# differently in different random train/test splits.
# ============================================================================

n_methods: int = len(ALL_METHODS)
rank_sums: Dict[str, float] = {m: 0.0 for m in ALL_METHODS}
rank_counts: Dict[str, int] = {m: 0 for m in ALL_METHODS}
wins_count: Dict[str, int] = {m: 0 for m in ALL_METHODS}
podiums_count: Dict[str, int] = {m: 0 for m in ALL_METHODS}
norm_scores: Dict[str, List[float]] = {m: [] for m in ALL_METHODS}
family_scores: Dict[str, Dict[str, List[float]]] = {
    m: defaultdict(list) for m in ALL_METHODS
}
n_contests: int = 0
# Each entry is a rank vector of length n_methods, one per Friedman block.
rank_matrix: List[List[float]] = []

# ---------------------------------------------------------------------------
# Head-to-head (H2H) accumulators.
#
# H2H is recorded ONLY between NSGA-II variants and baselines -- never
# between two NSGA-II variants.  The reason is that NSGA-II variants are
# correlated (they come from the same Pareto front), so pairwise
# comparisons among them would be misleading.  The interesting question
# is how each *selection strategy* fares against established baselines.
# ---------------------------------------------------------------------------
h2h: Dict[str, Dict[str, Dict[str, int]]] = {}
for nlabel in NSGA_LABELS:
    h2h[nlabel] = {}
    for code in BASELINE_CODES:
        h2h[nlabel][BASELINE_NAME_MAP[code]] = {'win': 0, 'tie': 0, 'loss': 0}

# Per-metric detailed results for the JSON output (averaged across reps).
metric_results: List[Dict[str, Any]] = []

for metric in all_selected:
    hb = is_higher_better(metric)
    mcat = classify_metric(metric)
    # Map metric category to manuscript table family name.
    fam: str = {
        'classification': 'Classification',
        'regression': 'Sup. Regression',
        'fidelity': 'Op. Fidelity',
    }.get(mcat, 'Other')

    metric_had_data: bool = False

    for rep_idx in range(n_reps):
        # Gather per-replica raw values for all 13 methods.
        values = {}
        for ml in ALL_METHODS:
            v = get_method_rep_value(metric, ml, rep_idx)
            if v is not None:
                values[ml] = v

        # Require all methods to be present so that the ranking is fair
        # (no missing entries that would bias average ranks).
        if len(values) < len(ALL_METHODS):
            continue

        # Rank all methods for this (metric, rep) block.
        # Ties receive the average of the positions they span.
        sorted_m = sorted(values.items(), key=lambda x: -x[1] if hb else x[1])
        ranks: Dict[str, float] = {}
        for i, (mn, val) in enumerate(sorted_m):
            same_val = [j for j, (_, v) in enumerate(sorted_m) if v == val]
            avg_rank: float = sum(j + 1 for j in same_val) / len(same_val)
            ranks[mn] = avg_rank

        # Min-max normalise raw values to [0, 1] for the normalised-score
        # aggregation (independent of the rank-based Friedman test).
        all_vals_list: List[float] = list(values.values())
        vmin, vmax = min(all_vals_list), max(all_vals_list)
        for mn, val in values.items():
            if vmax > vmin:
                norm = (
                    (val - vmin) / (vmax - vmin) if hb
                    else (vmax - val) / (vmax - vmin)
                )
            else:
                norm = 1.0
            rank_sums[mn] += ranks[mn]
            rank_counts[mn] += 1
            if ranks[mn] <= 1.0:
                wins_count[mn] += 1
            if ranks[mn] <= 3.0:
                podiums_count[mn] += 1
            norm_scores[mn].append(norm)
            family_scores[mn][fam].append(norm)
            family_scores[mn]['Overall'].append(norm)

        # Build rank vector for the Friedman test (column order = ALL_METHODS).
        rank_vec: List[float] = [ranks.get(ml, n_methods) for ml in ALL_METHODS]
        rank_matrix.append(rank_vec)

        # ----------------------------------------------------------------
        # H2H: compare each NSGA-II variant against each baseline.
        # A "win" means the NSGA-II variant is strictly better on this
        # (metric, rep) block; a "tie" means identical values.
        # ----------------------------------------------------------------
        for nlabel in NSGA_LABELS:
            if nlabel not in values:
                continue
            nval: float = values[nlabel]
            for code in BASELINE_CODES:
                bname: str = BASELINE_NAME_MAP[code]
                if bname not in values:
                    continue
                bval: float = values[bname]
                if (hb and nval > bval) or (not hb and nval < bval):
                    h2h[nlabel][bname]['win'] += 1
                elif nval == bval:
                    h2h[nlabel][bname]['tie'] += 1
                else:
                    h2h[nlabel][bname]['loss'] += 1

        n_contests += 1
        metric_had_data = True

    # Save per-metric detail (averaged across reps for the JSON report).
    if metric_had_data:
        avg_values: Dict[str, float] = {}
        for ml in ALL_METHODS:
            v = get_method_mean(metric, ml)
            if v is not None:
                avg_values[ml] = v
        metric_results.append({
            'metric': metric, 'category': fam,
            'values': {mn: round(val, 6) for mn, val in avg_values.items()},
        })


# ============================================================================
# Print Table 3: Championship leaderboard
# ============================================================================

print(f"\n{'='*80}")
print(f"TABLE 3: CHAMPIONSHIP ({n_contests} contests = {len(all_selected)} metrics "
      f"x {n_reps} reps, {n_methods} methods)")
print(f"{'='*80}")

table3: Dict[str, Dict[str, float]] = {}
sorted_methods: List[str] = sorted(
    ALL_METHODS,
    key=lambda mn: -(sum(norm_scores[mn]) / len(norm_scores[mn])
                     if norm_scores[mn] else 0),
)
for mn in sorted_methods:
    if rank_counts.get(mn, 0) == 0:
        continue
    ns: List[float] = norm_scores[mn]
    mean_ns: float = sum(ns) / len(ns) if ns else 0
    w_pct: float = 100.0 * wins_count[mn] / n_contests
    p_pct: float = 100.0 * podiums_count[mn] / n_contests
    fr: float = rank_sums[mn] / rank_counts[mn]
    table3[mn] = {
        "norm_score": round(mean_ns, 3),
        "win_pct": round(w_pct, 1),
        "podium_pct": round(p_pct, 1),
        "friedman_rank": round(fr, 3),
    }
    print(f"  {mn:24s} norm={mean_ns:.3f}  win={w_pct:5.1f}%  "
          f"podium={p_pct:5.1f}%  rank={fr:.3f}")

# ---------------------------------------------------------------------------
# Friedman test using scipy.
#
# Input: the rank matrix with one row per (metric, rep) block and one
# column per method.  ``friedmanchisquare`` computes the chi-squared
# statistic and its p-value under the null hypothesis that all methods
# have equal average ranks.
# ---------------------------------------------------------------------------
rank_matrix_np: np.ndarray = np.array(rank_matrix)
k: int = rank_matrix_np.shape[1]   # number of methods
N: int = rank_matrix_np.shape[0]   # number of blocks = metrics x reps
chi2: float = 0.0
friedman_p: float = 1.0
if N > 1 and k > 2:
    try:
        chi2, friedman_p = sp_stats.friedmanchisquare(
            *[rank_matrix_np[:, i] for i in range(k)]
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Nemenyi critical difference (CD) at alpha = 0.01.
#
# The CD tells us how far apart two methods' average ranks must be for
# the difference to be statistically significant.  The formula is:
#     CD = q_alpha * sqrt( k*(k+1) / (6*N) )
# where q_alpha is the studentised range quantile for k methods at the
# chosen significance level (tabulated in Demsar 2006, Table 5).
# For k=13, q_{0.01} = 4.151;  for k=9, q_{0.01} = 3.102.
# ---------------------------------------------------------------------------
q_alpha_001: float = 4.151 if k >= 13 else 3.102
cd_nemenyi: float = q_alpha_001 * math.sqrt(k * (k + 1) / (6.0 * N))
print(f"\nFriedman chi2={chi2:.2f}, p={friedman_p:.2e}, "
      f"Nemenyi CD={cd_nemenyi:.3f} (k={k}, N={N})")


# ============================================================================
# Print Table 4: Head-to-head -- each NSGA-II variant vs baselines
#
# Cliff's delta is computed as:
#     delta = (total_wins - total_losses) / total_comparisons
# Values near +1 indicate strong NSGA-II dominance; near 0 indicates
# no systematic advantage; near -1 indicates baseline dominance.
# ============================================================================

print(f"\n{'='*80}")
print("TABLE 4: HEAD-TO-HEAD (NSGA-II variants vs baselines only)")
print(f"{'='*80}")

table4: Dict[str, Dict[str, Any]] = {}
for nlabel in NSGA_LABELS:
    table4[nlabel] = {}
    total_w: int = 0
    total_t: int = 0
    total_l: int = 0
    for code in BASELINE_CODES:
        bname = BASELINE_NAME_MAP[code]
        c: Dict[str, int] = h2h[nlabel][bname]
        total: int = c['win'] + c['tie'] + c['loss']
        if total > 0:
            table4[nlabel][bname] = {
                'win': c['win'], 'tie': c['tie'], 'loss': c['loss'],
                'win_pct': round(100.0 * c['win'] / total, 1),
            }
            total_w += c['win']
            total_t += c['tie']
            total_l += c['loss']
    grand_total: int = total_w + total_t + total_l
    if grand_total > 0:
        # Cliff's delta: non-parametric effect-size measure.
        cliff: float = (total_w - total_l) / grand_total
        table4[nlabel]['_summary'] = {
            'total_wins': total_w, 'total_ties': total_t,
            'total_losses': total_l,
            'win_pct': round(100.0 * total_w / grand_total, 1),
            'cliff_d': round(cliff, 3),
        }
    print(f"\n  {nlabel}:")
    for code in BASELINE_CODES:
        bname = BASELINE_NAME_MAP[code]
        c = h2h[nlabel][bname]
        total = c['win'] + c['tie'] + c['loss']
        if total > 0:
            print(f"    vs {bname:18s}: W={c['win']:2d} T={c['tie']:2d} "
                  f"L={c['loss']:2d}  ({100.0*c['win']/total:.0f}%)")
    if grand_total > 0:
        print(f"    TOTAL: W={total_w} T={total_t} L={total_l} = "
              f"{100.0*total_w/grand_total:.1f}% win, cliff={cliff:.3f}")


# ============================================================================
# Print Table 5: Per-family normalised scores
# ============================================================================

print(f"\n{'='*80}")
print("TABLE 5: PER-FAMILY NORMALIZED SCORES")
print(f"{'='*80}")

table5: Dict[str, Dict[str, float]] = {}
for mn in sorted_methods:
    fam_dict: Dict[str, List[float]] = family_scores[mn]
    table5[mn] = {}
    for f in ['Classification', 'Op. Fidelity', 'Sup. Regression', 'Overall']:
        vals: List[float] = fam_dict.get(f, [])
        table5[mn][f] = round(sum(vals) / len(vals), 3) if vals else 0
    print(f"  {mn:24s}  cls={table5[mn]['Classification']:.3f}  "
          f"fid={table5[mn]['Op. Fidelity']:.3f}  "
          f"reg={table5[mn]['Sup. Regression']:.3f}  "
          f"overall={table5[mn]['Overall']:.3f}")


# ============================================================================
# Print Table 6: Best-achievable analysis
#
# Compares three entities on each metric:
#   - knee: the default balanced NSGA-II selection.
#   - oracle: the per-metric best across the entire Pareto front.
#   - baseline_ub: the single best baseline on that metric.
# For each metric we rank these three and report the distribution of ranks.
# ============================================================================

print(f"\n{'='*80}")
print("TABLE 6: BEST-ACHIEVABLE (knee vs envelope vs baseline UB)")
print(f"{'='*80}")

table6: Dict[str, Dict[str, Any]] = {}
best_ach: Dict[str, List[int]] = {'knee': [], 'oracle': [], 'baseline_ub': []}
for metric in all_selected:
    hb = is_higher_better(metric)
    knee_val: Optional[float] = nsga_sel_mean(metric, 'knee')
    oracle_val = nsga_oracle_mean(metric)
    if knee_val is None or oracle_val is None:
        continue

    bl_vals_list: List[Optional[float]] = [baseline_mean(metric, c) for c in BASELINE_CODES]
    bl_vals_clean: List[float] = [v for v in bl_vals_list if v is not None]
    bl_ub: float = (
        (max(bl_vals_clean) if hb else min(bl_vals_clean))
        if bl_vals_clean else knee_val
    )

    trio: List[Tuple[str, float]] = sorted(
        [('knee', knee_val), ('oracle', oracle_val), ('baseline_ub', bl_ub)],
        key=lambda x: -x[1] if hb else x[1],
    )
    for rank, (name, _) in enumerate(trio, 1):
        best_ach[name].append(rank)

for entity in ['knee', 'oracle', 'baseline_ub']:
    vals_list: List[int] = best_ach[entity]
    if vals_list:
        table6[entity] = {
            "mean_rank": round(sum(vals_list) / len(vals_list), 2),
            "median": sorted(vals_list)[len(vals_list) // 2],
            "dominant": sum(1 for v in vals_list if v == 1),
        }
        print(f"  {entity:15s}: mean_rank={table6[entity]['mean_rank']:.2f}  "
              f"median={table6[entity]['median']}  "
              f"dominant={table6[entity]['dominant']}/{len(vals_list)}")


# ============================================================================
# Print Table 9: Selection flexibility
#
# Demonstrates that having access to the full Pareto front enables
# choosing different selection strategies depending on the downstream
# task.  We rank the four fixed NSGA-II strategies plus the single best
# baseline on each metric, showing that no single strategy dominates
# everywhere but the front collectively covers more metrics than any
# individual baseline.
# ============================================================================

print(f"\n{'='*80}")
print("TABLE 9: SELECTION FLEXIBILITY (4 NSGA-II selections vs baseline UB)")
print(f"{'='*80}")

table9: Dict[str, Dict[str, Any]] = {}
sel_ranks: Dict[str, List[int]] = {
    k_name: [] for k_name in ['knee', 'best_mmd', 'best_sh', 'chebyshev', 'best_bl']
}
for metric in all_selected:
    hb = is_higher_better(metric)
    knee_m: Optional[float] = nsga_sel_mean(metric, 'knee')
    mmd_m: Optional[float] = nsga_sel_mean(metric, 'best_mmd')
    sh_m: Optional[float] = nsga_sel_mean(metric, 'best_sh')
    cheby_m: Optional[float] = nsga_sel_mean(metric, 'chebyshev')
    if knee_m is None:
        continue

    bl_vals_list2: List[Optional[float]] = [baseline_mean(metric, c) for c in BASELINE_CODES]
    bl_vals_clean2: List[float] = [v for v in bl_vals_list2 if v is not None]
    bl_best: float = (
        (max(bl_vals_clean2) if hb else min(bl_vals_clean2))
        if bl_vals_clean2 else knee_m
    )

    entities: List[Tuple[str, float]] = [
        ('knee', knee_m),
        ('best_mmd', mmd_m or knee_m),
        ('best_sh', sh_m or knee_m),
        ('chebyshev', cheby_m or knee_m),
        ('best_bl', bl_best),
    ]
    entities_sorted: List[Tuple[str, float]] = sorted(
        entities, key=lambda x: -x[1] if hb else x[1],
    )
    for rank, (name, _) in enumerate(entities_sorted, 1):
        sel_ranks[name].append(rank)

for name in ['knee', 'best_mmd', 'best_sh', 'chebyshev', 'best_bl']:
    vals_sr: List[int] = sel_ranks[name]
    mr: float = sum(vals_sr) / len(vals_sr) if vals_sr else 0
    med: int = sorted(vals_sr)[len(vals_sr) // 2] if vals_sr else 0
    table9[name] = {"mean_rank": round(mr, 2), "median": med}
    print(f"  {name:12s}: mean_rank={mr:.2f}  median={med}")


# ============================================================================
# Normalised score statistics (mean +/- std)
# ============================================================================

print(f"\n{'='*80}")
print("NORM SCORE STATS (mean +/- std)")
print(f"{'='*80}")
for mn in sorted_methods:
    ns = norm_scores[mn]
    if ns:
        m_val: float = sum(ns) / len(ns)
        s_val: float = (
            math.sqrt(sum((v - m_val) ** 2 for v in ns) / (len(ns) - 1))
            if len(ns) > 1 else 0
        )
        print(f"  {mn:24s}: {m_val:.3f} +/- {s_val:.3f}")


# ============================================================================
# Save all results as machine-readable JSON.
# ============================================================================

output: Dict[str, Any] = {
    "methodology": {
        "description": (
            "4 NSGA-II selections from front_metrics_vae.csv + oracle "
            "+ 8 baselines"
        ),
        "nsga_variants": [
            "knee (nearest ideal in normalized f_mmd/f_sh space)",
            "best-MMD (min f_mmd on front)",
            "best-SH (min f_sinkhorn on front)",
            "Chebyshev (min of max normalized objective)",
            "oracle (per-metric best across ~50 front members)",
        ],
        "metric_selection": (
            "Top 10 cls + 10 reg where oracle ranks best among 13 methods"
        ),
        "averaging": "Mean across 5 reps first, then rank 22 mean values",
        "h2h": "Only NSGA-II variants vs baselines, not vs each other",
    },
    "top_cls_metrics": [e["metric"] for e in top_cls],
    "top_reg_metrics": [e["metric"] for e in top_reg],
    "all_selected_metrics": all_selected,
    "n_metrics": len(all_selected),
    "n_reps": n_reps,
    "n_contests": n_contests,
    "n_methods": n_methods,
    "friedman_chi2": round(chi2, 2),
    "nemenyi_cd": round(cd_nemenyi, 3),
    "table3": table3,
    "table4": table4,
    "table5": table5,
    "table6": table6,
    "table9": table9,
    "metric_results": metric_results,
}

# Output lives inside the experiments directory so that consumers (analysis
# notebooks, build scripts) find it using the same path discipline as input
# data. Override via ``CORESET_EXPERIMENTS_DIR`` if needed.
outpath: str = os.path.join(BASE, "manuscript_final_v3.json")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {outpath}")
