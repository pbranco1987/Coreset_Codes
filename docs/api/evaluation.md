# API Reference — Evaluation (`coreset_selection.evaluation`)

The evaluation package contains every **metric computation** used in the pipeline: raw-space kernel metrics (Nyström error, kPCA distortion, KRR RMSE), geographic diagnostics (KL, L1, MaxDev), KPI stability, multi-model downstream models (KNN, RF, LR, GBT), classification metrics, QoS-specific tasks, R11 proxy-stability diagnostics, and cross-method comparison utilities. **This is where the S ∩ E overlap fix is enforced** — no other package computes metrics.

Public symbols are re-exported from `coreset_selection/evaluation/__init__.py`.

## Section Map

1. [Raw-Space Evaluator](#raw-space-evaluator)
2. [Geographic Diagnostics](#geographic-diagnostics)
3. [Coverage & Diversity Metrics](#coverage--diversity-metrics)
4. [R11 Diagnostics (Proxy Stability)](#r11-diagnostics-proxy-stability)
5. [KPI Stability (Manuscript Section VII)](#kpi-stability-manuscript-section-vii)
6. [Cross-Method Comparison](#cross-method-comparison)
7. [Classification Metrics (Phase 2)](#classification-metrics-phase-2)
8. [QoS Downstream Tasks](#qos-downstream-tasks)

---

## Raw-Space Evaluator

### `coreset_selection.evaluation.RawSpaceEvaluator`

**Kind:** class
**Source:** `coreset_selection/evaluation/raw_space.py`

**Summary:** Orchestrator for all kernel-based raw-space metrics on a given replicate and fixed evaluation set `E`.

**Description:** The most important evaluation class in the package. Given raw features `X_raw`, target vector `y`, evaluation splits `(eval_idx, eval_train_idx, eval_test_idx)`, and RBF bandwidth `sigma_sq`, it computes Nyström approximation error, kernel-PCA distortion, KRR RMSE per target, state-conditioned prediction stability, and multi-model downstream metrics. The internal `_NystromCache` is keyed by `S_idx` so repeated calls for the same mask are free. **All S ∩ E exclusions happen here**: when `S ∩ eval_idx ≠ ∅`, those overlapping indices are stripped before computing `K_EE_clean`, `Phi`, and downstream features — preventing landmarks from evaluating themselves.

**Construction (preferred):**
```python
RawSpaceEvaluator.build(
    X_raw: np.ndarray,
    y: Optional[np.ndarray],
    eval_idx: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    seed: int,
    target_names: Optional[List[str]] = None,
) -> RawSpaceEvaluator
```

The `build` classmethod computes `sigma_sq` from the median heuristic on `X_raw[eval_idx]` and instantiates the evaluator. Direct construction requires pre-computed `sigma_sq`.

**Key methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `.nystrom_error(S_idx)` | `float` | Relative Frobenius error `‖K_EE - K̂‖_F / ‖K_EE‖_F` (computed on `E_clean = E \ S`). |
| `.kpca_distortion(S_idx, r=20)` | `float` | Top-r spectral distortion between `K_EE` and `K̂` (on `E_clean`). |
| `.krr_rmse(S_idx)` | `Dict[str, float]` | Per-target KRR RMSE on Nyström features, using train-split of `E_clean`. |
| `.all_metrics(S_idx)` | `Dict[str, float]` | Combines `nystrom_error`, `kpca_distortion`, and `krr_rmse` into a single dict. |
| `.all_metrics_with_state_stability(S_idx, state_labels)` | `Dict[str, float]` | Above plus state-conditioned prediction stability metrics. |
| `.multi_model_downstream(S_idx, regression_targets, classification_targets, seed)` | `Dict[str, float]` | KNN/RF/LR/GBT on Nyström features for extra targets. |

**Attributes:**
| Name | Type | Description |
|------|------|-------------|
| `X_raw` | `np.ndarray` | Standardised feature matrix. |
| `y` | `np.ndarray or None` | Target matrix `(N, T)`. |
| `eval_idx` | `np.ndarray` | Evaluation set indices. |
| `eval_train_idx`, `eval_test_idx` | `np.ndarray` | Splits within `E`. |
| `sigma_sq` | `float` | RBF bandwidth squared. |
| `K_EE` | `np.ndarray` | Full-E Gram matrix (lazy, cached). |

**Example:**
```python
from coreset_selection.evaluation import RawSpaceEvaluator
from coreset_selection.data import load_replicate_cache

assets = load_replicate_cache("cache/rep00/assets.npz")
evaluator = RawSpaceEvaluator.build(
    X_raw=assets.X_scaled, y=assets.y,
    eval_idx=assets.eval_idx,
    eval_train_idx=assets.eval_train_idx,
    eval_test_idx=assets.eval_test_idx,
    seed=4200,
)
metrics = evaluator.all_metrics_with_state_stability(S_idx, assets.state_labels)
```

**See also:** `dual_geo_diagnostics`, `state_kpi_stability`, `multi_model_evaluator`.

---

## Geographic Diagnostics

### `coreset_selection.evaluation.geo_diagnostics`

**Kind:** function
**Source:** `coreset_selection/evaluation/geo_diagnostics.py`

**Summary:** Compute a standard set of geographic metrics (`geo_kl`, `geo_l1`, `geo_maxdev`) under a single weight mode.

### `coreset_selection.evaluation.geo_diagnostics_weighted`

**Kind:** function
**Source:** `coreset_selection/evaluation/geo_diagnostics.py`

**Summary:** Same as `geo_diagnostics` but accepts arbitrary per-point weights.

### `coreset_selection.evaluation.dual_geo_diagnostics`

**Kind:** function
**Source:** `coreset_selection/evaluation/geo_diagnostics.py`

**Summary:** Compute geographic metrics under **both** municipality-share and population-share weight modes in a single call.

**Description:** Returns a dict with both unsuffixed keys (`geo_kl`, `geo_l1`, `geo_maxdev`) and mode-suffixed keys (`geo_kl_muni`, `geo_kl_pop`, etc.). Called at stage [1/5] of `evaluate_coresets.py`.

**Signature:**
```python
def dual_geo_diagnostics(
    geo: GeoInfo, S_idx: np.ndarray, k: int, alpha: float = 1.0
) -> Dict[str, float]
```

**Returns:** dict with keys `geo_kl`, `geo_l1`, `geo_maxdev`, `geo_kl_muni`, `geo_l1_muni`, `geo_maxdev_muni`, `geo_kl_pop`, `geo_l1_pop`, `geo_maxdev_pop`.

### `coreset_selection.evaluation.compute_quota_satisfaction`

**Kind:** function — fraction of per-group counts exactly matching `c*(k)`.

### `coreset_selection.evaluation.state_coverage_report`

**Kind:** function — per-state summary (count, share, weight) for logs.

### `coreset_selection.evaluation.geographic_entropy`

**Kind:** function — Shannon entropy of the coreset's group distribution.

### `coreset_selection.evaluation.geographic_concentration_index`

**Kind:** function — Herfindahl-Hirschman index of the coreset's group distribution.

---

## Coverage & Diversity Metrics

Classic coreset quality signals; not the primary evaluation metrics in the manuscript but useful for ablations.

### `coreset_selection.evaluation.coverage_stats`

**Kind:** function — returns `(mean_dist, max_dist, std_dist)` of each non-S point to its nearest S point.

### `coreset_selection.evaluation.k_center_cost`

**Kind:** function — max over `i ∉ S` of min-distance to `S` (the k-center objective).

### `coreset_selection.evaluation.k_median_cost`

**Kind:** function — sum over `i ∉ S` of min-distance to `S`.

### `coreset_selection.evaluation.diversity_score`

**Kind:** function — average pairwise distance within `S`.

### `coreset_selection.evaluation.min_pairwise_distance`

**Kind:** function — smallest pairwise distance within `S`.

### `coreset_selection.evaluation.representation_error`

**Kind:** function — mean squared reconstruction error for each non-S point using its nearest S point.

### `coreset_selection.evaluation.all_metrics`

**Kind:** function — batch helper: returns a dict of all five metrics above.

---

## R11 Diagnostics (Proxy Stability)

Utilities for the manuscript's Section VIII.K diagnostics — studying how sensitive the method is to proxy approximations (RFF feature count, Sinkhorn anchor count, etc.).

### `coreset_selection.evaluation.SurrogateSensitivityConfig`

**Kind:** dataclass — configuration for the sensitivity sweep.

### `coreset_selection.evaluation.surrogate_sensitivity_analysis`

**Kind:** function — run the proxy sensitivity sweep and return results.

### `coreset_selection.evaluation.cross_space_evaluation`

**Kind:** function — evaluate a coreset across multiple representation spaces (VAE, PCA, raw).

### `coreset_selection.evaluation.objective_metric_alignment`

**Kind:** function — rank-correlate objective values with downstream metrics to study alignment.

### `coreset_selection.evaluation.compute_alignment_heatmap_data`

**Kind:** function — produce the pivot-table data underlying the alignment heatmap figure.

### `coreset_selection.evaluation.RepairDiagnostics`, `aggregate_repair_diagnostics`

**Kind:** dataclass + function — summarise NSGA-II repair activity across runs.

### `coreset_selection.evaluation.R11Results`

**Kind:** dataclass — return container for `run_r7_diagnostics`.

### `coreset_selection.evaluation.R6Results`

**Kind:** backward-compatible alias for `R11Results` (pre-reorganisation name).

### `coreset_selection.evaluation.run_r7_diagnostics`

**Kind:** function — top-level R11 diagnostics entry point (named R7 historically).

### CSV exporters

- `export_proxy_stability_csv`
- `export_objective_metric_alignment_csv`
- `export_alignment_heatmap_csv`

### Baseline loading helpers

- `load_baseline_indices_from_dir`
- `find_r10_results_dir`

---

## KPI Stability (Manuscript Section VII)

### `coreset_selection.evaluation.state_kpi_stability`

**Kind:** function
**Source:** `coreset_selection/evaluation/kpi_stability.py`

**Summary:** Per-state target-mean drift and ranking stability (Kendall's τ).

**Description:** Splits the evaluation test set in half, stratified by state, then measures how well the rank-order of state-level KPI means is preserved across splits. A high Kendall's τ indicates that the coreset's representation is stable at the state level. Called in stage [3/5] of `evaluate_coresets.py`.

### `coreset_selection.evaluation.state_krr_stability`

**Kind:** function — same protocol but using KRR-predicted means instead of raw target means.

### `coreset_selection.evaluation.per_state_kpi_drift_matrix`

**Kind:** function — produce a `(state × target)` drift matrix for heatmap visualisation.

### `coreset_selection.evaluation.export_state_kpi_drift_csv`

**Kind:** function — persist the drift matrix to CSV.

---

## Cross-Method Comparison

Utilities implementing the manuscript's kernel-k-means analysis protocol for comparing multiple methods.

### `coreset_selection.evaluation.load_result_rows`

**Kind:** function — load per-method, per-replicate rows from a directory of CSVs.

### `coreset_selection.evaluation.group_by_method`

**Kind:** function — group rows by method name.

### `coreset_selection.evaluation.mean_per_method`

**Kind:** function — per-method mean of numerical columns.

### `coreset_selection.evaluation.effect_isolation_table`

**Kind:** function — isolate the effect of a single factor via paired differences.

### `coreset_selection.evaluation.rank_table`

**Kind:** function — per-metric rank table (input to Friedman test).

### `coreset_selection.evaluation.pairwise_dominance_matrix`

**Kind:** function — pairwise "wins over" matrix across methods.

### `coreset_selection.evaluation.stability_summary`

**Kind:** function — across-replicates variability of each method.

### `coreset_selection.evaluation.build_comparison_report`

**Kind:** function — high-level orchestrator producing a multi-sheet report.

### Constants

- `DOWNSTREAM_LOWER` — set of downstream metrics where lower is better.
- `DOWNSTREAM_HIGHER` — set where higher is better.

---

## Classification Metrics (Phase 2)

Metrics for binary/multi-class classification targets in multi-model downstream evaluation.

### `coreset_selection.evaluation.infer_target_type`

**Kind:** function — decide whether a target vector is regression, binary classification, or multi-class.

### `coreset_selection.evaluation.accuracy`, `cohens_kappa`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`

**Kind:** function — standard classification metrics with stable implementations.

### `coreset_selection.evaluation.confusion_matrix_dict`

**Kind:** function — return a confusion matrix as a flat dict for CSV export.

### `coreset_selection.evaluation.full_classification_evaluation`

**Kind:** function — compute every classification metric on a single target.

### `coreset_selection.evaluation.multitarget_classification_evaluation`

**Kind:** function — batch over multiple classification targets.

---

## QoS Downstream Tasks

Telecom-specific evaluation where the downstream model is an ISG/IQS composite index regression.

### `coreset_selection.evaluation.QoSConfig`

**Kind:** dataclass — configuration: which models to run, whether to include fixed effects.

### `coreset_selection.evaluation.Demeaner`

**Kind:** class — entity/time fixed-effects demeaning transformer.

### `coreset_selection.evaluation.build_lagged_features`

**Kind:** function — build lagged versions of QoS-relevant features.

### `coreset_selection.evaluation.qos_coreset_evaluation`

**Kind:** function — run QoS regression on the coreset using OLS, Ridge, Elastic Net, PLS, constrained, and heuristic models.

### `coreset_selection.evaluation.qos_fullset_reference`

**Kind:** function — compute the QoS reference numbers on the full set.

### `coreset_selection.evaluation.qos_summary_table`

**Kind:** function — produce a summary table of QoS model performance.

---

## See Also

- [data](./data.md) — `ReplicateAssets` provides everything an evaluator consumes.
- [geo](./geo.md) — `GeoInfo` used by `dual_geo_diagnostics`.
- [scripts](./scripts.md#evaluate_coresets) — `evaluate_coresets.py` is the canonical caller.
- Manuscript Section V (evaluation protocol), Section VII (KPI stability), Section VIII.K (R11 diagnostics).
