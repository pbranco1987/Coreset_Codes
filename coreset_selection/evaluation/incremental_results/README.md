# Incremental Evaluation: Kernel K-Means vs MMD+Sinkhorn+NSGA-II

This directory contains the incremental evaluation outputs implementing
the metrics and comparison protocol recommended by the kernel k-means
vs MMD+Sinkhorn+NSGA-II analysis.

## New Source Modules (in `evaluation/`)

### `evaluation/downstream_metrics.py`
Additional supervised downstream metrics:
- **Tail absolute errors** (P90, P95, P99 of |error|) — captures coverage holes
- **Per-state RMSE / MAE / R²** — full group-level breakdown
- **Macro-averaged RMSE/MAE** — weights each state equally
- **Worst-group RMSE/R²** — robustness metric
- **Overall MAE and R²** — complementary to RMSE
- `evaluate_nystrom_landmarks_downstream()`: end-to-end Nyström → KRR → all metrics

### `evaluation/method_comparison.py`
Cross-method comparison and effect isolation:
- **`effect_isolation_table()`**: decomposes NSGA-II's improvement into
  *constraint effect* (Δ\_constraint = KKN − SKKN) and *objective effect*
  (Δ\_objective = SKKN − Pareto), with percentage attribution
- **`rank_table()`**: per-metric ranking across all methods + average rank
- **`pairwise_dominance_matrix()`**: Pareto dominance over metric subsets
- **`stability_comparison()`**: mean ± std across replicates, coefficient of variation
- **`comprehensive_comparison()`**: unified report tying everything together

### `evaluation/enhanced_evaluator.py`
Drop-in wrapper around `RawSpaceEvaluator` that adds all incremental metrics
to the standard evaluation pipeline.

### `scripts/incremental_evaluation.py`
Standalone runner that produces all CSV outputs. Supports:
- `--synthetic` mode: generates synthetic data mimicking the Brazil telecom
  structure (5569 municipalities, 27 states) for validation
- `--results-dir` mode: re-evaluates existing experiment results

### `tests/test_incremental_evaluation.py`
Unit tests for all new modules.

---

## Generated Output Files

### Core comparison tables

| File | Description |
|------|-------------|
| `all_downstream_metrics.csv` | Every metric × method × replicate (flat table) |
| `effect_isolation.csv` | Three-way decomposition: KKN vs SKKN vs Pareto |
| `rank_table_lower_is_better.csv` | Per-metric rank for RMSE/MAE/tail metrics |
| `rank_table_higher_is_better.csv` | Per-metric rank for R²-type metrics |
| `stability_comparison.csv` | Mean ± std ± CV across replicates |
| `pairwise_dominance.csv` | Pairwise Pareto dominance on all lower-is-better metrics |
| `evaluation_summary.json` | Summary with method list, k, replicate count, ranks |

### Per-state breakdowns

| Pattern | Description |
|---------|-------------|
| `per_state_{target}_{method}_rep{r}.csv` | RMSE / MAE / R² per state per replicate |

---

## Metrics Computed

### Global metrics (per target)
- `overall_rmse_{t}`, `overall_mae_{t}`, `overall_r2_{t}`

### Tail error metrics (per target)
- `abs_err_p90_{t}`, `abs_err_p95_{t}`, `abs_err_p99_{t}`, `abs_err_max_{t}`

### Group-level metrics (per target)
- `macro_rmse_{t}`: mean of per-state RMSEs (equal state weighting)
- `worst_group_rmse_{t}`: max per-state RMSE
- `best_group_rmse_{t}`: min per-state RMSE
- `rmse_dispersion_{t}`: std of per-state RMSEs
- `rmse_iqr_{t}`: IQR of per-state RMSEs
- `macro_mae_{t}`, `worst_group_mae_{t}`
- `macro_r2_{t}`, `worst_group_r2_{t}`, `best_group_r2_{t}`

### Cross-target summary
- `mean_macro_rmse`: mean of macro_rmse across targets
- `mean_worst_group_rmse`: mean of worst_group_rmse across targets

### Effect isolation (in `effect_isolation.csv`)
For each metric:
- `delta_total = KKN − Pareto` (total improvement)
- `delta_constraint = KKN − SKKN` (pure proportionality effect)
- `delta_objective = SKKN − Pareto` (MMD+Sinkhorn objective effect)
- `pct_constraint`, `pct_objective` (% attribution)

---

## How to integrate with the main experiment pipeline

### Option A: Post-hoc (recommended for existing results)
```python
from evaluation.enhanced_evaluator import EnhancedRawSpaceEvaluator

enhanced = EnhancedRawSpaceEvaluator.from_base(raw_evaluator)
metrics = enhanced.all_metrics_enhanced(S_idx, state_labels)
```

### Option B: In the experiment runner
Add to `_evaluate_coreset()` in `experiment/runner.py`:
```python
from evaluation.downstream_metrics import evaluate_nystrom_landmarks_downstream

downstream = evaluate_nystrom_landmarks_downstream(
    X_raw=assets.X_scaled, S_idx=idx_sel, y=assets.y,
    eval_train_idx=..., eval_test_idx=..., eval_idx=...,
    sigma_sq=raw_evaluator.sigma_sq,
    state_labels=assets.state_labels,
)
row.update(downstream)
```

### Option C: Standalone script
```bash
python scripts/incremental_evaluation.py --results-dir runs_out/ --output-dir incremental_eval/
```

---

## Relationship to the analysis documents

This implementation follows the protocol recommended in the pasted analyses:

1. **Fix k and downstream model** → All methods evaluated at same k with
   same KRR + 3-fold CV lambda selection
2. **Three landmarking baselines**: kernel k-means (KKN), kernel k-means
   with quota-matching (SKKN), NSGA-II Pareto (Pareto)
3. **Report**: overall RMSE/accuracy, macro/worst-group RMSE, variance across seeds
4. **Isolate** whether gains come from constraints vs objectives
