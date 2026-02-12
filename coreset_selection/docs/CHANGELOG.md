# Changelog

All notable changes to the coreset selection repository, organized by
upgrade-plan phase.

---

## Phase 2 (Refactor) — Type-Aware Preprocessing & Mixed-Type Support

Phase 2 refactors the preprocessing and evaluation pipelines to natively
support **categorical**, **ordinal**, and **numeric** feature types without
one-hot encoding.  Categorical variables are integer-encoded, preserving
compatibility with downstream classification metrics.

### Preprocessing (`data/cache.py`)

- **`_preprocess_fit_transform`** now accepts a `feature_types` list
  (parallel to `feature_names`).  Per-column behaviour is:
  - *Categorical*: mode imputation from I_train, no log1p, no
    missingness indicator (uses -1 missing code), no standardization
    by default.
  - *Ordinal*: rounded-median imputation, no log1p by default,
    optional standardization (`scale_ordinals`).
  - *Numeric*: unchanged (median imputation, log1p if heavy-tailed,
    missingness indicators, full standardization).
- **`_impute_typeaware`**: new function dispatching to median, mode,
  or rounded-median based on feature type.
- **`_impute_by_train_mode`**: new helper computing column mode on
  the training split.
- **`build_replicate_cache`**: passes feature types through the full
  pipeline, applies type-aware StandardScaler (partial scaling mask),
  stores `category_maps`, `scale_mask`, and `target_type` in the
  replicate .npz cache.

### Configuration (`config/dataclasses.py`)

- **`PreprocessingConfig`**: new Phase 2 fields:
  `scale_ordinals`, `scale_categoricals`, `log1p_categoricals`,
  `log1p_ordinals`, `categorical_impute_strategy`,
  `ordinal_impute_strategy`, `target_type`,
  `classification_cardinality_threshold`.
- **`ReplicateAssets`**: new fields `feature_types`, `category_maps`,
  `target_type`.

### Evaluation — Classification Metrics (`evaluation/classification_metrics.py`)

- New module providing classification-appropriate metrics:
  `accuracy`, `cohens_kappa` (κ), `macro_precision`, `macro_recall`,
  `macro_f1`, `weighted_f1`, `confusion_matrix_dict`.
- `full_classification_evaluation`: one-call per-target evaluation
  with optional per-state breakdown (mirrors
  `full_downstream_evaluation` API).
- `multitarget_classification_evaluation`: multi-target loop.
- `infer_target_type(y)`: heuristic auto-detection of regression vs
  classification based on dtype and cardinality.

### Evaluation — Unified Dispatch (`evaluation/downstream_metrics.py`)

- `evaluate_target_auto`: chooses between regression and
  classification metrics based on `target_type` (or auto-detection).
- `multitarget_evaluate_auto`: per-target type dispatch for
  multi-target setups.

### Feature Schema (`data/feature_schema.py`)

- Module docstring updated to reflect Phase 2 role.
- No functional changes (Phase 1 schema inference already supports
  categorical/ordinal/numeric classification).

### Tests (`tests/test_preprocessing.py`)

- **`TestTypeAwarePreprocessing`**: verifies categorical no-log1p,
  ordinal no-log1p, mode imputation for categoricals, no missingness
  indicators for categoricals, feature-type propagation.
- **`TestClassificationMetrics`**: verifies accuracy, Cohen's kappa,
  macro-F1, full classification evaluation, confusion matrix.
- **`TestTargetTypeDetection`**: verifies auto-detection of
  regression vs classification targets, dispatch routing.
- **`TestFeatureSchemaPhase2`**: verifies mixed-type DataFrame
  inference and explicit ordinal overrides.

---

## Phase 13 — Documentation & README Update

- **README.md**: Comprehensive rewrite reflecting all Phases 1–12.
  Updated run matrix table with purpose descriptions matching Table II.
  Updated artifact tables listing all manuscript, complementary, and new
  figures/tables. Added "Quick Start" section with minimal reproduction
  commands.
- **CHANGELOG.md**: Created (this file). Documents all changes from each
  phase.
- **ARTIFACTS.md**: Created. Comprehensive catalog of every generated
  artifact with filename, description, data source run IDs, and manuscript
  section references.
- **Docstring audit**: Ensured all key public functions reference the
  manuscript equation or section they implement.

---

## Phase 12 — Analysis Script & End-to-End Validation

- **`scripts/generate_all_artifacts.py`**: Top-level script that scans
  `runs_out/`, instantiates `ManuscriptArtifacts`, calls `generate_all()`,
  and reports missing data or failed artifacts.  Added `--validate` flag
  checking artifact existence, nonzero file size, and key metric value
  ranges.  Added `--validate-only` mode to skip generation.
- **`scripts/verify_compliance.py`**: Extended Phase 12 checks:
  `verify_coverage_targets()` (count = 10), `verify_output_coverage()`
  (all R0–R12 present), `verify_table_v_structure()` (10 rows),
  `verify_fig2_panel_structure()` (2×2), `verify_proxy_stability()` (3
  sections in Table IV).
- **`tests/test_end_to_end.py`**: Integration test creating a small
  synthetic dataset (N=100, G=5, D=20), running R1 at k=20 with minimal
  effort (pop=10, gen=10), evaluating, generating artifacts, and asserting
  all expected files are produced.

---

## Phase 11 — New/Enhanced Tables for Strengthened Narrative

- **Table N1** (`constraint_diagnostics_cross_config.tex`): Cross-config
  proportionality comparison at k=300 for R1/R4/R5/R6 + baselines.
  Companion CSV also generated.
- **Table N2** (`objective_ablation_summary.tex`): R1 vs R2 vs R3 at
  k=300 with e\_Nys, e\_kPCA, RMSE\_4G, RMSE\_5G, geo\_kl.
- **Table N3** (`representation_transfer_summary.tex`): R1 vs R8 vs R9
  transfer summary.
- **Table N4** (`skl_ablation_summary.tex`): R1 vs R7 bi-obj vs tri-obj
  comparison.
- **Table N5** (`multi_seed_statistics.tex`): R1/R5 mean ± std across 5
  seeds for key metrics.
- **Table N6** (`worst_state_rmse_by_k.tex`): Worst-state RMSE and
  dispersion at each k for equity analysis.
- **Table N7** (`baseline_paired_unconstrained_vs_quota.tex`): Paired
  comparison of each baseline method in exact-k vs quota-matched regimes.

---

## Phase 10 — New/Enhanced Figures for Strengthened Narrative

### Phase 10a (Figs N1–N6)

- **Fig N1** (`kl_floor_vs_k.pdf`): KL\_min(k) planning curve with
  horizontal τ thresholds. Data from R0.
- **Fig N2** (`pareto_front_mmd_sd_k300.pdf`): Pareto front scatter at
  k=300 with knee-point marker and optional R2/R3 overlay.
- **Fig N3** (`objective_ablation_bars_k300.pdf`): Grouped bar chart
  comparing R1-knee, R2, R3 across multiple metrics.
- **Fig N4** (`constraint_comparison_bars_k300.pdf`): Multi-panel
  constraint regime comparison (R1/R4/R5/R6) for geo\_kl, geo\_l1,
  e\_Nys, RMSE\_4G.
- **Fig N5** (`effort_quality_tradeoff.pdf`): Effort sweep scatter with
  wall-clock time vs downstream metric, colored by (P, T) config.
- **Fig N6** (`baseline_comparison_grouped.pdf`): Multi-metric baseline
  comparison with both unconstrained and quota-matched variants.

### Phase 10b (Figs N7–N12)

- **Fig N7** (`multi_seed_stability_boxplot.pdf`): Box-and-whisker plots
  of key metrics across 5 seeds for R1 and R5.
- **Fig N8** (`state_kpi_heatmap.pdf`): Per-state KPI drift heatmap with
  small-state annotations.
- **Fig N9** (`composition_shift_sankey.pdf`): Side-by-side bar chart of
  π\_g vs π̂\_g(S) for R6 unconstrained vs R1 constrained.
- **Fig N10** (`pareto_front_evolution.pdf`): Pareto front overlay at
  generation checkpoints showing convergence behavior.
- **Fig N11** (`nystrom_error_distribution.pdf`): Histogram/KDE of
  Nyström errors across all Pareto-front solutions at k=300.
- **Fig N12** (`krr_worst_state_rmse_vs_k.pdf`): Worst-state RMSE for
  4G/5G vs k, compared against average RMSE.

---

## Phase 9 — Manuscript Tables (Section VIII)

- **Table I** (`exp_settings.tex`): Auto-generated from
  `config/constants.py`, reproducing manuscript Table I layout.
- **Table II** (`run_matrix.tex`): Auto-generated from
  `config/run_specs.py`, reproducing manuscript Table II.
- **Table III** (`r1_by_k.tex`): R1 metric envelope vs k with columns
  k, e\_Nys, e\_kPCA, RMSE\_4G, RMSE\_5G.  Mean envelope across 5 seeds.
- **Table IV** (`proxy_stability.tex`): Three sections — MMD RFF sweep,
  Sinkhorn anchor sweep, cross-representation Spearman ρ.
- **Table V** (`krr_multitask_k300.tex`): 10-row × 3-column layout with
  "R1 knee", "R9 knee", "Best (pool)" RMSE values for all 10 coverage
  targets.

---

## Phase 8 — Manuscript Figures (Section VIII)

- **Fig 1** (`geo_ablation_tradeoff_scatter.pdf`): Scatter of composition
  drift vs Nyström error for R6 unconstrained Pareto front.  Log-scaled
  y-axis, constraint regimes distinguished by marker shape, constrained
  R1 knee-point overlaid.
- **Fig 2** (`distortion_cardinality_R1.pdf`): 2×2 panel showing
  (a) e\_Nys vs k, (b) e\_kPCA vs k, (c) RMSE\_4G vs k,
  (d) RMSE\_5G vs k.  Error bands (mean ± std) over 5 seeds.
- **Fig 3** (`regional_validity_k300.pdf`): State-conditioned KPI
  stability comparing R1 and R5 side-by-side at k=300 for both 4G and 5G
  targets.  Grouped bar layout.
- **Fig 4** (`objective_metric_alignment_heatmap.pdf`): Annotated Spearman
  rank correlation heatmap between optimization objectives and evaluation
  metrics.  Diverging colormap centered at 0.

---

## Phase 7 — Diagnostic Infrastructure (R11)

- **`evaluation/r7_diagnostics.py`**: Implements proxy stability sweep
  (RFF dimension m ∈ {500, 1000, 2000, 4000}, Sinkhorn anchor count
  A ∈ {50, 100, 200, 400}) and cross-representation rank correlations
  (VAE-vs-raw, VAE-vs-PCA, PCA-vs-raw).
- **Objective–metric alignment**: Computes Spearman rank correlations
  between optimization objectives (f\_MMD, f\_SD) and evaluation metrics
  (e\_Nys, e\_kPCA, RMSE\_4G, RMSE\_5G, geo\_kl, geo\_l1) over the
  Pareto front + R10 baseline pool.
- Structured CSV outputs: `proxy_stability.csv`, 
  `objective_metric_alignment.csv`, `objective_metric_alignment_heatmap.csv`.
- R10 baseline indices loaded to expand the candidate pool for alignment.

---

## Phase 6 — Effort Sweep & Time Complexity (R12)

- **`config/constants.py`**: Added `EFFORT_GRID` with 11 (pop\_size,
  n\_gen) pairs covering the manuscript specification.
- **`config/dataclasses.py`**: Added `EffortSweepGrid` dataclass with
  Cartesian product grid generation and `effort_grid` field on
  `SolverConfig`.
- **`experiment/runner.py`**: `_run_r12_effort_sweep()` iterates over the
  grid, records wall-clock time, evaluates knee-point downstream metrics,
  saves `effort_grid_config.csv` and `effort_sweep_summary.csv`.
- **`experiment/time_complexity.py`**: Per-phase timing (quota, objective
  setup, NSGA-II, evaluation, geo diagnostics) for each k ∈ K.  Baseline
  timing per method per k.  Theoretical complexity annotations.

---

## Phase 5 — Baseline Suite Completion (R10)

- **7 baseline methods verified**: uniform, k-means, herding,
  farthest-first, kernel thinning, ridge leverage-score, DPP.
- **Quota-matched variants**: `baselines/variant_generator.py` generates
  `_quota_matched` wrappers applying each baseline within each state
  selecting exactly c\*\_g(k) items.
- **Population-share feasibility**: Diagnostics computed for every
  baseline output.
- **Robust k-DPP**: Eigenvalue clamping and fallback to uniform added
  (`baselines/dpp.py`).
- **Structured comparison**: `BaselineVariantGenerator` produces paired
  comparison summary CSV for unconstrained vs quota-matched regimes.

---

## Phase 4 — Evaluation Protocol Hardening

- **`data/preprocessing.py`**: Verified `log(1+x)` applied only to
  continuous nonneg heavy-tailed variables (skewness > 2, all values ≥ 0).
  Binary missingness indicators per feature.  Standardization uses I\_train
  statistics only.
- **`evaluation/raw_space.py`**: Verified median heuristic for σ\_raw
  computed on E\_train only.  KRR tuning on E\_train only with no E\_test
  leakage.  Nyström stabilization λ\_nys = 1e−6 · tr(W)/k.  kPCA
  distortion with centered Gram matrix and top-20 eigenvalue extraction.
- **`evaluation/kpi_stability.py`**: Added `state_krr_rmse()` computing
  per-state RMSE, worst-state RMSE, and state RMSE dispersion.  Added
  `export_state_kpi_drift_csv()` for heatmap visualization.  Verified
  full-dataset state means (not E-subset).
- **`tests/test_evaluation_protocol.py`**: Comprehensive test with
  synthetic data verifying every metric against hand-calculated values.

---

## Phase 3 — Constraint Implementation Refinements

- **`constraints/proportionality.py`**: Verified `value()` implements
  Eq. (4) exactly: D^(w)(S) = KL(π^(w) || π̂^(w,α)(S)).  Verified
  Laplace smoothing: π̂\_g = (W\_g(S) + α) / (W(S) + αG).  Verified
  `total_violation()` sums max{D − τ, 0} over all active constraints.
- **`optimization/repair.py`**: Verified Algorithm 2 donor/recipient
  selection (smallest s\_g donor, largest s\_g recipient).  Verified
  acceptance only when V(S') decreases.  Early termination when V(S)=0.
- **`experiment/runner.py`**: Joint mode passes both population-share AND
  municipality-share constraints.  Municipality-share quota mode uses
  c\*(k) with within-group swap restriction.  None mode enforces only
  exact-k and capacity bounds.
- **`geo/kl.py`**: Verified lazy-heap greedy (Algorithm 1 lines 5–18),
  edge case ℓ\_g = 1, and KL floor formula.
- **`evaluation/geo_diagnostics.py`**: `dual_geo_diagnostics()` called
  for every run configuration (R1–R12) unconditionally.

---

## Phase 2 — Coverage Target Pipeline Alignment

- **`data/brazil_telecom_loader.py`**: Added `_discover_coverage_targets()`
  constructing all 10 Table V targets including 6 derived combined targets.
  Handles absent technology targets (fallback to single-technology).
  Stores in `BrazilTelecomData.extra_targets`.
- **`config/constants.py`**: `COVERAGE_TARGETS_TABLE_V` mapping with exact
  Table V names.  `COVERAGE_TARGET_NAMES` for ordered iteration.
  `_LEGACY_TARGET_KEY_MAP` for backward compatibility.
- **`experiment/runner.py`**: `_build_multitarget_y()` returns all 10
  targets in Table V order, normalizing legacy keys.
- **`data/cache.py`**: Extra targets stored in replicate cache metadata
  under `extra_targets` key.

---

## Phase 1 — Configuration & Constants Alignment

- **`config/constants.py`**: Verified N=5569, G=27, D=621, K\_GRID,
  NSGA-II defaults (P=150, T=700, p\_c=0.9, p\_m=0.2), RFF dim 2000,
  Sinkhorn (A=200, η=0.05, 100 iters), VAE (d\_z=32, 5000 epochs, bs=256,
  lr=1e-3), |E|=2000, r=20, α\_geo=1.0.  Added PCA\_DIM=32, EFFORT\_GRID.
- **`config/run_specs.py`**: All RunSpecs verified against Table II.
  R1 has sweep\_k=K\_GRID and n\_reps=5.  R5 has n\_reps=5 and joint mode.
  R7 uses VAE space with (mmd, sinkhorn, skl) objectives.
- **`config/dataclasses.py`**: `GeoConfig` has `constraint_mode`,
  `alpha_geo`, `tau_population`, `tau_municipality`.  `EffortSweepGrid`
  added for R12.  `SolverConfig` includes `effort_grid` field.
- **`tests/test_verify_constants.py`**: Imports every constant and asserts
  manuscript compliance.
