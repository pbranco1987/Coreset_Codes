# Coreset Selection — Manuscript-Aligned Repository

**Constrained Nyström Landmark Selection for Scalable Telecom Analytics**

---

## Overview

This repository implements the constrained coreset (landmark) selection
framework described in the manuscript. The codebase covers the complete
experimental pipeline: data loading, preprocessing, multi-objective
optimization, evaluation, baseline comparison, diagnostic infrastructure,
and automated artifact generation for all manuscript figures and tables.

### Core capabilities

- **KL-optimal integer quotas** with explicit feasibility floor KL\_min(k)
  via lazy-heap greedy (Algorithm 1, Theorem 1 — `geo/kl.py`).
- **Swap-based repair** for weighted proportionality constraints with
  donor/recipient selection and early feasibility termination (Algorithm 2
  — `optimization/repair.py`).
- **Constrained NSGA-II** with constraint-domination ordering and
  bi-objective (MMD + Sinkhorn divergence) optimization (Algorithm 3 —
  `optimization/nsga2_internal.py`).
- **Four constraint modes**: population-share (`w_i = pop_i`),
  municipality-share quota (`w_i ≡ 1`), joint (both), and none (exact-k
  only) — matching manuscript Section IV-B and Table II.
- **Tri-objective ablation**: symmetric KL drift in VAE latent space as a
  third objective (R7).
- **Comprehensive raw-space evaluation**: Nyström error (`e_Nys`), kPCA
  distortion (`e_kPCA`), KRR RMSE for 10 coverage targets (Table V),
  state-conditioned KPI stability, and dual proportionality diagnostics.
- **Seven baseline methods** each with exact-k and quota-matched variants
  (Section VII.E).
- **Automated artifact generation** for all 4 manuscript figures, 5
  manuscript tables, 12 narrative-strengthening figures (N1–N12), and 7
  narrative-strengthening tables (N1–N7).

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy pandas scikit-learn torch matplotlib seaborn

# 2. Prepare data (place the 3 input CSVs in data/)
#    - smp_main.csv
#    - metadata.csv
#    - city_populations.csv

# 3. Build replicate caches (VAE + PCA representations)
python -m coreset_selection.run_all --data-dir data --prep-only

# 4. Run all experiments (R0-R12)
python -m coreset_selection.run_all --data-dir data

# 5. Generate all manuscript artifacts (figures + tables)
python -m coreset_selection.scripts.generate_all_artifacts \
    --runs-root runs_out --out-dir artifacts_out

# 6. Validate compliance
python -m coreset_selection.scripts.verify_compliance \
    --output-dir runs_out --artifacts-dir artifacts_out
```

### Single-run execution

```bash
# Run a specific configuration at a specific k
python -m coreset_selection.run_all --data-dir data --run R1 --k 300

# Parallel execution (4 workers)
python -m coreset_selection.parallel_runner --n-workers 4 --data-dir data

# Artifacts only (assumes runs_out/ is populated)
python -m coreset_selection.run_all --data-dir data --artifacts-only

# Time complexity analysis
python -m coreset_selection.run_all --data-dir data --time-complexity
```

---

## Experiment Configurations (Manuscript Table II)

| ID  | k   | Repr | Constraints              | Objectives      | Seeds | Purpose                                |
|-----|-----|------|--------------------------|-----------------|-------|----------------------------------------|
| R0  | K   | —    | Count-quota c\*(k)       | —               | 1     | Quota path and KL\_min(k) floors       |
| R1  | K   | raw  | Population-share         | MMD, SD         | 5     | **Primary**: budget–fidelity profiles  |
| R2  | 300 | raw  | Population-share         | MMD only        | 1     | Objective ablation (MMD-only)          |
| R3  | 300 | raw  | Population-share         | SD only         | 1     | Objective ablation (SD-only)           |
| R4  | 300 | raw  | Municipality-share quota | MMD, SD         | 1     | Constraint swap                        |
| R5  | 300 | raw  | Joint (pop + muni-quota) | MMD, SD         | 5     | Joint constraints                      |
| R6  | 300 | raw  | None (exact-k only)      | MMD, SD         | 1     | Constraint ablation (unconstrained)    |
| R7  | 300 | VAE  | Population-share         | MMD, SD, SKL    | 1     | SKL ablation (tri-objective)           |
| R8  | 300 | PCA  | Population-share         | MMD, SD         | 1     | Representation transfer (PCA)          |
| R9  | 300 | VAE  | Population-share         | MMD, SD         | 1     | Representation transfer (VAE mean)     |
| R10 | 300 | all  | Population-share         | —               | 1     | Baseline suite (7 methods × 2 regimes)|
| R11 | 300 | raw  | Population-share         | —               | 1     | Diagnostics (proxy stability, alignment)|
| R12 | 300 | raw  | Population-share         | MMD, SD         | 1     | Effort sweep (P × T grid)             |

**K** = {50, 100, 200, 300, 400, 500} (manuscript cardinality grid).

---

## Manuscript-Referenced Artifacts

### Figures (Section VIII)

| Figure | Filename                                  | Section | Description                               |
|--------|-------------------------------------------|---------|-------------------------------------------|
| Fig 1  | `geo_ablation_tradeoff_scatter.pdf`       | VIII-B  | Composition drift vs Nyström error (R6)   |
| Fig 2  | `distortion_cardinality_R1.pdf`           | VIII-C  | 2×2 budget–fidelity profiles (R1)         |
| Fig 3  | `regional_validity_k300.pdf`              | VIII-D  | State-conditioned KPI stability (R1, R5)  |
| Fig 4  | `objective_metric_alignment_heatmap.pdf`  | VIII-K  | Spearman rank correlation heatmap (R11)   |

### Tables (Section VIII)

| Table   | Filename                    | Section | Description                          |
|---------|-----------------------------|---------|--------------------------------------|
| Tab I   | `exp_settings.tex`          | VII     | Hyperparameters (auto from constants)|
| Tab II  | `run_matrix.tex`            | VII     | Run matrix (auto from run\_specs)    |
| Tab III | `r1_by_k.tex`               | VIII-C  | R1 metric envelope vs k             |
| Tab IV  | `proxy_stability.tex`       | VIII-K  | Proxy stability diagnostics (R11)    |
| Tab V   | `krr_multitask_k300.tex`    | VIII-M  | Multi-target KRR RMSE (10 targets)   |

### Narrative-Strengthening Figures (N1–N12)

| Figure | Filename                                 | Purpose                                    |
|--------|------------------------------------------|--------------------------------------------|
| N1     | `kl_floor_vs_k.pdf`                     | KL feasibility planning curve (R0)         |
| N2     | `pareto_front_mmd_sd_k300.pdf`           | Pareto front at k=300 (R1)                 |
| N3     | `objective_ablation_bars_k300.pdf`       | Objective ablation bars (R1/R2/R3)         |
| N4     | `constraint_comparison_bars_k300.pdf`    | Constraint regime comparison (R1/R4/R5/R6) |
| N5     | `effort_quality_tradeoff.pdf`            | Effort sweep (R12)                         |
| N6     | `baseline_comparison_grouped.pdf`        | Baseline comparison (R10)                  |
| N7     | `multi_seed_stability_boxplot.pdf`       | Multi-seed robustness (R1, R5)             |
| N8     | `state_kpi_heatmap.pdf`                  | Per-state KPI drift heatmap                |
| N9     | `composition_shift_sankey.pdf`           | State composition shift (R6 vs R1)         |
| N10    | `pareto_front_evolution.pdf`             | Pareto front convergence by generation     |
| N11    | `nystrom_error_distribution.pdf`         | Nyström error distribution across Pareto   |
| N12    | `krr_worst_state_rmse_vs_k.pdf`          | Worst-state RMSE vs budget (equity)        |

### Narrative-Strengthening Tables (N1–N7)

| Table | Filename                                       | Purpose                                  |
|-------|-------------------------------------------------|------------------------------------------|
| N1    | `constraint_diagnostics_cross_config.tex`       | Cross-config proportionality comparison  |
| N2    | `objective_ablation_summary.tex`                | R1 vs R2 vs R3 ablation summary          |
| N3    | `representation_transfer_summary.tex`           | R1 vs R8 vs R9 transfer summary          |
| N4    | `skl_ablation_summary.tex`                      | R1 vs R7 SKL ablation summary            |
| N5    | `multi_seed_statistics.tex`                     | R1/R5 mean ± std across 5 seeds          |
| N6    | `worst_state_rmse_by_k.tex`                     | Worst-state RMSE equity analysis         |
| N7    | `baseline_paired_unconstrained_vs_quota.tex`    | Baseline paired comparison               |

See [ARTIFACTS.md](ARTIFACTS.md) for the complete catalog with data sources
and manuscript section references.

---

## Repository Structure

```
coreset_selection/
├── config/
│   ├── constants.py          # All manuscript constants (Table I)
│   ├── dataclasses.py        # Configuration dataclasses
│   └── run_specs.py          # Run specifications (Table II)
├── data/
│   ├── brazil_telecom_loader.py  # Data loader (3 CSV inputs, ZIP support)
│   ├── cache.py              # Replicate cache management
│   ├── preprocessing.py      # log(1+x), missingness, standardization
│   ├── target_columns.py     # Coverage target column discovery
│   └── split_persistence.py  # Persistent data splits
├── geo/
│   ├── kl.py                 # Algorithm 1: KL-optimal quotas, lazy-heap
│   ├── info.py               # Geographic group info construction
│   └── projector.py          # Geographic constraint projector
├── constraints/
│   ├── proportionality.py    # Eq. (4): smoothed KL constraints
│   └── calibration.py        # Constraint calibration utilities
├── optimization/
│   ├── nsga2_internal.py     # Algorithm 3: constrained NSGA-II
│   ├── repair.py             # Algorithm 2: swap-based repair
│   ├── operators.py          # Crossover and mutation operators
│   ├── selection.py          # Pareto representative selection
│   └── problem.py            # Problem formulation
├── objectives/
│   ├── mmd.py                # MMD with RFF approximation
│   ├── sinkhorn.py           # Sinkhorn divergence with anchors
│   ├── skl.py                # Symmetric KL divergence (VAE)
│   └── computer.py           # Objective computer factory
├── evaluation/
│   ├── raw_space.py          # Nystrom error, kPCA, KRR (Sec VII.C)
│   ├── geo_diagnostics.py    # Dual proportionality diagnostics
│   ├── kpi_stability.py      # State-conditioned KPI stability
│   ├── r7_diagnostics.py     # R11 proxy stability + alignment
│   └── metrics.py            # Metric utilities
├── baselines/
│   ├── uniform.py            # Stratified random sampling
│   ├── kmeans.py             # k-means representatives
│   ├── herding.py            # Kernel herding (RFF space)
│   ├── farthest_first.py     # Farthest-first traversal
│   ├── kernel_thinning.py    # Kernel thinning [Dwivedi et al.]
│   ├── leverage.py           # Ridge leverage-score sampling
│   ├── dpp.py                # Approximate k-DPP
│   └── variant_generator.py  # Quota-matched variant factory
├── models/
│   ├── vae.py                # beta-VAE (d_z=32, 1500 epochs, early stopping patience=50)
│   └── pca.py                # PCA (d_z=32)
├── artifacts/
│   ├── manuscript_artifacts.py  # All figures + tables generation
│   ├── tables.py             # LaTeX table utilities
│   ├── plots.py              # Plot style utilities
│   └── generator.py          # Legacy artifact generator
├── experiment/
│   ├── runner.py             # Experiment orchestrator (R0-R12)
│   ├── saver.py              # Results persistence
│   └── time_complexity.py    # Per-phase timing analysis
├── scripts/
│   ├── generate_all_artifacts.py  # Top-level artifact generation
│   ├── verify_compliance.py       # Manuscript compliance checks
│   └── ...
├── tests/
│   ├── test_algorithm1.py           # Algorithm 1 unit tests
│   ├── test_algorithm2.py           # Algorithm 2 unit tests
│   ├── test_constraint_modes.py     # Constraint mode integration
│   ├── test_evaluation_protocol.py  # Metric computation tests
│   ├── test_end_to_end.py           # End-to-end integration test
│   ├── test_verify_constants.py     # Constants verification
│   └── ...
├── utils/
│   ├── math.py               # Median heuristic, numerical helpers
│   ├── random.py             # Seed management
│   ├── debug_timing.py       # Per-phase timing instrumentation
│   └── ...
├── README.md                 # This file
├── CHANGELOG.md              # Phase-by-phase change log
└── ARTIFACTS.md              # Complete artifact catalog
```

---

## Key Design Decisions

**Evaluation in standardized raw space.** All evaluation metrics (Nyström
error, kPCA distortion, KRR RMSE) are computed in the standardized raw
attribute space on the evaluation set E (|E|=2000), regardless of which
representation space was used for optimization. This ensures fair
comparison across R1 (raw), R8 (PCA), and R9 (VAE) configurations.

**Bandwidth from E\_train only.** The raw kernel bandwidth σ\_raw is set via
the median heuristic computed on E\_train exclusively (no data leakage from
E\_test). KRR hyperparameters are likewise tuned on E\_train only.

**Dual diagnostics always computed.** Both municipality-share and
population-share proportionality diagnostics are computed for every run
configuration (R1–R12), not only when the corresponding constraint is
active. This supports the cross-configuration analyses in Section VIII.

**10 coverage targets.** The data pipeline constructs all 10 coverage
targets from manuscript Table V, including the 6 derived combined/averaged
targets (Area 4G+5G, Area All, Households 4G+5G, Households All, Residents
4G+5G, Residents All).

---

## Environment Variables

| Variable                    | Default | Description                              |
|-----------------------------|---------|------------------------------------------|
| `CORESET_R11_SOURCE_RUN`    | R1      | Source run for R11 diagnostics           |
| `CORESET_R11_K`             | 300     | k value for R11 diagnostics              |
| `CORESET_R11_SOURCE_SPACE`  | vae     | Source representation for R11            |
| `CORESET_R12_POP_SIZES`     | —       | Override effort sweep pop sizes (CSV)    |
| `CORESET_R12_N_GENS`        | —       | Override effort sweep generations (CSV)  |

---

## Testing

```bash
# Run all tests
python -m pytest coreset_selection/tests/ -v

# Run specific test suites
python -m pytest coreset_selection/tests/test_algorithm1.py -v
python -m pytest coreset_selection/tests/test_algorithm2.py -v
python -m pytest coreset_selection/tests/test_evaluation_protocol.py -v
python -m pytest coreset_selection/tests/test_end_to_end.py -v
python -m pytest coreset_selection/tests/test_verify_constants.py -v
```

---

## Citation

If you use this code, please cite:

> *Constrained Nyström Landmark Selection for Scalable Telecom Analytics*

---

## License

See LICENSE file for details.
