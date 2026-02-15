# Constrained Nystrom Landmark Selection for Scalable Telecom Analytics

## Abstract

This repository implements a **constrained multi-objective coreset selection framework** for scalable telecom analytics on Brazilian municipality-level data. The framework selects a small, representative subset (coreset) of **k** municipalities from a national dataset of **N = 5,570** municipalities across **G = 27** states, such that downstream kernel-based analytics (Nystrom approximation, kernel PCA, kernel ridge regression) remain faithful to the full dataset while satisfying geographic proportionality constraints.

The approach combines **representation learning** (Variational Autoencoder / PCA) with **multi-objective evolutionary optimization** (constrained NSGA-II) to simultaneously minimize distributional divergences (MMD and Sinkhorn divergence) subject to geographic equity constraints. The framework is evaluated across **15 experimental configurations** (R0--R14), **7 baseline methods**, **10 coverage prediction targets**, and **9 coreset cardinalities**, producing a total of **20 manuscript figures** and **13 manuscript tables**.

---

## Research Motivation

Telecom regulators and operators need to monitor service quality indicators across thousands of municipalities. Full-dataset kernel methods (Nystrom approximation, kernel PCA, kernel ridge regression) become computationally prohibitive at national scale. **Coreset selection** addresses this by identifying a compact subset of landmarks that approximate the full dataset's kernel structure.

However, naive coreset selection introduces **geographic composition drift**: small or remote states become underrepresented, biasing state-level analytics. This work introduces **proportionality constraints** that enforce fair geographic representation during subset selection, formulated as a constrained multi-objective optimization problem solved via a custom NSGA-II variant with swap-based constraint repair.

---

## Problem Formulation

Given a dataset **X** of N = 5,570 municipalities with D = 1,863 features, partitioned into G = 27 geographic groups (states), the goal is to select a subset S of cardinality k such that:

```
Minimize:   F(S) = [f_MMD(S), f_Sinkhorn(S)]
Subject to: |S| = k                                    (exact cardinality)
            D_KL(pi || pi_hat(S)) <= tau                (proportionality)
            c_g^- <= |S ∩ I_g| <= c_g^+   for all g    (per-group quotas)
```

Where:
- **f_MMD(S)**: Maximum Mean Discrepancy between S and X, approximated via Random Fourier Features (m = 2,000)
- **f_Sinkhorn(S)**: Sinkhorn divergence between S and X, approximated via anchors (A = 200)
- **pi**: target geographic distribution (population-share or municipality-share)
- **pi_hat(S)**: empirical distribution of the selected subset
- **tau = 0.02**: proportionality tolerance threshold

The decision variable is a binary mask **x** in {0,1}^N with exactly k ones.

For full mathematical details, see [METHODOLOGY.md](METHODOLOGY.md).

---

## Dataset Description

| Property | Value | Source |
|----------|-------|--------|
| Municipalities (N) | 5,570 | All Brazilian municipalities |
| States (G) | 27 | 26 states + Federal District |
| Features (D_total) | 1,863 | Total features after target exclusion |
| Features (D_non_miss) | 973 | Substantive features (excluding missingness indicators) |
| Missingness indicators (D_miss) | 890 | Binary indicators for columns with NaN/Inf values |
| Target snapshots | Coverage: Sep 2025; HHI: 2024; Densidade: 2025 | Per-target temporal reference |
| Primary targets | 2 | 4G and 5G area coverage (%) |
| Total coverage targets | 10 | Area, household, and resident coverage |
| Classification targets | 15 | Market concentration, urbanization, speed tiers, etc. |

**Input files:**

| File | Size | Content |
|------|------|---------|
| `smp_main.csv` | 90 MB | 1,919 columns: 1,011 substantive features + 905 missingness indicators + 3 identifiers (before target exclusion) |
| `metadata.csv` | 346 KB | Geographic coordinates (lat/lon), state codes, names |
| `city_populations.csv` | 168 KB | Population counts per municipality |

**Feature categories:** population, geographic coordinates, base station counts, per-operator coverage percentages (4G/5G), household and resident coverage, broadband speed statistics, income measures, school connectivity, service density, market concentration (HHI), urbanization indicators, and infrastructure quality metrics.

For complete data documentation, see [DATA_PIPELINE.md](DATA_PIPELINE.md).

---

## Preprocessing Pipeline

The preprocessing pipeline consists of 9 sequential steps, all fitted exclusively on the training split to prevent data leakage:

| Step | Operation | Input Shape | Output Shape |
|------|-----------|-------------|-------------|
| 1 | Load and merge 3 CSV files | Raw CSVs | N x (D + metadata) |
| 2 | Column name standardization | Brazilian numeric format | Standardized names |
| 3 | Feature schema inference | All columns | Per-column type assignment |
| 4 | Target column removal (67 regex patterns) | N x D_full | N x D' |
| 5 | Missingness indicators (binary per NaN column) | N x D' | N x (D' + n_missing) |
| 6 | Type-aware imputation (median/mode, fitted on I_train) | With NaNs | No NaNs |
| 7 | Log1p transform on heavy-tailed numeric columns | Skewed | Log-transformed |
| 8 | Selective standardization (zero-mean, unit-variance) | Unscaled | Standardized |
| 9 | Representation learning (VAE or PCA on I_train) | D-dimensional | d_z-dimensional |

**Split protocol (3-tier, stratified by state):**
- **(I_train, I_val)**: 80/20 -- preprocessing statistics, PCA fitting, VAE training
- **E**: |E| = 2,000 -- fixed evaluation set
- **(E_train, E_test)**: 80/20 within E -- KRR hyperparameter tuning and final metrics

---

## Representation Learning

### Variational Autoencoder (VAE)

| Parameter | Value |
|-----------|-------|
| Architecture | Input -> 128 (ReLU) -> 2x32 (mu, logvar) -> 32 -> 128 (ReLU) -> Input |
| Latent dimension (d_z) | 32 |
| Hidden dimension | 128 |
| KL weight (beta) | 0.1 |
| Optimizer | Adam, lr = 1e-3 |
| Training epochs | 1,500 (with early stopping, patience = 200) |
| Batch size | 256 |
| Acceleration | torch.compile() + AMP (automatic mixed precision) |

**Loss function:**

```
L = -MSE(x, x_hat) + 0.5 * beta * sum[1 + logvar - mu^2 - exp(logvar)]
```

### PCA

- **Components**: 32 (matching VAE latent dimension for fair comparison)
- **Implementation**: sklearn PCA, optional whitening

---

## Core Algorithms

Three custom algorithms form the optimization backbone. See [METHODOLOGY.md](METHODOLOGY.md) for pseudocode and full derivations.

### Algorithm 1: KL-Optimal Integer Quotas (`geo/kl.py`)

Computes the integer allocation **c*(k)** that minimizes KL divergence between target distribution pi and subset distribution pi_hat, subject to lower/upper bounds per group. Uses a **lazy-heap greedy** strategy with complexity **O(k log G)**.

Key formula -- marginal gain of adding one unit to group g at count t:

```
Delta_g(t) = pi_g * [log(t + alpha + 1) - log(t + alpha)]
```

### Algorithm 2: Swap-Based Repair (`optimization/repair.py`)

Projects arbitrary binary masks to satisfy both cardinality and quota constraints via **donor/recipient swaps**. Identifies excess groups (donors) and deficit groups (recipients), then performs targeted swaps with early feasibility termination.

### Algorithm 3: Constrained NSGA-II (`optimization/nsga2_internal.py`)

Multi-objective evolutionary optimization with:
- **Constraint-domination ordering** (feasible solutions always preferred)
- **Vectorized fast non-dominated sorting**
- **Uniform binary crossover** (p_c = 0.9) + **quota-swap mutation** (p_m = 0.2)
- **Post-variation repair** via Algorithm 2
- Population size P = 200, generations T = 1,000

---

## Objective Functions

### Maximum Mean Discrepancy (MMD) via Random Fourier Features

Approximates MMD^2 between subset S and full dataset X using m = 2,000 random Fourier features:

```
phi(x) = sqrt(2/m) * cos(Wx + b),   W ~ N(0, I/sigma^2),   b ~ Uniform(0, 2*pi)
MMD^2(S, X) ~ ||mu_S - mu_X||^2,   where mu = E[phi(X)]
```

Bandwidth sigma^2 via median heuristic. Subset evaluation: **O(m)** per candidate.

### Sinkhorn Divergence via Anchors

Approximates Sinkhorn divergence using A = 200 anchor points (k-means++):

```
S(P, Q) = OT_eps(P, Q) - 0.5 * OT_eps(P, P) - 0.5 * OT_eps(Q, Q)
```

With entropic regularization epsilon = eta * median(||r_i - r_j||^2), eta = 0.05, and 100 log-stabilized Sinkhorn iterations.

### Symmetric KL Divergence (R7 only)

Measures drift between moment-matched Gaussians of VAE posteriors:

```
SKL(G_N, G_S) = 0.5 * sum_j [v_Nj/v_Sj + v_Sj/v_Nj - 2 + (Delta_mu_j)^2 * (1/v_Nj + 1/v_Sj)]
```

With variance clamping to [exp(-10), exp(2)] for numerical stability.

---

## Constraint Modes

| Mode | Weight w_i | Target pi_g | Runs |
|------|-----------|-------------|------|
| **Population-share** | population_i | sum(pop in g) / sum(pop) | R1, R2, R3, R5, R7-R14 |
| **Municipality-share quota** | 1 (uniform) | n_g / N | R0, R4 |
| **Joint** | Both simultaneously | Both thresholds | R5 |
| **None** (exact-k only) | -- | -- | R6 |

Proportionality is enforced via **Dirichlet-smoothed KL divergence** with smoothing parameter alpha = 1.0 and tolerance tau = 0.02.

---

## Baseline Methods

Seven baseline coreset selection methods, each implemented in both exact-k and quota-matched variants:

| Method | Algorithm | Complexity | Source |
|--------|-----------|-----------|---------|
| **Uniform** | Stratified random sampling | O(N) | `baselines/uniform.py` |
| **K-means** | Lloyd's + closest to centroid | O(NkD) | `baselines/kmeans.py` |
| **Kernel Herding** | Greedy MMD minimization in RFF space | O(Nkm) | `baselines/herding.py` |
| **Farthest-First** | Greedy k-center | O(Nk) | `baselines/farthest_first.py` |
| **Ridge Leverage** | Importance sampling via RFF ridge scores | O(Nm) | `baselines/leverage.py` |
| **DPP** | Greedy MAP of k-DPP | O(Nk^2) | `baselines/dpp.py` |
| **Kernel Thinning** | Stein operator-based thinning | -- | `baselines/kernel_thinning.py` |

Quota-matched variants are generated via `variant_generator.py`, which applies post-hoc repair to enforce geographic quotas.

---

## Experimental Design

The framework includes **15 experimental configurations** (R0--R14) organized as systematic ablations across 4 axes: objectives, constraints, representations, and computational effort.

| ID | Space | Objectives | Constraints | k-sweep | Reps | Purpose |
|----|-------|-----------|-------------|---------|------|---------|
| R0 | -- | -- | Count-quota | K_GRID | 1 | Quota path & KL_min(k) |
| **R1** | **VAE** | **MMD, SD** | **Pop-share** | **K_GRID** | **5** | **Primary: budget-fidelity** |
| R2 | VAE | MMD | Pop-share | -- | 1 | Objective ablation (MMD-only) |
| R3 | VAE | SD | Pop-share | -- | 1 | Objective ablation (SD-only) |
| R4 | VAE | MMD, SD | Muni-quota | -- | 1 | Constraint swap |
| R5 | VAE | MMD, SD | Joint | -- | 1 | Joint constraints |
| R6 | VAE | MMD, SD | None | -- | 1 | Unconstrained baseline |
| R7 | VAE | MMD, SD, SKL | Pop-share | -- | 1 | Tri-objective ablation |
| R8 | Raw | MMD, SD | Pop-share | K_GRID | 1 | Representation: raw space |
| R9 | PCA | MMD, SD | Pop-share | K_GRID | 1 | Representation: PCA |
| R10 | All | -- | Pop-share | -- | 1 | Baseline suite (7 methods) |
| R11 | VAE | -- | Pop-share | -- | 1 | Diagnostics & stability |
| R12 | VAE | MMD, SD | Pop-share | -- | 1 | Effort sweep (7 configs) |
| R13 | VAE | MMD, SD | Pop-share | -- | 1 | VAE dim sweep (6 dims) |
| R14 | PCA | MMD, SD | Pop-share | -- | 1 | PCA dim sweep (6 dims) |

**K_GRID** = {20, 30, 40, 50, 100, 200, 300, 400, 500}. **D_GRID** = {4, 8, 16, 32, 64, 128}.

For exhaustive per-run documentation, see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Evaluation Protocol

All evaluation metrics are computed in the **standardized raw attribute space** on the evaluation set E (|E| = 2,000), regardless of which representation was used for optimization. This ensures fair comparison across configurations.

### Kernel-Based Metrics

| Metric | Formula | Parameters |
|--------|---------|-----------|
| **Nystrom error** | e_Nys = \|\|K_EE - K_hat\|\|_F / \|\|K_EE\|\|_F | lambda_nys = 1e-6 * tr(W)/k |
| **kPCA distortion** | e_kPCA = \|\|lambda[:r] - lambda_hat[:r]\|\|_2 / \|\|lambda[:r]\|\|_2 | r = 20 components |
| **KRR RMSE** | sqrt(MSE(y_test, Phi_test * w)) | 5-fold CV, 10 targets |

### Geographic Diagnostics

| Metric | Formula |
|--------|---------|
| **KL divergence** | sum_g pi_g * log(pi_g / pi_hat_g), Dirichlet-smoothed (alpha = 1.0) |
| **L1 distance** | sum_g \|pi_g - c_g/k\| |
| **Max deviation** | max_g \|pi_g - c_g/k\| |

### KPI Stability

| Metric | Definition |
|--------|-----------|
| **Max KPI drift** | max_g \|mu_g^S - mu_g^full\| per target |
| **Kendall's tau** | Rank correlation between state means |

### Downstream Models

Regression (KNN-5, RF-100, GBT-50) and classification (KNN, RF, LR, GBT) on Nystrom features. QoS models: OLS, Ridge, Elastic Net, PLS, Constrained OLS, Heuristic with fixed-effects panel variants.

For complete metric documentation, see [EVALUATION_METRICS.md](EVALUATION_METRICS.md).

---

## Generated Artifacts

### Manuscript Figures (8)

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig 1 | `geo_ablation_tradeoff_scatter.pdf` | Composition drift vs Nystrom error (R6 unconstrained) |
| Fig 2 | `distortion_cardinality_R1.pdf` | 2x2 budget-fidelity profiles (R1 primary) |
| Fig 3 | `regional_validity_k300.pdf` | State-conditioned KPI stability (R1 vs R5) |
| Fig 4 | `objective_metric_alignment_heatmap.pdf` | Spearman rank correlation heatmap (R11) |
| N1 | `kl_floor_vs_k.pdf` | KL_min(k) feasibility planning curve (R0) |
| N2 | `pareto_front_mmd_sd_k300.pdf` | Pareto front at k=300 with knee point (R1) |
| N6 | `baseline_comparison_grouped.pdf` | Multi-metric baseline comparison (R10) |
| N12 | `krr_worst_state_rmse_vs_k.pdf` | Worst-state RMSE vs budget (equity) |

### Manuscript Tables (6)

| Table | Filename | Description |
|-------|----------|-------------|
| I | `exp_settings.tex` | Hyperparameters (auto-generated from constants) |
| II | `run_matrix.tex` | Run matrix (auto-generated from run_specs) |
| III | `r1_by_k.tex` | R1 metric envelope vs k |
| IV | `proxy_stability.tex` | Proxy stability diagnostics (R11) |
| V | `krr_multitask_k300.tex` | Multi-target KRR RMSE (10 targets) |
| VI | `repr_timing.tex` | Representation timing |

Plus 12 narrative-strengthening figures (N1--N12) and 7 narrative-strengthening tables (N1--N7). Full catalog at [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Repository Structure

```
Coreset_Codes/
├── README.md                        # This file
├── METHODOLOGY.md                   # Mathematical formulations and algorithms
├── EXPERIMENTS.md                   # Detailed experimental design (R0-R14)
├── DATA_PIPELINE.md                 # Complete data documentation
├── EVALUATION_METRICS.md            # All metrics with formulas
├── ARCHITECTURE.md                  # Software architecture
│
├── data/                            # Input data directory
│   ├── smp_main.csv                 # Main features (91 MB)
│   ├── metadata.csv                 # Geographic metadata
│   └── city_populations.csv         # Population data
│
└── coreset_selection/               # Main Python package
    ├── config/
    │   ├── constants.py             # All manuscript constants (Table I)
    │   ├── dataclasses.py           # Configuration dataclasses
    │   └── run_specs.py             # Run specifications R0-R14 (Table II)
    ├── data/
    │   ├── brazil_telecom_loader.py # Data loader (3 CSVs, ZIP support)
    │   ├── cache.py                 # Replicate cache management
    │   ├── feature_schema.py        # Feature type inference
    │   ├── target_columns.py        # Target detection (67 patterns)
    │   └── derived_targets.py       # 12 regression + 15 classification targets
    ├── geo/
    │   ├── kl.py                    # Algorithm 1: KL-optimal quotas
    │   ├── info.py                  # Geographic group construction
    │   └── projector.py             # Constraint projector
    ├── constraints/
    │   ├── proportionality.py       # KL proportionality constraints
    │   └── calibration.py           # Tau feasibility estimation
    ├── optimization/
    │   ├── nsga2_internal.py        # Algorithm 3: constrained NSGA-II
    │   ├── repair.py                # Algorithm 2: swap-based repair
    │   ├── operators.py             # Crossover and mutation operators
    │   ├── selection.py             # Pareto front knee selection
    │   └── problem.py               # MOO problem formulation
    ├── objectives/
    │   ├── mmd.py                   # MMD via Random Fourier Features
    │   ├── sinkhorn.py              # Sinkhorn divergence via anchors
    │   ├── skl.py                   # Symmetric KL (VAE latent space)
    │   └── computer.py              # Objective computer factory
    ├── evaluation/
    │   ├── raw_space.py             # Nystrom, kPCA, KRR evaluation
    │   ├── enhanced_evaluator.py    # Extended downstream metrics
    │   ├── multi_model_evaluator.py # KNN, RF, LR, GBT evaluation
    │   ├── qos_tasks.py             # QoS prediction models
    │   ├── geo_diagnostics.py       # Proportionality diagnostics
    │   ├── kpi_stability.py         # State-conditioned KPI stability
    │   └── metrics.py               # Coverage and diversity metrics
    ├── baselines/
    │   ├── uniform.py               # Stratified random sampling
    │   ├── kmeans.py                # K-means representatives
    │   ├── herding.py               # Kernel herding (RFF space)
    │   ├── farthest_first.py        # Farthest-first traversal
    │   ├── leverage.py              # Ridge leverage-score sampling
    │   ├── dpp.py                   # k-DPP MAP approximation
    │   ├── kernel_thinning.py       # Kernel thinning
    │   └── variant_generator.py     # Quota-matched variant factory
    ├── models/
    │   ├── vae.py                   # beta-VAE (d_z=32, beta=0.1)
    │   └── pca.py                   # PCA (d_z=32)
    ├── artifacts/
    │   ├── manuscript_artifacts.py  # Figure + table generation
    │   ├── tables.py                # LaTeX table utilities
    │   └── plots.py                 # Plot style utilities
    ├── experiment/
    │   ├── runner.py                # Main experiment orchestrator
    │   ├── saver.py                 # Results persistence
    │   └── time_complexity.py       # Per-phase timing
    ├── cli/
    │   ├── _main.py                 # CLI entry point (argparse)
    │   └── _commands.py             # All CLI commands
    ├── runners/
    │   ├── scenario.py              # Standalone scenario runner
    │   └── parallel.py              # Parallel orchestration
    ├── tests/                       # 17 test files
    └── utils/                       # Math, seed, timing utilities
```

---

## Installation & Reproduction

### Prerequisites

```
Python >= 3.8
numpy >= 1.21, scipy >= 1.7, pandas >= 1.3
scikit-learn >= 1.0, torch >= 1.10
matplotlib >= 3.4, seaborn >= 0.11
```

### Installation

```bash
pip install -e .
# Or install dependencies directly:
pip install numpy scipy pandas scikit-learn torch matplotlib seaborn
```

### Step-by-Step Reproduction

```bash
# 1. Place input data in data/ directory
#    - smp_main.csv, metadata.csv, city_populations.csv

# 2. Pre-build replicate caches (VAE + PCA training)
python -m coreset_selection prep --n-replicates 5 --data-dir data

# 3. Run primary experiment (R1) at all k values
python -m coreset_selection scenario --run-id R1 --k-values 20,30,40,50,100,200,300,400,500

# 4. Run all experiments (R0-R14)
python -m coreset_selection all -k 20,30,40,50,100,200,300,400,500

# 5. Run experiments in parallel
python -m coreset_selection parallel r1 r2 r3 r4 r5 r6 r7 r8 r9 r10

# 6. Generate manuscript artifacts
python -m coreset_selection artifacts --output artifacts_out

# 7. Run test suite
python -m pytest coreset_selection/tests/ -v
```

### Individual Run Shortcuts

```bash
python -m coreset_selection r1 -k 300          # Primary at k=300
python -m coreset_selection r10 -k 300         # Baselines at k=300
python -m coreset_selection r6 -k 300 --source-run R1  # Unconstrained
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORESET_R11_SOURCE_RUN` | R1 | Source run for diagnostics |
| `CORESET_R11_K` | 300 | k value for diagnostics |
| `CORESET_EVAL_NJOBS` | 1 | Parallel evaluation workers |

---

## Companion Documentation

| Document | Description |
|----------|-------------|
| [METHODOLOGY.md](METHODOLOGY.md) | Mathematical formulations, algorithm pseudocode, objective derivations |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed experimental design for all 15 configurations (R0--R14) |
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | Complete data documentation, preprocessing, target definitions |
| [EVALUATION_METRICS.md](EVALUATION_METRICS.md) | All evaluation metrics with mathematical formulas |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Software architecture, CLI reference, testing infrastructure |

---

## Key Design Decisions

1. **Evaluation in standardized raw space.** All metrics (Nystrom error, kPCA distortion, KRR RMSE) are computed in standardized raw attribute space on the evaluation set E, regardless of which representation was used for optimization. This ensures fair comparison across R1 (VAE), R8 (raw), and R9 (PCA).

2. **Bandwidth from E_train only.** The raw kernel bandwidth sigma^2 is computed via the median heuristic on E_train exclusively, preventing data leakage from E_test. KRR hyperparameters are likewise tuned on E_train only.

3. **Dual diagnostics always computed.** Both municipality-share and population-share proportionality diagnostics are computed for every configuration, not only when the corresponding constraint is active. This supports cross-configuration analysis.

4. **10 coverage targets.** The data pipeline constructs all 10 coverage targets from manuscript Table V, including 6 derived combined/averaged targets across technologies and coverage types.

5. **Cache sharing across runs.** Replicate caches ensure identical VAE/PCA representations across all runs sharing the same replicate ID, guaranteeing that differences in results are attributable only to the experimental variable being ablated.

---

## Citation

```bibtex
@article{coreset_telecom_2025,
  title   = {Constrained {Nystr\"om} Landmark Selection for Scalable Telecom Analytics},
  year    = {2025},
  note    = {Repository: \url{https://github.com/...}}
}
```

---

## License

See LICENSE file for details.

<!-- Last verified: 2026-02-15 -->
