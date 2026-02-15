# Experimental Design: Complete Configuration Reference

This document provides exhaustive documentation of all 15 experimental configurations (R0--R14), hyperparameter settings, baseline methods, ablation rationale, seed management, and statistical methodology. It is designed to support the "Experimental Setup" and "Results" sections of a journal paper.

---

## 1. Experimental Framework Overview

### Design Philosophy

The experimental framework is organized as a **systematic ablation study** across four independent axes:

1. **Objective ablation** (R2, R3, R7): Which divergence measures drive selection quality?
2. **Constraint ablation** (R4, R5, R6): How do geographic constraints affect fidelity-equity tradeoffs?
3. **Representation ablation** (R8, R9, R13, R14): Does learned representation improve selection?
4. **Effort ablation** (R12): What are the diminishing returns of increased computation?

The primary configuration **R1** serves as the reference against which all ablations are compared. R10 provides external baselines, and R11 validates proxy-metric alignment.

### Proxy vs. Evaluation Metrics

A key architectural decision is the separation between:
- **Proxy objectives** (MMD, Sinkhorn divergence): computed in the embedding space (VAE/PCA/raw) during optimization
- **Evaluation metrics** (Nystrom error, kPCA distortion, KRR RMSE): computed in the standardized raw attribute space on an independent evaluation set E

This separation ensures that:
1. Optimization operates in a computationally tractable space
2. Evaluation is fair across representations (all evaluated in the same space)
3. Proxy-metric alignment can be validated empirically (R11)

---

## 2. Master Run Specification Table

| ID | Space | Objectives | Constraint Mode | k / Sweep | Dim Sweep | Reps | Baselines | Eval | Purpose |
|----|-------|-----------|----------------|-----------|-----------|------|-----------|------|---------|
| R0 | -- | -- | muni_quota | K_GRID | -- | 1 | No | No | Quota path & KL_min(k) |
| **R1** | **VAE** | **MMD, SD** | **pop_share** | **K_GRID** | -- | **5** | No | **Yes** | **Primary: budget-fidelity** |
| R2 | VAE | MMD | pop_share | single k | -- | 1 | No | Yes | Objective ablation (MMD-only) |
| R3 | VAE | SD | pop_share | single k | -- | 1 | No | Yes | Objective ablation (SD-only) |
| R4 | VAE | MMD, SD | muni_quota | single k | -- | 1 | No | Yes | Constraint swap |
| R5 | VAE | MMD, SD | joint | single k | -- | 5 | No | Yes | Joint constraints |
| R6 | VAE | MMD, SD | none | single k | -- | 1 | No | Yes | Unconstrained baseline |
| R7 | VAE | MMD, SD, SKL | pop_share | single k | -- | 1 | No | Yes | Tri-objective (SKL ablation) |
| R8 | Raw | MMD, SD | pop_share | K_GRID | -- | 1 | No | Yes | Representation: raw features |
| R9 | PCA | MMD, SD | pop_share | K_GRID | -- | 1 | No | Yes | Representation: PCA |
| R10 | All | -- | pop_share | single k | -- | 1 | **Yes** | Yes | 7 baselines x 2 regimes |
| R11 | VAE | -- | pop_share | single k | -- | 1 | No | Yes | Diagnostics & proxy stability |
| R12 | VAE | MMD, SD | pop_share | single k | -- | 1 | No | Yes | Effort sweep (7 configs) |
| R13 | VAE | MMD, SD | pop_share | single k | D_GRID | 1 | No | Yes | VAE dim sweep |
| R14 | PCA | MMD, SD | pop_share | single k | D_GRID | 1 | No | Yes | PCA dim sweep |

**K_GRID** = {50, 100, 200, 300, 400, 500}
**D_GRID** = {4, 8, 16, 32, 64, 128}

---

## 3. Detailed Per-Run Documentation

### R0: Quota Path Computation

**Purpose:** Establish the KL feasibility floor KL_min(k) and compute count-based quotas c*(k) over the full cardinality grid. This provides the planning curve that shows the minimum achievable KL divergence as a function of coreset size.

**Configuration:**
- Space: Raw (no optimization in embedding space)
- Objectives: None (pure quota computation)
- Constraint mode: municipality_share_quota (w_i = 1)
- k-sweep: K_GRID = {50, 100, 200, 300, 400, 500}
- Replicates: 1

**Outputs:**
- `quota_path.json`: Per-group optimal allocations c*(k) for each k
- `kl_floor.csv`: KL_min(k) values establishing feasibility thresholds
- Per-k `.npz` files with detailed allocation data

**Manuscript role:** Generates Figure N1 (KL_min(k) planning curve with tau thresholds).

---

### R1: Primary Configuration (Reference Run)

**Purpose:** The core budget-fidelity analysis. This is the reference run against which all ablations are compared. Evaluates how coreset quality scales with budget k across all evaluation metrics.

**Configuration:**
- Space: VAE (latent_dim = 32)
- Objectives: (MMD, Sinkhorn divergence) -- bi-objective
- Constraint mode: population_share (w_i = population_i)
- k-sweep: K_GRID = {50, 100, 200, 300, 400, 500}
- Replicates: 5 (n_reps = 5 for statistical robustness)
- **Total instances: 6 k-values x 5 replicates = 30 experiment runs**

**What is measured:**
- Nystrom error, kPCA distortion, KRR RMSE across all 10 coverage targets
- Geographic proportionality (KL, L1, max deviation) -- both muni-share and pop-share
- State-conditioned KPI stability (drift, Kendall's tau)
- Multi-model downstream (KNN, RF, GBT)
- Per-state KRR RMSE for equity analysis
- Pareto front statistics (cardinality, spread, hypervolume)
- Wall-clock timing per phase

**Manuscript role:** Generates Figure 2 (budget-fidelity profiles), Table III (metric envelope vs k), Table V (multi-target KRR), Figure N2 (Pareto front), Figure N7 (multi-seed stability), Figure N12 (worst-state RMSE equity).

---

### R2: Single-Objective Ablation (MMD-Only)

**Purpose:** Isolate the contribution of MMD to coreset quality. By removing Sinkhorn divergence, this tests whether a single distributional objective suffices.

**Configuration:**
- Space: VAE (latent_dim = 32)
- Objectives: (MMD,) -- single-objective optimization
- Constraint mode: population_share
- k: User-specified (typically 300)
- Replicates: 1

**Comparison:** Against R1 at matched k. Expected findings: single-objective solutions may achieve lower MMD but higher Sinkhorn divergence, demonstrating complementarity of the two objectives.

**Manuscript role:** Contributes to Figure N3 (objective ablation bars), Table N2 (ablation summary).

---

### R3: Single-Objective Ablation (Sinkhorn-Only)

**Purpose:** Isolate the contribution of Sinkhorn divergence. Complementary to R2.

**Configuration:** Same as R2, but objectives = (sinkhorn,).

**Comparison:** Against R1 and R2 at matched k. Tests whether optimal transport provides different selection pressure than kernel-based MMD.

---

### R4: Constraint Swap (Municipality-Share Quota)

**Purpose:** Test sensitivity to the constraint weighting scheme. Replaces population-share proportionality with count-based municipality-share quotas.

**Configuration:**
- Space: VAE
- Objectives: (MMD, Sinkhorn)
- Constraint mode: municipality_share_quota (w_i = 1 for all i)
- k: User-specified
- Replicates: 1

**Key difference from R1:** Under population-share, large states (SP, MG, RJ) receive more representatives proportional to their population. Under municipality-share, allocation is proportional to the number of municipalities per state instead.

**Manuscript role:** Contributes to Figure N4 (constraint comparison), Table N1 (cross-config proportionality).

---

### R5: Joint Constraints

**Purpose:** Test the strongest constraint regime -- enforcing both population-share and municipality-share quotas simultaneously.

**Configuration:**
- Space: VAE
- Objectives: (MMD, Sinkhorn)
- Constraint mode: joint (both population_share AND municipality_share_quota)
- k: User-specified
- Replicates: 5 (for stability analysis)

**Expected effect:** Tighter constraints reduce the feasible region, potentially increasing proxy objectives but improving geographic equity. The dual-constraint regime is the most restrictive.

**Manuscript role:** Contributes to Figure 3 (regional validity), Figure N4, Figure N7 (multi-seed stability with R1), Table N5 (multi-seed statistics).

---

### R6: Unconstrained Baseline

**Purpose:** Demonstrate the effect of removing all proportionality constraints. Only exact cardinality |S| = k is enforced.

**Configuration:**
- Space: VAE
- Objectives: (MMD, Sinkhorn)
- Constraint mode: none (exact-k only)
- k: User-specified
- Replicates: 1
- Special: Can accept `--source-run R1` and `--source-space vae` for post-hoc analysis of existing solutions

**Expected effect:** Without geographic constraints, optimization is free to concentrate selections in feature-dense regions, leading to **composition drift** where small/remote states become severely underrepresented.

**Manuscript role:** Generates Figure 1 (composition drift vs Nystrom error tradeoff scatter), Figure N9 (composition shift Sankey). This is the key evidence for why constraints are necessary.

---

### R7: Tri-Objective Ablation (SKL)

**Purpose:** Test whether symmetric KL divergence in VAE latent space provides complementary information as a third objective.

**Configuration:**
- Space: VAE
- Objectives: (MMD, Sinkhorn, SKL) -- **tri-objective**
- Constraint mode: population_share
- k: User-specified
- Replicates: 1

**SKL definition:** Symmetric KL between moment-matched Gaussians of VAE posteriors. Measures drift in the learned latent space distribution, potentially capturing information not captured by MMD or Sinkhorn.

**Manuscript role:** Table N4 (SKL ablation summary), Figure `skl_ablation_comparison.pdf`.

---

### R8: Representation Transfer (Raw Space)

**Purpose:** Ablate the value of representation learning. Optimizes directly in the D = 621 standardized raw feature space without any dimensionality reduction.

**Configuration:**
- Space: Raw (standardized features, no VAE/PCA)
- Objectives: (MMD, Sinkhorn)
- Constraint mode: population_share
- k-sweep: K_GRID (full sweep)
- Replicates: 1
- Does NOT require VAE training

**Key question:** Does the VAE's learned 32-dimensional representation improve coreset quality compared to operating directly in high-dimensional space?

**Manuscript role:** Table N3 (representation transfer), Figure `representation_transfer_bars.pdf`.

---

### R9: Representation Transfer (PCA Space)

**Purpose:** Test PCA as an alternative to VAE for representation learning. Uses the same dimensionality (d_z = 32) for fair comparison.

**Configuration:**
- Space: PCA (n_components = 32)
- Objectives: (MMD, Sinkhorn)
- Constraint mode: population_share
- k-sweep: K_GRID
- Replicates: 1
- Requires PCA fitting (requires_pca = True)

**Comparison:** R1 (VAE-32) vs R8 (raw-621) vs R9 (PCA-32). Isolates the effect of nonlinear vs linear dimensionality reduction.

---

### R10: Baseline Suite

**Purpose:** Compare the proposed NSGA-II framework against 7 established coreset selection methods.

**Configuration:**
- Space: VAE (for herding, leverage, DPP); raw for others
- Objectives: None (baseline algorithms have their own selection criteria)
- Constraint mode: population_share
- k: User-specified
- Replicates: 1
- Baselines enabled: True

**Methods (7 total, each in 2 regimes):**

| Method | Algorithm | Regime 1: Exact-k | Regime 2: Quota-matched |
|--------|-----------|-------------------|------------------------|
| Uniform | Stratified random sampling | k random points | Post-hoc quota repair |
| K-means | Lloyd's + closest to centroid | k centroids | Post-hoc quota repair |
| Herding | Greedy MMD minimization (RFF) | k greedy selections | Post-hoc quota repair |
| Farthest-First | Greedy k-center | k farthest points | Post-hoc quota repair |
| RLS | Ridge leverage-score sampling | k importance-sampled | Post-hoc quota repair |
| DPP | Greedy k-DPP MAP | k diverse points | Post-hoc quota repair |
| Kernel Thinning | Stein-based thinning | k thinned points | Post-hoc quota repair |

**Quota-matched variants** use `variant_generator.py` to apply swap-based repair (Algorithm 2) to the unconstrained selection, enforcing the same geographic quotas as R1.

**Manuscript role:** Figure N6 (baseline comparison grouped), Table `baseline_summary_k300.csv`, Table N7 (paired unconstrained vs quota comparison).

---

### R11: Diagnostics (Proxy Stability & Alignment)

**Purpose:** Validate the relationship between proxy objectives (computed during optimization) and evaluation metrics (computed post-hoc in raw space).

**Configuration:**
- Space: VAE
- Objectives: None (loads R1 solutions and re-evaluates)
- Constraint mode: population_share
- k: 300 (default, configurable via CORESET_R11_K env var)
- Source run: R1 (configurable via CORESET_R11_SOURCE_RUN)

**Analyses performed:**
1. **Spearman rank correlations** between proxy objectives (MMD, Sinkhorn in embedding space) and evaluation metrics (Nystrom error, kPCA distortion, KRR RMSE in raw space)
2. **Scatter plots** of proxy vs evaluation metrics across Pareto front solutions
3. **Proxy stability**: Do solutions that are Pareto-optimal in proxy space also perform well in evaluation space?

**Manuscript role:** Figure 4 (objective-metric alignment heatmap), Table IV (proxy stability diagnostics).

---

### R12: Effort Sweep (Diminishing Returns)

**Purpose:** Measure how solution quality improves with increased computational effort (population size x generations).

**Configuration:**
- Space: VAE
- Objectives: (MMD, Sinkhorn)
- Constraint mode: population_share
- k: User-specified (single k)
- Replicates: 1

**Effort grid (7 coupled pairs):**

| Configuration | pop_size (P) | n_gen (T) | Total evaluations (P*T) |
|--------------|-------------|-----------|------------------------|
| 1 (Very low) | 20 | 100 | 2,000 |
| 2 | 50 | 300 | 15,000 |
| 3 | 100 | 500 | 50,000 |
| 4 | 150 | 700 | 105,000 |
| 5 (Default) | 200 | 1,000 | 200,000 |
| 6 | 300 | 1,500 | 450,000 |
| 7 (Maximum) | 400 | 2,000 | 800,000 |

**Design rationale:** Population and generations are coupled (not a Cartesian grid) to reflect realistic scaling. The grid spans a 400x range in total evaluations.

**Manuscript role:** Figure N5 (effort-quality tradeoff), Table `effort_sweep_k300.tex`.

---

### R13: VAE Latent Dimension Sweep

**Purpose:** Test sensitivity to VAE latent dimensionality.

**Configuration:**
- Space: VAE (variable latent_dim)
- Objectives: (MMD, Sinkhorn)
- Constraint mode: population_share
- k: User-specified
- Dimension sweep: D_GRID = {4, 8, 16, **32**, 64, 128}
- Replicates: 1

**For each dimension d in D_GRID:**
1. Train a new VAE with latent_dim = d
2. Embed all N municipalities into d-dimensional space
3. Run NSGA-II optimization in d-dimensional space
4. Evaluate in raw standardized space (same as always)

**Note:** Representation training is re-done for each dimension. Only the NSGA-II optimization uses the new embeddings; evaluation is always in the original 621-dimensional standardized space.

---

### R14: PCA Dimension Sweep

**Purpose:** PCA counterpart to R13. Tests sensitivity to PCA component count.

**Configuration:** Same as R13, but space = PCA with n_components varying over D_GRID.

**Comparison:** R13 (VAE) vs R14 (PCA) at matched dimensions reveals whether nonlinear structure captured by the VAE provides consistent advantages across dimensionalities.

---

## 4. Ablation Study Design Rationale

### Objective Axis (R2, R3, R7 vs R1)

**Question:** Are both MMD and Sinkhorn divergence necessary?

| Comparison | Tests |
|-----------|-------|
| R1 vs R2 | Value of adding Sinkhorn to MMD |
| R1 vs R3 | Value of adding MMD to Sinkhorn |
| R1 vs R7 | Value of adding SKL as third objective |

**Expected insight:** If R2 and R3 both underperform R1, the two objectives provide complementary selection pressure. If R7 does not improve over R1, SKL is redundant.

### Constraint Axis (R4, R5, R6 vs R1)

**Question:** What is the cost of geographic constraints?

| Comparison | Tests |
|-----------|-------|
| R1 vs R6 | Cost of removing ALL constraints (composition drift) |
| R1 vs R4 | Sensitivity to weight scheme (population vs count) |
| R1 vs R5 | Effect of tightening constraints (joint regime) |

**Expected insight:** R6 should achieve better proxy objectives but worse geographic equity. R5 should achieve the best equity but potentially worse fidelity.

### Representation Axis (R8, R9 vs R1, plus R13, R14)

**Question:** Does VAE representation learning improve coreset selection?

| Comparison | Tests |
|-----------|-------|
| R1 vs R8 | VAE (32-d) vs raw (621-d) features |
| R1 vs R9 | VAE (32-d) vs PCA (32-d) -- nonlinear vs linear |
| R13 | VAE sensitivity to latent dimension |
| R14 | PCA sensitivity to component count |

### Effort Axis (R12)

**Question:** Where are the diminishing returns in computational budget?

R12 tests 7 effort levels spanning a 400x range. Expected: quality improves steeply at low effort, then flattens, identifying the minimum viable computation.

---

## 5. Complete Hyperparameter Summary

### NSGA-II Optimization Parameters

| Parameter | Symbol | Value | Source File |
|-----------|--------|-------|------------|
| Population size | P | 200 | `config/constants.py:NSGA2_POP_SIZE` |
| Number of generations | T | 1,000 | `config/constants.py:NSGA2_N_GENERATIONS` |
| Crossover probability | p_c | 0.9 | `config/constants.py:NSGA2_CROSSOVER_PROB` |
| Mutation probability | p_m | 0.2 | `config/constants.py:NSGA2_MUTATION_PROB` |
| Crossover type | -- | Uniform binary | `optimization/operators.py` |
| Mutation type | -- | Quota-swap | `optimization/operators.py` |

### MMD / Random Fourier Features

| Parameter | Symbol | Value | Source File |
|-----------|--------|-------|------------|
| RFF dimension | m | 2,000 | `config/constants.py:RFF_DIM_DEFAULT` |
| Bandwidth scaling | -- | 1.0 (median heuristic) | `config/constants.py` |
| Sensitivity grid | -- | {500, 1000, 2000, 4000} | `config/constants.py:RFF_DIM_SENSITIVITY` |

### Sinkhorn Divergence

| Parameter | Symbol | Value | Source File |
|-----------|--------|-------|------------|
| Number of anchors | A | 200 | `config/constants.py:SINKHORN_N_ANCHORS` |
| Entropy scale factor | eta | 0.05 | `config/constants.py:SINKHORN_ETA` |
| Max iterations | -- | 100 | `config/constants.py:SINKHORN_MAX_ITER` |
| Convergence threshold | -- | 1e-6 | `config/constants.py` |
| Anchor selection method | -- | k-means++ | `config/dataclasses.py` |
| Sensitivity grid | -- | {50, 100, 200, 400} | `config/constants.py:SINKHORN_ANCHOR_SENSITIVITY` |

### Symmetric KL Divergence (R7)

| Parameter | Value | Source File |
|-----------|-------|------------|
| Variance clamp min | exp(-10) ~ 4.54e-5 | `config/constants.py:SKL_VAR_CLAMP_MIN` |
| Variance clamp max | exp(2) ~ 7.389 | `config/constants.py:SKL_VAR_CLAMP_MAX` |

### Geographic Constraints

| Parameter | Symbol | Value | Source File |
|-----------|--------|-------|------------|
| Dirichlet smoothing | alpha | 1.0 | `config/constants.py:ALPHA_GEO` |
| Population-share tolerance | tau_pop | 0.02 | `config/dataclasses.py` |
| Municipality-share tolerance | tau_muni | 0.02 | `config/dataclasses.py` |
| Min one per group | -- | True | `config/dataclasses.py` |

### VAE Architecture

| Parameter | Value | Source File |
|-----------|-------|------------|
| Latent dimension (d_z) | 32 | `config/constants.py:VAE_LATENT_DIM` |
| Hidden dimension | 128 | `config/constants.py:VAE_HIDDEN_DIM` |
| Training epochs | 1,500 | `config/constants.py:VAE_EPOCHS` |
| Early stopping patience | 200 epochs | `config/constants.py:VAE_EARLY_STOPPING_PATIENCE` |
| Batch size | 256 | `config/constants.py:VAE_BATCH_SIZE` |
| Learning rate | 1e-3 | `config/constants.py:VAE_LR` |
| KL weight (beta) | 0.1 | `config/constants.py:VAE_KL_WEIGHT` |
| Optimizer | Adam | `models/vae.py` |

### PCA

| Parameter | Value | Source File |
|-----------|-------|------------|
| Components (d_z) | 32 | `config/constants.py:PCA_DIM` |
| Whitening | False (default) | `config/dataclasses.py` |

### Evaluation Protocol

| Parameter | Value | Source File |
|-----------|-------|------------|
| Evaluation set size | 2,000 | `config/constants.py:EVAL_SIZE` |
| Eval train fraction | 0.8 | `config/constants.py:EVAL_TRAIN_FRAC` |
| kPCA components (r) | 20 | `config/constants.py:KPCA_COMPONENTS` |
| Nystrom regularization | 1e-6 * tr(W)/k | `config/constants.py:NYSTROM_LAMBDA` |
| KRR lambda grid | logspace(-6, 6, 13) | `evaluation/raw_space.py` |
| KRR CV folds | 5 | `evaluation/raw_space.py` |

### Numerical Stability

| Parameter | Value | Source File |
|-----------|-------|------------|
| Normalization epsilon | 1e-12 | `config/constants.py:EPS_NORM` |
| Log epsilon | 1e-30 | `config/constants.py:EPS_LOG` |

---

## 6. Cardinality Grid Analysis

The cardinality grid K_GRID = {50, 100, 200, 300, 400, 500} spans approximately 1% to 9% of the full dataset:

| k | k/N (%) | Ratio to smallest | Interpretation |
|---|---------|-------------------|----------------|
| 50 | 0.90% | 1x | Extreme compression |
| 100 | 1.80% | 2x | High compression |
| 200 | 3.59% | 4x | Moderate compression |
| 300 | 5.39% | 6x | Standard operating point |
| 400 | 7.18% | 8x | Low compression |
| 500 | 8.98% | 10x | Minimal compression |

At k = 50, approximately 2 municipalities per state on average (50/27 ~ 1.85), creating tight constraints. At k = 500, roughly 18.5 per state, allowing more flexibility.

---

## 7. Seed Management & Reproducibility

### Seed Architecture

```
Base seed: 123 (default, configurable via --seed CLI flag)

Per-replicate seed derivation:
  rep_seed = base_seed + rep_id
  Example: rep_id=0 -> seed=123, rep_id=4 -> seed=127

Seed propagation:
  set_global_seed(rep_seed):
    numpy.random.seed(rep_seed)
    torch.manual_seed(rep_seed)
    random.seed(rep_seed)
    torch.cuda.manual_seed_all(rep_seed)  # if CUDA available
```

### Cache Sharing for Fair Comparison

Replicate caches are built once and shared across all runs using the same replicate ID. This ensures:
- R1, R2, R3, R4, R5, R6, R7 all use **identical** VAE embeddings for the same rep_id
- R8 uses the same raw features
- R9 uses the same PCA embeddings
- Differences in results are attributable **only** to the experimental variable being ablated

### Determinism

- VAE training is seeded but not fully deterministic due to CUDA non-determinism (unless torch.use_deterministic_algorithms is set)
- NSGA-II uses numpy random state, which is deterministic given the same seed
- Evaluation metrics are deterministic (no random operations)

---

## 8. Baseline Methods -- Detailed Descriptions

### 8.1 Uniform Random Sampling (`baselines/uniform.py`)

**Algorithm:** Stratified random sampling. For each geographic group g, sample proportional to the target distribution pi_g. If quotas are specified, sample exactly c_g* from each group.

**Complexity:** O(N) for sampling.

**Properties:** Unbiased estimator of the population distribution. No optimization of kernel or distributional properties.

### 8.2 K-Means Representatives (`baselines/kmeans.py`)

**Algorithm:** Run Lloyd's k-means with k clusters on the feature matrix X. For each cluster, select the point closest to the cluster centroid (medoid selection).

**Complexity:** O(N * k * D * iterations) for k-means.

**Properties:** Covers the feature space geometry but does not optimize kernel approximation quality.

### 8.3 Kernel Herding (`baselines/herding.py`)

**Algorithm:** Greedy MMD minimization in RFF feature space. At each step, select the point that most reduces the MMD between the selected subset and the full dataset:

```
s_{t+1} = argmax_{x in X \ S_t} phi(x)^T * (mu_X - mu_{S_t})
```

**Complexity:** O(N * k * m) where m = RFF dimension.

**Properties:** Directly minimizes MMD. Convergence rate O(1/k) for the MMD. Deterministic given a fixed RFF basis.

### 8.4 Farthest-First Traversal (`baselines/farthest_first.py`)

**Algorithm:** Greedy k-center. Start with a random point. Iteratively select the point farthest from the current selection:

```
s_{t+1} = argmax_{x in X \ S_t} min_{s in S_t} ||x - s||
```

**Complexity:** O(N * k) per iteration (maintaining nearest-neighbor distances).

**Properties:** Provides a 2-approximation to the optimal k-center cost. Maximizes coverage but may not optimize distributional properties.

### 8.5 Ridge Leverage Score Sampling (`baselines/leverage.py`)

**Algorithm:** Importance sampling based on RFF ridge leverage scores. Compute leverage scores:

```
l_i = phi(x_i)^T * (Phi^T * Phi + lambda * I)^{-1} * phi(x_i)
```

Sample k points proportionally to leverage scores.

**Complexity:** O(N * m + m^3) for leverage computation.

**Properties:** Theoretically optimal for kernel ridge regression approximation. The lambda parameter controls the bias-variance tradeoff.

### 8.6 Determinantal Point Process (`baselines/dpp.py`)

**Algorithm:** Greedy MAP approximation of k-DPP. Select points that maximize the determinant of the kernel submatrix, promoting diversity:

```
s_{t+1} = argmax_{x in X \ S_t} det(K_{S_t union {x}, S_t union {x}})
```

Uses the conditional gain formulation for efficiency.

**Complexity:** O(N * k^2) for greedy selection.

**Properties:** Balances quality (diagonal kernel entries) and diversity (off-diagonal repulsion). Strong theoretical guarantees for diverse subset selection.

### 8.7 Kernel Thinning (`baselines/kernel_thinning.py`)

**Algorithm:** Stein operator-based thinning. Uses the kernel Stein discrepancy framework to iteratively thin the dataset while preserving distributional properties.

**Properties:** Recent method with strong theoretical backing for distribution preservation.

### Quota-Variant Generation (`baselines/variant_generator.py`)

For each baseline, a quota-matched variant is generated by applying Algorithm 2 (swap-based repair) to the unconstrained selection. This enforces the same geographic quotas as R1, enabling fair comparison of selection quality under identical constraints.

---

## 9. Statistical Analysis Methodology

### Multi-Seed Aggregation (R1, R5)

For runs with multiple replicates (R1: 5 seeds, R5: 5 seeds):
- **Mean +/- std** reported for all metrics
- Box plots show inter-seed variability (Figure N7)
- Per-seed Pareto front statistics

### Proxy-Metric Alignment (R11)

**Spearman rank correlations** between:
- Proxy objectives: MMD (embedding), Sinkhorn (embedding)
- Evaluation metrics: Nystrom error (raw), kPCA distortion (raw), KRR RMSE (raw)

Computed across all solutions on the Pareto front at a fixed k. High positive correlation validates that optimizing proxy objectives improves downstream evaluation metrics.

### State Ranking Preservation

**Kendall's tau** between state-mean vectors computed on the full dataset vs. the coreset. Measures whether the relative ranking of states (by coverage, income, etc.) is preserved in the subset.

### Paired Comparisons

For ablation studies, results are compared at matched k values:
- R1 knee solution vs R2 best solution (at same k)
- R1 knee solution vs R6 knee solution (constrained vs unconstrained)
- Baseline vs baseline at matched k

### Pareto Front Statistics

For each optimization run:
- **Front cardinality**: number of non-dominated solutions
- **Spread**: maximum extent in each objective dimension
- **Hypervolume**: volume dominated by the Pareto front relative to a reference point

---

## 10. Computational Requirements

### Thread Management

The framework automatically tunes thread counts for parallel execution:

```
threads_per_job = max(2, min(16, n_cores // n_jobs))
```

Environment variables set per process:
- `OMP_NUM_THREADS`: OpenMP threads
- `MKL_NUM_THREADS`: Intel MKL threads
- `OPENBLAS_NUM_THREADS`: OpenBLAS threads
- `NUMEXPR_MAX_THREADS`: NumExpr threads
- `MKL_THREADING_LAYER=GNU`: Share OpenMP with PyTorch
- `OMP_MAX_ACTIVE_LEVELS=1`: Disable nested parallelism
- `KMP_BLOCKTIME=200`: Keep worker threads spinning (200ms)

### Execution Phases

Parallel execution follows a 3-phase protocol:

1. **Phase 0 (Sequential):** Pre-build all required replicate caches (VAE + PCA training). File-level locking prevents concurrent builds.
2. **Phase 1 (Parallel):** Launch all scenarios in parallel via ProcessPoolExecutor. Each process has isolated thread limits.
3. **Phase 2 (Sequential):** Collect results, print timing summary.

### Per-Run Timing

Each run records wall-clock timing per phase in `wall_clock.json`:
- Cache loading
- Objective computer construction
- NSGA-II optimization
- Representative selection
- Raw-space evaluation
- Result saving

### Cluster Submission

A SLURM job array template is provided in `scripts/slurm_array_job.sh` for cluster-scale execution.

---

## 11. Generated Artifacts Catalog

### Manuscript Figures

| ID | Filename | Source Runs | Description |
|----|----------|-----------|-------------|
| Fig 1 | `geo_ablation_tradeoff_scatter.pdf` | R1, R6 | Composition drift vs Nystrom error |
| Fig 2 | `distortion_cardinality_R1.pdf` | R1 | 2x2 budget-fidelity panel |
| Fig 3 | `regional_validity_k300.pdf` | R1, R5 | State-conditioned KPI stability |
| Fig 4 | `objective_metric_alignment_heatmap.pdf` | R11 | Spearman rank correlation heatmap |

### Narrative-Strengthening Figures (N1--N12)

| ID | Filename | Source Runs | Description |
|----|----------|-----------|-------------|
| N1 | `kl_floor_vs_k.pdf` | R0 | KL_min(k) planning curve + tau thresholds |
| N2 | `pareto_front_mmd_sd_k300.pdf` | R1 | Pareto front + knee + R2/R3 overlay |
| N3 | `objective_ablation_bars_k300.pdf` | R1, R2, R3 | Grouped bar ablation |
| N4 | `constraint_comparison_bars_k300.pdf` | R1, R4, R5, R6 | 2x2 constraint regime panel |
| N5 | `effort_quality_tradeoff.pdf` | R12 | Effort sweep diminishing returns |
| N6 | `baseline_comparison_grouped.pdf` | R10 | Multi-metric baseline comparison |
| N7 | `multi_seed_stability_boxplot.pdf` | R1, R5 | Seed robustness boxplots |
| N8 | `state_kpi_heatmap.pdf` | R1 | Per-state KPI drift heatmap |
| N9 | `composition_shift_sankey.pdf` | R1, R6 | pi vs pi_hat side-by-side |
| N10 | `pareto_front_evolution.pdf` | R1 | Front overlay by generation |
| N11 | `nystrom_error_distribution.pdf` | R1 | e_Nys histogram across Pareto |
| N12 | `krr_worst_state_rmse_vs_k.pdf` | R1 | Worst-state RMSE equity |

### Manuscript Tables

| ID | Filename | Source | Description |
|----|----------|--------|-------------|
| I | `exp_settings.tex` | constants.py | Hyperparameters (auto-generated) |
| II | `run_matrix.tex` | run_specs.py | Configuration matrix (auto-generated) |
| III | `r1_by_k.tex` | R1 | Metric envelope vs k |
| IV | `proxy_stability.tex` | R11 | Proxy stability diagnostics |
| V | `krr_multitask_k300.tex` | R1 | 10-target KRR RMSE |
| VI | `repr_timing.tex` | R1 | Representation timing |

### Narrative-Strengthening Tables (N1--N7)

| ID | Filename | Source | Description |
|----|----------|--------|-------------|
| N1 | `constraint_diagnostics_cross_config.tex` | R1, R4, R5, R6 | Cross-config proportionality |
| N2 | `objective_ablation_summary.tex` | R1, R2, R3 | Ablation summary |
| N3 | `representation_transfer_summary.tex` | R1, R8, R9 | Transfer summary |
| N4 | `skl_ablation_summary.tex` | R1, R7 | SKL increment |
| N5 | `multi_seed_statistics.tex` | R1, R5 | Mean +/- std across 5 seeds |
| N6 | `worst_state_rmse_by_k.tex` | R1 | Worst-state RMSE equity |
| N7 | `baseline_paired_unconstrained_vs_quota.tex` | R10 | Baseline paired comparison |

All figures are output at 300 dpi in PDF format with IEEE-compliant column widths (3.5" single-column, 7" double-column) and font sizes (>=8pt annotations, >=10pt axis labels). All tables are output in LaTeX format with companion CSV files for data access.
