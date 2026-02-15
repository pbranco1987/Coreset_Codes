# Evaluation Metrics: Complete Reference

This document provides exhaustive documentation of all evaluation metrics, including mathematical formulas, computation details, parameter settings, and interpretation. It is designed to support the "Evaluation Protocol" section of a journal paper.

---

## 1. Evaluation Protocol Overview

### Core Principle

All evaluation metrics are computed in the **standardized raw attribute space** (D = 1,863 dimensions) on a fixed evaluation set **E** (|E| = 2,000), regardless of which representation space (VAE, PCA, or raw) was used during optimization.

This design ensures:
1. **Fair comparison** across representation ablations (R1 vs R8 vs R9)
2. **No double-counting** of representation quality in metric computation
3. **Independence** between proxy objectives and evaluation metrics

### Evaluation Set Construction

The evaluation set E is constructed via stratified sampling:
- |E| = 2,000 municipalities (from N = 5,570)
- Stratified by state to ensure geographic representativeness
- Fixed per replicate (same E for all runs sharing a rep_id)
- Split internally: E_train (1,600) and E_test (400), also stratified by state

### RBF Kernel

All kernel-based metrics use the RBF (Gaussian) kernel:

```
K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
```

**Bandwidth estimation:** sigma^2 is computed via the **median heuristic** on E_train:

```
sigma^2 = median({||x_i - x_j||^2 : i, j in E_train, i < j}) / 2
```

To avoid O(|E|^2) computation, the median is estimated from a random subsample of 2,048 pairs.

**No data leakage:** The bandwidth is computed exclusively on E_train. E_test is not used for any fitted parameter.

**Implementation:** `coreset_selection/evaluation/raw_space.py` (`RawSpaceEvaluator` class)

### Target-to-Evaluator Mapping

The 37 target variables are not all evaluated by every component. Each evaluator uses a specific subset:

| Evaluator Component | Input Features (X) | Target(s) (y) | # Targets | Type |
|---|---|---|---|---|
| KRR (Section 4) | Nystrom features (k dims) | 10 coverage targets (2 primary + 8 derived) | 10 | Regression |
| Multi-Model Regression (Section 5) | Nystrom features (k dims) | 12 extra regression targets | 12 | Regression |
| Multi-Model Classification (Section 5) | Nystrom features (k dims) | 10--15 classification targets (validated) | 10--15 | Classification |
| KPI Stability (Section 8) | -- (direct state-mean comparison) | 10 coverage targets | 10 | Regression |
| QoS Prediction (Section 6) | **Raw features (all D = 1,863)** | `qf_mean` (Qualidade do Funcionamento) | 1 | Regression |
| Nystrom Error (Section 2) | Feature-space kernel | None | 0 | -- |
| kPCA Distortion (Section 3) | Feature-space kernel | None | 0 | -- |
| Coverage & Diversity (Section 9) | Feature-space distances | None | 0 | -- |
| Geographic Diagnostics (Section 7) | Index-space only | None | 0 | -- |

The target data flow is orchestrated in `_runner_eval.py`:
- Coverage targets (10) are built by `_build_multitarget_y()` from cache and passed to `RawSpaceEvaluator`
- Extra regression targets (12) and classification targets (10--15) are stored in cache metadata and passed separately to `multi_model_downstream()`
- The QoS target (`qf_mean`) is extracted from cache metadata and passed to `qos_coreset_evaluation()` with the full raw feature matrix

---

## 2. Nystrom Approximation Error

### Definition

The Nystrom error measures how well the coreset S approximates the full kernel matrix K_EE:

```
e_Nys(S) = ||K_EE - K_hat_EE||_F / ||K_EE||_F
```

Where ||.||_F is the Frobenius norm.

### Computation

**Step 1: Cross-kernel matrix**
```
C = K_{E,S} ∈ R^{|E| x k}
C_ij = K(x_{E_i}, x_{S_j})
```

**Step 2: Landmark kernel matrix**
```
W = K_{S,S} ∈ R^{k x k}
W_ij = K(x_{S_i}, x_{S_j})
```

**Step 3: Regularization**
```
lambda_nys = NYSTROM_LAMBDA * tr(W) / k
           = 1e-6 * tr(W) / k
```

The regularization prevents ill-conditioning when W is near-singular.

**Step 4: Nystrom approximation**
```
K_hat_EE = C * (W + lambda_nys * I)^{-1} * C^T
```

**Step 5: Error computation**
```
e_Nys = ||K_EE - K_hat_EE||_F / ||K_EE||_F
```

### Nystrom Features

The Nystrom approximation also produces a feature embedding:

```
Phi = C * (W + lambda_nys * I)^{-1/2} ∈ R^{|E| x k}
```

This embedding is used for downstream KRR and multi-model evaluation. The matrix square root is computed via eigendecomposition of W.

### Interpretation

- e_Nys = 0: Perfect approximation (coreset spans the kernel space)
- e_Nys -> 1: Poor approximation
- Lower is better; typically decreases with increasing k

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| lambda_nys scaling | 1e-6 | `config/constants.py:NYSTROM_LAMBDA` |
| Kernel | RBF | `evaluation/raw_space.py` |
| Bandwidth | Median heuristic (E_train) | `evaluation/raw_space.py` |

---

## 3. Kernel PCA Spectral Distortion

### Definition

Measures how well the top-r eigenvalues of the kernel matrix are preserved when using the Nystrom approximation:

```
e_kPCA(S) = ||lambda_real[:r] - lambda_approx[:r]||_2 / ||lambda_real[:r]||_2
```

### Computation

**Step 1: Center the kernel matrices**

Using the centering matrix J = I - (1/|E|) * 11^T:

```
K_c = J * K_EE * J          (centered true kernel)
K_hat_c = J * K_hat_EE * J  (centered approximate kernel)
```

**Step 2: Eigendecomposition**
```
lambda_real = top-r eigenvalues of K_c (sorted descending)
lambda_approx = top-r eigenvalues of K_hat_c (sorted descending)
```

**Step 3: Relative L2 error**
```
e_kPCA = ||lambda_real[:r] - lambda_approx[:r]||_2 / ||lambda_real[:r]||_2
```

### Interpretation

- e_kPCA = 0: Perfect spectral preservation
- e_kPCA -> 1: Major spectral distortion
- Captures whether the coreset preserves the principal modes of variation in kernel space

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Components (r) | 20 | `config/constants.py:KPCA_COMPONENTS` |

---

## 4. Kernel Ridge Regression (KRR) RMSE

### Definition

Evaluates whether the coreset-based Nystrom features produce accurate predictions for 10 coverage targets on the held-out evaluation test set.

### Computation

**Step 1: Construct Nystrom features**
```
Phi_train = Nystrom features for E_train (|E_train| x k)
Phi_test = Nystrom features for E_test (|E_test| x k)
```

**Step 2: Ridge regression with CV**

For each target t:
```
w_t = (Phi_train^T * Phi_train + lambda_ridge * I)^{-1} * Phi_train^T * y_train_t
```

Lambda selection: 5-fold cross-validation on E_train, searching over:
```
lambda_grid = logspace(-6, 6, 13) = [1e-6, 1e-5, ..., 1e5, 1e6]
```

**Step 3: Prediction and RMSE**
```
y_hat_t = Phi_test * w_t
RMSE_t = sqrt(mean((y_test_t - y_hat_t)^2))
```

### Target List (10 Coverage Targets)

| # | Target | Display Name |
|---|--------|-------------|
| 1 | cov_area_4G | Area (4G) |
| 2 | cov_area_5G | Area (5G) |
| 3 | cov_hh_4G | Households (4G) |
| 4 | cov_res_4G | Residents (4G) |
| 5 | cov_area_4G_5G | Area (4G + 5G) |
| 6 | cov_area_all | Area (All) |
| 7 | cov_hh_4G_5G | Households (4G + 5G) |
| 8 | cov_hh_all | Households (All) |
| 9 | cov_res_4G_5G | Residents (4G + 5G) |
| 10 | cov_res_all | Residents (All) |

**Scope note:** KRR evaluation uses only the 10 coverage targets (continuous, percentage-point scale). The 12 extra regression targets and 15 classification targets are evaluated separately by the multi-model downstream pipeline (Section 5).

### Output

Per target: `krr_rmse_{target_name}` and `krr_lambda_{target_name}` (selected regularization).

---

## 5. Multi-Model Downstream Evaluation

### Purpose

Tests whether Nystrom features from the coreset support diverse downstream learning algorithms, not just KRR.

**Implementation:** `coreset_selection/evaluation/multi_model_evaluator.py`

### Target Scope

The multi-model evaluator handles the 12 extra regression targets and 10--15 classification targets that are NOT covered by KRR. This separation exists because KRR uses Nystrom features with the RBF kernel (suited to coverage targets), while the multi-model pipeline tests whether Nystrom features generalise to diverse downstream tasks.

### Regression Targets (12)

Defined in `_EXTRA_REG_COLUMN_MAP` in `derived_targets.py`. Each is a single raw column from `smp_main.csv` with NaN replaced by 0.0.

| # | Target | Description |
|---|--------|-------------|
| 1 | velocidade_mediana_mean | Mean of median download speeds |
| 2 | velocidade_mediana_std | Std of median download speeds |
| 3 | pct_limite_mean | Mean % of connections with data caps |
| 4 | renda_media_mean | Mean average household income |
| 5 | renda_media_std | Std of average household income |
| 6 | HHI SMP_2024 | Herfindahl-Hirschman Index, mobile (SMP) market |
| 7 | HHI SCM_2024 | HHI, fixed broadband (SCM) market |
| 8 | pct_fibra_backhaul | % backhaul using fiber optics |
| 9 | pct_escolas_internet | % schools with internet |
| 10 | pct_escolas_fibra | % schools with fiber |
| 11 | Densidade_Banda Larga Fixa_2025 | Fixed broadband density (subscriptions per 100 inhabitants) |
| 12 | Densidade_Telefonia Movel_2025 | Mobile telephony density (subscriptions per 100 inhabitants) |

### Classification Targets (10--15, validated)

Derived in `derived_targets.py` via thresholding, binning, or cross-tabulation. Only targets that pass a minimum class-fraction threshold after binning are included; the actual count (10--15) depends on data distribution.

**Strict tier** (>= 5% minimum class fraction, 10 candidates):

| # | Target | Classes | Derivation |
|---|--------|---------|-----------|
| 1 | concentrated_mobile_market | 2 (binary) | HHI SMP >= 0.25 |
| 2 | high_fiber_backhaul | 2 (binary) | Median split on non-zero values |
| 3 | high_speed_broadband | 2 (binary) | Median split |
| 4 | has_5g_coverage | 2 (binary) | Direct binary column |
| 5 | urbanization_level | 3 (tercile) | Tercile bins of pct_urbano |
| 6 | broadband_speed_tier | 3 (tercile) | Tercile bins of velocidade_mediana_mean |
| 7 | income_tier | 3 (tercile) | Tercile bins of renda_media_mean |
| 8 | mobile_penetration_tier | 4 (quartile) | Quartile bins of mobile density |
| 9 | infra_density_tier | 5 (quintile) | Quintile bins of broadband density |
| 10 | road_coverage_4g_tier | 5 (quintile) | Quintile bins of road 4G coverage |

**Relaxed tier** (>= 2% minimum class fraction with failsafe binning, 5 candidates):

| # | Target | Classes | Derivation |
|---|--------|---------|-----------|
| 11 | income_speed_class | 4 | Cross-tab: income x speed (2x2) |
| 12 | urban_rural_extremes | 4 | Extreme bins (p3, p50, p97) of pct_urbano |
| 13 | income_extremes | 4 | Extreme bins of renda_media_mean |
| 14 | speed_extremes | 4 | Extreme bins of velocidade_mediana_mean |
| 15 | pop_5g_digital_divide | 4 | Cross-tab: population tier x 5G presence |

### Regression Models

| Model | Configuration | Metrics |
|-------|--------------|---------|
| KNN | n_neighbors=5, weights='distance' | RMSE, MAE, R^2 |
| Random Forest | n_estimators=100, max_depth=10 | RMSE, MAE, R^2 |
| Gradient Boosted Trees | n_estimators=50, max_depth=5 | RMSE, MAE, R^2 |

All models are trained on Phi_train and evaluated on Phi_test.

### Classification Models

| Model | Configuration | Metrics |
|-------|--------------|---------|
| KNN | n_neighbors=5, weights='distance' | Accuracy, Balanced Accuracy, Macro F1 |
| Random Forest | n_estimators=100, max_depth=10 | Accuracy, Balanced Accuracy, Macro F1 |
| Logistic Regression | max_iter=1000 | Accuracy, Balanced Accuracy, Macro F1 |
| Gradient Boosted Trees | n_estimators=50, max_depth=5 | Accuracy, Balanced Accuracy, Macro F1 |

### Target-Model Pairing

- **Regression models** (KNN, RF, GBT) evaluate all **12 extra regression targets** -> output keys: `{model}_rmse_{target}`, `{model}_mae_{target}`, `{model}_r2_{target}`
- **Classification models** (KNN, RF, LR, GBT) evaluate all **validated classification targets** -> output keys: `{model}_accuracy_{target}`, `{model}_bal_accuracy_{target}`, `{model}_macro_f1_{target}`

### Output Key Format

```
{model}_rmse_{target}      # Regression RMSE
{model}_mae_{target}       # Regression MAE
{model}_r2_{target}        # Regression R^2
{model}_accuracy_{target}  # Classification accuracy
{model}_bal_accuracy_{target}  # Balanced accuracy
{model}_macro_f1_{target}  # Macro F1 score
```

### Parallelization

- Controlled by `CORESET_EVAL_NJOBS` environment variable
- Uses joblib with loky backend
- Default: sequential (1 worker)

---

## 6. QoS Prediction Models

### Purpose

Evaluates coreset quality for Quality-of-Service (QoS) prediction, a key telecom analytics task. Tests 6 regression models with optional panel-data fixed-effects variants.

**Implementation:** `coreset_selection/evaluation/qos_tasks.py`

### Feature and Target Scope

Unlike KRR and multi-model evaluation (which use Nystrom features as input), QoS prediction uses the **full raw standardized feature matrix** (all D = 1,863 dimensions) as predictors. The target variable is `qf_mean` -- the Qualidade do Funcionamento (technical quality of service) from Anatel's ISG (Indice de Satisfacao Geral) satisfaction survey (scale [0, 1], 4,946 / 5,570 municipalities with data, NaN filled with 0.0).

- **Features (X):** `assets.X_scaled` -- all 1,863 raw standardized features
- **Training data:** Coreset rows (`X[S_idx]`, `y[S_idx]`)
- **Test data:** Held-out evaluation indices (`X[eval_test_idx]`, `y[eval_test_idx]`)
- **Target (y):** `qf_mean` -- extracted from `smp_main.csv` and stored in cache metadata as `qos_target`

This design answers a different question than KRR: can a small coreset serve as a representative training set for classical regression models operating on the original high-dimensional feature space?

### Models

| Model | Method | Hyperparameter Selection |
|-------|--------|------------------------|
| **OLS** | sklearn LinearRegression | None |
| **Ridge** | sklearn Ridge | 6 alphas: [0.01, 0.1, 1.0, 10, 100, 1000], TimeSeriesSplit CV |
| **Elastic Net** | sklearn ElasticNet | alpha=0.01, l1_ratio=0.5 |
| **PLS** | sklearn PLSRegression | CV for n_components (max_components=None, cv_folds=5) |
| **Constrained OLS** | SLSQP optimization | epsilon=0.05, max_weight=0.50, variance_strength=0.01 |
| **Heuristic** | ISG composite weights | Default ISG weighting scheme |

### Constrained OLS Details

The constrained OLS formulation uses scipy's SLSQP optimizer:

```
Minimize:   ||y - X*w||^2 + variance_strength * sum(w_i^2)
Subject to: epsilon <= w_i <= max_weight   for all i
            sum(w_i) = 1
```

When D >> N (underdetermined), PCA reduction is applied first to reduce dimensionality.

### Fixed-Effects (Panel Data) Variants

For panel data structures, a **Demeaner** class provides within-transformation:

```
Fit:       Compute per-entity means: mu_e = mean(X_e)
Transform: X_demeaned = X - mu_e  (subtract entity means)
Reinflate: y_hat = y_hat_demeaned + mu_e  (add back means)
```

Each model is evaluated in two variants:
- **Pooled:** Standard regression ignoring entity structure
- **Fixed-effects (FE):** After within-transformation

### Autoregressive Distributed Lag (ADL/ARX)

For time-series panel data, `build_lagged_features()` constructs:
```
X_lag = [X_{t-1}, X_{t-2}, ..., X_{t-n_lags}, y_{t-1}, y_{t-2}, ..., y_{t-n_lags}]
```

Rows with incomplete lags are dropped.

### Output Metrics Per Model

| Metric | Description |
|--------|-------------|
| `{model}_pooled_rmse` | Pooled RMSE |
| `{model}_pooled_mae` | Pooled MAE |
| `{model}_pooled_r2` | Pooled R^2 |
| `{model}_fe_rmse` | Fixed-effects RMSE |
| `{model}_fe_mae` | Fixed-effects MAE |
| `{model}_fe_r2` | Fixed-effects R^2 |
| `{model}_tail_p90` | 90th percentile absolute error |
| `{model}_tail_p95` | 95th percentile absolute error |
| `{model}_tail_p99` | 99th percentile absolute error |
| `{model}_tail_max` | Maximum absolute error |

### Reference Baseline

`qos_fullset_reference()` evaluates the same models using the **full training set** instead of the coreset, providing an upper-bound baseline for comparison.

---

## 7. Geographic Proportionality Diagnostics

### Purpose

Measures whether the coreset preserves the geographic distribution of the full dataset. Computed for **every** run configuration, not only when constraints are active.

**Implementation:** `coreset_selection/evaluation/geo_diagnostics.py`

### KL Divergence (Dirichlet-Smoothed)

```
D_KL(pi || pi_hat^alpha) = sum_{g=1}^{G} pi_g * log(pi_g / pi_hat_g^alpha)
```

Where:
```
pi_hat_g^alpha = (c_g(S) + alpha) / (|S| + alpha * G)
```

- `pi_g`: Target distribution (population-share or municipality-share)
- `c_g(S)`: Count of selected municipalities in state g
- `alpha = 1.0`: Dirichlet smoothing (prevents infinite KL when c_g = 0)
- `G = 27`: Number of states

### L1 Distance (Unsmoothed)

```
||pi - pi_hat(S)||_1 = sum_{g=1}^{G} |pi_g - c_g(S) / |S||
```

### Maximum Absolute Deviation

```
max_g |pi_g - c_g(S) / |S||
```

Identifies the single most over- or under-represented state.

### Shannon Entropy

```
H(pi_hat) = -sum_{g=1}^{G} pi_hat_g * log(pi_hat_g)
```

Higher entropy indicates more uniform distribution. Maximum entropy = log(G) = log(27) ~ 3.30.

### Herfindahl-Hirschman Index (HHI)

```
HHI = sum_{g=1}^{G} pi_hat_g^2
```

Lower HHI indicates more dispersed selection. Minimum HHI = 1/G ~ 0.037 (perfectly uniform).

### Quota Satisfaction

Per-group check whether hard quotas are satisfied:
```
satisfied_g = (c_g^- <= c_g(S) <= c_g^+)
```

Returns per-group boolean and aggregate satisfaction rate.

### Dual Diagnostics

The function `dual_geo_diagnostics()` computes **both** municipality-share and population-share variants simultaneously:

| Output Key | Description |
|-----------|-------------|
| `geo_kl_muni` | KL divergence (municipality-share target) |
| `geo_kl_pop` | KL divergence (population-share target) |
| `geo_l1_muni` | L1 distance (municipality-share) |
| `geo_l1_pop` | L1 distance (population-share) |
| `geo_maxdev_muni` | Max deviation (municipality-share) |
| `geo_maxdev_pop` | Max deviation (population-share) |
| `geo_entropy` | Shannon entropy of coreset distribution |
| `geo_hhi` | HHI of coreset distribution |

### State Coverage Report

Per-state breakdown with:
- Count c_g in coreset
- Target count c_g* (from quota allocation)
- Fraction c_g / k
- Target fraction pi_g
- Deviation |c_g/k - pi_g|

---

## 8. KPI Stability Analysis

### Purpose

Measures whether the coreset preserves state-level summary statistics (KPIs). If regulators compute mean coverage per state, these means should be similar whether computed on the full dataset or the coreset.

**Implementation:** `coreset_selection/evaluation/kpi_stability.py`

### Target Scope

KPI stability is computed over the **10 coverage targets** (the same set used by KRR). Per-state means, drifts, and Kendall's tau are computed independently for each of the 10 targets. The extra regression and classification targets are NOT included in KPI stability analysis.

### Per-State Mean Computation

```
mu_g^full(t) = (1/n_g) * sum_{i: g_i=g} y_i^t       # Full dataset state mean
mu_g^S(t)    = (1/c_g(S)) * sum_{i in S: g_i=g} y_i^t  # Coreset state mean
```

Where y_i^t is the value of target t for municipality i.

### Max KPI Drift

```
drift_max(t) = max_{g=1..G} |mu_g^S(t) - mu_g^full(t)|
```

Identifies the worst-case state-level bias for each target.

### Average KPI Drift

```
drift_avg(t) = (1/G) * sum_{g=1}^{G} |mu_g^S(t) - mu_g^full(t)|
```

Mean absolute deviation across all states.

### Kendall's Tau

```
tau(t) = Kendall's rank correlation between {mu_g^full(t)}_g and {mu_g^S(t)}_g
```

Measures whether the relative ranking of states is preserved. tau = 1 means perfect preservation; tau = -1 means complete reversal.

### Per-State Drift Matrix

The full (G x T) matrix of drifts is computed and can be exported as CSV for heatmap visualization:

```
drift_matrix[g, t] = |mu_g^S(t) - mu_g^full(t)|
```

### State-Conditioned KRR RMSE

In addition to aggregate KRR RMSE, per-state RMSE is computed:

```
RMSE_g(t) = sqrt(mean({(y_hat_i^t - y_i^t)^2 : i in E_test, g_i = g}))
```

### Aggregate Stability Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `macro_rmse` | mean_g(RMSE_g) | Average per-state RMSE |
| `worst_group_rmse` | max_g(RMSE_g) | Worst-state RMSE (equity metric) |
| `best_group_rmse` | min_g(RMSE_g) | Best-state RMSE |
| `rmse_dispersion` | std_g(RMSE_g) | Inter-state RMSE variability |
| `rmse_iqr` | IQR_g(RMSE_g) | Robust inter-state variability |

### Worst-State RMSE (Equity Analysis)

The worst-state RMSE is a key equity metric tracked across all k values:

```
worst_state_rmse(t, k) = max_{g=1..G} RMSE_g(t) at cardinality k
```

Plotted as a function of k in Figure N12 to show how increasing budget reduces worst-case state bias.

---

## 9. Coverage & Diversity Metrics

### Purpose

Measures the geometric coverage and diversity of the coreset in feature space.

**Implementation:** `coreset_selection/evaluation/metrics.py`

### k-Center Cost (Minimax)

```
cost_kcenter(S) = max_{i=1..N} min_{j in S} ||x_i - x_j||
```

The maximum distance from any point in the dataset to its nearest coreset point. Lower is better. Related to the k-center clustering objective.

### k-Median Cost

```
cost_kmedian(S) = sum_{i=1..N} min_{j in S} ||x_i - x_j||
```

Optionally weighted: `cost_kmedian(S) = sum_i w_i * min_{j in S} ||x_i - x_j||`

### Mean/Max Coverage Distance

Computed on the evaluation set E rather than the full dataset:

```
mean_coverage(S) = (1/|E|) * sum_{i in E} min_{j in S} ||x_i - x_j||
max_coverage(S)  = max_{i in E} min_{j in S} ||x_i - x_j||
```

**Implementation:** Uses chunked computation for memory efficiency on large datasets.

### Diversity Score

```
diversity(S) = sum_{i,j in S, i<j} ||x_i - x_j||
```

Sum of all pairwise distances within the coreset. Higher indicates greater diversity. Relates to the determinantal point process objective.

### Min Pairwise Distance

```
min_pairwise(S) = min_{i,j in S, i<j} ||x_i - x_j||
```

Detects **redundancy**: if min_pairwise is very small, two coreset points are nearly identical, wasting budget.

### Representation Error

**Mean error:**
```
err_mean(S) = ||mean(X) - mean(X_S)||_2
```

**Covariance error:**
```
err_cov(S) = ||Cov(X) - Cov(X_S)||_F / ||Cov(X)||_F
```

---

## 10. Pareto Front Analysis & Solution Selection

### Purpose

After NSGA-II optimization, the Pareto front contains multiple non-dominated solutions. Analysis and selection methods identify representative solutions for evaluation.

**Implementation:** `coreset_selection/optimization/selection.py`

### Knee Point Selection

The knee point balances all objectives equally, avoiding solutions that are extreme in any single objective:

**Step 1: Min-max normalization**
```
f_tilde_j(s) = (f_j(s) - f_j^min) / (f_j^max - f_j^min + eps_norm)
```

Where eps_norm = 1e-12 prevents division by zero.

**Step 2: L2-norm minimization**
```
s_knee = argmin_{s in Pareto front} ||f_tilde(s)||_2
```

The knee is the solution closest to the utopia point (origin after normalization).

### Per-Objective Minimizers

For each objective j:
```
s_best_j = argmin_{s in Pareto front} f_j(s)
```

Named `best_mmd`, `best_sinkhorn`, etc.

### Pairwise Knees

For multi-objective problems (m > 2), additional knee points are computed for each 2D projection:
```
For each pair (i, j) with i < j:
  s_knee_ij = knee point considering only objectives (f_i, f_j)
```

### Feasibility Filtering

Only feasible solutions (satisfying all constraints) are considered for selection. Feasibility is checked via:
```
feasible(s) = (total_constraint_violation(s) == 0)
```

### Pareto Front Statistics

| Statistic | Definition |
|-----------|-----------|
| Front cardinality | Number of non-dominated solutions |
| Spread | Range of each objective on the front |
| Hypervolume | Volume dominated by the front (relative to reference point) |
| Crowding distance | NSGA-II crowding distance per solution |

---

## 11. Metric Computation Implementation Details

### Bandwidth Estimation

```python
def median_sq_dist(X, sample_size=2048, seed=0):
    # Sample random pairs
    idx = rng.choice(len(X), size=(sample_size, 2), replace=True)
    diffs = X[idx[:, 0]] - X[idx[:, 1]]
    sq_dists = (diffs ** 2).sum(axis=1)
    return np.median(sq_dists)

sigma_sq = median_sq_dist(X_eval_train) / 2
```

### Memory-Efficient Computation

Pairwise distance computations use chunked processing:
```python
chunk_size = 500
for i in range(0, n, chunk_size):
    for j in range(0, n, chunk_size):
        D_chunk = cdist(X[i:i+chunk_size], X[j:j+chunk_size])
```

This prevents O(N^2) memory allocation for large evaluation sets.

### Numerical Precision

- All metric computations use **float64** for numerical stability
- Kernel matrices are computed in float64 even if inputs are float32
- Eigendecompositions use scipy's `eigh` for symmetric positive definite matrices
- Log computations use eps_log = 1e-30 to prevent log(0)

### Timing Instrumentation

Each evaluation phase is timed and recorded in `wall_clock.json`:

| Phase | Description |
|-------|-------------|
| `eval_setup` | Kernel bandwidth estimation, feature construction |
| `eval_nystrom` | Nystrom error computation |
| `eval_kpca` | kPCA distortion computation |
| `eval_krr` | KRR fitting + prediction + RMSE |
| `eval_geo` | Geographic diagnostics |
| `eval_kpi` | KPI stability analysis |
| `eval_downstream` | Multi-model evaluation |
| `eval_total` | Total evaluation wall time |

---

## 12. Metric Summary Table

| Metric | Formula | Range | Better | Source File |
|--------|---------|-------|--------|------------|
| e_Nys | \|\|K-K_hat\|\|_F / \|\|K\|\|_F | [0, 1+] | Lower | `evaluation/raw_space.py` |
| e_kPCA | \|\|lambda-lambda_hat\|\|_2 / \|\|lambda\|\|_2 | [0, 1+] | Lower | `evaluation/raw_space.py` |
| KRR RMSE | sqrt(MSE) per target | [0, inf) | Lower | `evaluation/raw_space.py` |
| KL (geo) | D_KL(pi \|\| pi_hat^alpha) | [0, inf) | Lower | `evaluation/geo_diagnostics.py` |
| L1 (geo) | \|\|pi - pi_hat\|\|_1 | [0, 2] | Lower | `evaluation/geo_diagnostics.py` |
| Max dev (geo) | max_g\|pi_g - c_g/k\| | [0, 1] | Lower | `evaluation/geo_diagnostics.py` |
| KPI drift (max) | max_g\|mu_g^S - mu_g^full\| | [0, inf) | Lower | `evaluation/kpi_stability.py` |
| Kendall's tau | Rank correlation | [-1, 1] | Higher | `evaluation/kpi_stability.py` |
| k-center cost | max min distance | [0, inf) | Lower | `evaluation/metrics.py` |
| k-median cost | sum min distances | [0, inf) | Lower | `evaluation/metrics.py` |
| Diversity | sum pairwise distances | [0, inf) | Higher | `evaluation/metrics.py` |
| Min pairwise | min coreset distance | [0, inf) | Higher | `evaluation/metrics.py` |
| Worst-state RMSE | max_g RMSE_g | [0, inf) | Lower | `evaluation/kpi_stability.py` |
| RMSE dispersion | std_g(RMSE_g) | [0, inf) | Lower | `evaluation/kpi_stability.py` |
| Multi-model regression | {model}_rmse/mae/r2 per target | varies | RMSE/MAE lower, R^2 higher | `evaluation/multi_model_evaluator.py` |
| Multi-model classification | accuracy/bal_accuracy/macro_f1 per target | [0, 1] | Higher | `evaluation/multi_model_evaluator.py` |
| QoS downstream | qos_{model}_{variant}_{metric} | varies | RMSE/MAE lower, R^2 higher | `evaluation/qos_tasks.py` |
