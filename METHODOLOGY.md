# Methodology: Mathematical Formulations and Algorithms

This document provides the complete mathematical reference for all algorithms, objective functions, and constraint formulations implemented in the repository.

---

## Notation Table

| Symbol | Definition | Value / Range |
|--------|-----------|---------------|
| N | Number of municipalities | 5,569 |
| G | Number of geographic groups (states) | 27 |
| D | Number of features (after preprocessing) | 621 |
| d_z | Latent/embedding dimension | 32 (default) |
| k | Coreset cardinality | {50, 100, 200, 300, 400, 500} |
| S | Selected subset (coreset) | S subset of {1,...,N}, \|S\| = k |
| x | Binary decision mask | x in {0,1}^N |
| I_g | Index set of group g | I_g = {i : g_i = g} |
| n_g | Size of group g | n_g = \|I_g\| |
| c_g(S) | Count of selected in group g | c_g = \|S intersect I_g\| |
| pi_g | Target share for group g | Depends on weight mode |
| pi_hat_g | Empirical share of S in group g | (c_g + alpha) / (k + alpha*G) |
| alpha | Dirichlet smoothing parameter | 1.0 |
| tau | Proportionality tolerance | 0.02 |
| w_i | Per-municipality weight | population_i or 1 |
| P | NSGA-II population size | 200 |
| T | Number of generations | 1,000 |
| m | Number of Random Fourier Features | 2,000 |
| A | Number of Sinkhorn anchors | 200 |
| sigma^2 | RBF kernel bandwidth | Median heuristic |
| E | Evaluation set | \|E\| = 2,000 |
| r | kPCA components retained | 20 |

---

## 1. Problem Formulation

The constrained multi-objective coreset selection problem:

```
Minimize:    F(S) = [f_1(S), f_2(S), ..., f_m(S)]
Subject to:  |S| = k                                         (C1: exact cardinality)
             D_KL(pi || pi_hat^alpha(S)) <= tau               (C2: proportionality)
             c_g^- <= c_g(S) <= c_g^+   for all g in {1..G}  (C3: per-group quotas)
```

Where:
- **C1** enforces exactly k municipalities are selected
- **C2** bounds the KL divergence between target and achieved geographic distributions
- **C3** provides hard per-group bounds computed by Algorithm 1

The decision variable is a binary mask x in {0,1}^N. The objectives f_1, f_2 are typically MMD and Sinkhorn divergence (optionally f_3 = SKL in R7).

---

## 2. Algorithm 1: KL-Optimal Integer Quotas

**Purpose:** Compute integer allocation c*(k) = (c_1*,...,c_G*) minimizing KL(pi || pi_hat^alpha(c)) subject to bounds.

**Input:** Target distribution pi, cardinality k, smoothing alpha, bounds [lb_g, ub_g] per group.

**Approach:** Lazy-heap greedy. The marginal gain of incrementing group g from count t to t+1 is:

```
Delta_g(t) = pi_g * [log(t + alpha + 1) - log(t + alpha)]
```

This is **non-increasing** in t, enabling a max-heap-based greedy strategy:

```
ALGORITHM 1: KL-Optimal Integer Quotas
────────────────────────────────────────
Input:  pi (G,), k, alpha, lb (G,), ub (G,)
Output: c* (G,) integer counts

1. Initialize c_g = lb_g for all g
2. Build max-heap H keyed by Delta_g(c_g) for all g where c_g < ub_g
3. While sum(c) < k:
   a. Pop (g, delta) from H
   b. If delta != Delta_g(c_g):           // Lazy key update
      Push (g, Delta_g(c_g)) to H
      Continue
   c. c_g = c_g + 1
   d. If c_g < ub_g: Push (g, Delta_g(c_g)) to H
4. Return c*
```

**Complexity:** O(k log G)

**Feasibility floor:** KL_min(k) = KL(pi || pi_hat^alpha(c*(k))) provides the minimum achievable KL for a given k, establishing feasibility thresholds.

---

## 3. Algorithm 2: Swap-Based Repair

**Purpose:** Project an arbitrary binary mask to satisfy both cardinality and quota constraints.

```
ALGORITHM 2: Swap-Based Repair
────────────────────────────────
Input:  mask (N,) binary, target counts c* (G,), rng
Output: repaired mask satisfying |S| = k and c_g = c_g* for all g

1. Compute current counts c_g for each group
2. Identify donor groups: D = {g : c_g > c_g*}
3. Identify recipient groups: R = {g : c_g < c_g*}
4. While D != {} and R != {}:
   a. Pick donor g_d from D, recipient g_r from R
   b. In g_d: deselect one selected municipality (random)
   c. In g_r: select one unselected municipality (random)
   d. Update counts; remove satisfied groups from D, R
5. Fix cardinality if |S| != k:
   - If |S| > k: randomly deselect from largest groups
   - If |S| < k: randomly select from smallest groups
6. Return repaired mask
```

**Complexity:** O(N * G) in the worst case.

---

## 4. Algorithm 3: Constrained NSGA-II

**Purpose:** Multi-objective optimization of coreset selection with constraint handling.

```
ALGORITHM 3: Constrained NSGA-II
─────────────────────────────────
Input:  Problem (N, k, objectives, constraints), P, T
Output: Pareto-optimal set of solutions

1. Initialize population of P random feasible masks (via Algorithm 2)
2. Evaluate objectives F(x) and constraints G(x) for all x
3. For generation t = 1 to T:
   a. SELECTION: Binary tournament with constraint-domination
   b. VARIATION:
      - Uniform binary crossover (p_c = 0.9)
      - Quota-swap mutation (p_m = 0.2)
   c. REPAIR: Apply Algorithm 2 to each offspring
   d. EVALUATION: Compute F(x) and G(x) for offspring
   e. ENVIRONMENTAL SELECTION:
      i.   Merge parent + offspring (2P individuals)
      ii.  Constraint-dominated sort (feasible > infeasible)
      iii. Fast non-dominated sorting within feasible set
      iv.  Crowding distance tie-breaking
      v.   Select top P individuals
4. Return Pareto front of final population
```

**Constraint-domination:** Solution a constraint-dominates b if:
1. a is feasible and b is not, OR
2. Both infeasible and a has smaller total constraint violation, OR
3. Both feasible and a Pareto-dominates b

**Genetic operators:**
- **UniformBinaryCrossover:** Each bit independently swapped with probability 0.5
- **QuotaSwapMutation:** For each selected bit, swap with an unselected bit in a random group with probability p_m, preserving quota structure

---

## 5. Objective Functions

### 5.1 Maximum Mean Discrepancy via Random Fourier Features

The RBF kernel k(x, y) = exp(-||x-y||^2 / (2*sigma^2)) is approximated by random Fourier features:

```
phi(x) = sqrt(2/m) * cos(Wx + b)
```

Where W in R^{m x d} with rows ~ N(0, I/sigma^2) and b ~ Uniform(0, 2*pi).

The MMD^2 between distributions P and Q is approximated as:

```
MMD^2(P, Q) ~ ||mu_P - mu_Q||^2
where mu_P = (1/|P|) * sum_{x in P} phi(x)
```

**Parameters:** m = 2,000 features, sigma^2 via median heuristic (median of pairwise squared distances on a 2,048-sample subset).

**Subset evaluation:** Given precomputed mu_X for the full dataset, evaluating a subset S requires O(m) to compute mu_S and the squared norm difference.

### 5.2 Sinkhorn Divergence via Anchors

The Sinkhorn divergence provides a smooth, positive, metrized approximation to optimal transport:

```
S_eps(P, Q) = OT_eps(P, Q) - 0.5 * OT_eps(P, P) - 0.5 * OT_eps(Q, Q)
```

Where OT_eps is the entropy-regularized optimal transport cost:

```
OT_eps(P, Q) = min_{pi in Pi(P,Q)} <pi, C> + eps * KL(pi || a x b)
```

**Anchor approximation:** Instead of computing on all N points, A = 200 anchor points are selected via k-means++. Cost matrices use squared Euclidean distances, scaled by median cost: C_ij = ||r_i - r_j||^2 / median(||r_i - r_j||^2).

**Entropic regularization:** eps = eta * median(||r_i - r_j||^2), with eta = 0.05.

**Solver:** 100 log-stabilized Sinkhorn iterations with convergence threshold 1e-6.

### 5.3 Symmetric KL Divergence (VAE Latent Space)

For the tri-objective ablation (R7), symmetric KL measures drift between moment-matched Gaussians of VAE posterior distributions:

```
G_N = N(mu_N, diag(v_N)),  where mu_N = mean(mu_i), v_N = mean(var_i) + Var(mu_i)
G_S = N(mu_S, diag(v_S)),  similarly for subset S
```

```
SKL(G_N, G_S) = 0.5 * sum_j [v_Nj/v_Sj + v_Sj/v_Nj - 2 + (mu_Nj - mu_Sj)^2 * (1/v_Nj + 1/v_Sj)]
```

**Variance clamping:** v_j is clamped to [exp(-10), exp(2)] = [4.54e-5, 7.389] for numerical stability.

### 5.4 Geographic KL Divergence

Measures divergence between target and achieved geographic distributions:

```
D_KL(pi || pi_hat^alpha) = sum_g pi_g * log(pi_g / pi_hat_g^alpha)
where pi_hat_g^alpha = (c_g + alpha) / (k + alpha * G)
```

**Dirichlet smoothing** (alpha = 1.0) prevents infinite KL when c_g = 0.

---

## 6. Constraint Modes

### Population-share (default)

```
w_i = population_i
pi_g = sum_{i in I_g} w_i / sum_i w_i
```

Ensures representation proportional to state populations.

### Municipality-share quota

```
w_i = 1 for all i
pi_g = n_g / N = |I_g| / N
```

Ensures representation proportional to municipality counts.

### Joint

Both population-share and municipality-share constraints applied simultaneously. Requires satisfying both:
- D_KL(pi_pop || pi_hat_pop) <= tau
- D_KL(pi_muni || pi_hat_muni) <= tau

### None (exact-k only)

Only the cardinality constraint |S| = k is enforced. No geographic proportionality.

---

## 7. Representation Learning

### Variational Autoencoder

**Architecture:**

```
Encoder: D -> 128 (ReLU) -> [mu (32), logvar (32)]
Decoder: 32 -> 128 (ReLU) -> D (linear)
```

**Loss:**

```
L = Reconstruction + beta * KL
  = MSE(x, x_hat) + 0.5 * beta * sum_j [1 + logvar_j - mu_j^2 - exp(logvar_j)]
```

With beta = 0.1 (down-weighted KL term).

**Training:** Adam optimizer (lr = 1e-3), batch size 256, up to 1,500 epochs with early stopping (patience = 200). Full-batch mode for N <= 50,000. AMP (automatic mixed precision) on CUDA. torch.compile() for PyTorch 2.x.

### PCA

Standard PCA with n_components = 32 (matching VAE for fair comparison). Fitted on I_train only.

---

## 8. Pareto Front Analysis

### Knee Point Selection

The knee point balances all objectives equally:

```
f_tilde_j(s) = (f_j(s) - f_j^min) / (f_j^max - f_j^min + eps_norm)
s_knee = argmin_s ||f_tilde(s)||_2
```

Where eps_norm = 1e-12 prevents division by zero.

### Representative Selection

For each Pareto front, the following representatives are identified:
- **Per-objective minimizers:** best_{objective_name} for each objective
- **Overall knee:** Balanced solution via L2-norm minimization
- **Pairwise knees** (for m > 2 objectives): Knee in each 2D projection
