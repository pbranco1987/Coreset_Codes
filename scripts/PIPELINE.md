# Scripts Pipeline Documentation

Complete reference for how every script in `scripts/` works, what it consumes, what it produces, and how the pieces connect into an end-to-end experiment-to-manuscript pipeline.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Flow Diagram](#2-data-flow-diagram)
3. [Script Reference](#3-script-reference)
   - [3.1 Launchers](#31-launchers)
   - [3.2 Analysis](#32-analysis)
   - [3.3 Bootstrap](#33-bootstrap)
   - [3.4 Infrastructure](#34-infrastructure)
   - [3.5 Deployment](#35-deployment)
4. [External Dependencies](#4-external-dependencies)
5. [Folder Layout](#5-folder-layout)
6. [Typical Workflow](#6-typical-workflow)

---

## 1. Pipeline Overview

The pipeline has four sequential phases. Each phase's output feeds directly into the next:

| Phase | Scripts | Input | Output |
|-------|---------|-------|--------|
| **Phase 1: Data Preparation** | `build_caches.py` | Raw CSVs in `data/` | Replicate caches (`assets.npz`) |
| **Phase 2: Coreset Construction** | `adaptive_tau.py`, `run_baselines.py` | Replicate caches | `coreset.npz` per experiment |
| **Phase 3: Evaluation** | `evaluate_coresets.py`, `bootstrap_reeval.py` | Saved coresets + caches | `metrics.csv` per experiment |
| **Phase 4: Analysis** | `championship.py` | All metrics CSVs | Championship JSON + stdout tables |

**Design principle:** Construction (Phase 2) and evaluation (Phase 3) are strictly separated. Coresets are saved as index arrays; metrics are computed in a separate pass. This means:
- Metrics can be recomputed without re-running the optimiser.
- Every row in every results table traces back to the same evaluation functions.
- The S ∩ E overlap fix (excluding coreset points from the evaluation set) is enforced in one place.

---

## 2. Data Flow Diagram

```
data/smp_main.csv ──────────────────────────────────────────────────────┐
data/metadata.csv ──────────────────────────────────────────────────────┤
data/city_populations.csv ──────────────────────────────────────────────┤
                                                                        ▼
                                                              ┌─────────────────┐
                                                              │  build_caches.py │
                                                              └────────┬────────┘
                                                                       │
                                        replicate_cache/rep{00-04}/assets.npz
                                                                       │
                            ┌──────────────────────────────────────────┼──────────────────┐
                            ▼                                          ▼                  ▼
                  ┌──────────────────┐                    ┌───────────────────┐  ┌──────────────────┐
                  │  adaptive_tau.py  │                    │  run_baselines.py  │  │bootstrap_reeval.py│
                  │  (NSGA-II)        │                    │  (8 methods)       │  │  (B draws)        │
                  └────────┬─────────┘                    └────────┬──────────┘  └────────┬─────────┘
                           │                                       │                      │
         {exp}/coreset.npz │                          {exp}/coreset.npz           bootstrap_results/
         {exp}/representatives/                                    │              bootstrap_raw_*.csv
         {exp}/manifest.json                                       │
                           │                                       │
                           └───────────────┬───────────────────────┘
                                           ▼
                                ┌────────────────────────┐
                                │  evaluate_coresets.py   │
                                │  (5-stage metric eval)  │
                                └───────────┬────────────┘
                                            │
                                   {exp}/metrics.csv
                                   {exp}/metrics-representatives.csv
                                            │
                                            ▼
                                ┌────────────────────────┐
                                │ championship.py        │
                                │ (Friedman, H2H,        │
                                │  Cliff's delta)        │
                                └────────────┬───────────┘
                                             │
                                   championship JSON
                                   + stdout tables
```

The manuscript source (`manuscript/updated.tex`, `manuscript/main_vs2.tex`,
`manuscript/references.bib`) is maintained manually. The pipeline terminates
at the championship output; there is intentionally no automated LaTeX/figure
generation in this repository.

---

## 3. Script Reference

### 3.1 Launchers

These scripts create coresets. They produce index arrays only — no downstream metrics.

#### `launchers/build_caches.py` — Replicate Cache Builder

**Purpose:** Pre-compute all data transformations (preprocessing, VAE embedding, PCA projection, evaluation splits) so that experiments are reproducible and efficient.

**What a replicate cache contains:**
- `X_scaled` — standardised raw features (N × D)
- `Z_vae` — VAE latent embeddings (N × 32)
- `Z_logvar` — VAE log-variances
- `Z_pca` — PCA projections (N × 32)
- `state_labels` — geographic group labels (N,)
- `population` — per-municipality population weights (N,)
- `eval_idx`, `eval_train_idx`, `eval_test_idx` — fixed evaluation split (~2000 points, 80/20 stratified)
- Metadata: removed target columns (audit trail), feature names, extra regression/classification targets

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| Raw features | `data/smp_main.csv` | 5,569 municipalities × 1,700+ columns |
| Feature metadata | `data/metadata.csv` | Column types (continuous, ordinal, categorical) |
| Population data | `data/city_populations.csv` | Municipal populations for constraint weighting |

**Outputs:**
| Output | Path | Description |
|--------|------|-------------|
| Cache file | `{experiment_dir}/replicate_cache/rep{id}/assets.npz` | Complete replicate snapshot |
| Seed manifest | `{experiment_dir}/seed_manifest.json` | Seed-to-replicate mapping |

**Usage:**
```bash
python scripts/launchers/build_caches.py                   # all 5 replicas
python scripts/launchers/build_caches.py --reps 0 1        # replicas 0 and 1 only
python scripts/launchers/build_caches.py --device cuda      # GPU for VAE training
```

**Safety guarantees:**
- `validate_no_leakage()` is called before VAE training and PCA fitting — raises an error if any downstream evaluation target leaks into the feature matrix.
- StandardScaler is fit on the training partition only (not the full dataset).

---

#### `launchers/adaptive_tau.py` — NSGA-II Coreset Construction

**Purpose:** Run constrained bi-objective NSGA-II with adaptive tau calibration to find a Pareto front of coreset solutions that balance MMD and Sinkhorn divergence under geographic proportionality constraints.

**Adaptive tau calibration (3 phases):**
1. **Probe:** Start at the greedy KL floor, double tau every 15 generations until ≥50% of the population is feasible. Records tau_lo (last failure) and tau_hi (first success).
2. **Bisect:** Binary search between tau_lo and tau_hi with adaptive patience. Trend-aware: if feasibility is still climbing at the patience limit, grants extra generations. Converges when the gap is < 5%.
3. **Production:** Run 1,500 generations at the calibrated tau_hi.

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| Replicate cache | `{cache_dir}/rep{id}/assets.npz` | From `build_caches.py` |

**Outputs (per experiment):**
| Output | Path | Description |
|--------|------|-------------|
| Pareto front | `{output_dir}/{exp_name}/coreset.npz` | Boolean mask matrix X (n_solutions × N) + objective values F |
| Representatives | `{output_dir}/{exp_name}/representatives/knee.npz` | Named selections (knee, best-mmd, best-sinkhorn, chebyshev) |
| Tau log | `{output_dir}/{exp_name}/adaptive-tau-log.json` | Per-generation tau, feasibility count, objective values |
| Timing | `{output_dir}/{exp_name}/wall-clock.json` | Total seconds and generations |
| Manifest | `{output_dir}/{exp_name}/manifest.json` | Seed, config, git commit, hyperparameters |

**Usage:**
```bash
# Single experiment
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --rep 0 \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

# All 5 replicas
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --all \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

# Single-objective ablation
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --objectives mmd --all \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics
```

**Experiment naming convention:** Automatically generates descriptive folder names:
- `nsga2-vae-popsoft-k100-rep0` (standard bi-objective)
- `ablation-mmdonly-k100-rep0` (single-objective MMD)
- `ablation-triobjective-k100-rep0` (tri-objective with NystromLogDet)

---

#### `launchers/run_baselines.py` — Baseline Coreset Construction + Repair

**Purpose:** Run 8 baseline coreset selection methods (Uniform, k-Medoids, Kernel Herding, FastForward, RLS-Nystrom, k-DPP, Kernel Thinning, KKN) with post-hoc geographic constraint repair to make them directly comparable to NSGA-II.

**Constraint repair logic:** For each baseline whose constraint mode includes a soft KL component, applies the same swap-based proportionality repair that NSGA-II uses during evolution. This ensures apples-to-apples comparison — baselines satisfy the same geographic constraints as the NSGA-II solutions.

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| Baseline coresets | `experiments_v2/B_{space}_{suffix}/rep{id}/coresets/*.npz` | Pre-computed baseline selections |
| Replicate cache | `replicate_cache/rep{id}/assets.npz` | For evaluation and repair |

**Outputs:**
| Output | Path | Description |
|--------|------|-------------|
| Repaired results | `results/repaired_baselines/all_results.csv` | All metrics after repair |

**Usage:**
```bash
python scripts/launchers/run_baselines.py
python scripts/launchers/run_baselines.py --modes ps ms     # specific constraint modes
python scripts/launchers/run_baselines.py --spaces vae pca  # specific spaces
```

---

### 3.2 Analysis

These scripts compute metrics and statistical comparisons. They never modify coresets.

#### `analysis/evaluate_coresets.py` — Standalone Metric Evaluation

**Purpose:** The single authoritative source for all downstream metric computation. Loads saved coreset indices and computes the complete 5-stage evaluation pipeline without any optimiser or runner.

**Five evaluation stages:**
| Stage | Metrics Computed | Key Function |
|-------|-----------------|--------------|
| [1/5] Geographic | KL divergence (muni + pop-share), L1 deviation, max state-share deviation | `dual_geo_diagnostics()` |
| [2/5] Operator fidelity | Nystrom error, kernel-PCA distortion, KRR RMSE (per target), state-conditioned stability | `RawSpaceEvaluator.all_metrics_with_state_stability()` |
| [3/5] KPI stability | Per-state target-mean drift, Kendall's tau ranking stability | `state_kpi_stability()` |
| [4/5] Multi-model downstream | KNN, Random Forest, Logistic Regression, GBT accuracy/RMSE on Nystrom features | `multi_model_downstream()` |
| [5/5] QoS | (Reserved — disabled in clean pipeline) | — |

**S ∩ E overlap fix:** Before computing any metric, coreset indices that also appear in the evaluation set are excluded. This prevents landmarks from evaluating themselves, which would artificially reduce Nystrom error and inflate KRR accuracy.

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| Pareto front | `{experiment_dir}/coreset.npz` | Boolean mask matrix or index array |
| Representatives | `{experiment_dir}/representatives/*.npz` | Named selections |
| Replicate cache | `{cache_path}` (assets.npz) | Features, targets, eval splits |

**Outputs:**
| Output | Path | Description |
|--------|------|-------------|
| Full metrics | `{experiment_dir}/metrics.csv` | One row per coreset (all front members + named selections) |
| Representative metrics | `{experiment_dir}/metrics-representatives.csv` | Subset: knee, best-mmd, best-sinkhorn only |

**Usage:**
```bash
# Single experiment
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir EXPERIMENTS-.../nsga2-vae-popsoft-k100-rep0 \
    --cache-path replicate_cache_seed4200/rep00/assets.npz

# Batch: all experiments matching a pattern
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir EXPERIMENTS-.../ \
    --cache-path replicate_cache_seed4200/rep00/assets.npz \
    --batch --pattern "nsga2-vae-*"

# Force re-evaluation
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir EXPERIMENTS-.../ \
    --cache-path replicate_cache_seed4200/rep00/assets.npz \
    --batch --force
```

---

#### `analysis/championship.py` — Statistical Method Ranking

**Purpose:** Compute the championship ranking of 13 coreset selection methods (5 NSGA-II variants + 8 baselines) using the Friedman test with Nemenyi post-hoc and pairwise head-to-head comparisons with Cliff's delta.

**Methodology:**
- **Metric selection:** Selects top-10 classification + top-10 regression metrics where the NSGA-II oracle (per-metric best from front) ranks highest. Plus 2 operator-fidelity metrics. Total: 22 metrics.
- **Per-replica blocking:** Each (metric, replica) pair is one Friedman block, giving N = 22 × 5 = 110 contests. This preserves within-replicate variability.
- **Statistical tests:** `scipy.stats.friedmanchisquare` for omnibus test; Nemenyi critical difference at α = 0.01.
- **Head-to-head:** For each NSGA-II variant vs each baseline, counts wins/ties/losses and computes Cliff's delta (δ_C).

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| NSGA-II front | `experiments_v2/K_vae_k100/rep*/results/front_metrics_vae.csv` | Full Pareto front evaluation |
| Baseline results | `experiments_v2/B_v_ps/rep*/results/all_results.csv` | 8 baselines × 5 replicas |

**Outputs:**
| Output | Path | Description |
|--------|------|-------------|
| Rankings | `experiments_v2/manuscript_final_v3.json` | Full championship tables in JSON |
| Console output | (stdout) | Tables 3–6 formatted for manuscript |

---

### 3.3 Bootstrap

Robustness evaluation via random target sampling.

#### `bootstrap_reeval.py` — Bootstrap Target Re-evaluation

**Purpose:** For each of B random draws, sample a different set of regression + classification targets, then evaluate all coreset methods on those targets. This tests whether method rankings are robust to the choice of downstream tasks.

**Inputs:**
| Input | Path | Description |
|-------|------|-------------|
| Experiment config | `experiments_v2/{run_id}/rep{id}/config.json` | Run configuration |
| Pareto front | `experiments_v2/{run_id}/rep{id}/results/{space}_pareto.npz` | NSGA-II front |
| Replicate cache | `{cache_dir}/rep{id}/assets.npz` | Features and targets |

**Outputs:**
| Output | Path | Description |
|--------|------|-------------|
| Bootstrap results | `bootstrap_results/bootstrap_raw_{run_id}_rep{id}.csv` | One row per method × draw |
| Metadata | `bootstrap_results/bootstrap_meta_{run_id}_rep{id}.json` | Draw count, seed, timing |

**Usage:**
```bash
python scripts/bootstrap_reeval.py --run-id K_vae_k100 --rep-id 0 --n-bootstrap 50 --n-reg 5 --n-cls 5
```

#### `bootstrap/dispatcher.sh` — Parallel Bootstrap Orchestrator

**Purpose:** Launch many `bootstrap_reeval.py` jobs in parallel tmux windows with automatic failure detection and re-queueing.

**Usage:**
```bash
bash scripts/bootstrap/dispatcher.sh              # auto-discover jobs, 64 parallel
bash scripts/bootstrap/dispatcher.sh 32            # limit to 32 parallel
bash scripts/bootstrap/dispatcher.sh 64 jobs.txt   # use pre-built job list
```

#### `bootstrap/run_one.sh` — Single Bootstrap Worker

**Purpose:** Run one `bootstrap_reeval.py` invocation with controlled threading. Called by `dispatcher.sh`.

**Usage:**
```bash
bash scripts/bootstrap/run_one.sh K_vae_k100 0    # run_id=K_vae_k100, rep_id=0
```

---

### 3.4 Infrastructure

Server management and result collection.

#### `infra/server_setup.sh` — One-Time Server Provisioning

Clones the repository, creates a virtualenv, installs the package, and validates data files. Run once on each new server.

```bash
bash scripts/infra/server_setup.sh
```

#### `infra/run_parallel_tmux.sh` — Multi-Session Experiment Launcher

Launches N concurrent tmux sessions, each running all 15 experiment scenarios (R0–R14) with auto-scaled threading and deterministic seed assignment.

```bash
bash scripts/infra/run_parallel_tmux.sh 2                  # 2 sessions
bash scripts/infra/run_parallel_tmux.sh 2 --k1 100 --k2 300  # custom k per session
```

#### `infra/collect_and_merge.sh` — Result Collection + Merging

Pulls results from LABGELE and Lessonia via SCP, then merges all CSVs into a single combined file tagged by server of origin.

```bash
bash scripts/infra/collect_and_merge.sh              # both servers
bash scripts/infra/collect_and_merge.sh --labgele     # LABGELE only
```

#### `infra/monitor.sh` — Real-Time Dashboard

Creates a tmux session with one window per experiment scenario showing live log output.

```bash
bash scripts/infra/monitor.sh ~/Coreset_Codes/runs_out_labgele 300
```

---

### 3.5 Deployment

One-command setup + launch for remote servers.

#### `deploy/labgele.sh` — LABGELE Deployment Template

Complete end-to-end: clone repo, install package, launch all experiments at the specified k. Uses seed 123.

```bash
bash scripts/deploy/labgele.sh 300     # k=300
```

#### `deploy/lessonia.sh` — Lessonia Deployment Template

Mirror of `labgele.sh` for the second server. Uses seed 456.

```bash
bash scripts/deploy/lessonia.sh 300    # k=300
```

---

## 4. External Dependencies

### `coreset_selection/` Package Modules

| Module | Used By | Purpose |
|--------|---------|---------|
| `data.cache` | `build_caches.py`, `evaluate_coresets.py`, `adaptive_tau.py` | Build/load replicate caches |
| `data.target_columns` | `data.cache` (internal) | Detect and exclude target columns |
| `geo.info` | `adaptive_tau.py`, `run_baselines.py`, `evaluate_coresets.py` | Geographic group structure |
| `geo.projector` | `adaptive_tau.py`, `run_baselines.py` | Constraint projection and repair |
| `constraints.proportionality` | `adaptive_tau.py`, `run_baselines.py` | Soft KL constraint enforcement |
| `objectives.computer` | `adaptive_tau.py` | MMD, Sinkhorn, NystromLogDet objectives |
| `optimization.nsga2_internal` | `adaptive_tau.py` | NSGA-II operators (crossover, mutation, repair, sorting) |
| `optimization.selection` | `adaptive_tau.py` | Pareto front representative selection (knee, best-per-objective) |
| `evaluation.raw_space` | `evaluate_coresets.py`, `run_baselines.py` | Nystrom, kPCA, KRR evaluation |
| `evaluation.geo_diagnostics` | `evaluate_coresets.py`, `run_baselines.py` | KL, L1, MaxDev geographic metrics |
| `evaluation.kpi_stability` | `evaluate_coresets.py` | Per-state target drift |
| `evaluation.multi_model_evaluator` | `evaluate_coresets.py`, `bootstrap_reeval.py` | KNN/RF/LR/GBT downstream |
| `evaluation.bootstrap_targets` | `bootstrap_reeval.py` | Random target sampling |
| `baselines.*` | `run_baselines.py`, `bootstrap_reeval.py` | 8 baseline selection methods |

### `data/` Folder

| File | Used By | Content |
|------|---------|---------|
| `smp_main.csv` | `build_caches.py` | 5,569 municipalities × 1,700+ features |
| `metadata.csv` | `build_caches.py` | Feature types and metadata |
| `city_populations.csv` | `build_caches.py` | Population weights for geographic constraints |
| `df_indicators_flat_by_municipality.csv` | `build_caches.py` | Extra regression/classification targets |
| `qf_mean_by_*.csv` | `build_caches.py` | QoS survey targets (optional) |

### `manuscript/` Folder

The repository stores only manuscript **source files**. There is no
automated LaTeX/figure generation; the pre-generated `tables/*.tex` and
`figures/*.pdf` under `manuscript/generated/` are committed snapshots from
the last paper compile and are not regenerated by any script in this
repository.

| Item | Producer | Consumer |
|------|----------|----------|
| `manuscript/generated/tables/*.tex` | Committed artifact (from past compile) | LaTeX compiler |
| `manuscript/generated/figures/*.pdf` | Committed artifact (from past compile) | LaTeX compiler |
| `manuscript/figure_data/*.csv` | Committed artifact | External plotting tools |
| `manuscript/manuscript_final.json` | `championship.py` | External analysis tools |
| `manuscript/updated.tex` | Manual editing | LaTeX compiler |
| `manuscript/main_vs2.tex` | Manual editing | LaTeX compiler |
| `manuscript/references.bib` | Manual editing | LaTeX compiler |

---

## 5. Folder Layout

```
Coreset_Codes/
├── coreset_selection/                    # Core Python package (238 source files)
│   ├── baselines/                        #   8 baseline selection methods
│   ├── constraints/                      #   KL proportionality constraints
│   ├── data/                             #   Cache builder, target column detection
│   ├── evaluation/                       #   All metric computation
│   ├── experiment/                       #   Runner, saver, evaluation mixin
│   ├── geo/                              #   Geographic group info + projector
│   ├── objectives/                       #   MMD, Sinkhorn, NystromLogDet computers
│   └── optimization/                     #   NSGA-II operators + selection
│
├── scripts/                              # All runnable scripts (15 files)
│   ├── launchers/                        #   Coreset construction
│   │   ├── adaptive_tau.py               #     NSGA-II (core)
│   │   ├── build_caches.py               #     Cache builder (core)
│   │   └── run_baselines.py              #     Baseline methods
│   ├── analysis/                         #   Metric evaluation + ranking
│   │   ├── evaluate_coresets.py          #     Standalone metric engine (core)
│   │   └── championship.py              #     Statistical championship
│   ├── bootstrap/                        #   Robustness evaluation
│   │   ├── dispatcher.sh                #     Parallel orchestrator
│   │   └── run_one.sh                   #     Single-job worker
│   ├── infra/                            #   Server operations
│   │   ├── server_setup.sh              #     One-time provisioning
│   │   ├── run_parallel_tmux.sh         #     Multi-session launcher
│   │   ├── collect_and_merge.sh         #     SCP + CSV merge
│   │   └── monitor.sh                   #     Live dashboard
│   ├── deploy/                           #   One-command server deployment
│   │   ├── labgele.sh                   #     LABGELE template
│   │   └── lessonia.sh                  #     Lessonia template
│   ├── bootstrap_reeval.py              #   Bootstrap evaluation
│   └── PIPELINE.md                      #   This documentation
│
├── data/                                 # Raw input data (11 files)
├── manuscript/                           # Paper files
│   ├── updated.tex                       #   Primary manuscript
│   ├── main_vs2.tex                      #   Previous version
│   ├── references.bib                    #   Bibliography
│   ├── manuscript_final.json             #   Championship results
│   ├── generated/                        #   Auto-generated content
│   │   ├── tables/*.tex                  #     LaTeX table bodies
│   │   └── figures/*.pdf                 #     Publication figures
│   └── figure_data/                      #   Intermediate CSV data
├── docs/                                 # Project documentation (5 files)
├── EXPERIMENTS-tau_fixed-...             # Clean experiment outputs
│   └── {experiment-name}/
│       ├── coreset.npz                   #   Pareto front (construction output)
│       ├── representatives/              #   Named selections from front
│       ├── metrics.csv                   #   Evaluation output
│       ├── manifest.json                 #   Reproducibility metadata
│       └── adaptive-tau-log.json         #   Tau calibration trajectory
├── updated.tex                           # Root copy of manuscript
├── README.md                             # Project overview
└── .gitignore
```

---

## 6. Typical Workflow

### Starting from scratch on LABGELE:

```bash
# 1. Deploy to server (one-time)
bash scripts/deploy/labgele.sh

# 2. Build replicate caches (once, ~30 min)
python scripts/launchers/build_caches.py

# 3. Run NSGA-II experiments (per batch, ~8-18 hours each)
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --all \
    --cache-dir experiments/adaptive_tau_k100_ps_vae/replicate_cache \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

# 4. Evaluate coresets (per batch, ~1-2 hours)
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics \
    --cache-path experiments/adaptive_tau_k100_ps_vae/replicate_cache/rep00/assets.npz \
    --batch --pattern "nsga2-vae-popsoft-k100-*"

# 5. Assess: run championship analysis
python scripts/analysis/championship.py

# 6. Collect results from server (from local machine)
bash scripts/infra/collect_and_merge.sh --labgele

# 7. Monitor running experiments (optional)
bash scripts/infra/monitor.sh ~/Coreset_Codes/runs_out_labgele
```

> **Note:** the repository does not include a manuscript-build step — paper
> tables/figures are generated outside this repo. The pipeline ends at the
> championship output (JSON + stdout tables).

### Iterative optimization loop:

```
Run batch → Evaluate → Championship → Diagnose weakness → Tune → Repeat
```

At each assessment gate, compare NSGA-II vs baselines on:
- Overall championship score (Friedman rank)
- Per-family breakdown (Classification, Op. Fidelity, Supervised Regression)
- Head-to-head win rates and Cliff's delta
- Pareto envelope vs knee vs baselines (improvement margin)
