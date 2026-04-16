# API Reference — Scripts (CLI Entry Points)

This page documents the **command-line interface** of every entry-point script under `scripts/`. Each script is self-contained and invoked directly (`python scripts/…`). For the programmatic API of underlying library functions, see the other pages in `docs/api/`.

## Script Map

| Entry point | Purpose | Inputs | Outputs |
|-------------|---------|--------|---------|
| `scripts/launchers/build_caches.py` | Build replicate caches (VAE, PCA, splits) | Raw CSVs in `data/` | `{experiment_dir}/replicate_cache/rep{id}/assets.npz` |
| `scripts/launchers/adaptive_tau.py` | Run NSGA-II with adaptive tau calibration | Replicate cache | `{output_dir}/{exp_name}/{coreset.npz, representatives/*, manifest.json}` |
| `scripts/launchers/run_baselines.py` | Run & soft-KL-repair baselines | Baseline coresets + cache | `results/repaired_baselines/all_results.csv` |
| `scripts/analysis/evaluate_coresets.py` | Compute all downstream metrics | `coreset.npz` + cache | `metrics.csv`, `metrics-representatives.csv` |
| `scripts/analysis/championship.py` | Friedman/Nemenyi/H2H ranking | Per-method metric CSVs | Championship JSON + stdout tables |
| `scripts/bootstrap_reeval.py` | Bootstrap target resampling | Pareto front + cache | `bootstrap_raw_*.csv` + `bootstrap_meta_*.json` |

---

## `build_caches`

**Source:** `scripts/launchers/build_caches.py`

**Purpose:** Build 5 fresh replicate caches (or a subset) for the adaptive-tau experiment.

**Usage:**
```bash
python scripts/launchers/build_caches.py                 # all 5 replicas
python scripts/launchers/build_caches.py --reps 0 1 2    # subset
python scripts/launchers/build_caches.py --device cuda    # GPU for VAE
```

**Arguments:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--reps` | `int[+]` | `0 1 2 3 4` | Replicate IDs to build. |
| `--device` | `str` | `cpu` | PyTorch device for VAE training (`cpu` or `cuda`). |

**Outputs:**
- `experiments/adaptive_tau_k100_ps_vae/replicate_cache/rep{id}/assets.npz`
- `experiments/adaptive_tau_k100_ps_vae/seed_manifest.json`

**Implementation:** wraps `coreset_selection.data.build_replicate_cache` with a seed table (see `SEEDS` constant at top of file). Seed arithmetic compensates for the internal `seed + rep_id` offset so each replicate lands on its canonical seed.

**See also:** [data layer API](./data.md).

---

## `adaptive_tau`

**Source:** `scripts/launchers/adaptive_tau.py`

**Purpose:** The **primary construction launcher**. Runs constrained bi-objective NSGA-II with the 3-phase adaptive-tau calibration protocol (probe → bisect → production).

**Usage:**
```bash
# Single replica
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --rep 0 \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

# All 5 replicas
python scripts/launchers/adaptive_tau.py \
    --k 300 --space raw --constraint-mode popsoft --all \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics

# Ablation: MMD-only objective
python scripts/launchers/adaptive_tau.py \
    --k 100 --space vae --constraint-mode popsoft --objectives mmd --all \
    --cache-dir replicate_cache_seed4200 \
    --output-dir EXPERIMENTS-tau_fixed-leakage_fixed-separated_construction_metrics
```

**Arguments:**
| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--k` | `int` | yes | — | Coreset cardinality. |
| `--space` | `{vae, raw, pca}` | yes | — | Representation space. |
| `--constraint-mode` | `{popsoft, munisoft, unconstrained}` | yes | — | Geographic constraint. |
| `--objectives` | `str[+]` | no | `mmd sinkhorn` | Active objectives. |
| `--rep` | `int` | one of these | — | Single replica ID. |
| `--all` | flag | one of these | — | Run all 5 replicas. |
| `--cache-dir` | `str` | yes | — | Directory containing `rep{id}/assets.npz`. |
| `--output-dir` | `str` | yes | — | Base directory for experiment outputs. |

**Outputs (per experiment):**
```
{output_dir}/{exp_name}/
├── coreset.npz                # Pareto front (X, F, objectives)
├── representatives/           # Named selections from front
│   ├── knee.npz
│   ├── best-mmd.npz
│   └── best-sinkhorn.npz
├── adaptive-tau-log.json     # Probe/bisect/production log per generation
├── wall-clock.json           # Total seconds, total generations
└── manifest.json             # Seed, config, git commit, hyperparameters
```

Experiment names are auto-generated to be descriptive:
- `nsga2-vae-popsoft-k100-rep0` (standard bi-objective run)
- `ablation-mmdonly-k100-rep0` (MMD-only)
- `ablation-triobjective-k100-rep0` (tri-objective)

**Construction only.** No metrics are computed here; call `evaluate_coresets.py` afterwards.

**See also:** [optimization API](./optimization.md), [PIPELINE.md](../../scripts/PIPELINE.md).

---

## `run_baselines`

**Source:** `scripts/launchers/run_baselines.py`

**Purpose:** Post-hoc soft-KL repair for baseline coresets, making them directly comparable to NSGA-II runs that enforced the same soft constraints during optimisation.

**Usage:**
```bash
python scripts/launchers/run_baselines.py
python scripts/launchers/run_baselines.py --modes ps ms
python scripts/launchers/run_baselines.py --spaces vae raw pca
```

**Arguments:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--modes` | `str[+]` | all | Constraint modes to repair (`ps`, `ms`, `sh`, `hs`). |
| `--spaces` | `str[+]` | all | Feature spaces to process (`vae`, `raw`, `pca`). |

**Outputs:** `results/repaired_baselines/all_results.csv` with columns: method, constraint_mode, space, k, rep_id, KL-before/after, items changed, all evaluation metrics.

**See also:** [baselines API](./baselines.md), [constraints API](./constraints.md).

---

## `evaluate_coresets`

**Source:** `scripts/analysis/evaluate_coresets.py`

**Purpose:** The **single authoritative source** for all downstream metric computation. Loads saved coreset indices from disk and runs the 5-stage evaluation pipeline (geo diagnostics → raw-space operators → KPI stability → multi-model downstream → QoS) without any runner or config object. Applies the S ∩ E overlap fix.

**Usage:**
```bash
# Evaluate one experiment
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir runs_out/nsga2-vae-popsoft-k100-rep0 \
    --cache-path replicate_cache_seed4200/rep00/assets.npz

# Batch-evaluate all experiments matching a pattern
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir runs_out/ \
    --cache-path replicate_cache_seed4200/rep00/assets.npz \
    --batch --pattern "nsga2-vae-*"

# Force re-evaluation (overwrite metrics.csv)
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir runs_out/ \
    --cache-path replicate_cache_seed4200/rep00/assets.npz \
    --batch --force
```

**Arguments:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment-dir` | `str` | required | Path to a single experiment dir **or** a parent dir containing many experiments (with `--batch`). |
| `--cache-path` | `str` | required | Path to `assets.npz`. |
| `--batch` | flag | off | Evaluate every subdirectory of `--experiment-dir` that contains a `coreset.npz`. |
| `--force` | flag | off | Re-evaluate even if `metrics.csv` already exists. |
| `--pattern` | `str` | `*` | Glob filter for `--batch` mode. |

**Outputs (per experiment):**
- `{experiment-dir}/metrics.csv` — every front member + named representatives.
- `{experiment-dir}/metrics-representatives.csv` — subset for only knee, best-mmd, etc.

**See also:** [evaluation API](./evaluation.md).

---

## `championship`

**Source:** `scripts/analysis/championship.py`

**Purpose:** Statistical ranking of 13 coreset methods (5 NSGA-II variants + 8 baselines) using Friedman test, Nemenyi post-hoc CD, per-replica Cliff's delta head-to-head, and oracle analysis. Produces the Tables 3/4/5/6/9 numbers for the manuscript.

**Usage:**
```bash
python scripts/analysis/championship.py
```

**Arguments:** (the audit recommends switching `BASE` constant to a `--experiments-dir` CLI arg; currently hardcoded.)

**Outputs:** `manuscript_final_v3.json` (machine-readable rankings) plus stdout tables. Does **not** write LaTeX files.

**See also:** [evaluation API](./evaluation.md#cross-method-comparison).

---

## `bootstrap_reeval`

**Source:** `scripts/bootstrap_reeval.py`

**Purpose:** Bootstrap target-variable re-evaluation — draws `B` random target subsets and evaluates all NSGA-II Pareto front members + 8 baselines on each draw. Produces confidence intervals for downstream metrics.

**Usage:**
```bash
python scripts/bootstrap_reeval.py --run-id K_vae_k100 --rep-id 0
python scripts/bootstrap_reeval.py --run-id K_vae_k100 --rep-id 0 \
    --n-bootstrap 50 --n-reg 5 --n-cls 5 \
    --output-dir bootstrap_results/
```

**Arguments:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--run-id` | `str` | required | Experiment directory name. |
| `--rep-id` | `int` | required | Replica ID. |
| `--n-bootstrap` | `int` | `50` | Number of target draws. |
| `--n-reg` | `int` | `5` | Regression targets per draw. |
| `--n-cls` | `int` | `5` | Classification targets per draw. |
| `--output-dir` | `str` | `bootstrap_results/` | Output directory. |

**Outputs:**
- `bootstrap_results/bootstrap_raw_{run_id}_rep{id}.csv` — one row per (method, draw).
- `bootstrap_results/bootstrap_meta_{run_id}_rep{id}.json` — metadata sidecar with B, seed, timing, completion marker for dispatcher.

**Crash safety:** writes partial results atomically and resumes from the last completed draw. The dispatcher relies on the metadata sidecar's `complete: true` field to mark a job finished.

**See also:** orchestration via `scripts/bootstrap/dispatcher.sh`.

---

## Shell Orchestrators (Not Detailed)

These are minor orchestration scripts; see their inline `--help` or `scripts/PIPELINE.md` for usage:

- `scripts/bootstrap/dispatcher.sh` — parallel tmux dispatcher for `bootstrap_reeval.py` jobs.
- `scripts/bootstrap/run_one.sh` — single-job wrapper called by dispatcher.
- `scripts/deploy/labgele.sh`, `lessonia.sh` — one-command server deployment.
- `scripts/infra/server_setup.sh` — one-time server provisioning.
- `scripts/infra/run_parallel_tmux.sh` — multi-session launcher.
- `scripts/infra/collect_and_merge.sh` — SCP + CSV merge.
- `scripts/infra/monitor.sh` — tmux dashboard.

---

## See Also

- [scripts/PIPELINE.md](../../scripts/PIPELINE.md) — pipeline-level documentation with data-flow diagrams.
- [docs/METHODOLOGY.md](../METHODOLOGY.md) — theoretical background and implementation mapping.
