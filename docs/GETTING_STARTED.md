# Getting Started

A 10-minute orientation for newcomers. If you have never seen this repository before, read this first.

## What is this repo?

This repository contains the code for a research project on **geographically-constrained coreset selection**. In one sentence: given a large dataset partitioned into geographic groups (e.g., Brazilian municipalities by state), we select a *small* representative subset (a "coreset") that (a) preserves the distributional properties of the full dataset under kernel measures like MMD and Sinkhorn divergence, and (b) satisfies hard or soft geographic proportionality constraints.

The **primary algorithm** is a multi-objective NSGA-II that jointly minimises `MMD²` and `Sinkhorn divergence` under population-share or municipality-share constraints. The output is a *Pareto front* of coresets, each balancing the two objectives differently.

The **primary application** is telecom analytics on Brazilian municipality data: the selected municipalities become "landmarks" for downstream tasks (kernel regression, classification of infrastructure outcomes, QoS analysis) with provably bounded approximation error.

## What makes it worth publishing?

Three contributions:

1. **Adaptive τ calibration.** Soft geographic constraints require a tolerance `τ`; we automatically calibrate `τ` via a probe/bisect/production protocol so that roughly 50 % of the NSGA-II population is feasible — the sweet spot for evolutionary search.
2. **Target-leakage-safe preprocessing.** Every downstream evaluation target is automatically detected and excluded from the feature matrix before representation learning, so the VAE/PCA embeddings cannot leak downstream targets. A `validate_no_leakage` guard enforces this.
3. **S ∩ E overlap fix.** When a landmark point coincides with an evaluation point, it would artificially reduce the Nyström error by evaluating itself. We exclude overlapping indices from the evaluation set at metric-computation time.

## How does the pipeline work?

Four phases:

```
Phase 1 (Data prep)         Raw CSVs  →  build_caches.py   →  replicate cache (assets.npz)
Phase 2 (Construction)      Cache     →  adaptive_tau.py    →  coreset.npz (Pareto front)
Phase 3 (Evaluation)        Coresets  →  evaluate_coresets.py →  metrics.csv
Phase 4 (Analysis)          Metrics   →  championship.py    →  rankings (CSV/JSON)
```

Construction and evaluation are **intentionally decoupled**: coresets are saved as index arrays, and metrics are computed in a separate pass. This means the S ∩ E overlap fix is enforced in one place, metrics can be recomputed without re-running the optimiser, and every row in every results table traces to the same code path.

For a detailed, per-phase tour, see [../scripts/PIPELINE.md](../scripts/PIPELINE.md).

## 5-minute install

```bash
# 1. Clone the repo
git clone <repo-url> Coreset_Codes
cd Coreset_Codes

# 2. Create a virtualenv and install
python -m venv venv
source venv/bin/activate          # or: venv\Scripts\activate on Windows
pip install -e .

# 3. Verify
python -c "import coreset_selection; print(coreset_selection.__version__)"
```

If step 3 prints `0.1.0` without error, you're ready.

For GPU setup, platform-specific notes, or dependency troubleshooting, see [INSTALL.md](./INSTALL.md).

## Smallest possible experiment

Once installed, run a tiny end-to-end experiment to confirm everything works:

```bash
# 1. Build a single replicate cache (fastest possible settings).
python scripts/launchers/build_caches.py --reps 0

# 2. Run one NSGA-II experiment with k=30 (small coreset, fast).
python scripts/launchers/adaptive_tau.py \
    --k 30 --space vae --constraint-mode popsoft --rep 0 \
    --cache-dir experiments/adaptive_tau_k100_ps_vae/replicate_cache \
    --output-dir runs_out_quickstart

# 3. Evaluate the coreset.
python scripts/analysis/evaluate_coresets.py \
    --experiment-dir runs_out_quickstart/nsga2-vae-popsoft-k30-rep0 \
    --cache-path experiments/adaptive_tau_k100_ps_vae/replicate_cache/rep00/assets.npz

# 4. Look at the result.
head runs_out_quickstart/nsga2-vae-popsoft-k30-rep0/metrics.csv
```

The whole sequence takes roughly 10 minutes on a modern laptop (most of it is VAE training inside `build_caches.py`). The `metrics.csv` has one row per Pareto-front member with columns for every metric (Nyström error, KRR RMSE per target, geographic KL, etc.).

## Understanding the outputs

After a successful run, each experiment directory contains:

```
{exp_name}/
├── coreset.npz               Pareto front (binary masks + objective values)
├── representatives/          Named selections from the front
│   ├── knee.npz              Balanced compromise point
│   ├── best-mmd.npz          Front member with lowest MMD
│   └── best-sinkhorn.npz     Front member with lowest Sinkhorn
├── metrics.csv               All metrics for every front member + named reps
├── metrics-representatives.csv   Subset: only named reps (for quick inspection)
├── adaptive-tau-log.json     Probe/bisect/production log per generation
├── wall-clock.json           Total seconds and generations
└── manifest.json             Seed, config, git commit, hyperparameters
```

**Key columns in `metrics.csv`:**
| Group | Columns | Meaning |
|-------|---------|---------|
| Geographic | `geo_kl_muni`, `geo_kl_pop`, `geo_l1_*`, `geo_maxdev_*` | Distance from target group-share distributions. |
| Operator fidelity | `nystrom_error`, `kpca_distortion` | How well `S` approximates kernel operations. |
| KRR | `krr_rmse_cov_area_4G`, `krr_rmse_cov_area_5G` | Downstream kernel ridge regression RMSE. |
| Multi-model | `knn_rmse_*`, `rf_rmse_*`, `lr_*`, `gbt_*` | KNN / RF / logistic / GBT on Nyström features. |

## Next steps

- For hands-on tutorials, see the Jupyter notebooks in [../examples/](../examples/).
- For a deeper understanding of the pipeline, read [../scripts/PIPELINE.md](../scripts/PIPELINE.md).
- To look up a specific function, use the [API Reference](./api/index.md).
- To run the full experiment matrix (15 configs across multiple k values), see [EXPERIMENTS.md](./EXPERIMENTS.md).
- To contribute code, read [../CONTRIBUTING.md](../CONTRIBUTING.md).
