# API Reference — Experiment & Models (`coreset_selection.experiment`, `coreset_selection.models`)

Two subpackages covering (a) **experiment orchestration** — the `ExperimentRunner` that wires together config, data, optimisation, and evaluation into a single run — and (b) **representation-learning models** — `TabularVAE` and PCA utilities used to build the VAE and PCA embeddings stored in each replicate cache.

Public symbols are re-exported from `coreset_selection/experiment/__init__.py` and `coreset_selection/models/__init__.py`.

---

## Part 1: Experiment Orchestration

### `coreset_selection.experiment.ExperimentRunner`

**Kind:** class
**Source:** `coreset_selection/experiment/runner.py`

**Summary:** Top-level orchestrator that ties together config, cache loading, NSGA-II, and evaluation for a single (run_id, rep_id) pair.

**Description:** The legacy entry point used by the original pipeline. Instantiated with an `ExperimentConfig`, it performs: (1) ensure the replicate cache exists (trigger `build_replicate_cache` if not), (2) load assets via `load_replicate_cache`, (3) build `GeoInfo`, constraints, and `SpaceObjectiveComputer`, (4) run `nsga2_optimize`, (5) evaluate representatives via `EvalMixin._evaluate_coreset`, (6) persist everything via `ResultsSaver`. The new clean pipeline uses `scripts/launchers/adaptive_tau.py` directly and skips the runner, but `ExperimentRunner` remains available for anyone who wants a one-shot callable.

**Key methods:**
- `.run()` — execute the full pipeline.
- `._evaluate_coreset(S_idx, ...)` — internal evaluator (shared logic with `scripts/analysis/evaluate_coresets.py`).

### `coreset_selection.experiment.run_single_experiment`

**Kind:** function
**Source:** `coreset_selection/experiment/runner.py`

**Summary:** Functional wrapper that instantiates `ExperimentRunner(cfg).run()` in one call.

**Signature:** `def run_single_experiment(cfg: ExperimentConfig) -> ExperimentResult`

### `coreset_selection.experiment.run_sweep`

**Kind:** function
**Source:** `coreset_selection/experiment/runner.py`

**Summary:** Iterate `run_single_experiment` across a grid of configs and return aggregated results.

---

### `coreset_selection.experiment.ResultsSaver`

**Kind:** class
**Source:** `coreset_selection/experiment/saver.py`

**Summary:** Unified writer for all experiment outputs (Pareto front, coreset indices, metrics CSV, timing JSON, run manifest).

**Description:** Given an output directory and an experiment identifier, produces the canonical file layout used by the legacy pipeline (`results/`, `coresets/`, `wall_clock.json`, `run_manifest.json`). Exposes:
- `.save_pareto_front(space, pareto_data)` — persist `{space}_pareto.npz`.
- `.save_front_metrics(space, rows)` — persist `front_metrics_{space}.csv`.
- `.save_all_results(rows)` — persist `all_results.csv`.
- `.save_coreset(name, indices, mask)` — persist a named coreset `.npz`.

### `coreset_selection.experiment.ParetoFrontData`

**Kind:** dataclass
**Source:** `coreset_selection/experiment/saver.py`

**Summary:** Container carrying the Pareto-front outputs from NSGA-II to the saver.

**Attributes (typical):**
| Name | Type | Description |
|------|------|-------------|
| `X` | `np.ndarray, shape (n_front, N)` | Boolean masks, one row per Pareto-front member. |
| `F` | `np.ndarray, shape (n_front, n_obj)` | Objective values. |
| `objectives` | `List[str]` | Objective names in column order of `F`. |
| `rep_names` | `List[str]` | Names of selected representatives (knee, best-mmd, ...). |
| `rep_indices` | `List[int]` | Row indices in `X`/`F` corresponding to each representative. |

### `coreset_selection.experiment.load_pareto_front`

**Kind:** function
**Source:** `coreset_selection/experiment/saver.py`

**Summary:** Deserialize a `{space}_pareto.npz` back into a `ParetoFrontData`.

---

## Part 2: Representation-Learning Models

### `coreset_selection.models.TabularVAE`

**Kind:** class (PyTorch `nn.Module`)
**Source:** `coreset_selection/models/vae.py`

**Summary:** Mixed-type tabular variational autoencoder used to produce `Z_vae` embeddings in the replicate cache.

**Description:** Architecture: two hidden layers of 128 ReLU units, latent dim = 32 (configurable). Trained with MSE reconstruction loss plus KL regularisation weighted by 0.1, using Adam and early stopping with patience 200. The log-variance head produces per-dimension posterior variances (`Z_logvar`), which feed the SKL objective. Training is called only inside `build_replicate_cache`; for downstream runs, the trained model is discarded and only the per-point `Z_vae`/`Z_logvar` are cached.

**Constructor signature (representative):**
```python
TabularVAE(
    n_features: int,
    latent_dim: int = 32,
    hidden_dims: List[int] = [128, 128],
    dropout: float = 0.0,
) -> TabularVAE
```

**Forward signature:** `.forward(x) -> (x_recon, mu, logvar)`.

### `coreset_selection.models.VAETrainer`

**Kind:** class
**Source:** `coreset_selection/models/vae.py`

**Summary:** Helper that handles training loop, early stopping, and embedding extraction.

**Key methods:**
- `.train(X_train, X_val, max_epochs, patience, lr)` — train with early stopping.
- `.embed(X) -> (Z, Z_logvar)` — extract embeddings and log-variances.

### `coreset_selection.models.fit_pca`

**Kind:** function
**Source:** `coreset_selection/models/pca.py`

**Summary:** Fit a PCA on a training matrix.

**Signature:** `def fit_pca(X_train: np.ndarray, n_components: int = 32, random_state: int = 0) -> PCA`

### `coreset_selection.models.pca_embed`

**Kind:** function
**Source:** `coreset_selection/models/pca.py`

**Summary:** Project any matrix through a fitted PCA.

### `coreset_selection.models.explained_variance_ratio`

**Kind:** function — ratio of variance explained per component.

### `coreset_selection.models.cumulative_explained_variance`

**Kind:** function — cumulative variance curve (used for scree plots).

### `coreset_selection.models.components_for_variance`

**Kind:** function — smallest `k` such that cumulative variance ≥ threshold.

### `coreset_selection.models.IncrementalPCAWrapper`

**Kind:** class — batched/out-of-core PCA fallback for very large `N`.

---

## See Also

- [data](./data.md) — `build_replicate_cache` invokes both `VAETrainer` and `fit_pca`.
- [objectives](./objectives.md) — VAE embeddings are the default input to `SpaceObjectiveComputer`.
- [scripts](./scripts.md#build_caches) — `build_caches.py` invokes these through `build_replicate_cache`.
- Manuscript Section VI-A (representation choices).
