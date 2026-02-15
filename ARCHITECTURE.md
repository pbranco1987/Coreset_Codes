# Software Architecture: Complete Reference

This document provides exhaustive documentation of the software architecture, module dependencies, class hierarchies, configuration system, CLI interface, parallel execution infrastructure, result persistence, testing, and dependencies. It is designed to support reproducibility and understanding of the codebase.

---

## 1. Module Dependency Graph

```
                         ┌──────────────────┐
                         │   CLI (_main.py)  │
                         │   _commands.py    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌────▼────┐ ┌──────▼──────┐
              │  runners/  │ │ config/ │ │  artifacts/ │
              │ scenario   │ │ specs   │ │ manuscript  │
              │ parallel   │ │ consts  │ │ tables/plot │
              └─────┬──────┘ └────┬────┘ └──────┬──────┘
                    │             │              │
              ┌─────▼─────────────▼──────────────┘
              │        experiment/runner.py
              │        (ExperimentRunner)
              └──┬──────┬──────┬──────┬──────┬──────┐
                 │      │      │      │      │      │
           ┌─────▼──┐ ┌─▼────┐│ ┌────▼───┐ ┌▼─────┐│
           │ data/  │ │geo/  ││ │objecti-│ │eval/ ││
           │manager │ │kl.py ││ │ves/    │ │raw   ││
           │cache   │ │info  ││ │mmd     │ │space ││
           │loader  │ │proj  ││ │sinkh   │ │geo   ││
           └────────┘ └──────┘│ │skl     │ │kpi   ││
                              │ └────────┘ └──────┘│
                        ┌─────▼──────┐       ┌─────▼──────┐
                        │optimization│       │ baselines/ │
                        │ nsga2      │       │ 7 methods  │
                        │ repair     │       │ variant_gen│
                        │ operators  │       └────────────┘
                        │ selection  │
                        │ problem    │
                        └─────┬──────┘
                              │
                        ┌─────▼──────┐
                        │constraints/│
                        │proportion  │
                        │calibration │
                        └────────────┘

Supporting:
  ┌──────────┐  ┌──────────┐
  │ models/  │  │  utils/  │
  │ vae.py   │  │ math.py  │
  │ pca.py   │  │ seed.py  │
  └──────────┘  │ timing   │
                └──────────┘
```

### Data Flow

```
Input CSVs ──► BrazilTelecomDataLoader ──► DataManager ──► Cache Builder
                                                              │
                                              ┌───────────────┤
                                              │               │
                                         VAE/PCA          Preprocessing
                                         Training         (impute, log1p,
                                              │            scale)
                                              ▼               │
                                         Embeddings           ▼
                                              │          Standardized
                                              ▼          Features
                                         Replicate Cache (.npz)
                                              │
                      ┌───────────────────────┤
                      │                       │
                Objective Computer      ExperimentRunner
                (RFFMMD, AnchorSink)          │
                      │               ┌───────┴───────┐
                      ▼               │               │
                NSGA-II Optimizer   Baselines     Diagnostics
                      │               │               │
                      ▼               ▼               ▼
                Pareto Front      Selections     Alignment
                      │               │           Analysis
                      ▼               ▼               │
                Representative    Evaluation          │
                Selection         (raw space)         │
                      │               │               │
                      ▼               ▼               ▼
                Results (JSON/NPZ) ──► Artifact Generation
                                           │
                                           ▼
                                    Figures (PDF) + Tables (LaTeX/CSV)
```

---

## 2. Complete Class Hierarchy

### Core Orchestration

| Class | File | Responsibility |
|-------|------|---------------|
| `ExperimentRunner` | `experiment/runner.py` | Main experiment orchestrator. Mixins: R0Mixin, DiagnosticsMixin, EffortMixin, EvalMixin |
| `ResultsSaver` | `experiment/saver.py` | Persists results, manifests, configs to JSON/NPZ |

### Data Management

| Class | File | Responsibility |
|-------|------|---------------|
| `DataManager` | `data/manager.py` | Unified data loading, preprocessing, target extraction |
| `BrazilTelecomDataLoader` | `data/brazil_telecom_loader.py` | 3-CSV Brazil telecom data loading with fuzzy matching |
| `ReplicateAssets` | `config/dataclasses.py` | Dataclass holding all cached replicate data |

### Geographic Constraints

| Class | File | Responsibility |
|-------|------|---------------|
| `GeoInfo` | `geo/info.py` | Geographic group metadata (27 states, proportions, indices) |
| `GeographicConstraintProjector` | `geo/projector.py` | Computes KL-optimal quotas, projects masks to satisfaction |

### Constraints

| Class | File | Responsibility |
|-------|------|---------------|
| `ProportionalityConstraint` | `constraints/proportionality.py` | Single constraint: D(S) <= tau |
| `ProportionalityConstraintSet` | `constraints/proportionality.py` | Aggregates multiple constraints, checks feasibility |

### Optimization

| Class | File | Responsibility |
|-------|------|---------------|
| `CoresetMOOProblem` | `optimization/problem.py` | pymoo Problem formulation (variables, objectives, constraints) |
| `QuotaAndCardinalityRepair` | `optimization/repair.py` | Algorithm 2: swap-based repair for quota + cardinality |
| `ExactKRepair` | `optimization/repair.py` | Simpler repair for cardinality-only |
| `RepairActivityTracker` | `optimization/repair.py` | Tracks repair operations and violation statistics |
| `UniformBinaryCrossover` | `optimization/operators.py` | Binary crossover operator (p_c = 0.9) |
| `BitFlipMutation` | `optimization/operators.py` | Standard bit-flip mutation |
| `QuotaSwapMutation` | `optimization/operators.py` | Quota-preserving swap mutation (p_m = 0.2) |

### Objective Functions

| Class | File | Responsibility |
|-------|------|---------------|
| `RFFMMD` | `objectives/mmd.py` | MMD computation via Random Fourier Features (m=2000) |
| `AnchorSinkhorn` | `objectives/sinkhorn.py` | Sinkhorn divergence via anchor approximation (A=200) |
| `SpaceObjectiveComputer` | `objectives/computer.py` | Factory dispatching to MMD/Sinkhorn/SKL based on config |

### Models

| Class | File | Responsibility |
|-------|------|---------------|
| `TabularVAE` | `models/vae.py` | beta-VAE architecture (encoder + decoder) |
| `VAETrainer` | `models/vae.py` | Training loop with early stopping, AMP, torch.compile |

### Evaluation

| Class | File | Responsibility |
|-------|------|---------------|
| `RawSpaceEvaluator` | `evaluation/raw_space.py` | Nystrom error, kPCA distortion, KRR RMSE |
| `EnhancedRawSpaceEvaluator` | `evaluation/enhanced_evaluator.py` | Adds downstream models, tail errors, per-state metrics |

### Baselines

| Class/Function | File | Algorithm |
|----------------|------|-----------|
| `uniform_sample()` | `baselines/uniform.py` | Stratified random sampling |
| `kmeans_select()` | `baselines/kmeans.py` | K-means + closest to centroid |
| `herding_select()` | `baselines/herding.py` | Greedy MMD minimization (RFF) |
| `farthest_first_select()` | `baselines/farthest_first.py` | Greedy k-center |
| `leverage_select()` | `baselines/leverage.py` | Ridge leverage-score importance sampling |
| `dpp_select()` | `baselines/dpp.py` | Greedy k-DPP MAP approximation |
| `kernel_thinning_select()` | `baselines/kernel_thinning.py` | Stein operator-based thinning |
| `generate_quota_variant()` | `baselines/variant_generator.py` | Post-hoc quota repair for any baseline |

### Artifact Generation

| Class | File | Responsibility |
|-------|------|---------------|
| `ManuscriptArtifacts` | `artifacts/manuscript_artifacts.py` | Multi-mixin figure/table orchestrator |

Mixins: `GenerateAllMixin`, `GeoMapsMixin`, `ParetoFigsMixin`, `MetricFigsMixin`, `ComparisonFigsMixin`, `TablesMixin`

---

## 3. Configuration System

### Dataclass Hierarchy

```
ExperimentConfig
├── FilesConfig
│   ├── data_dir: str = "data"
│   ├── use_brazil_telecom: bool = True
│   ├── main_data_file: str = "smp_main.csv"
│   ├── metadata_file: str = "metadata.csv"
│   └── population_file: str = "city_populations.csv"
│
├── PreprocessingConfig
│   ├── categorical_columns: List[str] = []
│   ├── ordinal_columns: List[str] = []
│   ├── ignore_columns: List[str] = []
│   ├── target_columns: List[str] = []
│   ├── treat_low_cardinality_int_as_categorical: bool = True
│   ├── low_cardinality_threshold: int = 25
│   ├── scale_ordinals: bool = True
│   ├── scale_categoricals: bool = False
│   ├── log1p_categoricals: bool = False
│   ├── log1p_ordinals: bool = False
│   ├── categorical_impute_strategy: str = "mode"
│   ├── ordinal_impute_strategy: str = "median"
│   └── target_type: str = "auto"
│
├── VAEConfig
│   ├── latent_dim: int = 32
│   ├── hidden_dim: int = 128
│   ├── epochs: int = 1500
│   ├── batch_size: int = 256
│   ├── lr: float = 1e-3
│   ├── kl_weight: float = 0.1
│   ├── early_stopping_patience: int = 200
│   ├── torch_threads: int = 4
│   └── torch_interop_threads: int = 4
│
├── PCAConfig
│   ├── n_components: int = 32
│   └── whiten: bool = False
│
├── GeoConfig
│   ├── constraint_mode: str = "population_share"
│   ├── alpha_geo: float = 1.0
│   ├── min_one_per_group: bool = True
│   ├── tau_population: float = 0.02
│   └── tau_municipality: float = 0.02
│
├── MMDConfig
│   ├── rff_dim: int = 2000
│   └── bandwidth_mult: float = 1.0
│
├── SinkhornConfig
│   ├── n_anchors: int = 200
│   ├── eta: float = 0.05
│   ├── max_iter: int = 100
│   ├── stop_thr: float = 1e-6
│   └── anchor_method: str = "kmeans"
│
├── SolverConfig
│   ├── k: int = None  (required)
│   ├── pop_size: int = 200
│   ├── n_gen: int = 1000
│   ├── crossover_prob: float = 0.9
│   ├── mutation_prob: float = 0.2
│   ├── objectives: Tuple[str] = ("mmd", "sinkhorn")
│   ├── enforce_exact_k: bool = True
│   └── verbose: int = 0
│
├── EvalConfig
│   ├── enabled: bool = True
│   ├── eval_size: int = 2000
│   ├── eval_train_frac: float = 0.8
│   ├── n_kpca_components: int = 20
│   ├── nystrom_enabled: bool = True
│   ├── kpca_enabled: bool = True
│   ├── krr_enabled: bool = True
│   ├── multi_model_enabled: bool = True
│   ├── qos_enabled: bool = True
│   └── qos_models: List[str] = ["ols","ridge","elastic_net","pls","constrained","heuristic"]
│
├── BaselinesConfig
│   ├── enabled: bool = True
│   └── methods: List[str] = ["uniform","kmeans","herding","farthest_first","rls","dpp","kernel_thinning"]
│
├── AblationsConfig
│   ├── geo_ablation: bool = False
│   ├── objective_ablation: bool = False
│   └── representation_ablation: bool = False
│
├── SweepConfig
│   ├── k_values: List[int] = K_GRID
│   └── n_replicates: int = 1
│
└── ManuscriptFiguresConfig
    ├── format: str = "pdf"
    ├── dpi: int = 300
    └── which_plots: List[str] = ["all"]
```

### Run Specification System

**RunSpec dataclass** (`config/run_specs.py`):

```python
@dataclass(frozen=True)
class RunSpec:
    run_id: str              # "R0" through "R14"
    description: str         # Human-readable purpose
    space: str               # "vae", "raw", or "pca"
    objectives: Tuple[str]   # e.g., ("mmd", "sinkhorn")
    constraint_mode: str     # "population_share", "municipality_share_quota", "joint", "none"
    enforce_exact_k: bool
    k: Optional[int]         # Fixed k (None = use sweep)
    sweep_k: Optional[Tuple[int]]  # K_GRID for sweep runs
    sweep_dim: Optional[Tuple[int]]  # D_GRID for dim sweeps
    n_reps: int              # Number of replicates
    baselines_enabled: bool
    eval_enabled: bool
    requires_vae: bool
    requires_pca: bool
    cache_build_mode: str    # "lazy" or "skip"
    depends_on_runs: Tuple[str]
```

**`apply_run_spec(base_cfg, spec, rep_id, dim_override=None)`** transforms a base ExperimentConfig according to a RunSpec:
1. Sets cache path from rep_id
2. Applies constraint_mode to GeoConfig
3. Sets solver.k, solver.objectives
4. Enables/disables eval and baselines
5. Sets representation space
6. For dimension sweeps: overrides latent_dim or n_components

### Configuration Construction

**`build_base_config(args)`** (`cli/_config.py`) constructs the default ExperimentConfig from CLI arguments, populating all sub-configs with their default values.

---

## 4. CLI Interface -- Complete Command Reference

**Entry point:** `python -m coreset_selection <command> [options]`

### `prep` -- Pre-build Replicate Caches

```
python -m coreset_selection prep
    --n-replicates N      Number of replicate caches to build (default: 10)
    --output-dir DIR      Output directory (default: runs_out)
    --cache-dir DIR       Cache directory (default: replicate_cache)
    --data-dir DIR        Data directory (default: data)
    --seed N              Base random seed (default: 123)
    --device cpu|cuda     Compute device (default: cpu)
    --fail-fast           Stop on first failure
```

**Purpose:** Train VAE and PCA for each replicate, build cache files.

### `run` -- Single Experiment Instance

```
python -m coreset_selection run
    --run-id STR          Required: R0 through R14
    --rep-id N            Replicate ID (default: 0)
    --k N                 Required: coreset size
    --output-dir DIR      (default: runs_out)
    --cache-dir DIR       (default: replicate_cache)
    --data-dir DIR        (default: data)
    --seed N              (default: 123)
    --device cpu|cuda     (default: cpu)
```

### `scenario` -- Full Scenario Execution

```
python -m coreset_selection scenario
    --run-id STR          Required: R0 through R14
    --k-values K1,K2,...  Override k values (comma-separated)
    --rep-ids ID1,ID2,... Override replicate IDs
    --output-dir DIR      (default: runs_out)
    --cache-dir DIR       (default: replicate_cache)
    --data-dir DIR        (default: data)
    --seed N              (default: 123)
    --device cpu|cuda     (default: cpu)
    --fail-fast           Stop on first failure
    # R6 specific:
    --source-run STR      Source run for filtering (default: R1)
    --source-space STR    Source space: vae|pca|raw (default: vae)
```

### `r0` through `r14` -- Convenience Aliases

```
python -m coreset_selection r1 -k 100,200,300
python -m coreset_selection r6 -k 300 --source-run R1
python -m coreset_selection r10 -k 300
```

### `sweep` -- K-Value Sweep (Legacy)

```
python -m coreset_selection sweep
    --run-id STR          Required
    --k-values K1,K2,...  Required
    --n-replicates N      (default: 10)
    ...
```

### `parallel` (`par`) -- Multi-Process Execution

```
python -m coreset_selection parallel [RUNS...]
    RUNS                  Space-separated run IDs (e.g., r1 r2 r3) or "all"
    -t, --threads N       Threads per job (auto-calculated if not set)
    -l, --log-dir DIR     Log directory (default: logs/)
    --output-dir DIR      (default: runs_out)
    --cache-dir DIR       (default: replicate_cache)
    --data-dir DIR        (default: data)
```

### `all` -- Run All Experiments Sequentially

```
python -m coreset_selection all -k K1,K2,...
    -k STR                Required (single or comma-separated)
    --rep-ids ID1,ID2,... (overrides run spec)
    --fail-fast           Stop on first failure
    ...
```

### `seq` -- Run Selected Experiments Sequentially

```
python -m coreset_selection seq R1 R2 R6 -k K1,K2,...
    RUNS                  Run IDs to execute
    -k STR                Required
    ...
```

### `artifacts` (`figs`, `plots`) -- Generate Manuscript Artifacts

```
python -m coreset_selection artifacts [RUNS_DIR]
    RUNS_DIR              Auto-detects if not specified
    -o, --output DIR      Output directory (default: artifacts/)
    --rep REP_FOLDER      Replicate folder (default: rep00)
    --data-dir DIR        (default: data)
```

### `list-runs` -- Show All Run Specifications

```
python -m coreset_selection list-runs
```

Prints a formatted table of all 15 run specifications with descriptions.

---

## 5. Parallel Execution Infrastructure

### Three-Phase Protocol

**Phase 0: Cache Pre-Build (Sequential)**

```
For each unique rep_id needed across all scenarios:
    acquire_build_lock(cache_dir / rep{rep_id:02d})
    if cache does not exist or is incomplete:
        build_replicate_cache(cfg, rep_id)
    release_build_lock()
```

File-level locking (`_acquire_build_lock()` / `_release_build_lock()`) prevents concurrent builds from corrupting cache files.

**Phase 1: Experiment Execution (Parallel)**

```python
with ProcessPoolExecutor(max_workers=len(runs)) as pool:
    futures = {}
    for run in runs:
        env = {
            'OMP_NUM_THREADS': str(threads_per_job),
            'MKL_NUM_THREADS': str(threads_per_job),
            'OPENBLAS_NUM_THREADS': str(threads_per_job),
            'NUMEXPR_MAX_THREADS': str(threads_per_job),
            'MKL_THREADING_LAYER': 'GNU',
            'OMP_MAX_ACTIVE_LEVELS': '1',
            'KMP_BLOCKTIME': '200',
        }
        cmd = ['python', '-m', 'coreset_selection', run.run_id, ...]
        futures[pool.submit(subprocess.run, cmd, env=env)] = run
```

**Phase 2: Result Collection (Sequential)**

```python
for future in as_completed(futures):
    run = futures[future]
    result = future.result()
    print(f"{run.run_id}: {'OK' if result.returncode == 0 else 'FAILED'}")
```

### Thread Tuning

```python
n_cores = multiprocessing.cpu_count()
n_jobs = len(runs)
threads_per_job = args.threads or max(2, min(16, n_cores // n_jobs))
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OMP_NUM_THREADS` | OpenMP threads | Auto-calculated |
| `MKL_NUM_THREADS` | Intel MKL threads | Auto-calculated |
| `OPENBLAS_NUM_THREADS` | OpenBLAS threads | Auto-calculated |
| `NUMEXPR_MAX_THREADS` | NumExpr threads | Auto-calculated |
| `MKL_THREADING_LAYER` | MKL-OpenMP compatibility | GNU |
| `OMP_MAX_ACTIVE_LEVELS` | Nested parallelism | 1 (disabled) |
| `KMP_BLOCKTIME` | Worker thread spin time (ms) | 200 |

---

## 6. Result Persistence

### Output Directory Structure

```
runs_out/
├── R0/
│   └── rep00/
│       ├── quota_path.json        # Per-group allocations for each k
│       ├── kl_floor.csv           # KL_min(k) values
│       └── k{k}/                  # Per-k results
│           └── quota_counts.npz
├── R1/
│   ├── rep00/
│   │   ├── k50/
│   │   │   ├── pareto_front.json
│   │   │   ├── selection_summary.json
│   │   │   ├── selected_indices.json
│   │   │   ├── evaluation_report.json
│   │   │   ├── proportionality.json
│   │   │   ├── run_manifest.json
│   │   │   └── wall_clock.json
│   │   ├── k100/
│   │   │   └── ...
│   │   └── ...
│   ├── rep01/
│   │   └── ...
│   └── ...
├── R10/
│   └── rep00/
│       └── k300/
│           ├── baseline_uniform_exactk.json
│           ├── baseline_uniform_quota.json
│           ├── baseline_kmeans_exactk.json
│           └── ...
└── ...
```

### Result File Formats

| File | Format | Contents |
|------|--------|---------|
| `pareto_front.json` | JSON | Solutions array with objective values, binary masks |
| `selection_summary.json` | JSON | Representative solutions, knee point, per-objective bests |
| `selected_indices.json` | JSON | Municipality indices in the selected coreset |
| `evaluation_report.json` | JSON | All computed metrics (Nystrom, kPCA, KRR, etc.) |
| `proportionality.json` | JSON | Geographic diagnostics (KL, L1, max_dev, dual) |
| `run_manifest.json` | JSON | Full configuration dump + seed + timing |
| `wall_clock.json` | JSON | Per-phase wall-clock timing |

---

## 7. Artifact Generation System

### ManuscriptArtifacts Architecture

The artifact generator uses a **multi-mixin pattern** for modularity:

```
ManuscriptArtifacts(GenerateAllMixin, GeoMapsMixin, ParetoFigsMixin,
                    MetricFigsMixin, ComparisonFigsMixin, TablesMixin)
```

| Mixin | Responsibility |
|-------|---------------|
| `GenerateAllMixin` | Orchestrates all figure/table generation |
| `GeoMapsMixin` | Geographic choropleth maps |
| `ParetoFigsMixin` | Pareto front visualizations (2D, 3D) |
| `MetricFigsMixin` | Metric vs k curves, distortion plots |
| `ComparisonFigsMixin` | Baseline/ablation comparisons, regional analysis |
| `TablesMixin` | LaTeX table generation |

### Output Structure

```
artifacts_out/
├── figures/
│   ├── geo_ablation_tradeoff_scatter.pdf    # Fig 1
│   ├── distortion_cardinality_R1.pdf        # Fig 2
│   ├── regional_validity_k300.pdf           # Fig 3
│   ├── objective_metric_alignment_heatmap.pdf  # Fig 4
│   ├── kl_floor_vs_k.pdf                   # N1
│   ├── pareto_front_mmd_sd_k300.pdf         # N2
│   └── ...                                 # 20 total figures
└── tables/
    ├── exp_settings.tex                     # Table I
    ├── run_matrix.tex                       # Table II
    ├── r1_by_k.tex                          # Table III
    └── ...                                  # 13+ total tables
```

### IEEE Compliance

| Constraint | Requirement |
|-----------|------------|
| Single-column width | 3.5 inches |
| Double-column width | 7.0 inches |
| Minimum annotation font | 8pt |
| Minimum axis label font | 10pt |
| Resolution | 300 dpi |
| Format | PDF (vector) |

---

## 8. Testing Infrastructure

### Test File Inventory (17 files)

| Test File | Coverage |
|-----------|----------|
| `test_algorithm1_core.py` | Algorithm 1: KL-optimal quota computation, lazy-heap greedy |
| `test_algorithm1_edge.py` | Algorithm 1: edge cases, boundary conditions |
| `test_algorithm2.py` | Algorithm 2: swap-based repair, donor/recipient logic |
| `test_constraint_modes_basic.py` | Basic constraint mode integration (pop_share, muni_quota) |
| `test_constraint_modes_advanced.py` | Advanced constraint handling (joint, none, edge cases) |
| `test_end_to_end.py` | Full pipeline integration (data -> optimize -> evaluate) |
| `test_eval_protocol_raw.py` | Raw-space evaluation metrics (Nystrom, kPCA, KRR) |
| `test_eval_protocol_geo.py` | Geographic evaluation (KL, L1, quota satisfaction) |
| `test_incremental_evaluation.py` | Incremental result evaluation |
| `test_objectives.py` | Objective function computation (MMD, Sinkhorn, SKL) |
| `test_metrics.py` | Coverage and diversity metrics |
| `test_preproc_loading.py` | Data loading and merging |
| `test_preproc_schema.py` | Feature schema inference |
| `test_preproc_transforms.py` | Preprocessing transformations (imputation, log1p, scaling) |
| `test_verify_constants.py` | Constants consistency checks |
| `conftest.py` | Shared test fixtures and helpers |
| `__init__.py` | Package marker |

### Running Tests

```bash
# All tests
python -m pytest coreset_selection/tests/ -v

# Specific test suites
python -m pytest coreset_selection/tests/test_algorithm1_core.py -v
python -m pytest coreset_selection/tests/test_end_to_end.py -v

# With coverage
python -m pytest coreset_selection/tests/ --cov=coreset_selection --cov-report=html
```

### Test Design Patterns

- **Synthetic data:** Tests use `load_synthetic_data()` to generate small datasets (N=100, D=10, G=5) for fast execution
- **Deterministic seeds:** All tests set fixed seeds for reproducibility
- **Property-based testing:** Algorithm tests verify mathematical properties (e.g., KL optimality, constraint satisfaction)
- **Integration tests:** End-to-end tests run the full pipeline on synthetic data

---

## 9. Dependencies & Environment

### Core Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | >= 1.21 | Array operations, linear algebra |
| scipy | >= 1.7 | Optimization (SLSQP), eigendecomposition, distances |
| pandas | >= 1.3 | Data loading, manipulation |
| scikit-learn | >= 1.0 | StandardScaler, Ridge, KNN, RF, PCA, train_test_split |
| torch | >= 1.10 | VAE training (neural network), CUDA support |
| matplotlib | >= 3.4 | Figure generation |
| seaborn | >= 0.11 | Statistical visualization |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| geopandas | >= 0.12 | Geographic choropleth maps |
| pytest | >= 7.0 | Test execution |
| pytest-cov | -- | Coverage reporting |

### Python Version

- **Minimum:** Python >= 3.8
- **Recommended:** Python 3.10+ (for torch.compile support)

### Hardware Acceleration

| Feature | Condition | Benefit |
|---------|-----------|---------|
| CUDA GPU | `torch.cuda.is_available()` | VAE training acceleration |
| torch.compile() | PyTorch >= 2.0 | Optimized VAE forward/backward passes |
| AMP (Automatic Mixed Precision) | CUDA available | Reduced memory, faster training |

### Installation

```bash
# From source (editable mode)
pip install -e .

# Or install dependencies directly
pip install numpy scipy pandas scikit-learn torch matplotlib seaborn

# Optional: geographic maps
pip install geopandas

# Optional: testing
pip install pytest pytest-cov
```

---

## 10. Shell Script Templates

### `scripts/run_all_parallel.sh`

Runs all scenarios in parallel using the best available method:
1. GNU Parallel (if installed)
2. Background jobs with `&` and `wait`
3. Sequential fallback

### `scripts/slurm_array_job.sh`

SLURM job array template for cluster submission:
```bash
#SBATCH --job-name=coreset
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

RUNS=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14)
RUN_ID=${RUNS[$SLURM_ARRAY_TASK_ID]}
python -m coreset_selection scenario --run-id $RUN_ID ...
```

### `scripts/tmux_experiments.sh`

Creates a tmux session with panes for monitoring experiment progress:
- One pane per run
- Tail log files in real-time
- Summary pane showing overall status

---

## 11. Key Design Patterns

### Lazy Evaluation

- Cache validation before rebuild (`validate_cache()`)
- Cache augmentation for missing keys only
- Lazy heap in Algorithm 1 (avoid recomputing marginal gains)

### Separation of Concerns

- **Proxy space** (VAE/PCA/raw) for optimization
- **Raw space** for evaluation (independent of representation choice)
- **Configuration** separated from execution (dataclasses vs runner logic)

### Reproducibility by Design

- Global seed setting before each replicate
- Cache sharing ensures identical representations
- Deterministic index sets persisted in cache
- Full configuration dump in run manifests

### Memory Efficiency

- Chunked pairwise distance computation
- Full-batch VAE training for small datasets (avoids DataLoader overhead)
- Lazy loading of evaluation set kernel matrices
- Numpy operations preferred over Python loops

### Extensibility

- New baselines: implement function with `(X, k, rng)` signature, register in `BASELINE_METHODS`
- New objectives: implement class with `build()` and `evaluate_subset()`, register in objective computer
- New constraint modes: implement in `GeoConfig.constraint_mode`, handle in projector
- New evaluation metrics: add to `RawSpaceEvaluator.all_metrics()` or `EnhancedRawSpaceEvaluator`
