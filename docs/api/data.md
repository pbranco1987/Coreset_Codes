# API Reference — Data Layer (`coreset_selection.data`)

The data layer is the **foundation of reproducibility** in the pipeline. It turns raw CSVs into standardised, leakage-free, deterministically-split replicate caches that every downstream phase (construction, evaluation, analysis) consumes.

Public symbols are re-exported from `coreset_selection/data/__init__.py`.

## Section Map

1. [Cache Building & Loading](#cache-building--loading)
2. [Target-Leakage Guards (Phase 4.3)](#target-leakage-guards)
3. [Feature Schema](#feature-schema)
4. [Split Persistence (Phase 4.2)](#split-persistence)
5. [Preprocessing Primitives](#preprocessing-primitives)
6. [Raw Loaders](#raw-loaders)
7. [Brazilian Telecom Loader](#brazilian-telecom-loader)
8. [DataManager](#datamanager)

---

## Cache Building & Loading

A **replicate cache** is a single `.npz` file capturing every expensive preprocessing artifact for one independent replicate: standardised features, VAE embedding, PCA embedding, stratified evaluation splits, geographic metadata, and targets. Building a cache is typically the longest step (VAE training dominates wall-clock); loading is instant. All downstream code only needs the cache.

### `coreset_selection.data.build_replicate_cache`

**Kind:** function
**Source:** `coreset_selection/data/cache.py:161`

**Summary:** Build and persist a single replicate's assets from raw data.

**Description:** Constructs `X_raw`, `X_scaled`, VAE embeddings, PCA projections, stratified evaluation splits, and target arrays for replicate `rep_id`, then saves them to a single `.npz` file under `cfg.files.cache_dir/rep{rep_id:02d}/assets.npz`. The RNG seed is `cfg.seed + rep_id`, making each replicate reproducible independently. Target columns are detected and removed *before* VAE/PCA are fit, so the learned representations cannot leak downstream targets (Phase 4.3 protection). This is the preferred entry point for batch cache generation.

**Signature:**
```python
def build_replicate_cache(
    cfg: ExperimentConfig,
    rep_id: int,
    data_manager: Optional[DataManager] = None,
) -> str
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cfg` | `ExperimentConfig` | required | Configuration dataclass; fields `cfg.seed`, `cfg.files`, `cfg.preprocessing`, `cfg.models.vae`, `cfg.models.pca` control all behaviour. |
| `rep_id` | `int` | required | Replicate identifier (0, 1, 2, …). Effective seed is `cfg.seed + rep_id`. |
| `data_manager` | `DataManager` or `None` | `None` | Optional pre-loaded manager (skips the raw CSV read). Useful when building many replicates from the same raw data. |

**Returns:**
| Type | Description |
|------|-------------|
| `str` | Path to the saved `assets.npz` file. |

**Raises:** `FileNotFoundError` if the raw data source referenced in `cfg.files` is missing; `RuntimeError` if `validate_no_leakage` rejects the feature matrix before VAE/PCA training.

**Example:**
```python
from coreset_selection.config import build_base_config
from coreset_selection.data import build_replicate_cache

cfg = build_base_config(output_dir="runs_out", cache_dir="cache",
                        data_dir="data", seed=4200)
cache_path = build_replicate_cache(cfg, rep_id=0)
print(cache_path)  # e.g., cache/rep00/assets.npz
```

**See also:** `load_replicate_cache`, `prebuild_full_cache`, `ensure_replicate_cache`.

---

### `coreset_selection.data.load_replicate_cache`

**Kind:** function
**Source:** `coreset_selection/data/cache.py:1075`

**Summary:** Load a replicate cache into a `ReplicateAssets` dataclass.

**Description:** Instant loader — reads the `.npz`, wires the fields into `ReplicateAssets`, and populates `metadata` with optional extras (y_4G, y_5G, extra coverage targets, classification targets, QoS target, removed-target-columns audit trail). This is the single entry point used by every evaluator, baseline runner, and experiment launcher.

**Signature:**
```python
def load_replicate_cache(asset_path: str) -> ReplicateAssets
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `asset_path` | `str` | required | Path to the replicate cache `.npz` file (as returned by `build_replicate_cache`). |

**Returns:**
| Type | Description |
|------|-------------|
| `ReplicateAssets` | Dataclass with `X_raw`, `X_scaled`, `Z_vae`, `Z_logvar`, `Z_pca`, `state_labels`, `train_idx`, `val_idx`, `eval_idx`, `eval_train_idx`, `eval_test_idx`, `y`, `population`, `metadata`. See `ReplicateAssets` in `config/_dc_results.py`. |

**Raises:** `FileNotFoundError` if the file does not exist.

**Example:**
```python
from coreset_selection.data import load_replicate_cache

assets = load_replicate_cache("cache/rep00/assets.npz")
print(assets.X_scaled.shape, assets.eval_idx.shape)
print(assets.metadata.get("removed_target_columns"))  # leakage audit
```

**See also:** `build_replicate_cache`, `ReplicateAssets` (data class).

---

### `coreset_selection.data.prebuild_full_cache`

**Kind:** function
**Source:** `coreset_selection/data/cache.py:979`

**Summary:** Build caches for a range of replicates in a single call.

**Description:** Convenience wrapper around `build_replicate_cache` that iterates over multiple `rep_id` values, optionally reusing one `DataManager` instance so the raw data is only read once. Typically called from batch scripts such as `scripts/launchers/build_caches.py`.

**Signature:**
```python
def prebuild_full_cache(
    cfg: ExperimentConfig,
    rep_ids: Iterable[int] = range(5),
    force_rebuild: bool = False,
) -> List[str]
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cfg` | `ExperimentConfig` | required | Configuration dataclass. |
| `rep_ids` | `Iterable[int]` | `range(5)` | Which replicates to build. |
| `force_rebuild` | `bool` | `False` | If `True`, re-run even when a cache already exists. |

**Returns:**
| Type | Description |
|------|-------------|
| `List[str]` | Paths to all produced `assets.npz` files. |

**See also:** `build_replicate_cache`, `ensure_replicate_cache`, `scripts/launchers/build_caches.py`.

---

## Target-Leakage Guards

A suite introduced in Phase 4.3 to **guarantee** that downstream evaluation targets never leak into feature matrices used for representation learning or kernel computation.

### `coreset_selection.data.detect_target_columns`

**Kind:** function
**Source:** `coreset_selection/data/target_columns.py`

**Summary:** Return the list of column names that look like downstream evaluation targets.

**Description:** Matches the provided column names against `TARGET_COLUMN_PATTERNS` (35+ regex rules covering coverage, QoS, classification-source, and market-concentration targets). Used internally by `build_replicate_cache` before VAE/PCA training.

**Signature:**
```python
def detect_target_columns(feature_names: Iterable[str]) -> List[str]
```

**Parameters:** `feature_names` — column names of the candidate feature matrix.

**Returns:** list of target names detected.

**See also:** `remove_target_columns`, `validate_no_leakage`, `TARGET_COLUMN_PATTERNS`.

---

### `coreset_selection.data.remove_target_columns`

**Kind:** function
**Source:** `coreset_selection/data/target_columns.py`

**Summary:** Drop target columns from a feature matrix and return the cleaned matrix + audit trail.

**Description:** Strips columns identified by `detect_target_columns`. Returns both the cleaned array and the list of removed columns so callers can record an audit trail in cache metadata.

**Signature:**
```python
def remove_target_columns(
    X: np.ndarray, feature_names: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]
```

**Returns:** `(X_clean, kept_names, removed_names)`.

---

### `coreset_selection.data.validate_no_leakage`

**Kind:** function
**Source:** `coreset_selection/data/target_columns.py`

**Summary:** Hard-fail if any known target column remains in the feature set.

**Description:** Intended as a pre-flight assertion immediately before fitting PCA or training the VAE. Raises `ValueError` listing the offending columns. Called twice inside `build_replicate_cache`.

**Signature:**
```python
def validate_no_leakage(feature_names: Iterable[str]) -> None
```

**Raises:** `ValueError` — if any target pattern matches.

---

### `coreset_selection.data.TARGET_COLUMN_PATTERNS`

**Kind:** constant (`List[str]`, regex patterns)
**Source:** `coreset_selection/data/target_columns.py`

**Summary:** Canonical regex list of leakage-risk columns.

**Description:** Edit this list when adding a new evaluation target to the pipeline. Patterns are case-insensitive substrings/regexes (e.g., `cov_area_4g`, `qf_mean`, `hhi_smp`). Every item here is treated as an evaluation target and will be excluded from selection features by `remove_target_columns`.

---

## Feature Schema

### `coreset_selection.data.FeatureType`

**Kind:** enum/class
**Source:** `coreset_selection/data/feature_schema.py`

**Summary:** Enumeration of semantic feature types (continuous, ordinal, categorical, binary).

### `coreset_selection.data.FeatureSchema`

**Kind:** dataclass
**Source:** `coreset_selection/data/feature_schema.py`

**Summary:** Complete schema describing column semantics, used for mixed-type preprocessing and constraint building.

### `coreset_selection.data.infer_schema`

**Kind:** function
**Source:** `coreset_selection/data/feature_schema.py`

**Summary:** Automatically infer a `FeatureSchema` by scanning column contents (heuristic cutoffs on integrality, cardinality, uniqueness).

### `coreset_selection.data.build_schema_from_config`

**Kind:** function
**Source:** `coreset_selection/data/feature_schema.py`

**Summary:** Build a schema from a configuration dict (explicit user overrides).

---

## Split Persistence

Utilities ensuring that train/val/eval splits are saved with the cache and validated on reload so analysis stays reproducible.

### `coreset_selection.data.save_splits`

**Kind:** function
**Source:** `coreset_selection/data/split_persistence.py`

**Summary:** Persist train/val/eval index arrays into an `.npz`.

### `coreset_selection.data.load_splits`

**Kind:** function
**Source:** `coreset_selection/data/split_persistence.py`

**Summary:** Load previously-saved splits; validates shapes and disjointness.

### `coreset_selection.data.validate_splits`

**Kind:** function
**Source:** `coreset_selection/data/split_persistence.py`

**Summary:** Check that train/val/eval are disjoint and cover the expected row set.

---

## Preprocessing Primitives

Low-level helpers used by the loaders. Most users do not invoke these directly; they are exposed for custom pipelines.

### `coreset_selection.data.resolve_column`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Case- and whitespace-tolerant resolution of a column name against a DataFrame's actual columns.

### `coreset_selection.data._br_to_float`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Parse Brazilian-format decimals (`"1.234,56"` → `1234.56`). Underscore prefix indicates a historically private helper that is intentionally re-exported for data-loading scripts.

### `coreset_selection.data._to_int_id`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Coerce potentially-float municipality/state IDs to clean integers.

### `coreset_selection.data.parse_period_mm_yyyy`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Parse a Brazilian period string `"MM/YYYY"` into a `(month, year)` tuple.

### `coreset_selection.data.period_mm_tag`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Produce a canonical tag (e.g., `"2024_09"`) from a parsed period.

### `coreset_selection.data.period_suffix_monYYYY`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Produce a human-readable suffix (e.g., `"Sep2024"`) from a parsed period.

### `coreset_selection.data.detect_numeric_columns`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Identify columns that contain numeric data despite object dtype (typically the result of Brazilian-format decimals).

### `coreset_selection.data.standardize_state_code`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Normalise Brazilian state codes to canonical 2-letter form (e.g., `"Sao Paulo"` → `"SP"`).

### `coreset_selection.data.impute_missing_values`

**Kind:** function
**Source:** `coreset_selection/data/preprocessing.py`

**Summary:** Median/mode/forward-fill imputation dispatch based on feature type.

---

## Raw Loaders

Per-source CSV readers. Most users call `BrazilTelecomDataLoader` or `DataManager` instead.

### `coreset_selection.data.load_population_muni_csv`

**Source:** `coreset_selection/data/loaders.py` — loads municipality population table.

### `coreset_selection.data.load_cobertura_features_and_targets`

**Source:** `coreset_selection/data/loaders.py` — loads ANATEL coverage features and their target columns (4G/5G).

### `coreset_selection.data.load_atendidos_features`

**Source:** `coreset_selection/data/loaders.py` — loads served-customers features.

### `coreset_selection.data.load_setores_features`

**Source:** `coreset_selection/data/loaders.py` — loads census-tract features.

### `coreset_selection.data.load_synthetic_data`

**Source:** `coreset_selection/data/loaders.py` — produces a small synthetic dataset for tests and tutorials.

---

## Brazilian Telecom Loader

### `coreset_selection.data.BrazilTelecomDataLoader`

**Kind:** class
**Source:** `coreset_selection/data/brazil_telecom_loader.py`

**Summary:** Orchestrator combining the four per-source loaders into a unified feature matrix with aligned targets.

### `coreset_selection.data.BrazilTelecomData`

**Kind:** dataclass
**Source:** `coreset_selection/data/brazil_telecom_loader.py`

**Summary:** Return container for the loader (features, targets, metadata, geographic labels).

### `coreset_selection.data.load_brazil_telecom_data`

**Kind:** function
**Source:** `coreset_selection/data/brazil_telecom_loader.py`

**Summary:** Functional wrapper around the class for one-shot loading.

### `coreset_selection.data.BRAZILIAN_STATES`

**Kind:** constant (`List[str]`) — canonical ordered list of the 27 state codes.

### `coreset_selection.data.STATE_TO_IDX`

**Kind:** constant (`Dict[str, int]`) — map from state code to 0-indexed group ID.

---

## DataManager

### `coreset_selection.data.DataManager`

**Kind:** class
**Source:** `coreset_selection/data/manager.py`

**Summary:** Unified preprocessing orchestrator used by `build_replicate_cache`.

**Description:** Holds the `ExperimentConfig.files` object, owns lazy loading of raw CSVs, applies the preprocessing pipeline (impute → log1p → standardise, with optional target removal), and caches intermediate products so multiple replicates share the raw-data read. Instantiate once, call `load()` once, then reuse across `build_replicate_cache(cfg, rep_id, data_manager=...)` calls.

**Key methods:** `load()`, `get_features()`, `get_targets()`, `get_state_labels()`, `get_feature_names()`.

---

## See Also

- [geo](./geo.md) — builds `GeoInfo` from the `state_labels` produced here.
- [evaluation](./evaluation.md) — consumes `eval_idx`, `eval_train_idx`, `eval_test_idx`.
- [scripts](./scripts.md#build_caches) — CLI wrapper.
