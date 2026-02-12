# Data Version Specification

## Required Input Files

This section documents the expected input data files for the coreset selection
pipeline, per manuscript Section VII.

### Primary Mode: Brazil Telecom

| File | Description | Expected Columns |
|------|-------------|------------------|
| `smp_main.csv` | Main features file containing municipality-level telecom indicators | 621 numeric covariates + identifiers |
| `metadata.csv` | Metadata with geographic coordinates (lat/lon), municipality names | CO_MUNICIPIO_IBGE, LATITUDE, LONGITUDE, MUNICIPIO |
| `city_populations.csv` | Population counts per municipality | CO_MUNICIPIO_IBGE, POPULACAO |

### Expected Dataset Properties

| Property | Value | Source |
|----------|-------|--------|
| N (municipalities) | 5,569 | Section V.A |
| G (states/UFs) | 27 | Section V.A |
| D (numeric covariates) | 621 | Section V.A |
| Snapshot date | September 2025 | Section V.A |

### Target Columns (must NOT appear in feature matrix)

Per Section VII, the following target-defining columns are excluded from the
feature matrix before any representation learning (PCA/VAE):

- `cov_area_4G` — Area coverage (4G) [primary target]
- `cov_area_5G` — Area coverage (5G) [primary target]
- `y_4G`, `y_5G` — Internal target aliases
- Any column matching `cov_hh_*`, `cov_res_*`, `cov_area_*` patterns

Detection logic is in `data/target_columns.py`.

### Preprocessing Pipeline (Section VII)

Applied in strict order in `data/cache.py::build_replicate_cache`:

1. **Missingness indicators**: For each feature column with any NaN/Inf values,
   a binary indicator `{col}__missing` is appended.
2. **Imputation**: NaN values replaced with column medians computed on **I_train
   only** (no test leakage).
3. **Log-transform**: `log(1 + x)` applied to heavy-tailed non-negative columns
   identified on **I_train only** (skewness heuristic).
4. **Standardization**: Zero-mean unit-variance using **I_train** statistics.

### Split Protocol (3-tier, Section VII)

Per seed:

- **(I_train, I_val)**: 80/20 stratified by state — used for preprocessing
  stats, PCA fitting, and VAE training.
- **E**: |E| = 2,000 stratified by state — fixed evaluation set.
- **(E_train, E_test)**: 80/20 within E, stratified by state — KRR
  hyper-parameter selection and final evaluation.

Splits are persisted to `cache/repXX/splits.npz`.

### Data Integrity

To verify data integrity, compute SHA-256 checksums of input files and compare
against known values.  The checksums for the canonical dataset snapshot used in
the manuscript are recorded in `data/checksums.txt` (if present).

### Column Naming Assumptions

- State code column: `UF` (2-letter uppercase)
- Municipality IBGE code: `CO_MUNICIPIO_IBGE` (7-digit integer)
- Population: `POPULACAO` (positive integer)
