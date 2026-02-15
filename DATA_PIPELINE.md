# Data Pipeline: Complete Documentation

This document provides exhaustive documentation of the data sources, preprocessing pipeline, feature schema, target variable definitions, geographic grouping, split protocol, and caching mechanism. It is designed to support the "Data" and "Preprocessing" sections of a journal paper.

---

## 1. Input Data Sources

### Primary Mode: Brazil Telecom

The framework operates on **municipality-level telecom indicators** from Brazil, sourced from regulatory data (Anatel). Three input files are required:

| File | Size | Rows | Description |
|------|------|------|-------------|
| `smp_main.csv` | 90 MB | 5,570 | 1,919 columns: 1,011 feature columns + 905 missingness indicators + 3 identifiers |
| `metadata.csv` | 346 KB | 5,570 | Geographic metadata: coordinates, state codes, names |
| `city_populations.csv` | 168 KB | 5,570 | Population counts per municipality |

**Target snapshots:** Targets reference multiple temporal snapshots depending on the variable (coverage targets: March/June/September 2025; HHI: 2024; Densidade: 2025; speed/income/schools: cross-sectional, no temporal suffix).

**Data loader:** `coreset_selection/data/brazil_telecom_loader.py` (`BrazilTelecomDataLoader` class)

**File resolution features:**
- Fuzzy filename matching (handles spaces, underscores, "(1)" suffixes)
- Case-insensitive fallback
- ZIP bundle support: data_dir can point to a `.zip` file containing the CSVs
- Brazilian number format conversion: "1.234,56" is parsed as 1234.56

### Column Standardization

| Expected Column | Variants Accepted | Type |
|----------------|-------------------|------|
| `codigo_ibge` | CO_MUNICIPIO_IBGE, IBGE, cod_ibge | 7-digit integer |
| `uf` | UF, estado, state | 2-letter uppercase string |
| `municipio` | MUNICIPIO, nome, city | String |
| `longitude` | LONGITUDE, lon, long | Float |
| `latitude` | LATITUDE, lat | Float |
| `populacao` | POPULACAO, populacao_2025, pop | Positive integer |

---

## 2. Dataset Statistics

| Property | Value | Source |
|----------|-------|--------|
| Municipalities (N) | 5,570 | All Brazilian municipalities |
| States (G) | 27 | 26 states + Federal District (DF) |
| Features (D_total) | 1,863 | Total features after target exclusion |
| Substantive features (D_non_miss) | 973 | Original feature columns (excluding missingness indicators) |
| Missingness indicators (D_miss) | 890 | Binary indicators for columns with NaN/Inf values |
| Primary targets | 2 | 4G and 5G area coverage |
| Total coverage targets | 10 | Table V targets |
| Extra regression targets | 12 | Speed, income, HHI, infrastructure |
| Classification targets | 15 | 10 strict + 5 relaxed |

### All 27 Brazilian State Codes

```
AC (Acre), AL (Alagoas), AP (Amapa), AM (Amazonas),
BA (Bahia), CE (Ceara), DF (Distrito Federal), ES (Espirito Santo),
GO (Goias), MA (Maranhao), MT (Mato Grosso), MS (Mato Grosso do Sul),
MG (Minas Gerais), PA (Para), PB (Paraiba), PR (Parana),
PE (Pernambuco), PI (Piaui), RJ (Rio de Janeiro), RN (Rio Grande do Norte),
RS (Rio Grande do Sul), RO (Rondonia), RR (Roraima), SC (Santa Catarina),
SP (Sao Paulo), SE (Sergipe), TO (Tocantins)
```

---

## 3. Feature Categories -- Complete Inventory

The 1,863 features (after target exclusion) span the following categories:

### Population & Geography

| Feature | Description | Type |
|---------|-------------|------|
| `populacao_2025` | Municipal population estimate (2025) | Numeric |
| `longitude` | Geographic longitude | Numeric |
| `latitude` | Geographic latitude | Numeric |

### Telecom Infrastructure

| Feature Pattern | Description | Count |
|----------------|-------------|-------|
| `n_estacoes_smp` | Number of mobile base stations | 1 |
| `pct_fibra_backhaul` | Percentage of fiber backhaul connections | 1 |
| `rod_pct_cob_todas_4g` | Road 4G coverage percentage | 1 |
| `att09_any_present_5G` | 5G presence indicator | 1 |

### Per-Operator Coverage (Area)

| Feature Pattern | Description | Variables |
|----------------|-------------|-----------|
| `cov_pct_area_coberta__tec_{4g,5g}__op_{operator}__2025_09` | Area coverage percentage by technology and operator | Multiple per operator |

Coverage is reported per operator (e.g., Claro, Vivo, TIM, Oi) and per technology (4G, 5G). The aggregate "op_todas" (all operators) is excluded from target computation to avoid double-counting.

### Per-Operator Coverage (Households & Residents)

| Feature Pattern | Description |
|----------------|-------------|
| `cov_pct_domicilios__tec_{tech}__op_{operator}__2025_09` | Household coverage by technology and operator |
| `cov_pct_populacao__tec_{tech}__op_{operator}__2025_09` | Resident (population) coverage by technology and operator |

### Broadband Speed

| Feature | Description |
|---------|-------------|
| `velocidade_mediana_mean` | Median broadband speed (mean across measurements) |
| `velocidade_mediana_std` | Median broadband speed (standard deviation) |
| `pct_limite_mean` | Speed cap ratio (% of contracted speed achieved) |

### Socioeconomic

| Feature | Description |
|---------|-------------|
| `renda_media_mean` | Mean household income |
| `renda_media_std` | Income variability |

### Education Infrastructure

| Feature | Description |
|---------|-------------|
| `pct_escolas_internet` | Percentage of schools with internet access |
| `pct_escolas_fibra` | Percentage of schools with fiber connection |

### Service Density

| Feature | Description |
|---------|-------------|
| `Densidade_Banda Larga Fixa_2025` | Fixed broadband density (subscriptions per capita) |
| `Densidade_Telefonia Movel_2025` | Mobile telephony density (subscriptions per capita) |

### Market Concentration

| Feature | Description |
|---------|-------------|
| `HHI SMP_2024` | Herfindahl-Hirschman Index for mobile services (2024) |
| `HHI SCM_2024` | Herfindahl-Hirschman Index for fixed services (2024) |

### Urbanization

| Feature | Description |
|---------|-------------|
| `pct_urbano` | Percentage of urban population |
| `pct_agl_alta_velocidade` | Percentage with high-speed broadband access |

### Income-Speed Quadrant Features

| Feature | Description |
|---------|-------------|
| `pct_cat_low_renda_low_vel` | Low income, low speed quadrant |
| `pct_cat_low_renda_high_vel` | Low income, high speed quadrant |
| `pct_cat_high_renda_low_vel` | High income, low speed quadrant |
| `pct_cat_high_renda_high_vel` | High income, high speed quadrant |

---

## 4. Feature Schema & Typing System

### FeatureType Enumeration

The preprocessing pipeline classifies each column into one of five types:

| Type | Imputation | Log1p | Scaling | Missingness Indicator |
|------|-----------|-------|---------|----------------------|
| **NUMERIC** | Median (train) | Yes (if heavy-tailed) | Yes (StandardScaler) | Yes (binary) |
| **ORDINAL** | Rounded median (train) | No (default) | Yes (optional, default=True) | Yes |
| **CATEGORICAL** | Mode (train) | No | No (default) | No (uses -1 code) |
| **IGNORE** | -- | -- | -- | -- |
| **TARGET** | -- | -- | -- | -- |

**Implementation:** `coreset_selection/data/feature_schema.py`

### Schema Inference Priority

The feature typing system follows a strict priority order:

1. **Explicit target_columns** override (highest priority)
2. **Explicit ignore_columns** override
3. **Explicit categorical_columns** override
4. **Explicit ordinal_columns** override
5. **Built-in identifier ignore list**: {codigo_ibge, uf, municipio}
6. **Regex-based target detection** via `target_columns.py` patterns
7. **Heuristic inference** (lowest priority):
   - Object/string dtype -> CATEGORICAL (or NUMERIC if coercible to float)
   - Boolean dtype -> CATEGORICAL (if `treat_bool_as_categorical=True`)
   - Integer with <= 25 unique values -> CATEGORICAL
   - Numeric dtype -> NUMERIC

### Configurable Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `treat_low_cardinality_int_as_categorical` | True | Auto-classify low-cardinality integers |
| `low_cardinality_threshold` | 25 | Maximum unique values for categorical |
| `high_cardinality_drop_threshold` | None | Drop high-cardinality categoricals |
| `treat_bool_as_categorical` | True | Classify booleans as categorical |
| `auto_detect_targets` | True | Enable regex-based target detection |

---

## 5. Target Variables -- Complete Catalog

There are **37 total targets** organized in four categories: 2 primary coverage, 8 derived coverage, 12 extra regression, and 15 classification targets.

### How Targets Are Obtained

Targets are constructed in three distinct ways, depending on their nature:

**1. Operator-mean coverage targets (10 targets: 2 primary + 8 derived).** Computed as the **arithmetic mean across operators** at a **specific temporal snapshot**. For each municipality, the per-operator coverage columns (e.g., `cov_pct_area_coberta__tec_4g__op_claro__2025_09`, `...__op_vivo__2025_09`, `...__op_tim__2025_09`) are averaged row-wise with `skipna=True` -- excluding the aggregate `__op_todas__` column to avoid double-counting. When the maximum absolute value of the result is <= 1.5 (indicating fractional encoding), values are multiplied by 100 to convert to percentage points. The snapshot `2025_09` (September 2025) is used for all 10 coverage targets. These are **cross-operator statistics at a point in time**. If no operator-specific columns are found, the loader searches for pre-computed fallback columns (e.g., `y_cov_area_pct_meanops_4G_sep2025`); if none exist, it defaults to a zero vector.

**2. Extra regression targets (12 targets).** Taken **directly from individual columns** in the CSV -- each is a single pre-computed value per municipality, with NaN values replaced by 0.0. No cross-operator averaging or binning is applied. Their temporal reference varies by variable (see Section 5.3).

**3. Classification targets (15 targets).** **Derived** from source columns (either raw CSV columns or the extra regression targets above) via binning or thresholding. The derivation method varies per target: domain thresholds (e.g., HHI >= 0.25), median splits, percentile-based binning (tercile/quartile/quintile), extreme-tail binning (p3/p50/p97), or cross-tabulation of two variables. These are **engineered labels**, not raw snapshots.

---

### 5.1 Primary Regression Targets (2)

These are the two first-class targets, stored as `y_4G` and `y_5G` in the cache. Computed in `_compute_cov_area_target()` in `brazil_telecom_loader.py`.

#### `y_4G` (a.k.a. `cov_area_4G`)

- **What it measures:** Mean 4G area-coverage percentage across operators for each municipality.
- **Temporal snapshot:** September 2025 (`2025_09`).
- **Source columns:** All columns matching `cov_pct_area_coberta__tec_4g__op_{operator}__2025_09`, excluding any column containing `__op_todas__` (the "all operators" aggregate).
- **Computation:**
  1. Find all columns with prefix `cov_pct_area_coberta__tec_4g__op_` and suffix `__2025_09`.
  2. Remove the `__op_todas__` aggregate column if operator-specific columns exist.
  3. Coerce remaining columns to numeric (non-numeric values become NaN).
  4. Compute row-wise `mean(axis=1, skipna=True)`, filling any remaining NaN with 0.0.
  5. If `max(|values|) <= 1.5` (fractional encoding), multiply by 100 to get percentage points.
- **Fallback order:** If no operator columns found, searches for: `y_cov_area_pct_meanops_4G_sep2025`, then `y_cov_area_pct_meanops_4G`, then `y_4G`. Defaults to zero vector if none found.
- **Scale:** Percentage points [0, 100].
- **Cache key:** `y_4G`

#### `y_5G` (a.k.a. `cov_area_5G`)

- **What it measures:** Mean 5G area-coverage percentage across operators for each municipality.
- **Temporal snapshot:** September 2025 (`2025_09`).
- **Source columns:** All columns matching `cov_pct_area_coberta__tec_5g__op_{operator}__2025_09`, excluding `__op_todas__`.
- **Computation:** Identical to `y_4G` but with `tec_5g` instead of `tec_4g`.
- **Fallback order:** `y_cov_area_pct_meanops_5G_sep2025`, `y_cov_area_pct_meanops_5G`, `y_5G`. Defaults to zeros.
- **Scale:** Percentage points [0, 100].
- **Cache key:** `y_5G`

---

### 5.2 Derived Coverage Targets (8)

Computed in `_discover_coverage_targets()` in `brazil_telecom_loader.py`. All use the same `_extract_operator_mean()` helper: find columns matching a prefix/suffix pattern, exclude `__op_todas__`, coerce to numeric, take row-wise mean, convert fractions to percentage points if `max <= 1.5`. After computation, `cov_area_4G` and `cov_area_5G` are removed (they duplicate the primary targets). Stored under cache keys `y_extra_{name}`.

#### `cov_hh_4G` -- Household 4G Coverage

- **Source columns:** `cov_pct_domicilios__tec_4g__op_{operator}__2025_09`, excluding `__op_todas__`.
- **Computation:** Row-wise mean across operators, fractional-to-percentage conversion.
- **Fallback:** `y_cov_domicilios_pct_meanops_4G_202509`, `y_cov_domicilios_4G`, or `y_cov_households_4G`.
- **Snapshot:** `2025_09`.

#### `cov_res_4G` -- Resident/Population 4G Coverage

- **Source columns:** `cov_pct_populacao__tec_4g__op_{operator}__2025_09`, excluding `__op_todas__`.
- **Computation:** Row-wise mean across operators, fractional-to-percentage conversion.
- **Fallback:** `y_cov_populacao_pct_meanops_4G_202509`, `y_cov_populacao_4G`, or `y_cov_residents_4G`.
- **Snapshot:** `2025_09`.

#### `cov_area_4G_5G` -- Combined Area Coverage

- **Formula:** `(y_4G + y_5G) / 2.0`
- **Source:** Targets 1 and 2 (no additional CSV columns).

#### `cov_area_all` -- Mean Area Coverage Across All Technologies

- **Formula:** `mean([y_4G, y_5G, cov_area_3G, cov_area_2G])` -- only technologies with data present are stacked.
- **Additional source columns:** `cov_pct_area_coberta__tec_3g__op_{X}__2025_09` and `cov_pct_area_coberta__tec_2g__op_{X}__2025_09` (when present).

#### `cov_hh_4G_5G` -- Combined Household Coverage

- **Formula:** `(cov_hh_4G + cov_hh_5G) / 2.0` if 5G household data exists; otherwise equals `cov_hh_4G` alone.

#### `cov_hh_all` -- Mean Household Coverage Across All Technologies

- **Formula:** `mean([all available household coverage technologies])` stacked column-wise.

#### `cov_res_4G_5G` -- Combined Resident Coverage

- **Formula:** `(cov_res_4G + cov_res_5G) / 2.0` if 5G resident data exists; otherwise equals `cov_res_4G` alone.

#### `cov_res_all` -- Mean Resident Coverage Across All Technologies

- **Formula:** `mean([all available resident coverage technologies])` stacked column-wise.

**Total for KRR evaluation:** 10 targets (2 primary + 8 derived)

---

### 5.3 Extra Regression Targets (12)

Defined in `_EXTRA_REG_COLUMN_MAP` in `derived_targets.py` and extracted by `extract_extra_regression_targets()`. Each target is a **single raw column** read directly from `smp_main.csv` -- no cross-operator averaging or binning. NaN values are replaced with 0.0. Stored under cache keys `y_extreg_{name}`.

| Target | Source Column(s) | Temporal Reference | Description |
|--------|-----------------|-------------------|-------------|
| `velocidade_mediana_mean` | `velocidade_mediana_mean` | Cross-sectional | Mean of median download speeds |
| `velocidade_mediana_std` | `velocidade_mediana_std` | Cross-sectional | Std of median download speeds |
| `pct_limite_mean` | `pct_limite_mean` | Cross-sectional | Mean percentage of connections with data caps |
| `renda_media_mean` | `renda_media_mean` | Cross-sectional | Mean average household income |
| `renda_media_std` | `renda_media_std` | Cross-sectional | Std of average household income |
| `HHI SMP_2024` | `HHI SMP_2024` or `HHI_SMP_2024` | Annual 2024 | Herfindahl-Hirschman Index, mobile (SMP) market |
| `HHI SCM_2024` | `HHI SCM_2024` or `HHI_SCM_2024` | Annual 2024 | Herfindahl-Hirschman Index, fixed broadband (SCM) market |
| `pct_fibra_backhaul` | `pct_fibra_backhaul` | Cross-sectional | Percentage of backhaul using fiber optics |
| `pct_escolas_internet` | `pct_escolas_internet` | Cross-sectional | Percentage of schools with internet |
| `pct_escolas_fibra` | `pct_escolas_fibra` | Cross-sectional | Percentage of schools with fiber |
| `Densidade_Banda Larga Fixa_2025` | Multiple name variants (spaces/underscores) | Annual 2025 | Fixed broadband density (subscriptions per 100 inhabitants) |
| `Densidade_Telefonia Movel_2025` | Multiple name variants (accent on 'o', spaces/underscores) | Annual 2025 | Mobile telephony density (subscriptions per 100 inhabitants) |

---

### 5.4 Classification Targets (15)

Derived in `derived_targets.py`. Each target is engineered from one or two source columns via thresholding, binning, or cross-tabulation.

#### Strict Tier (10 targets, >= 5% minimum class fraction)

Built in `_build_strict_candidates()`. All NaN values are filled with 0.0 before thresholding/binning.

**1. `concentrated_mobile_market`** (Binary, 2 classes)
- **Source column:** `HHI SMP_2024` (or `HHI_SMP_2024`)
- **Derivation:** Domain threshold (Anatel regulatory definition of concentrated market)
- **Formula:** `(HHI_SMP_2024 >= 0.25).astype(int64)`
- **Classes:** 0 = competitive (HHI < 0.25), 1 = concentrated (HHI >= 0.25)

**2. `high_fiber_backhaul`** (Binary, 2 classes)
- **Source column:** `pct_fibra_backhaul`
- **Derivation:** Median split on non-zero values
- **Formula:** Compute `median_val = median(arr[arr > 0])`, then `(arr >= median_val).astype(int64)`
- **Classes:** 0 = below-median fiber, 1 = above-median fiber

**3. `high_speed_broadband`** (Binary, 2 classes)
- **Source column:** `pct_agl_alta_velocidade`
- **Derivation:** Median split on overall values
- **Formula:** `(arr > median(arr)).astype(int64)` (strict `>`, not `>=`)
- **Classes:** 0 = below-median high-speed broadband share, 1 = above-median

**4. `has_5g_coverage`** (Binary, 2 classes)
- **Source column:** `att09_any_present_5G`
- **Derivation:** Direct use of binary column
- **Formula:** `nan_to_num(arr, nan=0.0).astype(int64)` (column is already 0/1)
- **Classes:** 0 = no 5G operator present, 1 = at least one 5G operator

**5. `urbanization_level`** (3 classes)
- **Source column:** `pct_urbano`
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Formula:** Compute p33.33 and p66.67 percentiles on finite values. `<=p33 -> 0`, `p33 < x <= p67 -> 1`, `>p67 -> 2`
- **Classes:** 0 = low urbanization, 1 = medium, 2 = high

**6. `broadband_speed_tier`** (3 classes)
- **Source column:** `velocidade_mediana_mean`
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Classes:** 0 = low speed, 1 = medium, 2 = high

**7. `income_tier`** (3 classes)
- **Source column:** `renda_media_mean`
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Classes:** 0 = low income, 1 = medium, 2 = high

**8. `mobile_penetration_tier`** (4 classes)
- **Source column:** `Densidade_Telefonia Movel_2025` (tries multiple spelling variants including accent on 'o')
- **Derivation:** Quartile binning via `_quartile_bin()`
- **Formula:** Compute p25, p50, p75 on finite values. `<=p25 -> 0`, `p25 < x <= p50 -> 1`, `p50 < x <= p75 -> 2`, `>p75 -> 3`
- **Classes:** 0 = very low (Q1), 1 = low (Q2), 2 = medium (Q3), 3 = high (Q4)

**9. `infra_density_tier`** (5 classes)
- **Source column:** `n_estacoes_smp`
- **Derivation:** Quintile binning via `_quintile_bin()`
- **Formula:** Compute p20, p40, p60, p80 on finite values. Five bins assigned 0-4
- **Classes:** 0 = very low density (Q1) through 4 = very high density (Q5)

**10. `road_coverage_4g_tier`** (5 classes)
- **Source column:** `rod_pct_cob_todas_4g`
- **Derivation:** Quintile binning via `_quintile_bin()`
- **Classes:** 0 = very low road 4G coverage (Q1) through 4 = very high (Q5)

#### Relaxed Tier (5 targets, >= 2% minimum class fraction, with failsafe binning)

Built in `_build_relaxed_candidates()`. Each has a primary derivation and a failsafe alternative. The primary is used if every class contains at least 2% of samples; otherwise the failsafe is substituted automatically.

**11. `income_speed_class`** (4-class primary, 2-class failsafe)
- **Source columns:** `pct_cat_low_renda_low_vel`, `pct_cat_low_renda_high_vel`, `pct_cat_high_renda_low_vel`, `pct_cat_high_renda_high_vel`
- **Primary derivation:** Stack the 4 columns into an (N,4) matrix, assign each row to whichever quadrant has the highest share via `argmax(axis=1)`
- **Primary classes:** 0 = dominant low-income/low-speed, 1 = dominant low-income/high-speed, 2 = dominant high-income/low-speed, 3 = dominant high-income/high-speed
- **Failsafe derivation:** Binary collapse. `low_income = col0 + col1`, `high_income = col2 + col3`. `(high_income > low_income) -> 1, else 0`
- **Why failsafe may trigger:** The high-income/low-speed quadrant (class 2) is rare (~0.9%), failing the 2% threshold

**12. `urban_rural_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `pct_urbano`
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()`. Compute p3, p50, p97 percentiles. `<=p3 -> 0 (extreme rural)`, `p3 < x <= p50 -> 1`, `p50 < x < p97 -> 2`, `>=p97 -> 3 (extreme urban)`. Extreme classes each contain approximately 3% of data
- **Failsafe derivation:** Standard tercile binning via `_tercile_bin()` (3 balanced classes of ~33% each)

**13. `income_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `renda_media_mean`
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()` using p3/p50/p97
- **Failsafe derivation:** Standard tercile binning on `renda_media_mean`

**14. `speed_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `velocidade_mediana_mean`
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()` using p3/p50/p97
- **Failsafe derivation:** Standard tercile binning on `velocidade_mediana_mean`

**15. `pop_5g_digital_divide`** (4-class primary, 2-class failsafe)
- **Source columns:** `populacao_2025` (from `city_populations.csv`) and `att09_any_present_5G` (from `smp_main.csv`)
- **Primary derivation:** Cross-tabulation. `is_large = (pop > median(pop)).astype(int)`, `has_5g = att09_any_present_5G`. Label = `is_large * 2 + has_5g`
- **Primary classes:** 0 = small pop + no 5G, 1 = small pop + has 5G (~2.5%, naturally rare), 2 = large pop + no 5G, 3 = large pop + has 5G
- **Failsafe derivation:** Simple binary `has_5g` (0 vs 1)

#### Binning Functions

| Function | Classes | Boundaries |
|----------|---------|-----------|
| `_tercile_bin()` | 3 | p33.33, p66.67 |
| `_quartile_bin()` | 4 | p25, p50, p75 |
| `_quintile_bin()` | 5 | p20, p40, p60, p80 |
| `_extreme_4class_bin()` | 4 | p3, p50, p97 (extreme low, low-mid, mid-high, extreme high) |

All binning functions compute percentiles on finite values only. NaN is filled with 0.0 before binning.

#### Cache Storage

| Key pattern | Contents |
|---|---|
| `y_4G`, `y_5G` | Primary coverage targets (float64) |
| `y_extra_{name}` | Derived coverage targets (float64) |
| `y_extreg_{name}` | Extra regression targets (float64) |
| `y_cls_{name}` | Classification targets (int64) |
| `extra_target_names` | Array of derived coverage target names |
| `extra_reg_target_names` | Array of extra regression target names |
| `cls_target_names` | Array of classification target names |

**Metadata:** Each classification target produces rich JSON metadata including source columns, derivation method, class labels, class semantics, min_class_fraction check, and `is_engineered` flag. Saved as `.cls_metadata.json` alongside the cache.

---

## 6. Target Leakage Prevention

**67 regex patterns** in `coreset_selection/data/target_columns.py` prevent target information from appearing in the feature matrix:

### Pattern Categories

| Category | Example Patterns | Count |
|----------|-----------------|-------|
| Primary targets | `cov_area_4g`, `cov_area_5g`, `y_4G`, `y_5G` | ~10 |
| Multi-target coverage | `cov_hh_*`, `cov_res_*`, `cov_area_*` | ~15 |
| Speed targets | `velocidade_mediana_*` | ~5 |
| Income targets | `renda_media_*` | ~5 |
| Market concentration | `HHI_smp_*`, `HHI_scm_*` | ~5 |
| Infrastructure | `pct_fibra_backhaul`, `pct_escolas_*`, `densidade_*` | ~10 |
| Urbanization | `pct_agl_alta_velocidade`, `pct_urbano` | ~5 |
| Classification sources | `pct_cat_low_renda_*`, `n_estacoes_smp`, `rod_pct_cob_*` | ~12 |

### Leakage Prevention Functions

| Function | Purpose |
|----------|---------|
| `detect_target_columns(feature_names)` | Returns list of matched target column names |
| `remove_target_columns(X, feature_names, explicit_targets)` | Returns (X_clean, kept_names, removed_names) |
| `validate_no_leakage(feature_names)` | Raises ValueError if any target column remains |

**Additional safeguard:** Missingness indicators of target columns (`{target}_missing`) are also excluded from the feature matrix.

---

## 7. Preprocessing Pipeline -- 9 Steps in Detail

### Step 1: Load and Merge

```
Input:  smp_main.csv + metadata.csv + city_populations.csv
Output: N x (D_full + metadata_cols) DataFrame
```

- Load each CSV with column name standardization
- Inner-join on `codigo_ibge` (7-digit IBGE municipality code)
- Verify N = 5,570 rows after merge

### Step 2: Column Name Standardization

- Convert Brazilian number format: "1.234,56" -> 1234.56 (period as thousands separator, comma as decimal)
- Standardize column names to lowercase with underscores
- Map known column name variants to canonical names

### Step 3: Feature Schema Inference

```
Input:  All columns
Output: Per-column FeatureType assignment (NUMERIC, ORDINAL, CATEGORICAL, IGNORE, TARGET)
```

- Apply priority-ordered inference rules (see Section 4)
- Separate features from targets and metadata
- Assign type to each feature column

### Step 4: Target Column Removal

```
Input:  N x D_full feature matrix
Output: N x D' feature matrix (D' < D_full)
```

- Apply 67 regex patterns from `target_columns.py`
- Remove matched columns from feature matrix
- Also remove any explicitly specified target columns
- Record removed columns for audit trail
- Validate no leakage remains

### Step 5: Missingness Indicators

```
Input:  N x D' feature matrix (may contain NaN/Inf)
Output: N x (D' + n_missing) augmented matrix
```

- For each numeric column with any NaN values, create a binary indicator column
- Append indicators as new columns with prefix `missingness_indicator_of_`
- Categorical columns use -1 missing code instead of indicators

### Step 6: Type-Aware Imputation

```
Input:  N x D'' matrix with NaN values
Output: N x D'' matrix (no NaN)
```

All imputation statistics are computed on **I_train only** (training split) and applied to the full dataset:

| Feature Type | Strategy | Details |
|-------------|----------|---------|
| NUMERIC | Median | Column median on I_train |
| ORDINAL | Rounded median | Round(median) on I_train; fallback to mode |
| CATEGORICAL | Mode | Most frequent value on I_train; fallback to -1 |

**Implementation:** `_impute_typeaware()` in `data/_cache_preprocessing.py`

### Step 7: Log1p Transform

```
Input:  Imputed matrix
Output: Transformed matrix (selected columns log-transformed)
```

- Applied **only** to NUMERIC columns (not ordinal or categorical)
- Detects heavy-tailed non-negative columns via skewness heuristic on I_train
- Transform: x -> log(1 + x) for selected columns
- Selection criterion: column is non-negative and has high positive skewness

**Implementation:** `_detect_log1p_cols()` in `data/_cache_preprocessing.py`

### Step 8: Selective Standardization

```
Input:  Transformed matrix
Output: Standardized matrix (zero mean, unit variance for selected columns)
```

- StandardScaler fitted on **I_train only**
- Applied to all data (I_train, I_val, E, etc.)
- Selective scaling based on feature type:

| Feature Type | Scaled? | Configurable |
|-------------|---------|-------------|
| NUMERIC | Always | No |
| ORDINAL | Default: Yes | `scale_ordinals` parameter |
| CATEGORICAL | Default: No | `scale_categoricals` parameter |
| Missingness indicators | Always | No |

- Scale mask: boolean array indicating which columns are scaled
- Stored in cache for reproducible transform/inverse-transform

### Step 9: Representation Learning

```
Input:  Standardized N x D matrix
Output: N x d_z embedding matrix (d_z = 32 default)
```

- **VAE**: Train TabularVAE on I_train, embed all N municipalities
  - Produces: Z_vae (means, N x 32) + Z_logvar (log-variances, N x 32)
- **PCA**: Fit PCA on I_train, transform all N municipalities
  - Produces: Z_pca (N x 32)
- Training uses I_train only; I_val used for early stopping (VAE)

---

## 8. Geographic Grouping

### GeoInfo Dataclass

**Implementation:** `coreset_selection/geo/info.py`

| Attribute | Type | Description |
|-----------|------|-------------|
| `groups` | List[str] | Sorted list of state codes (27 elements) |
| `group_ids` | ndarray (N,) | Per-municipality state assignment |
| `group_to_indices` | List[ndarray] | Index arrays per state |
| `pi` | ndarray (G,) | Municipality-share proportions: pi_g = n_g / N |
| `pi_pop` | ndarray (G,) | Population-share proportions: pi_g = pop_g / pop_total |
| `population_weights` | ndarray (N,) | Per-municipality population (optional) |

### Two Weight Types

| Mode | Weight w_i | Target pi_g | Use Case |
|------|-----------|-------------|----------|
| **Municipality ("muni")** | 1 for all i | n_g / N | Equal municipality representation |
| **Population ("pop")** | population_i | pop_g / pop_total | Representation proportional to population |

### State Size Distribution

The 27 states have highly unequal sizes:
- **Largest by municipalities:** MG (~853), SP (~645), BA (~417), RS (~497)
- **Smallest by municipalities:** DF (1), RR (~15), AP (~16), AC (~22)
- **Largest by population:** SP (~46M), MG (~21M), RJ (~17M), BA (~15M)
- **Smallest by population:** RR (~650K), AP (~870K), AC (~900K)

This asymmetry makes geographic constraints crucial: without them, small states may receive zero or one representative in small coresets.

---

## 9. Split Protocol -- 3-Tier, Stratified

### Tier 1: Training/Validation Split

```
(I_train, I_val) = stratified_split(X, test_size=0.2, stratify=state_labels)
```

- **Purpose:** Preprocessing statistics, PCA fitting, VAE training
- **I_train** (~80%): Used for all fitted transformations
- **I_val** (~20%): Used for VAE early stopping validation
- **Stratification:** By state labels to ensure all states represented in both splits

### Tier 2: Evaluation Set

```
E = stratified_sample(X, n=2000, stratify=state_labels)
```

- **Purpose:** Fixed evaluation set for all downstream metrics
- **Size:** |E| = 2,000 (configurable via EVAL_SIZE)
- **Stratification:** By state to ensure geographic representativeness
- **Fixed:** Same E for all runs sharing a replicate ID

### Tier 3: Evaluation Train/Test Split

```
(E_train, E_test) = stratified_split(E, test_size=0.2, stratify=state_labels)
```

- **Purpose:** KRR hyperparameter tuning (E_train) and final metric computation (E_test)
- **E_train** (~1,600 points): Kernel bandwidth estimation, KRR lambda CV
- **E_test** (~400 points): Final RMSE, MAE, R^2 evaluation
- **No data leakage:** Bandwidth and hyperparameters fitted on E_train only

### Fallback for Small Groups

If any state has fewer than 2 samples in a split, the system falls back to **non-stratified splitting**. This can occur for DF (1 municipality) or very small states at small evaluation set sizes.

---

## 10. Caching & Persistence Mechanism

### Per-Replicate Cache

Each replicate produces a single `.npz` file containing all preprocessed data and embeddings:

```
replicate_cache/
├── rep00/
│   ├── assets.npz          # All data, embeddings, splits, metadata
│   └── .cls_metadata.json  # Classification target metadata
├── rep01/
│   ├── assets.npz
│   └── .cls_metadata.json
└── ...
```

### Cache Contents (assets.npz)

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `X_raw` | (N, D') | float32 | Imputed/transformed, unscaled features |
| `X_scaled` | (N, D') | float32 | Standardized features |
| `state_labels` | (N,) | string | State code per municipality |
| `y` | (N,) or (N,T) | float64 | Primary target(s) |
| `y_4G` | (N,) | float64 | 4G area coverage |
| `y_5G` | (N,) | float64 | 5G area coverage |
| `y_extra_{name}` | (N,) | float64 | Extra target (per name) |
| `extra_target_names` | list | string | Names of extra targets |
| `Z_vae` | (N, 32) | float32 | VAE latent means |
| `Z_logvar` | (N, 32) | float32 | VAE log-variances |
| `Z_pca` | (N, 32) | float32 | PCA embeddings |
| `train_idx` | (n_train,) | int | Training split indices |
| `val_idx` | (n_val,) | int | Validation split indices |
| `eval_idx` | (2000,) | int | Evaluation set indices |
| `eval_train_idx` | (1600,) | int | Eval training indices |
| `eval_test_idx` | (400,) | int | Eval test indices |
| `feature_names` | list | string | Final feature names |
| `feature_types` | list | string | Feature type per column |
| `missing_feature_names` | list | string | Columns with missingness indicators |
| `log1p_feature_names` | list | string | Log1p-transformed columns |
| `removed_target_columns` | list | string | Audit trail of removed targets |
| `scale_mask` | (D',) | bool | Which columns were scaled |
| `categorical_columns` | list | string | Categorical feature names |
| `ordinal_columns` | list | string | Ordinal feature names |
| `numeric_columns` | list | string | Numeric feature names |
| `category_maps_json` | string | JSON | {col: {original: encoded}} maps |
| `impute_values_json` | string | JSON | {col_idx: fill_value} |
| `cls_target_names` | list | string | Classification target names |
| `y_cls_{name}` | (N,) | int64 | Classification target arrays |
| `latitude` | (N,) | float | Coordinates |
| `longitude` | (N,) | float | Coordinates |
| `population` | (N,) | float64 | Population weights |

### Concurrency Control

- **Lock file:** `.assets_build.lock` in cache directory
- **Mechanism:** `_acquire_build_lock()` / `_release_build_lock()`
- **Purpose:** Prevents concurrent cache builds from parallel processes
- **Safe for:** Multi-process experiment execution

### Cache Augmentation

The system supports non-destructive cache augmentation:
- `ensure_replicate_cache()` checks for existing cache with required keys
- If keys are missing (e.g., PCA embeddings not yet computed), adds them without rebuilding
- Preserves all existing data during augmentation

### Cache Validation

```python
validate_cache(cache_path, required_keys) -> (valid: bool, missing: List[str])
```

Checks that all required keys exist in the `.npz` file before loading.

---

## 11. Column Name Reference Table

| Column Name | Meaning | Category |
|------------|---------|----------|
| `codigo_ibge` | 7-digit IBGE municipality code | Identifier |
| `uf` | 2-letter state code (e.g., SP, MG) | Metadata |
| `municipio` | Municipality name | Metadata |
| `populacao_2025` | 2025 population estimate | Feature + weight |
| `longitude`, `latitude` | Geographic coordinates | Feature |
| `n_estacoes_smp` | Mobile base station count | Feature |
| `pct_fibra_backhaul` | Fiber backhaul % | Feature + target |
| `velocidade_mediana_mean` | Median broadband speed | Feature + target |
| `renda_media_mean` | Mean household income | Feature + target |
| `HHI SMP_2024` | Mobile market HHI | Feature + target |
| `pct_escolas_internet` | Schools with internet % | Feature + target |
| `pct_urbano` | Urban population % | Feature |
| `cov_pct_area_coberta__tec_4g__op_*` | 4G area coverage by operator | Feature |
| `y_4G`, `cov_area_4G` | Primary 4G target | Target |
| `y_5G`, `cov_area_5G` | Primary 5G target | Target |

---

## 12. Data Integrity

### Expected Shapes After Preprocessing

| Array | Expected Shape |
|-------|---------------|
| X_scaled | (5570, 1863) |
| state_labels | (5570,) with 27 unique values |
| y_4G, y_5G | (5570,) each |
| Z_vae | (5570, 32) |
| Z_pca | (5570, 32) |
| eval_idx | (2000,) |

### Validation Checks

1. **Row count:** N = 5,570 after merge
2. **State count:** G = 27 unique state codes
3. **Feature count:** D_total = 1,863 after target exclusion (D_non_miss = 973 substantive features + D_miss = 890 missingness indicators)
4. **No NaN:** X_scaled contains no NaN or Inf values after imputation
5. **Target range:** Coverage targets in [0, 100] percentage range
6. **Population:** All positive, sum > 200 million (Brazil's total population)
