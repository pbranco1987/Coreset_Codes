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

**Pre-processing provenance:** `smp_main.csv` is itself the **output** of a multi-stage pre-processing pipeline (`pre_processing_nsga_coreset_input`) that merges **18 heterogeneous data sources** (Stages A–R) into a single municipality-level flat file. Each column in `smp_main.csv` originates from a specific Anatel/regulatory data source with its own temporal reference. See **Section 1.5** for full per-column temporal provenance.

**Target snapshots:** Coverage targets use September 2025 (`2025_09`). HHI targets use 2024 (from IBC, Stage E). Densidade targets use the 2025 snapshot (from Acessos, Stage F). Speed and income targets are static cross-sectional aggregations from sector-level data (Stage G). School connectivity targets reference the September 2025 snapshot (Stage O). Backhaul fiber data uses the 2025 latest-year detail (Stage P). RNI measurement data aggregates over the 2000–2025 measurement period (Stage L). Satisfaction survey indicators (ISG, QIC, QF) come from the 2024 survey wave (Stage R).

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

## 1.5. Pre-Processing Pipeline: Data Sources & Temporal Provenance

`smp_main.csv` is produced by a **multi-stage pre-processing pipeline** (`pre_processing_nsga_coreset_input`) that ingests 18 heterogeneous data sources from Anatel and related regulatory databases. Each pipeline stage loads a specific source file, aggregates or flattens the data to the municipality level, and merges it into a combined DataFrame keyed on `codigo_ibge`. **Every feature column in the final dataset traces back to one of these stages and has a specific temporal reference.**

### Quick-Reference Summary

| Stage | Source File | Time Period | Temporal Type | Column Suffix |
|-------|------------|-------------|---------------|---------------|
| **A** RQUAL | `Tabela_CSV_Indicadores_RQUAL.csv` | Mar 2022 – Jan 2025 (35 months) | Monthly time-series | _(raw rows)_ |
| **B** Flatten RQUAL | Stage A output | Mar 2022 – Jan 2025 | Flattened monthly | `{feat}_{YYYY}_{MM}` |
| **C** Merge SMP | `processed_smp_data_enriched_sep2025_plus_timeseries.csv` | Sep 2025 baseline | Mixed | Inherited |
| **D** NaN Filter | Stage C output | — | Filter only (≥10% NaN dropped) | — |
| **E** IBC | `IBC_municipios_indicadores_originais.csv` | 2021–2024 (4 yearly) | Yearly | `{feat}_{YYYY}` |
| **F** Acessos | `Meu_Municipio_Acessos.csv` | Dec 2019–2024, Nov 2025 (7 yearly) | Yearly | `{feat}_{YYYY}` |
| **G** Setores | `categorias_setores.csv` | Static | Cross-sectional | None |
| **H** Rodovias | `cobertura_rodovias.zip` | Dec 2025 | Snapshot | None |
| **I** Estações Licenciadas | `estacoes_licenciadas.zip` (3 CSVs) | Static | Registry snapshot | None |
| **J** Terrenas & VSATs | `dados_coletas_vsats.csv` + `estacoes_terrenas_individuais.csv` | VSATs: Jun 2023–Dec 2025; Terrenas: static | Aggregated / static | None |
| **K** GAISPI & Lei Antenas | `gaispi.zip` + `leidasantenas.zip` | GAISPI: 2016–2025; Lei: static | Policy snapshot | None |
| **L** Mapa RNI | `mapa_rni.zip` → `medicoes_rni.csv` | 2000–2025 (25 years) | Aggregated historical | None |
| **M** Estações SMP | `ESTACOES_SMP.csv` | Static | Registry snapshot | None |
| **N** Renda & Velocidade | `renda_velocidade_smp.zip` → `categorias_aglomerados.csv` | Static | Cross-sectional | None |
| **O** Escolas | `conectividade_escolas.zip` → `Conectividade_Escolas_2025-09.csv` | Sep 2025 | Single snapshot | None |
| **P** Backhaul | `mapeamento_rede_transporte.zip` (2 CSVs) | Evolution: 2016–2025; Detail: 2025 | Historical + latest-year | None |
| **Q** Prestadoras | `prestadoras_servicos_telecomunicacoes.csv` | Static | Registry snapshot | None |
| **R** Pesquisa Satisfação | `pesquisa_de_satisfacao.zip` → `pesquisa_dados_brutos.csv` | 2024 survey wave | Survey snapshot | None |

**Common merge key:** `codigo_ibge` (7-digit IBGE municipality code) across all stages.
**NaN threshold:** Columns with >10% missing values are dropped at each stage.

---

### Stage A: RQUAL Indicators (Mobile Telephony Quality)

**Source:** `Tabela_CSV_Indicadores_RQUAL.csv`
**Rows:** Variable (multi-row per municipality × month × indicator × operator) | **Municipalities:** 5,570
**Time Period:** March 2022 – January 2025 (35 monthly snapshots)
**Temporal Type:** Monthly time-series
**Service Filter:** "Telefonia Móvel" (mobile telephony only)

**Processing:**
1. Filter to mobile telephony service
2. Values converted from percentage (0–100) to decimal (0–1)
3. Time series harmonized to a complete monthly grid (2022-03 through 2025-01)
4. Columns with values >1 normalized via MinMaxScaler

**Features produced:** Multiple RQUAL quality indicators per operator (Prestadora), in long format (one row per municipality × date). Column naming: `{Indicador}_Resultado_{Operador}`.

---

### Stage B: Flatten RQUAL to Municipality Level

**Source:** Stage A output (no new external file)
**Time Period:** March 2022 – January 2025 (inherited)
**Temporal Type:** Flattened monthly — each indicator becomes 35 columns

**Column naming convention:** `{Indicador}_Resultado_{Operador}_{YYYY}_{MM}` (zero-padded month)

**Example columns:**
- `Acessibilidade_Resultado_CLARO_2022_03` (March 2022)
- `Acessibilidade_Resultado_CLARO_2024_12` (December 2024)
- `Taxa_de_Queda_Resultado_TIM_2025_01` (January 2025)

**Output:** One row per municipality; (# indicators × # operators × 35 months) feature-time columns.

---

### Stage C: Merge SMP Enriched Data

**Source:** `processed_smp_data_enriched_sep2025_plus_timeseries.csv`
**Time Period:** September 2025 (enriched baseline with embedded time-series features)
**Temporal Type:** Mixed — the SMP file contains both point-in-time columns (e.g., `2025_09` coverage snapshots) and time-series features inherited from prior processing.

**Merge:** OUTER join on `codigo_ibge` — combines all SMP features with flattened RQUAL features.

**Key SMP features and their temporal references:**
- `cov_pct_area_coberta__tec_{4g,5g}__op_{operator}__2025_09` — **Sep 2025**
- `cov_pct_domicilios__tec_{tech}__op_{operator}__2025_09` — **Sep 2025**
- `cov_pct_populacao__tec_{tech}__op_{operator}__2025_09` — **Sep 2025**
- `att09_any_present_5G` — **Sep 2025**

---

### Stage D: NaN Column Filter

**Source:** Stage C output
**Processing:** Drops any column where >10% of values are NaN. The `codigo_ibge` key is always retained. No new features are created.

---

### Stage E: IBC (Índice Brasil de Conectividade)

**Source:** `IBC_municipios_indicadores_originais.csv`
**Rows:** 5,570 municipalities × 4 years = ~22,280 | **Municipalities:** 5,570
**Time Period:** 2021, 2022, 2023, 2024 (4 yearly snapshots)
**Temporal Type:** Yearly — flattened to one row per municipality with year-stamped columns

**Column naming convention:** `{indicator}_{YYYY}` (integer year)

**Features produced:**

| Indicator | Years | Example Columns | Temporal Reference |
|-----------|-------|-----------------|-------------------|
| IBC (connectivity index) | 2021–2024 | `IBC_2021`, `IBC_2022`, `IBC_2023`, `IBC_2024` | Annual, each year |
| Cobertura Pop. 4G5G | 2021–2024 | `Cobertura Pop. 4G5G_2021`, … | Annual |
| Densidade SMP | 2021–2024 | `Densidade SMP_2021`, … | Annual |
| HHI SMP | 2021–2024 | `HHI SMP_2021`, …, `HHI SMP_2024` | Annual |
| Densidade SCM | 2021–2024 | `Densidade SCM_2021`, … | Annual |
| HHI SCM | 2021–2024 | `HHI SCM_2021`, …, `HHI SCM_2024` | Annual |
| Adensamento Estações | 2021–2024 | `Adensamento Estações_2021`, … | Annual |
| Cobertura área agricultável | 2021–2024 | `Cobertura área agricultável_2021`, … | Annual |
| ibc_indicador_fibra_* (one-hot) | 2021–2024 | `ibc_indicador_fibra_{category}_{YYYY}` | Annual |

**Total columns:** ~(8 indicators + N fibra dummies) × 4 years

**Processing:** Numeric columns cleaned (comma→dot), agricultural coverage imputed by municipality mean, Fibra one-hot encoded, MinMaxScaler applied, flattened by year.
**Merge:** OUTER join on `codigo_ibge`.

---

### Stage F: Acessos (Service Access & Density)

**Source:** `Meu_Municipio_Acessos.csv`
**Rows:** 5,570 municipalities × 4 services × 7 yearly snapshots | **Municipalities:** 5,570
**Time Period:** December 2019, December 2020, December 2021, December 2022, December 2023, December 2024, November 2025 (7 yearly snapshots)
**Temporal Type:** Yearly — flattened to one row per municipality with year-stamped columns

**Column naming convention:** `{metric}_{service}_{YYYY}` (integer year)

**Services (Serviço):** 4 distinct service types (e.g., Banda Larga Fixa, Telefonia Fixa, Telefonia Móvel, TV por Assinatura)

**Features produced (per service × per year):**

| Metric | Service Types | Years | Example | Temporal Reference |
|--------|--------------|-------|---------|-------------------|
| Acessos (access count) | 4 services | 2019–2025 | `Acessos_Banda Larga Fixa_2019` | Dec of each year (Nov for 2025) |
| Densidade (density/rate) | 4 services | 2019–2025 | `Densidade_Telefonia Movel_2025` | Dec of each year (Nov for 2025) |

**Total columns:** 4 services × 2 metrics × 7 years = 56 feature-time columns

**Processing:** Pivoted by Serviço, Densidade comma→dot conversion, MinMaxScaler applied, flattened by year.
**Merge:** OUTER join on `codigo_ibge`.

---

### Stage G: Setores (Census Sector Aggregations)

**Source:** `categorias_setores.csv`
**Rows:** ~170,000 sector-level rows (~31 sectors per municipality) | **Municipalities:** 5,570
**Time Period:** Static (cross-sectional snapshot, no time dimension)
**Temporal Type:** No time dimension — sector-level data aggregated directly to municipality level

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `velocidade_mediana_mean` | Mean of median download speeds across sectors | Static |
| `velocidade_mediana_median` | Median of median download speeds | Static |
| `velocidade_mediana_std` | Std dev of median download speeds | Static |
| `renda_media_mean` | Mean of average household income across sectors | Static |
| `renda_media_median` | Median of average household income | Static |
| `renda_media_std` | Std dev of average household income | Static |
| `n_setores` | Number of census sectors | Static |
| `pct_urbano` | Proportion of urban sectors | Static |
| `pct_rural` | Proportion of rural sectors | Static |
| `pct_cat_low_renda_low_vel` | Low income, low speed quadrant share | Static |
| `pct_cat_low_renda_high_vel` | Low income, high speed quadrant share | Static |
| `pct_cat_high_renda_high_vel` | High income, high speed quadrant share | Static |
| `pct_cat_high_renda_low_vel` | High income, low speed quadrant share | Static |

**Processing:** Numeric columns cleaned, aggregated to municipality level via groupby, proportion columns kept as [0,1], others scaled.
**Merge:** LEFT join on `codigo_ibge`.

---

### Stage H: Rodovias (Federal Highway Coverage)

**Source:** `cobertura_rodovias.zip` containing:
- `Cobertura_Rodovias_Federais_2025_12.csv` — **December 2025** highway coverage data (~460K road segments)
- `SNV202511A.kml` — **November 2025** highway segment coordinates

**Municipalities:** Matched via spatial proximity (Haversine BallTree) to IBGE municipality centroids
**Time Period:** December 2025
**Temporal Type:** Static snapshot

**Operators:** Todas, CLARO, TIM, VIVO
**Technologies:** Todas, 4G, 5G, 4G5G

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `rod_pct_cob_{operator}_{technology}` | % of federal highway km covered by operator×technology | Dec 2025 |

**Examples:** `rod_pct_cob_claro_4g`, `rod_pct_cob_vivo_5g`, `rod_pct_cob_todas_4g5g`, etc.

**Processing:** Spatial matching of road segments to municipalities, aggregated by (municipality, operator, technology), pivoted to wide format.
**Merge:** LEFT join on `codigo_ibge`.

---

### Stage I: Estações Licenciadas (Licensed Telecom Stations)

**Source:** `estacoes_licenciadas.zip` containing 3 files:

| File | Rows | Service | Format |
|------|------|---------|--------|
| `Estacoes_Banda_Larga_Fixa.csv` | 45K | Broadband (BLF) | Mosaico (direct IBGE code) |
| `Estacoes_Telefonia_Fixa.csv` | 301K | Fixed telephony (TF) | Mosaico (direct IBGE code) |
| `Estacoes_Geral.csv` | 1.3M | General/mixed (Geral) | Legacy (UF + name matching) |

**Municipalities:** 5,570 | **Time Period:** Static (registry snapshot, no date specified)
**Temporal Type:** No time dimension

**Features produced (per service suffix: `_blf`, `_tf`, `_geral`):**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_estacoes_{suf}` | Total station count | Static (registry) |
| `n_estacoes_ativas_{suf}` | Active station count | Static (registry) |
| `n_entidades_{suf}` | Distinct operators | Static (registry) |
| `potencia_mean_{suf}` | Mean transmitter power (W) | Static (registry) |
| `ganho_antena_mean_{suf}` | Mean antenna gain (dB) | Static (registry) |
| `altura_antena_mean_{suf}` | Mean antenna height (m) | Static (registry) |
| `pct_ativas_{suf}` | Proportion of active stations | Static (registry) |

**Total columns:** 7 features × 3 service types = 21 columns.
**Merge:** LEFT join on `codigo_ibge`.

---

### Stage J: Terrenas & VSATs (Satellite Ground Stations)

**Sources:**

| File | Rows | Municipalities | Time Period |
|------|------|----------------|-------------|
| `dados_coletas_vsats.csv` | 584K | 5,542 | Jun 2023 – Dec 2025 (time-series) |
| `estacoes_terrenas_bloco.csv` | 72 | — | Static (used for CNPJ flags only) |
| `estacoes_terrenas_individuais.csv` | 421K | 4,295 | Static (registry snapshot) |

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_vsats` | Total VSAT station records | Aggregated Jun 2023 – Dec 2025 |
| `n_entidades_vsat` | Distinct VSAT operators | Aggregated Jun 2023 – Dec 2025 |
| `n_satellites_vsat` | Distinct satellites used | Aggregated Jun 2023 – Dec 2025 |
| `pct_starlink` | Proportion of Starlink stations | Aggregated Jun 2023 – Dec 2025 |
| `n_coleta_months_vsat` | Distinct collection months | Aggregated Jun 2023 – Dec 2025 |
| `n_terrenas_indiv` | Total individual ground stations | Static (registry) |
| `n_entidades_terrena` | Distinct ground station entities | Static (registry) |
| `pct_licenciadas_terrena` | Proportion of licensed stations | Static (registry) |
| `pct_vsat_terrena` | Proportion VSAT-type stations | Static (registry) |
| `n_services_terrena` | Distinct service types | Static (registry) |

**Merge:** OUTER merge of VSAT + individuais aggregations, then LEFT join on `codigo_ibge`.

---

### Stage K: GAISPI & Lei das Antenas

**Sources:**

| File | Rows | Municipalities | Time Period |
|------|------|----------------|-------------|
| `gaispi_migracaoTVRO_liberacao.csv` (inside `gaispi.zip`) | 5,570 | 5,570 | Historical 2016–2025 (current state = 2025) |
| `LeidasAntenas.csv` (inside `leidasantenas.zip`) | 5,570 | 5,570 | Static (legislation status) |

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `gaispi_fase` | TVRO migration phase (ordinal 1–6, scaled to [0,1]) | 2025 current state |
| `gaispi_liberado` | 3.5 GHz band liberated (binary) | 2025 status |
| `gaispi_has_migration_date` | Migration start date exists (binary) | Historical (2016–2025) |
| `lei_antenas_aprovada` | Municipal antenna law approved (binary) | Static (legislation) |

**Merge:** OUTER merge of GAISPI + Lei, then LEFT join on `codigo_ibge`.

---

### Stage L: Mapa RNI (Non-Ionizing Radiation Measurements)

**Source:** `medicoes_rni.csv` (inside `mapa_rni.zip`)
**Rows:** 3.2M measurements | **Municipalities:** 5,465
**Time Period:** 2000–2025 (25 years of cumulative measurements)
**Temporal Type:** Aggregated across entire measurement history

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_medicoes_rni` | Total RNI measurements | Aggregated 2000–2025 |
| `n_medicoes_faixa_larga` | Broadband (Faixa Larga) measurement count | Aggregated 2000–2025 |
| `n_medicoes_fixas` | Fixed (Fixas) measurement count | Aggregated 2000–2025 |
| `pct_limite_mean` | Mean % of RNI regulatory limit | Aggregated 2000–2025 |
| `pct_limite_max` | Max % of RNI limit (worst case) | Aggregated 2000–2025 |
| `valor_medio_mean` | Mean measured radiation value | Aggregated 2000–2025 |
| `n_anos_rni` | Distinct years with measurements (temporal span) | Aggregated 2000–2025 |
| `pct_acima_50_limite` | Proportion of measurements >50% of limit | Aggregated 2000–2025 |

**Merge:** LEFT join on `codigo_ibge`.

---

### Stage M: Estações SMP (Mobile Base Stations)

**Source:** `ESTACOES_SMP.csv`
**Rows:** 1.7M station records | **Municipalities:** 5,569
**Time Period:** Static (registry snapshot, no date specified)
**Temporal Type:** No time dimension

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_estacoes_smp` | Total SMP station count | Static (registry) |
| `n_entidades_smp` | Distinct operators | Static (registry) |
| `n_estacoes_4g` | Count of 4G (LTE) stations | Static (registry) |
| `n_estacoes_5g` | Count of 5G (NR) stations | Static (registry) |
| `n_estacoes_2g` | Count of 2G (GSM/EDGE/CDMA) stations | Static (registry) |
| `n_estacoes_3g` | Count of 3G (WCDMA) stations | Static (registry) |
| `pct_4g_smp` | Proportion of 4G stations | Static (registry) |
| `pct_5g_smp` | Proportion of 5G stations | Static (registry) |
| `freq_mean_smp` | Mean frequency (MHz) | Static (registry) |
| `n_tecnologias_smp` | Distinct technology generations | Static (registry) |

**Merge:** LEFT join on `codigo_ibge`.

---

### Stage N: Renda & Velocidade (Urban Agglomerate Speed)

**Source:** `categorias_aglomerados.csv` (inside `renda_velocidade_smp.zip`)
**Rows:** 13,151 (urban agglomerates only) | **Municipalities:** 734
**Time Period:** Static (cross-sectional snapshot)
**Temporal Type:** No time dimension

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_aglomerados` | Number of urban agglomerate areas | Static |
| `velocidade_mediana_agl_mean` | Mean of median download speed across agglomerates | Static |
| `pct_agl_alta_velocidade` | Proportion of agglomerates above speed threshold | Static |

**Coverage:** Only 734 municipalities (urban agglomerates); remaining municipalities have NaN.
**Merge:** LEFT join on `codigo_ibge`.

---

### Stage O: Conectividade Escolas (School Connectivity)

**Source:** `Conectividade_Escolas_2025-09.csv` (inside `conectividade_escolas.zip`)
**Rows:** 137,847 schools | **Municipalities:** 5,570
**Time Period:** September 2025 (latest available snapshot)
**Temporal Type:** Single snapshot

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_escolas` | Total number of schools | Sep 2025 |
| `pct_escolas_internet` | Proportion of schools with internet | Sep 2025 |
| `pct_escolas_conect_adequada` | Proportion with adequate connectivity | Sep 2025 |
| `pct_escolas_fibra` | Mean fiber access indicator | Sep 2025 |
| `pct_4g_rural_ativada` | Proportion with activated rural 4G | Sep 2025 |
| `vel_contratada_enec_mean` | Mean contracted speed (ENEC program) | Sep 2025 |

**Merge:** LEFT join on `codigo_ibge`.

---

### Stage P: Backhaul / Rede de Transporte

**Sources (inside `mapeamento_rede_transporte.zip`):**

| File | Rows | Municipalities | Time Period |
|------|------|----------------|-------------|
| `backhaul_municipios_evolucao.csv` | 5,570 | 5,570 | 2016–2025 (annual evolution, year columns) |
| `backhaul_municipios_2023-2025.csv` | 79,555 | 5,526 | 2023–2025 (provider-level, latest year used) |

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `backhaul_fibra_2025` | Binary: 1 = fiber optics in 2025 | 2025 (from evolution) |
| `backhaul_ano_fibra` | First year municipality got fiber (0 if never) | Historical 2016–2025 |
| `n_prestadoras_backhaul` | Distinct backhaul providers | 2025 (latest year from detail) |
| `n_meios_transporte` | Distinct transport media types | 2025 (latest year from detail) |
| `pct_fibra_backhaul` | Proportion of provider entries using fiber | 2025 (latest year from detail) |

**Merge:** LEFT join on `codigo_ibge`.

---

### Stage Q: Prestadoras Telecom (Service Provider Registry)

**Source:** `prestadoras_servicos_telecomunicacoes.csv`
**Rows:** 266,542 registrations | **Municipalities:** 5,173
**Time Period:** Static (registry snapshot, no date specified)
**Temporal Type:** No time dimension

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| `n_prestadoras` | Total telecom service registrations | Static (registry) |
| `n_entidades_prestadoras` | Distinct entities (by CNPJ) | Static (registry) |
| `n_servicos_prestados` | Distinct service types offered | Static (registry) |
| `pct_sir` | Proportion Serviço de Interesse Restrito | Static (registry) |
| `pct_sic` | Proportion Serviço de Interesse Coletivo | Static (registry) |
| `pct_dispensada` | Proportion dispensed from license | Static (registry) |

**Merge:** LEFT join on `codigo_ibge`.

---

### Stage R: Pesquisa de Satisfação (Consumer Satisfaction Survey)

**Source:** `pesquisa_dados_brutos.csv` (inside `pesquisa_de_satisfacao.zip`)
**Rows:** 306,404 survey responses | **Municipalities:** 4,949
**Time Period:** 2024 survey wave
**Temporal Type:** Single survey wave
**Services covered:** SCM (broadband), POS/PRE (mobile), STFC (fixed phone), SEAC (pay TV)

**Features produced:**

| Column | Description | Temporal Reference |
|--------|-------------|--------------------|
| **Overall (all services):** | | |
| `isg_mean` | Weighted mean General Satisfaction Index (0–10→[0,1]) | 2024 survey |
| `qic_mean` | Weighted mean Internet/Connection Quality | 2024 survey |
| `qf_mean` | Weighted mean Functioning Quality | 2024 survey |
| `n_respostas_pesquisa` | Total survey responses | 2024 survey |
| **SCM (broadband only):** | | |
| `isg_scm_mean` | Weighted ISG for broadband | 2024 survey |
| `qic_scm_mean` | Weighted QIC for broadband | 2024 survey |
| `qf_scm_mean` | Weighted QF for broadband | 2024 survey |
| **Mobile (POS+PRE):** | | |
| `isg_movel_mean` | Weighted ISG for mobile | 2024 survey |
| `qic_movel_mean` | Weighted QIC for mobile | 2024 survey |
| `qf_movel_mean` | Weighted QF for mobile | 2024 survey |

**Note:** `qf_mean` (Qualidade do Funcionamento) is used as the **QoS evaluation target** in the coreset evaluation pipeline (see EVALUATION_METRICS.md Section 6).

**Merge:** LEFT join on `codigo_ibge`.

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

The 1,863 features (after target exclusion) span the following categories. Each table includes the **temporal reference** (source pipeline stage and time period) for every column.

### Population & Geography

| Feature | Description | Type | Source & Period |
|---------|-------------|------|-----------------|
| `populacao_2025` | Municipal population estimate (2025) | Numeric | **2025 estimate** (city_populations.csv) |
| `longitude` | Geographic longitude | Numeric | **Static** (metadata.csv) |
| `latitude` | Geographic latitude | Numeric | **Static** (metadata.csv) |

### Telecom Infrastructure

| Feature Pattern | Description | Count | Source & Period |
|----------------|-------------|-------|-----------------|
| `n_estacoes_smp` | Number of mobile base stations | 1 | **Static registry** (Stage M) |
| `pct_fibra_backhaul` | Percentage of fiber backhaul connections | 1 | **2025** (Stage P, latest year) |
| `rod_pct_cob_todas_4g` | Road 4G coverage percentage | 1 | **Dec 2025** (Stage H) |
| `att09_any_present_5G` | 5G presence indicator | 1 | **Sep 2025** (Stage C, SMP enriched) |

### Per-Operator Coverage (Area)

| Feature Pattern | Description | Variables | Source & Period |
|----------------|-------------|-----------|-----------------|
| `cov_pct_area_coberta__tec_{4g,5g}__op_{operator}__2025_09` | Area coverage percentage by technology and operator | Multiple per operator | **Sep 2025** (Stage C, SMP enriched) |

Coverage is reported per operator (e.g., Claro, Vivo, TIM, Oi) and per technology (4G, 5G). The aggregate "op_todas" (all operators) is excluded from target computation to avoid double-counting.

### Per-Operator Coverage (Households & Residents)

| Feature Pattern | Description | Source & Period |
|----------------|-------------|-----------------|
| `cov_pct_domicilios__tec_{tech}__op_{operator}__2025_09` | Household coverage by technology and operator | **Sep 2025** (Stage C, SMP enriched) |
| `cov_pct_populacao__tec_{tech}__op_{operator}__2025_09` | Resident (population) coverage by technology and operator | **Sep 2025** (Stage C, SMP enriched) |

### Broadband Speed

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `velocidade_mediana_mean` | Median broadband speed (mean across sectors) | **Static** (Stage G, sector-level cross-sectional) |
| `velocidade_mediana_std` | Median broadband speed (standard deviation) | **Static** (Stage G, sector-level cross-sectional) |
| `pct_limite_mean` | Mean % of RNI regulatory limit | **Aggregated 2000–2025** (Stage L, RNI measurements) |

### Socioeconomic

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `renda_media_mean` | Mean household income | **Static** (Stage G, sector-level cross-sectional) |
| `renda_media_std` | Income variability | **Static** (Stage G, sector-level cross-sectional) |

### Education Infrastructure

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `pct_escolas_internet` | Percentage of schools with internet access | **Sep 2025** (Stage O) |
| `pct_escolas_fibra` | Percentage of schools with fiber connection | **Sep 2025** (Stage O) |

### Service Density

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `Densidade_Banda Larga Fixa_2025` | Fixed broadband density (subscriptions per capita) | **2025 annual** (Stage F, latest Acessos snapshot) |
| `Densidade_Telefonia Movel_2025` | Mobile telephony density (subscriptions per capita) | **2025 annual** (Stage F, latest Acessos snapshot) |

### Market Concentration

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `HHI SMP_2024` | Herfindahl-Hirschman Index for mobile services (2024) | **Annual 2024** (Stage E, IBC) |
| `HHI SCM_2024` | Herfindahl-Hirschman Index for fixed services (2024) | **Annual 2024** (Stage E, IBC) |

### Urbanization

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `pct_urbano` | Percentage of urban population | **Static** (Stage G, sector-level cross-sectional) |
| `pct_agl_alta_velocidade` | Percentage with high-speed broadband access | **Static** (Stage N, urban agglomerate cross-sectional) |

### Income-Speed Quadrant Features

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `pct_cat_low_renda_low_vel` | Low income, low speed quadrant | **Static** (Stage G, sector-level cross-sectional) |
| `pct_cat_low_renda_high_vel` | Low income, high speed quadrant | **Static** (Stage G, sector-level cross-sectional) |
| `pct_cat_high_renda_low_vel` | High income, low speed quadrant | **Static** (Stage G, sector-level cross-sectional) |
| `pct_cat_high_renda_high_vel` | High income, high speed quadrant | **Static** (Stage G, sector-level cross-sectional) |

### RQUAL Quality Indicators (Time-Series)

| Feature Pattern | Description | Source & Period |
|----------------|-------------|-----------------|
| `{Indicador}_Resultado_{Operador}_{YYYY}_{MM}` | Mobile telephony quality indicators per operator per month | **Mar 2022 – Jan 2025, monthly** (Stages A–B) |

35 monthly columns per indicator×operator combination. Examples: `Acessibilidade_Resultado_CLARO_2022_03`, `Taxa_de_Queda_Resultado_TIM_2025_01`.

### IBC Indicators (Time-Series)

| Feature Pattern | Description | Source & Period |
|----------------|-------------|-----------------|
| `{indicator}_{YYYY}` | IBC connectivity indicators per year | **2021–2024, yearly** (Stage E) |

4 yearly columns per indicator. Indicators include: IBC, Cobertura Pop. 4G5G, Densidade SMP, HHI SMP, Densidade SCM, HHI SCM, Adensamento Estações, Cobertura área agricultável, ibc_indicador_fibra_* (one-hot).

### Acessos (Time-Series)

| Feature Pattern | Description | Source & Period |
|----------------|-------------|-----------------|
| `Acessos_{Serviço}_{YYYY}` | Service access counts per year | **Dec 2019–2024, Nov 2025, yearly** (Stage F) |
| `Densidade_{Serviço}_{YYYY}` | Service density rates per year | **Dec 2019–2024, Nov 2025, yearly** (Stage F) |

7 yearly columns per service×metric combination. 4 services × 2 metrics × 7 years = 56 feature-time columns.

### Satisfaction Survey

| Feature | Description | Source & Period |
|---------|-------------|-----------------|
| `isg_mean` | Weighted General Satisfaction Index (all services) | **2024 survey** (Stage R) |
| `qic_mean` | Weighted Internet/Connection Quality (all services) | **2024 survey** (Stage R) |
| `qf_mean` | Weighted Functioning Quality (all services) | **2024 survey** (Stage R) |
| `isg_scm_mean`, `qic_scm_mean`, `qf_scm_mean` | Quality indicators for broadband (SCM) | **2024 survey** (Stage R) |
| `isg_movel_mean`, `qic_movel_mean`, `qf_movel_mean` | Quality indicators for mobile (POS+PRE) | **2024 survey** (Stage R) |

### Additional Stage Features

| Feature Group | Sample Columns | Source & Period |
|--------------|----------------|-----------------|
| Licensed stations (BLF/TF/Geral) | `n_estacoes_blf`, `pct_ativas_tf`, `potencia_mean_geral` | **Static registry** (Stage I) |
| VSAT / Ground stations | `n_vsats`, `pct_starlink`, `n_terrenas_indiv` | **VSATs: Jun 2023–Dec 2025 aggregated; Terrenas: static** (Stage J) |
| GAISPI & Lei Antenas | `gaispi_fase`, `gaispi_liberado`, `lei_antenas_aprovada` | **2025 policy status / static** (Stage K) |
| RNI measurements | `n_medicoes_rni`, `pct_acima_50_limite`, `n_anos_rni` | **Aggregated 2000–2025** (Stage L) |
| SMP station breakdown | `n_estacoes_4g`, `pct_5g_smp`, `freq_mean_smp` | **Static registry** (Stage M) |
| Urban agglomerates | `n_aglomerados`, `velocidade_mediana_agl_mean` | **Static** (Stage N, 734 municipalities only) |
| School connectivity | `n_escolas`, `pct_escolas_conect_adequada`, `pct_4g_rural_ativada` | **Sep 2025** (Stage O) |
| Backhaul | `backhaul_fibra_2025`, `backhaul_ano_fibra`, `n_prestadoras_backhaul` | **Evolution: 2016–2025; Detail: 2025** (Stage P) |
| Prestadoras | `n_prestadoras`, `n_servicos_prestados`, `pct_sir` | **Static registry** (Stage Q) |
| Highway coverage | `rod_pct_cob_claro_4g`, `rod_pct_cob_vivo_5g`, etc. | **Dec 2025** (Stage H) |

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

### NaN / Missing Value Treatment

NaN values are handled differently depending on the context (targets vs. features) and the pipeline stage:

#### In target computation

| Context | NaN Treatment | Rationale |
|---------|--------------|-----------|
| **Coverage targets** (primary + derived) | `skipna=True` in row-wise mean across operators, then any remaining NaN filled with **0.0** | A municipality with NaN for all operators of a given technology has zero coverage for that technology. `skipna=True` ensures that if some operators report coverage but others have NaN, the mean is computed over available operators only. |
| **Extra regression targets** (12 raw columns) | NaN replaced with **0.0** via `np.nan_to_num(arr, nan=0.0)` | For each municipality, a missing value in a target column (e.g., `velocidade_mediana_mean`) is interpreted as the absence of the measured quantity (no broadband speed data = 0 speed reported). |
| **Classification targets** (15 derived) | Source array NaN filled with **0.0** before thresholding/binning. Percentile boundaries computed on **finite values only** (`arr[np.isfinite(arr)]`). | Ensures that NaN municipalities are assigned to the lowest bin (class 0), consistent with the "absence = zero" convention. Computing percentile boundaries on finite values avoids bias from NaN-heavy columns. |

#### In feature preprocessing (Pipeline Steps 5-6)

| Stage | NaN Treatment | Details |
|-------|--------------|---------|
| **Step 5: Missingness indicators** | For each numeric column containing any NaN or Inf, a binary indicator column `missingness_indicator_of_{col}` is created (1 = value was missing, 0 = value was present). Inf values are also treated as missing. Categorical columns use a **-1 sentinel code** instead of a separate indicator column. | Preserves missingness information as features so the model can learn from patterns of missing data. |
| **Step 6: Type-aware imputation** | **Numeric**: column median (computed on I_train). **Ordinal**: rounded median (computed on I_train); fallback to mode. **Categorical**: mode (most frequent value on I_train); fallback to -1. | All imputation statistics are fitted on **I_train only** and applied to the full dataset to prevent data leakage. After this step, the feature matrix contains no NaN or Inf values. |

#### Summary of NaN conventions

- **Targets:** NaN -> 0.0 (zero imputation). The interpretation is that a missing measurement means the quantity is absent or unreported.
- **Features:** NaN -> missingness indicator (binary column) + median/mode imputation. The interpretation is that the absence of a feature value is informative and should be captured as a separate signal.
- **No NaN survives preprocessing:** After Step 6, `X_scaled` is guaranteed to contain no NaN or Inf values. After target computation, all target arrays are guaranteed finite.

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

| Target | Source Column(s) | Temporal Reference | Pipeline Stage | Description |
|--------|-----------------|-------------------|----------------|-------------|
| `velocidade_mediana_mean` | `velocidade_mediana_mean` | **Static** (sector-level cross-sectional) | Stage G (Setores) | Mean of median download speeds across census sectors |
| `velocidade_mediana_std` | `velocidade_mediana_std` | **Static** (sector-level cross-sectional) | Stage G (Setores) | Std of median download speeds across census sectors |
| `pct_limite_mean` | `pct_limite_mean` | **Aggregated 2000–2025** (RNI measurements) | Stage L (Mapa RNI) | Mean % of RNI regulatory limit across all measurements |
| `renda_media_mean` | `renda_media_mean` | **Static** (sector-level cross-sectional) | Stage G (Setores) | Mean of average household income across census sectors |
| `renda_media_std` | `renda_media_std` | **Static** (sector-level cross-sectional) | Stage G (Setores) | Std of average household income across census sectors |
| `HHI SMP_2024` | `HHI SMP_2024` or `HHI_SMP_2024` | **Annual 2024** | Stage E (IBC) | Herfindahl-Hirschman Index, mobile (SMP) market |
| `HHI SCM_2024` | `HHI SCM_2024` or `HHI_SCM_2024` | **Annual 2024** | Stage E (IBC) | Herfindahl-Hirschman Index, fixed broadband (SCM) market |
| `pct_fibra_backhaul` | `pct_fibra_backhaul` | **2025** (latest year from provider-level data) | Stage P (Backhaul) | Percentage of backhaul using fiber optics |
| `pct_escolas_internet` | `pct_escolas_internet` | **Sep 2025** (school connectivity snapshot) | Stage O (Escolas) | Percentage of schools with internet |
| `pct_escolas_fibra` | `pct_escolas_fibra` | **Sep 2025** (school connectivity snapshot) | Stage O (Escolas) | Percentage of schools with fiber |
| `Densidade_Banda Larga Fixa_2025` | Multiple name variants (spaces/underscores) | **2025 annual** (latest Acessos snapshot) | Stage F (Acessos) | Fixed broadband density (subscriptions per 100 inhabitants) |
| `Densidade_Telefonia Movel_2025` | Multiple name variants (accent on 'o', spaces/underscores) | **2025 annual** (latest Acessos snapshot) | Stage F (Acessos) | Mobile telephony density (subscriptions per 100 inhabitants) |

---

### 5.4 Classification Targets (15)

Derived in `derived_targets.py`. Each target is engineered from one or two source columns via thresholding, binning, or cross-tabulation. The **temporal provenance** of each classification target is inherited from its source column(s) — see the "Source Temporal Reference" annotation below each target.

#### Strict Tier (10 targets, >= 5% minimum class fraction)

Built in `_build_strict_candidates()`. All NaN values are filled with 0.0 before thresholding/binning.

**1. `concentrated_mobile_market`** (Binary, 2 classes)
- **Source column:** `HHI SMP_2024` (or `HHI_SMP_2024`)
- **Source temporal reference:** Annual 2024 (Stage E, IBC)
- **Derivation:** Domain threshold (Anatel regulatory definition of concentrated market)
- **Formula:** `(HHI_SMP_2024 >= 0.25).astype(int64)`
- **Classes:** 0 = competitive (HHI < 0.25), 1 = concentrated (HHI >= 0.25)

**2. `high_fiber_backhaul`** (Binary, 2 classes)
- **Source column:** `pct_fibra_backhaul`
- **Source temporal reference:** 2025 (Stage P, Backhaul latest year)
- **Derivation:** Median split on non-zero values
- **Formula:** Compute `median_val = median(arr[arr > 0])`, then `(arr >= median_val).astype(int64)`
- **Classes:** 0 = below-median fiber, 1 = above-median fiber

**3. `high_speed_broadband`** (Binary, 2 classes)
- **Source column:** `pct_agl_alta_velocidade`
- **Source temporal reference:** Static (Stage N, urban agglomerate cross-sectional)
- **Derivation:** Median split on overall values
- **Formula:** `(arr > median(arr)).astype(int64)` (strict `>`, not `>=`)
- **Classes:** 0 = below-median high-speed broadband share, 1 = above-median

**4. `has_5g_coverage`** (Binary, 2 classes)
- **Source column:** `att09_any_present_5G`
- **Source temporal reference:** Sep 2025 (Stage C, SMP enriched)
- **Derivation:** Direct use of binary column
- **Formula:** `nan_to_num(arr, nan=0.0).astype(int64)` (column is already 0/1)
- **Classes:** 0 = no 5G operator present, 1 = at least one 5G operator

**5. `urbanization_level`** (3 classes)
- **Source column:** `pct_urbano`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Formula:** Compute p33.33 and p66.67 percentiles on finite values. `<=p33 -> 0`, `p33 < x <= p67 -> 1`, `>p67 -> 2`
- **Classes:** 0 = low urbanization, 1 = medium, 2 = high

**6. `broadband_speed_tier`** (3 classes)
- **Source column:** `velocidade_mediana_mean`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Classes:** 0 = low speed, 1 = medium, 2 = high

**7. `income_tier`** (3 classes)
- **Source column:** `renda_media_mean`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Derivation:** Tercile binning via `_tercile_bin()`
- **Classes:** 0 = low income, 1 = medium, 2 = high

**8. `mobile_penetration_tier`** (4 classes)
- **Source column:** `Densidade_Telefonia Movel_2025` (tries multiple spelling variants including accent on 'o')
- **Source temporal reference:** 2025 annual (Stage F, Acessos latest snapshot)
- **Derivation:** Quartile binning via `_quartile_bin()`
- **Formula:** Compute p25, p50, p75 on finite values. `<=p25 -> 0`, `p25 < x <= p50 -> 1`, `p50 < x <= p75 -> 2`, `>p75 -> 3`
- **Classes:** 0 = very low (Q1), 1 = low (Q2), 2 = medium (Q3), 3 = high (Q4)

**9. `infra_density_tier`** (5 classes)
- **Source column:** `n_estacoes_smp`
- **Source temporal reference:** Static registry (Stage M, SMP stations)
- **Derivation:** Quintile binning via `_quintile_bin()`
- **Formula:** Compute p20, p40, p60, p80 on finite values. Five bins assigned 0-4
- **Classes:** 0 = very low density (Q1) through 4 = very high density (Q5)

**10. `road_coverage_4g_tier`** (5 classes)
- **Source column:** `rod_pct_cob_todas_4g`
- **Source temporal reference:** Dec 2025 (Stage H, highway coverage)
- **Derivation:** Quintile binning via `_quintile_bin()`
- **Classes:** 0 = very low road 4G coverage (Q1) through 4 = very high (Q5)

#### Relaxed Tier (5 targets, >= 2% minimum class fraction, with failsafe binning)

Built in `_build_relaxed_candidates()`. Each has a primary derivation and a failsafe alternative. The primary is used if every class contains at least 2% of samples; otherwise the failsafe is substituted automatically.

**11. `income_speed_class`** (4-class primary, 2-class failsafe)
- **Source columns:** `pct_cat_low_renda_low_vel`, `pct_cat_low_renda_high_vel`, `pct_cat_high_renda_low_vel`, `pct_cat_high_renda_high_vel`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Primary derivation:** Stack the 4 columns into an (N,4) matrix, assign each row to whichever quadrant has the highest share via `argmax(axis=1)`
- **Primary classes:** 0 = dominant low-income/low-speed, 1 = dominant low-income/high-speed, 2 = dominant high-income/low-speed, 3 = dominant high-income/high-speed
- **Failsafe derivation:** Binary collapse. `low_income = col0 + col1`, `high_income = col2 + col3`. `(high_income > low_income) -> 1, else 0`
- **Why failsafe may trigger:** The high-income/low-speed quadrant (class 2) is rare (~0.9%), failing the 2% threshold

**12. `urban_rural_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `pct_urbano`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()`. Compute p3, p50, p97 percentiles. `<=p3 -> 0 (extreme rural)`, `p3 < x <= p50 -> 1`, `p50 < x < p97 -> 2`, `>=p97 -> 3 (extreme urban)`. Extreme classes each contain approximately 3% of data
- **Failsafe derivation:** Standard tercile binning via `_tercile_bin()` (3 balanced classes of ~33% each)

**13. `income_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `renda_media_mean`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()` using p3/p50/p97
- **Failsafe derivation:** Standard tercile binning on `renda_media_mean`

**14. `speed_extremes`** (4-class primary, 3-class failsafe)
- **Source column:** `velocidade_mediana_mean`
- **Source temporal reference:** Static (Stage G, sector-level cross-sectional)
- **Primary derivation:** Extreme 4-class binning via `_extreme_4class_bin()` using p3/p50/p97
- **Failsafe derivation:** Standard tercile binning on `velocidade_mediana_mean`

**15. `pop_5g_digital_divide`** (4-class primary, 2-class failsafe)
- **Source columns:** `populacao_2025` (from `city_populations.csv`) and `att09_any_present_5G` (from `smp_main.csv`)
- **Source temporal reference:** `populacao_2025` = 2025 estimate; `att09_any_present_5G` = Sep 2025 (Stage C, SMP enriched)
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
