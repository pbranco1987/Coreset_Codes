"""
Brazil Telecom Infrastructure Data Loader.

This loader is the *Brazil telecom mode* data backend used by the manuscript
experiments (R0-R8). It merges exactly **three** inputs:

1. smp_main.csv
   - Main feature table indexed by municipality IBGE code.
   - May include auxiliary columns like UF/state, municipality name, and/or
     population. If present, they are used as metadata (not features).

2. metadata.csv
   - Municipality-level geographic metadata: IBGE code, longitude, latitude,
     and UF/state.

3. city_populations.csv
   - Municipality-level population estimates.

The loader supports two input layouts:
- **Directory layout**: `data_dir` is a folder containing the 3 CSVs.
- **ZIP layout**: `data_dir` is a `.zip` file that contains the 3 CSVs anywhere
  inside (nested paths are OK). This is useful when "the input has to be a zip".

Per manuscript Section 5.4:
- Each entity is a Brazilian municipality identified by IBGE code.
- Geographic groups are the 27 UFs (including DF).

Targets:
- If target columns exist in the main file (e.g., y_cov_*), they are extracted.
  If absent, targets are set to zeros (the optimization pipeline does not
  strictly require targets unless you enable evaluation that uses them).
"""

from __future__ import annotations

import os
import re
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .feature_schema import FeatureSchema, FeatureType, infer_schema

# --- Imports from split-out sub-modules (backward-compat re-exports) --------
from ._btl_constants import (
    BRAZILIAN_STATES,
    STATE_TO_IDX,
    EXPECTED_N,
    EXPECTED_G,
)
from ._btl_data import BrazilTelecomData


class BrazilTelecomDataLoader:
    """
    Loader for the Brazil telecom infrastructure dataset.

    Parameters
    ----------
    data_dir : str
        Either a directory containing the input CSVs *or* a ZIP file containing
        the input CSVs.
    main_file : str
        Main features CSV name.
    metadata_file : str
        Metadata CSV name (coordinates + UF).
    population_file : str
        Population CSV name.
    """

    def __init__(
        self,
        data_dir: str,
        main_file: str = "smp_main.csv",
        metadata_file: str = "metadata.csv",
        population_file: str = "city_populations.csv",
        *,
        preprocessing_cfg=None,
    ):
        self.data_dir = str(data_dir)
        self.main_file = main_file
        self.metadata_file = metadata_file
        self.population_file = population_file
        self.preprocessing_cfg = preprocessing_cfg  # PreprocessingConfig (or None)

        self._zip_path: Optional[str] = None
        self._zip_members: Optional[List[str]] = None

        # Treat data_dir as a zip "data bundle" if it is a zip file.
        if os.path.isfile(self.data_dir) and zipfile.is_zipfile(self.data_dir):
            self._zip_path = self.data_dir

    @property
    def is_zip_source(self) -> bool:
        return self._zip_path is not None

    # ------------------------------------------------------------------
    # Source resolution (directory vs zip)
    # ------------------------------------------------------------------
    def _resolve_path(self, filename: str) -> str:
        """Resolve a file path inside a directory source."""
        import glob

        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            return path

        # Try alternative names with various transformations
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        alternatives = [
            # Direct variations
            filename.replace(" ", "_"),
            filename.replace("_", " "),
            filename.replace("(1)", "").strip(),
            filename.replace(" (1)", ""),
            filename.replace("__1_", " (1)"),
            filename.replace(" (1)", "__1_"),
            filename.replace("(1)", "__1_").replace(" ", ""),
            # Handle double underscore variations
            filename.replace(" ", "__").replace("(1)", "_1_"),
            # Without any suffix
            base_name.replace(" (1)", "") + ext,
            base_name.replace("__1_", "") + ext,
            base_name.replace("_1_", "") + ext,
        ]

        for alt in alternatives:
            alt_path = os.path.join(self.data_dir, alt)
            if os.path.exists(alt_path):
                return alt_path

        # Try glob pattern matching for partial matches
        patterns = [
            os.path.join(self.data_dir, f"*{base_name.split('_')[0]}*{ext}"),
            os.path.join(self.data_dir, f"*{base_name.replace(' ', '*').replace('(1)', '*')}*{ext}"),
        ]

        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        raise FileNotFoundError(f"Could not find file: {filename} in {self.data_dir}")

    def _zip_namelist(self) -> List[str]:
        if not self.is_zip_source:
            raise RuntimeError("Internal error: _zip_namelist called for non-zip source.")
        if self._zip_members is None:
            with zipfile.ZipFile(self._zip_path, "r") as zf:
                self._zip_members = list(zf.namelist())
        return self._zip_members

    def _resolve_zip_member(self, filename: str) -> str:
        """
        Resolve a member name inside a ZIP source.

        Matching is done against basenames, so nested folder structure in the zip
        does not matter.
        """
        members = self._zip_namelist()

        # Build basename->members map (there may be duplicates; we choose the first)
        base_map = {}
        for m in members:
            b = os.path.basename(m)
            if not b:
                continue
            base_map.setdefault(b, []).append(m)

        def _pick(candidate: str) -> Optional[str]:
            # Exact
            if candidate in base_map:
                return sorted(base_map[candidate], key=len)[0]
            # Case-insensitive
            cand_lower = candidate.lower()
            for b, fulls in base_map.items():
                if b.lower() == cand_lower:
                    return sorted(fulls, key=len)[0]
            return None

        # Candidate alternatives (mirror directory logic)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        alternatives = [
            filename,
            filename.replace(" ", "_"),
            filename.replace("_", " "),
            filename.replace("(1)", "").strip(),
            filename.replace(" (1)", ""),
            filename.replace("__1_", " (1)"),
            filename.replace(" (1)", "__1_"),
            filename.replace("(1)", "__1_").replace(" ", ""),
            filename.replace(" ", "__").replace("(1)", "_1_"),
            base_name.replace(" (1)", "") + ext,
            base_name.replace("__1_", "") + ext,
            base_name.replace("_1_", "") + ext,
        ]

        for alt in alternatives:
            hit = _pick(alt)
            if hit is not None:
                return hit

        # Fuzzy fallback: choose a CSV whose basename contains the base_name tokens
        base_lower = base_name.lower()
        csv_candidates = [m for m in members if os.path.basename(m).lower().endswith(ext.lower())]
        for m in csv_candidates:
            if base_lower in os.path.basename(m).lower():
                return m

        raise FileNotFoundError(f"Could not find file: {filename} inside zip: {self._zip_path}")

    def _read_csv(self, filename: str, **read_csv_kwargs) -> pd.DataFrame:
        """Read a CSV from either directory or zip source."""
        if self.is_zip_source:
            member = self._resolve_zip_member(filename)
            print(f"  Loading {filename} from ZIP: {self._zip_path} :: {member}")
            with zipfile.ZipFile(self._zip_path, "r") as zf:
                with zf.open(member) as f:
                    return pd.read_csv(f, **read_csv_kwargs)
        else:
            path = self._resolve_path(filename)
            print(f"  Loading {filename} from: {path}")
            return pd.read_csv(path, **read_csv_kwargs)

    # ------------------------------------------------------------------
    # File-specific loaders
    # ------------------------------------------------------------------
    def _load_main_data(self) -> pd.DataFrame:
        """Load main features file."""
        df = self._read_csv(self.main_file, low_memory=False)
        print(f"  Loaded main: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata file with coordinates."""
        df = self._read_csv(self.metadata_file, low_memory=False)

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'codigo' in col_lower or 'ibge' in col_lower:
                rename_map[col] = 'codigo_ibge'
            elif 'long' in col_lower:
                rename_map[col] = 'longitude'
            elif 'lat' in col_lower:
                rename_map[col] = 'latitude'
            elif col_lower == 'uf':
                rename_map[col] = 'uf'

        df = df.rename(columns=rename_map)

        # Keep expected columns if available
        keep = [c for c in ['codigo_ibge', 'longitude', 'latitude', 'uf'] if c in df.columns]
        if 'codigo_ibge' not in keep:
            raise KeyError("Metadata file must contain an IBGE code column (codigo_ibge).")

        df = df[keep].drop_duplicates()

        # Ensure numeric for join
        df['codigo_ibge'] = pd.to_numeric(df['codigo_ibge'], errors='coerce')
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

        return df.dropna(subset=['codigo_ibge'])

    def _load_population(self) -> pd.DataFrame:
        """Load population file."""
        # Handle BOM in some CSV exports
        df = self._read_csv(self.population_file, low_memory=False, encoding='utf-8-sig')

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'codigo' in col_lower or 'ibge' in col_lower:
                rename_map[col] = 'codigo_ibge'
            elif 'pop' in col_lower:
                rename_map[col] = 'populacao_2025'
            elif col_lower == 'uf':
                rename_map[col] = 'uf'
            elif 'munic' in col_lower:
                rename_map[col] = 'municipio'

        df = df.rename(columns=rename_map)

        if 'codigo_ibge' not in df.columns:
            raise KeyError("Population file must contain an IBGE code column (codigo_ibge).")
        if 'populacao_2025' not in df.columns:
            raise KeyError("Population file must contain a population column (populacao_2025).")

        # Ensure numeric
        df['codigo_ibge'] = pd.to_numeric(df['codigo_ibge'], errors='coerce')
        df['populacao_2025'] = pd.to_numeric(df['populacao_2025'], errors='coerce')

        df = df.dropna(subset=['codigo_ibge'])
        df = df.drop_duplicates(subset=['codigo_ibge'])

        return df

    # ------------------------------------------------------------------
    # Feature handling
    # ------------------------------------------------------------------
    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify numeric feature columns.

        In the manuscript, the design matrix uses all numeric covariates after
        merging the three required inputs:

        - ``smp_main.csv``
        - ``metadata.csv``
        - ``city_populations.csv``

        Therefore we apply only a *minimal* exclusion rule set:

        - identifiers/categoricals: ``codigo_ibge``, ``uf``/``UF``, and
          municipality name columns (e.g., ``municipio``)
        - targets/labels: any column starting with ``y_`` or containing
          ``y_cov_``

        Notably, **population and coordinates are retained as numeric
        covariates** (they contribute to the manuscript's $D=621$).
        """
        exclude_exact = {
            'codigo_ibge',
            'uf', 'UF',
            'municipio', 'MUNICIPIO',
        }

        feature_cols: List[str] = []

        for col in df.columns:
            col_lower = str(col).lower()

            # Exclude obvious non-features
            if col_lower in exclude_exact:
                continue
            if col_lower.startswith("y_") or "y_cov_" in col_lower:
                continue

            # Keep numeric (or numeric-convertible) columns
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
                continue

            # Try coercion for numeric-ish object columns
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                continue
            feature_cols.append(col)

        return feature_cols

    def _compute_cov_area_target(
        self,
        df: pd.DataFrame,
        *,
        tech: str,
        snapshot: str = "2025_09",
    ) -> Optional[np.ndarray]:
        """Compute September-2025 coverage-area targets in percentage points.

        The manuscript defines targets as the mean covered-area percentage
        across operators providing that technology (4G/5G) in the municipality.

        We implement this using columns that match the preprocessed naming
        convention in the provided dataset:

        ``cov_pct_area_coberta__tec_{tech}__op_{operator}__{snapshot}``

        where values are commonly stored as *fractions* in [0, 1]. We convert
        to percentage points by multiplying by 100 when the scale suggests a
        fractional encoding.

        Returns
        -------
        Optional[np.ndarray]
            Target vector of shape (N,) if matching columns exist, otherwise
            None.
        """

        tech_norm = tech.lower().strip()
        if tech_norm in {"4g", "tec_4g"}:
            tech_norm = "4g"
        elif tech_norm in {"5g", "tec_5g"}:
            tech_norm = "5g"

        snap_norm = snapshot.replace("-", "_").strip()

        prefix = f"cov_pct_area_coberta__tec_{tech_norm}__op_"
        suffix = f"__{snap_norm}".lower()

        cols = [c for c in df.columns if str(c).lower().startswith(prefix) and str(c).lower().endswith(suffix)]
        if not cols:
            return None

        # Prefer operator-specific columns; drop any "op_todas" aggregate
        op_cols = [c for c in cols if "__op_todas__" not in str(c).lower()]
        use_cols = op_cols if len(op_cols) > 0 else cols

        vals = df[use_cols].apply(pd.to_numeric, errors="coerce")
        y = vals.mean(axis=1, skipna=True).fillna(0.0).to_numpy(dtype=np.float64)

        # Convert fractions to percentage points when appropriate
        max_abs = float(np.nanmax(np.abs(y))) if y.size > 0 else 0.0
        if max_abs <= 1.5:
            y = y * 100.0

        return y

    def _standardize_features(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Standardize features to zero mean and unit variance."""
        if mask is None:
            mask = np.ones(len(X), dtype=bool)

        X = X.astype(np.float64)

        mean = np.nanmean(X[mask], axis=0)
        std = np.nanstd(X[mask], axis=0)
        std[std < 1e-10] = 1.0

        X_std = (X - mean) / std
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        return X_std

    def _impute_missing(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Impute missing values with column median."""
        df = df.copy()
        for col in cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        return df

    def _discover_coverage_targets(
        self,
        df: pd.DataFrame,
        *,
        y_4G: np.ndarray,
        y_5G: np.ndarray,
        snapshot: str = "2025_09",
    ) -> Dict[str, np.ndarray]:
        """Discover all available coverage targets for manuscript Table V.

        Produces exactly the 10 targets from Table V:

        ===  =================  ========================================
        #    Key                Construction
        ===  =================  ========================================
        1    cov_area_4G        operator mean area-coverage 4G (primary)
        2    cov_area_5G        operator mean area-coverage 5G (primary)
        3    cov_hh_4G          operator mean household-coverage 4G
        4    cov_res_4G         operator mean resident-coverage 4G
        5    cov_area_4G_5G     (area_4G + area_5G) / 2
        6    cov_area_all       mean area-cov across all technologies
        7    cov_hh_4G_5G       (hh_4G + hh_5G) / 2   (hh_4G if 5G absent)
        8    cov_hh_all         mean hh-cov across all technologies
        9    cov_res_4G_5G      (res_4G + res_5G) / 2  (res_4G if 5G absent)
        10   cov_res_all        mean res-cov across all technologies
        ===  =================  ========================================

        Returns a dict ``{canonical_name: (N,) array}``.
        """
        N = len(df)

        # ------------------------------------------------------------------
        # Step 1: Collect per-technology base targets
        # ------------------------------------------------------------------
        # Primary targets (always present)
        base: Dict[str, np.ndarray] = {
            "cov_area_4G": y_4G,
            "cov_area_5G": y_5G,
        }

        # ---- Helper: extract coverage from operator columns ----
        def _extract_operator_mean(
            prefix: str,
            suffix: str,
        ) -> Optional[np.ndarray]:
            """Mean across operator columns matching prefix...suffix pattern."""
            cols = [
                c for c in df.columns
                if str(c).lower().startswith(prefix.lower())
                and str(c).lower().endswith(suffix.lower())
            ]
            if not cols:
                return None
            # Exclude "op_todas" aggregate when operator-specific cols exist
            op_cols = [c for c in cols if "__op_todas__" not in str(c).lower()]
            use = op_cols if op_cols else cols
            vals = df[use].apply(pd.to_numeric, errors="coerce")
            y = vals.mean(axis=1, skipna=True).fillna(0.0).to_numpy(dtype=np.float64)
            if y.size > 0 and float(np.nanmax(np.abs(y))) <= 1.5:
                y = y * 100.0
            return y

        snap = snapshot.replace("-", "_").strip()
        snap_suffix = f"__{snap}"

        # --- Legacy area-coverage technologies ---
        area_techs: Dict[str, np.ndarray] = {}
        for tech, key in [("3g", "cov_area_3G"), ("2g", "cov_area_2G")]:
            y = _extract_operator_mean(
                f"cov_pct_area_coberta__tec_{tech}__op_", snap_suffix,
            )
            if y is not None:
                area_techs[key] = y

        # --- Household coverage (domicilios) ---
        hh_techs: Dict[str, np.ndarray] = {}
        for tech, key in [("4g", "cov_hh_4G"), ("5g", "cov_hh_5G")]:
            y = _extract_operator_mean(
                f"cov_pct_domicilios__tec_{tech}__op_", snap_suffix,
            )
            if y is None:
                col = self._find_column(df, [
                    f"y_cov_domicilios_pct_meanops_{tech.upper()}_{snapshot.replace('_', '')}",
                    f"y_cov_domicilios_{tech.upper()}",
                    f"y_cov_households_{tech.upper()}",
                ])
                if col:
                    y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(np.float64)
                    if y.size > 0 and float(np.nanmax(np.abs(y))) <= 1.5:
                        y = y * 100.0
            if y is not None and len(y) == N:
                hh_techs[key] = y

        # --- Resident / population coverage ---
        res_techs: Dict[str, np.ndarray] = {}
        for tech, key in [("4g", "cov_res_4G"), ("5g", "cov_res_5G")]:
            y = _extract_operator_mean(
                f"cov_pct_populacao__tec_{tech}__op_", snap_suffix,
            )
            if y is None:
                col = self._find_column(df, [
                    f"y_cov_populacao_pct_meanops_{tech.upper()}_{snapshot.replace('_', '')}",
                    f"y_cov_populacao_{tech.upper()}",
                    f"y_cov_residents_{tech.upper()}",
                ])
                if col:
                    y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(np.float64)
                    if y.size > 0 and float(np.nanmax(np.abs(y))) <= 1.5:
                        y = y * 100.0
            if y is not None and len(y) == N:
                res_techs[key] = y

        # ------------------------------------------------------------------
        # Step 2: Build the 10 Table V targets
        # ------------------------------------------------------------------
        targets: Dict[str, np.ndarray] = {}

        # Row 1-2: primary area targets (always present)
        targets["cov_area_4G"] = y_4G
        targets["cov_area_5G"] = y_5G

        # Row 3: Households (4G) --- only if available
        if "cov_hh_4G" in hh_techs:
            targets["cov_hh_4G"] = hh_techs["cov_hh_4G"]

        # Row 4: Residents (4G) --- only if available
        if "cov_res_4G" in res_techs:
            targets["cov_res_4G"] = res_techs["cov_res_4G"]

        # Row 5: Area (4G + 5G)  = (area_4G + area_5G) / 2
        targets["cov_area_4G_5G"] = (y_4G + y_5G) / 2.0

        # Row 6: Area (All) = mean of area-cov across ALL available technologies
        all_area = [y_4G, y_5G] + list(area_techs.values())
        targets["cov_area_all"] = np.column_stack(all_area).mean(axis=1).astype(np.float64)

        # Row 7: Households (4G + 5G) = (hh_4G + hh_5G) / 2  (hh_4G if 5G absent)
        if "cov_hh_4G" in hh_techs:
            if "cov_hh_5G" in hh_techs:
                targets["cov_hh_4G_5G"] = (hh_techs["cov_hh_4G"] + hh_techs["cov_hh_5G"]) / 2.0
            else:
                targets["cov_hh_4G_5G"] = hh_techs["cov_hh_4G"].copy()

        # Row 8: Households (All) = mean of hh-cov across all technologies
        if hh_techs:
            targets["cov_hh_all"] = np.column_stack(list(hh_techs.values())).mean(axis=1).astype(np.float64)

        # Row 9: Residents (4G + 5G) = (res_4G + res_5G) / 2  (res_4G if 5G absent)
        if "cov_res_4G" in res_techs:
            if "cov_res_5G" in res_techs:
                targets["cov_res_4G_5G"] = (res_techs["cov_res_4G"] + res_techs["cov_res_5G"]) / 2.0
            else:
                targets["cov_res_4G_5G"] = res_techs["cov_res_4G"].copy()

        # Row 10: Residents (All) = mean of res-cov across all technologies
        if res_techs:
            targets["cov_res_all"] = np.column_stack(list(res_techs.values())).mean(axis=1).astype(np.float64)

        # Validate shapes
        targets = {k: v for k, v in targets.items() if v is not None and len(v) == N}

        return targets

    # ------------------------------------------------------------------
    # Public load
    # ------------------------------------------------------------------
    def load(
        self,
        standardize: bool = True,
        *,
        impute_missing: bool = True,
    ) -> BrazilTelecomData:
        """
        Load and merge all data files.
        """
        print("[BrazilTelecomDataLoader] Loading data...")

        # Load all files
        df_main = self._load_main_data()
        df_meta = self._load_metadata()
        df_pop = self._load_population()

        # Ensure codigo_ibge is numeric in main data
        if 'codigo_ibge' not in df_main.columns:
            # Attempt to detect IBGE column if named differently
            candidates = [c for c in df_main.columns if 'ibge' in str(c).lower() or 'codigo' in str(c).lower()]
            if candidates:
                df_main = df_main.rename(columns={candidates[0]: 'codigo_ibge'})
            else:
                raise KeyError("Main data file must contain 'codigo_ibge' (IBGE municipality code).")

        df_main['codigo_ibge'] = pd.to_numeric(df_main['codigo_ibge'], errors='coerce')
        df_main = df_main.dropna(subset=['codigo_ibge'])

        # ------------------------------------------------------------------
        # Merge metadata (UF + coordinates)
        # ------------------------------------------------------------------
        meta_cols = ['codigo_ibge']
        for c in ['uf', 'longitude', 'latitude']:
            if c in df_meta.columns:
                meta_cols.append(c)

        # Merge with suffix to avoid clobbering existing columns
        df_main = df_main.merge(df_meta[meta_cols], on='codigo_ibge', how='left', suffixes=('', '_meta'))

        # UF: prefer main, fill missing from metadata
        if 'uf' not in df_main.columns and 'uf_meta' in df_main.columns:
            df_main = df_main.rename(columns={'uf_meta': 'uf'})
        elif 'uf' in df_main.columns and 'uf_meta' in df_main.columns:
            df_main['uf'] = df_main['uf'].fillna(df_main['uf_meta'])
            df_main = df_main.drop(columns=['uf_meta'])

        # Coordinates: prefer main, fill missing from metadata
        for c in ['longitude', 'latitude']:
            meta_c = f"{c}_meta"
            if c not in df_main.columns and meta_c in df_main.columns:
                df_main = df_main.rename(columns={meta_c: c})
            elif c in df_main.columns and meta_c in df_main.columns:
                df_main[c] = df_main[c].fillna(df_main[meta_c])
                df_main = df_main.drop(columns=[meta_c])

        # Drop rows with missing state (UF)
        if 'uf' not in df_main.columns:
            raise KeyError("Could not determine 'uf' (state) column after merging metadata.")
        df_main = df_main.dropna(subset=['uf'])

        # Standardize UF casing
        df_main['uf'] = df_main['uf'].astype(str).str.upper()

        # ------------------------------------------------------------------
        # Merge population (always, to ensure the 3rd input is used)
        # ------------------------------------------------------------------
        df_main = df_main.merge(
            df_pop[['codigo_ibge', 'populacao_2025']],
            on='codigo_ibge',
            how='left',
            suffixes=('', '_pop'),
        )

        # If main already had population, fill missing; otherwise take from pop file
        if 'populacao_2025' not in df_main.columns and 'populacao_2025_pop' in df_main.columns:
            df_main = df_main.rename(columns={'populacao_2025_pop': 'populacao_2025'})
        elif 'populacao_2025' in df_main.columns and 'populacao_2025_pop' in df_main.columns:
            # If both exist, prefer main but fill missing from population file
            df_main['populacao_2025'] = df_main['populacao_2025'].fillna(df_main['populacao_2025_pop'])
            df_main = df_main.drop(columns=['populacao_2025_pop'])

        # ------------------------------------------------------------------
        # Identify and prepare features (schema-aware, Phase 1)
        # ------------------------------------------------------------------
        # Build schema kwargs from preprocessing config if available
        schema_kwargs: Dict = {}
        if self.preprocessing_cfg is not None:
            pcfg = self.preprocessing_cfg
            schema_kwargs.update(
                categorical_columns=getattr(pcfg, "categorical_columns", []),
                ordinal_columns=getattr(pcfg, "ordinal_columns", []),
                ignore_columns=getattr(pcfg, "ignore_columns", []),
                target_columns=getattr(pcfg, "target_columns", []),
                treat_low_cardinality_int_as_categorical=getattr(
                    pcfg, "treat_low_cardinality_int_as_categorical", True,
                ),
                low_cardinality_threshold=getattr(pcfg, "low_cardinality_threshold", 25),
                high_cardinality_drop_threshold=getattr(pcfg, "high_cardinality_drop_threshold", None),
                treat_bool_as_categorical=getattr(pcfg, "treat_bool_as_categorical", True),
                auto_detect_targets=getattr(pcfg, "auto_detect_targets", True),
            )

        # Rename pre-engineered missingness indicator columns so their role
        # is explicit:  "Foo_missing" → "missingness_indicator_of_Foo"
        # This must happen BEFORE schema inference so the new names propagate
        # through the entire pipeline.
        _rename_map = {}
        for col in df_main.columns:
            if col.endswith("_missing"):
                base = col[: -len("_missing")]
                _rename_map[col] = f"missingness_indicator_of_{base}"
        if _rename_map:
            df_main.rename(columns=_rename_map, inplace=True)

        schema = infer_schema(df_main, **schema_kwargs)

        # Also exclude missingness indicators of target columns to prevent
        # leakage (e.g. missingness_indicator_of_pct_limite_mean hints that
        # pct_limite_mean was missing).
        target_col_names = schema.target_columns
        _miss_prefix = "missingness_indicator_of_"
        target_base_names = set(target_col_names)
        missingness_of_targets = [
            c for c in schema.feature_columns
            if c.startswith(_miss_prefix) and c[len(_miss_prefix):] in target_base_names
        ]
        for c in missingness_of_targets:
            schema.column_types[c] = FeatureType.TARGET

        # Save excluded columns (targets + their missingness indicators) to CSV
        # so the data is not lost.
        all_excluded = target_col_names + missingness_of_targets
        if all_excluded:
            excluded_df = df_main[['codigo_ibge'] + all_excluded].copy()
            excl_path = os.path.join(
                os.path.dirname(self.files.smp_main_path),
                "excluded_target_columns.csv",
            )
            excluded_df.to_csv(excl_path, index=False)
            print(f"  Saved {len(all_excluded)} excluded columns to {excl_path}")
            print(f"    Target columns ({len(target_col_names)}): {target_col_names[:5]}{'...' if len(target_col_names) > 5 else ''}")
            print(f"    Missingness of targets ({len(missingness_of_targets)}): {missingness_of_targets[:5]}{'...' if len(missingness_of_targets) > 5 else ''}")

        schema.print_summary()

        # Feature columns = numeric + ordinal + categorical (in schema order)
        feature_cols = schema.feature_columns
        feature_types_list = [t.value for t in schema.feature_types]
        print(f"  Identified {len(feature_cols)} feature columns")

        # --- Legacy-compatible path for pure-numeric extraction ---
        # Separate numeric/ordinal columns (coerce to float) from categoricals
        # (integer-encode from raw values).
        numeric_ordinal_cols = [
            c for c, t in zip(feature_cols, schema.feature_types)
            if t in (FeatureType.NUMERIC, FeatureType.ORDINAL)
        ]
        categorical_cols = schema.categorical_columns

        # Impute numeric/ordinal columns if requested
        if impute_missing:
            df_main = self._impute_missing(df_main, numeric_ordinal_cols)

        # Build the numeric portion (coerce to float)
        X_parts: List[np.ndarray] = []
        final_feature_names: List[str] = []
        final_feature_types: List[str] = []
        category_maps: Dict[str, Dict] = {}

        if numeric_ordinal_cols:
            X_num = df_main[numeric_ordinal_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
            if impute_missing:
                X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
            X_parts.append(X_num)
            final_feature_names.extend(numeric_ordinal_cols)
            final_feature_types.extend(
                [schema.column_types[c].value for c in numeric_ordinal_cols]
            )

        # Build categorical portion: stable integer encoding (sort unique -> 0..K-1)
        for col in categorical_cols:
            raw_vals = df_main[col].copy()
            # Build stable mapping: sort unique non-null values (stringified) -> 0..K-1
            non_null = raw_vals.dropna().unique()
            sorted_cats = sorted(non_null, key=lambda v: str(v))
            cat_map = {v: i for i, v in enumerate(sorted_cats)}
            category_maps[col] = cat_map

            # Encode: known -> code, unknown/NaN -> -1
            codes = raw_vals.map(cat_map)
            codes = codes.fillna(-1).astype(np.float64)
            X_parts.append(codes.values.reshape(-1, 1))
            final_feature_names.append(col)
            final_feature_types.append("categorical")

        # Concatenate all parts into a single feature matrix
        if X_parts:
            X = np.concatenate(X_parts, axis=1)
        else:
            X = np.empty((len(df_main), 0), dtype=np.float64)

        # In non-imputation mode, preserve NaNs for downstream missingness indicators
        if impute_missing:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if standardize:
            X = self._standardize_features(X)

        # ------------------------------------------------------------------
        # Targets (coverage-area % for September 2025)
        # ------------------------------------------------------------------
        # Per manuscript Section 5.5 and Section 5.9.3 we evaluate predictive
        # performance on 4G/5G covered-area percentages (percentage points) at
        # the September-2025 snapshot.
        #
        # Preferred extraction: compute from per-operator columns in the
        # processed dataset. Fallback: use explicit y_* columns if present.
        y_4G = self._compute_cov_area_target(df_main, tech="4g", snapshot="2025_09")
        y_5G = self._compute_cov_area_target(df_main, tech="5g", snapshot="2025_09")

        if y_4G is None:
            y_4G_col = self._find_column(
                df_main,
                [
                    'y_cov_area_pct_meanops_4G_sep2025',
                    'y_cov_area_pct_meanops_4G',
                    'y_4G',
                ],
            )
            y_4G = df_main[y_4G_col].values.astype(np.float64) if y_4G_col else np.zeros(len(df_main), dtype=np.float64)

        if y_5G is None:
            y_5G_col = self._find_column(
                df_main,
                [
                    'y_cov_area_pct_meanops_5G_sep2025',
                    'y_cov_area_pct_meanops_5G',
                    'y_5G',
                ],
            )
            y_5G = df_main[y_5G_col].values.astype(np.float64) if y_5G_col else np.zeros(len(df_main), dtype=np.float64)

        # ------------------------------------------------------------------
        # Metadata outputs
        # ------------------------------------------------------------------
        state_labels = df_main['uf'].values.astype(str)
        state_indices = np.array([STATE_TO_IDX.get(s.upper(), 0) for s in state_labels])

        ibge_codes = df_main['codigo_ibge'].values
        population = df_main.get('populacao_2025', pd.Series(np.ones(len(df_main)))).fillna(0).values

        coords = np.column_stack([
            df_main.get('longitude', pd.Series(np.zeros(len(df_main)))).fillna(0).values,
            df_main.get('latitude', pd.Series(np.zeros(len(df_main)))).fillna(0).values
        ])

        muni_col = self._find_column(df_main, ['municipio', 'NomMunicipio', 'MUNICIPIO'])
        municipality_names = df_main[muni_col].values if muni_col else np.array([''] * len(df_main))

        # Validate dimensions
        N, D = X.shape
        G = len(np.unique(state_indices))
        print(f"  Final dataset: N={N} municipalities, D={D} features, G={G} states")

        if N != EXPECTED_N:
            print(f"  WARNING: Expected N={EXPECTED_N}, got N={N}")
        if G != EXPECTED_G:
            print(f"  WARNING: Expected G={EXPECTED_G}, got G={G}")

        # ------------------------------------------------------------------
        # Multi-target coverage discovery (manuscript Table IV)
        # ------------------------------------------------------------------
        extra_targets = self._discover_coverage_targets(
            df_main, y_4G=y_4G, y_5G=y_5G, snapshot="2025_09",
        )
        # Remove the two primary targets already stored as y_4G / y_5G
        extra_targets.pop("cov_area_4G", None)
        extra_targets.pop("cov_area_5G", None)
        n_extra = len(extra_targets)
        if n_extra:
            print(f"  Discovered {n_extra} extra coverage targets: {list(extra_targets.keys())}")

        return BrazilTelecomData(
            X=X,
            state_labels=state_labels,
            state_indices=state_indices,
            y_4G=y_4G,
            y_5G=y_5G,
            ibge_codes=ibge_codes,
            population=population,
            coords=coords,
            feature_names=final_feature_names,
            municipality_names=municipality_names,
            extra_targets=extra_targets,
            feature_types=final_feature_types,
            category_maps=category_maps,
        )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None


def load_brazil_telecom_data(
    data_dir: str,
    standardize: bool = True,
    **kwargs
) -> BrazilTelecomData:
    """
    Convenience function to load Brazil telecom data.

    Notes
    -----
    `data_dir` may be either:
    - a directory containing the three CSVs, or
    - a zip file containing the three CSVs.
    """
    loader = BrazilTelecomDataLoader(data_dir, **kwargs)
    return loader.load(standardize=standardize)
