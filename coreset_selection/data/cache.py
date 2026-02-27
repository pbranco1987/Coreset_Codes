r"""
Replicate cache building and loading utilities.

This cache is the mechanism that enforces *replicate-level reuse* of:
- preprocessing statistics (imputation + standardization),
- representation fitting (PCA + VAE),
- raw-space evaluation sampling.

Per manuscript Section 5.8 (Splits, stratification, and random control):
1) Representation-learning split (I_train, I_val) is used to compute preprocessing
   statistics and to fit PCA / train the VAE.
2) Raw-space evaluation index set E has fixed size |E|=2000 and is sampled
   stratified by state using rounded state counts \tilde{c} that approximate
   |E|·pi_g and satisfy \sum_g \tilde{c}_g = |E|.
3) Evaluation train/test split (E_train, E_test) is an 80/20 stratified split
   *within each state*.

The resulting arrays are stored in a single compressed .npz file per replicate.

Implementation is split across sub-modules for maintainability:
- ``_cache_io``: atomic save, validation, locking
- ``_cache_preprocessing``: imputation, log1p, standardization pipeline
- ``_cache_sampling``: rounded counts, group sampling, within-group splitting

This module re-exports every helper so that existing ``from .cache import ...``
statements continue to work unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config.dataclasses import ExperimentConfig, ReplicateAssets
from ..utils.io import ensure_dir
from ..utils.debug_timing import timer
from .target_columns import detect_target_columns, remove_target_columns, validate_no_leakage
from .split_persistence import save_splits
from .derived_targets import extract_extra_regression_targets, derive_classification_targets

# ---- Re-exports from sub-modules (backward compatibility) ----
from ._cache_io import (          # noqa: F401
    atomic_savez,
    validate_cache,
    _acquire_build_lock,
    _release_build_lock,
)
from ._cache_preprocessing import (  # noqa: F401
    _stratified_split,
    _impute_by_train_median,
    _impute_by_train_mode,
    _impute_typeaware,
    _preprocess_fit_transform,
    _detect_log1p_cols,
)
from ._cache_sampling import (    # noqa: F401
    _rounded_counts,
    _sample_by_group,
    _split_within_groups,
)


# ---------------------------------------------------------------------------
# Column specification builder for mixed-type VAE
# ---------------------------------------------------------------------------

def _build_column_spec(
    feature_types: List[str],
    scale_mask: List[bool],
    X_raw: np.ndarray,
    feature_names: List[str],
    category_maps: Optional[Dict] = None,
):
    """Build a ColumnSpec from preprocessing metadata.

    Classifies each column in the preprocessed feature matrix into one of
    three likelihood families for :class:`MixedTypeVAE`:

    - **continuous** — numeric/ordinal columns (scaled) and missingness
      indicators.  These use Gaussian NLL (≡ MSE when σ²=1).
    - **binary** — categorical columns with K ≤ 2 (unscaled 0/1 codes).
      These use BCE with logits.
    - **categorical** — categorical columns with K > 2 (unscaled integer
      codes).  These use CE with softmax logits.

    Parameters
    ----------
    feature_types : list of str
        Per-column type ("numeric", "ordinal", "categorical") including
        appended missingness indicators (which are "numeric").
    scale_mask : list of bool
        Per-column flag: True if the column was standardised, False if not.
        Categorical columns are NOT scaled.
    X_raw : np.ndarray
        The imputed (but unscaled) feature matrix — used to determine
        cardinality of categorical columns when ``category_maps`` is absent.
    feature_names : list of str
        Column names matching the column order in ``X_raw``.
    category_maps : dict, optional
        ``{col_name: {value: code}}`` mappings from DataManager.  Used to
        determine cardinality K for each categorical column.

    Returns
    -------
    ColumnSpec
        Column specification for MixedTypeVAE.
    """
    from ..models._vae_networks import ColumnSpec

    n_total = X_raw.shape[1]
    continuous_idx: List[int] = []
    binary_idx: List[int] = []
    categorical_specs: List[Tuple[int, int]] = []

    for j in range(n_total):
        ft = feature_types[j] if j < len(feature_types) else "numeric"
        is_scaled = bool(scale_mask[j]) if j < len(scale_mask) else True

        if ft == "categorical" and not is_scaled:
            # Determine cardinality K
            col_name = feature_names[j] if j < len(feature_names) else f"x{j}"
            K = None
            if category_maps and col_name in category_maps:
                K = len(category_maps[col_name])
            if K is None:
                # Fallback: infer from data (max integer code + 1)
                vals = X_raw[:, j]
                finite = vals[np.isfinite(vals)]
                K = int(np.max(finite)) + 1 if finite.size > 0 else 2

            if K <= 2:
                binary_idx.append(j)
            else:
                categorical_specs.append((j, K))
        else:
            continuous_idx.append(j)

    spec = ColumnSpec(
        continuous_idx=np.array(continuous_idx, dtype=np.int64),
        binary_idx=np.array(binary_idx, dtype=np.int64),
        categorical_specs=categorical_specs,
        n_total=n_total,
    )

    print(
        f"[cache] ColumnSpec: {spec.n_continuous} continuous, "
        f"{spec.n_binary} binary, {spec.n_categorical} multi-class "
        f"(total {n_total})",
        flush=True,
    )

    return spec


def build_replicate_cache(
    cfg: ExperimentConfig,
    rep_id: int,
    data_manager=None,
) -> str:
    """
    Build and save cached assets for a single replicate.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration
    rep_id : int
        Replicate identifier
    data_manager : Optional[DataManager]
        Pre-loaded data manager. If None, loads from cfg.files.

    Returns
    -------
    str
        Path to saved cache file (assets.npz)
    """
    import torch

    from .manager import DataManager
    from ..geo.info import build_geo_info
    from ..models.vae import VAETrainer

    with timer.section("build_replicate_cache", rep_id=rep_id):
        seed = int(cfg.seed + rep_id)
        rng = np.random.default_rng(seed)

        # Load data
        with timer.section("load_data_manager"):
            preproc_cfg = getattr(cfg, 'preprocessing', None)
            if data_manager is None:
                data_manager = DataManager(cfg.files, seed, preprocessing_cfg=preproc_cfg)
                data_manager.load()

        # Raw (unscaled) features and labels
        # IMPORTANT: X_unscaled may contain NaNs; preprocessing below must
        # add missingness indicators and impute using I_train statistics only.
        with timer.section("extract_raw_data"):
            X_unscaled = np.asarray(data_manager.X_numeric_unscaled(), dtype=np.float64)
            state_labels = np.asarray(data_manager.state_labels())
            y = data_manager.targets()

            # Individual targets if available (Brazil telecom)
            y_4G = data_manager.targets_4G()
            y_5G = data_manager.targets_5G()

            # Multi-target coverage (manuscript Table IV)
            extra_targets_dict = data_manager.targets_all_dict() if hasattr(data_manager, 'targets_all_dict') else {}
            # Remove the primary pair (saved separately as y_4G / y_5G)
            extra_targets_dict.pop("cov_area_4G", None)
            extra_targets_dict.pop("cov_area_5G", None)

            # Derived downstream targets (extra regression + classification)
            # Must use the FULL raw DataFrame (before feature/target separation)
            # because columns like HHI, pct_fibra_backhaul, velocidade_mediana,
            # etc. are classified as TARGET by the schema and stripped from df.
            raw_df = getattr(data_manager, "_raw_df", None)
            if raw_df is None:
                raw_df = getattr(data_manager, "df", None)
            derived_extra_reg: dict = {}
            derived_cls: dict = {}
            if raw_df is not None:
                derived_extra_reg = extract_extra_regression_targets(raw_df)
                # NOTE: metadata_path deferred until cache_path is known (line ~393)
                derived_cls = derive_classification_targets(raw_df)

                # Include 4G coverage as a supervised regression target so that
                # multi-model downstream learners (GBT, RF, KNN, Ridge, SVR)
                # are also trained on the primary 4G coverage outcome.
                if y_4G is not None:
                    derived_extra_reg["cov_area_4G"] = np.nan_to_num(
                        np.asarray(y_4G, dtype=np.float64), nan=0.0
                    )

                timer.checkpoint(
                    "Derived targets extracted",
                    n_extra_reg=len(derived_extra_reg),
                    n_cls=len(derived_cls),
                )

            # QoS target: qf_mean — Qualidade do Funcionamento (Technical
            # Quality of Service) from Anatel satisfaction survey (Stage R).
            # QF is a sub-component of ISG (= weighted composite of QF + QIC
            # + QCR).  Used as the QoS evaluation target in qos_tasks.py.
            qos_target = None
            if raw_df is not None and "qf_mean" in raw_df.columns:
                _qf = pd.to_numeric(raw_df["qf_mean"], errors="coerce")
                qos_target = np.nan_to_num(_qf.to_numpy(dtype=np.float64), nan=0.0)
                timer.checkpoint("QoS target extracted (qf_mean)", n_valid=int(np.count_nonzero(qos_target)))

            # Optional geo metadata for plotting / diagnostics
            lat, lon = data_manager.latlon()
            pop = data_manager.population()

        N = X_unscaled.shape[0]
        timer.checkpoint("Data extracted", N=N, n_features=X_unscaled.shape[1])

        # Geographic groups for stratification
        with timer.section("build_geo_info_for_cache"):
            geo = build_geo_info(state_labels)

        # ------------------------------------------------------------
        # (1) Representation-learning split (I_train, I_val)
        # ------------------------------------------------------------
        with timer.section("stratified_split"):
            train_idx, val_idx = _stratified_split(
                np.arange(N, dtype=int),
                geo.group_ids,
                test_frac=float(cfg.vae.validation_frac),
                seed=seed,
            )

        timer.checkpoint("Split created", n_train=len(train_idx), n_val=len(val_idx))

        # ------------------------------------------------------------
        # Preprocessing and standardization (manuscript Section 5.7)
        #   - add missingness indicators for columns with missing entries
        #   - impute NaNs with TRAIN-split medians
        #   - apply log1p to heavy-tailed non-negative variables (fit on TRAIN)
        #   - standardize using TRAIN-split mean/std
        # ------------------------------------------------------------
        with timer.section("preprocessing"):
            feature_names = list(getattr(data_manager, "feature_names", lambda: [])())
            if not feature_names:
                feature_names = list(getattr(data_manager, "_feature_cols", []))
            if not feature_names:
                feature_names = [f"x{j}" for j in range(X_unscaled.shape[1])]

            # Phase 1: Capture feature types from DataManager (if available)
            raw_feature_types = list(
                getattr(data_manager, "_feature_types", [])
            )
            if len(raw_feature_types) != len(feature_names):
                # Fallback: mark all as numeric
                raw_feature_types = ["numeric"] * len(feature_names)

            # ---- Phase 4.3: Target leakage prevention ----
            # Per manuscript Section VII, target-defining columns MUST be
            # excluded from the feature matrix before PCA / VAE training.
            # Phase 4: Also remove explicitly declared target columns (so
            # users can swap targets via config without editing regexes).
            explicit_targets = (
                list(getattr(preproc_cfg, "target_columns", []))
                if preproc_cfg else []
            )
            target_cols_detected = detect_target_columns(feature_names)
            # Union explicit + regex-detected targets
            all_target_cols = set(target_cols_detected) | set(explicit_targets)
            # Filter to columns actually present in feature_names
            all_target_cols &= set(feature_names)
            if all_target_cols:
                X_unscaled, feature_names, removed_cols = remove_target_columns(
                    X_unscaled, feature_names,
                    explicit_targets=explicit_targets,
                )
                # Also remove matching entries from raw_feature_types
                removed_set = set(removed_cols)
                raw_feature_types = [
                    t for name, t in zip(
                        list(getattr(data_manager, "feature_names", lambda: [])()) or feature_names,
                        raw_feature_types,
                    )
                    if name not in removed_set
                ]
                # Ensure alignment after removal
                if len(raw_feature_types) != len(feature_names):
                    raw_feature_types = ["numeric"] * len(feature_names)

                timer.checkpoint(
                    "Target columns removed",
                    n_removed=len(removed_cols),
                    removed=removed_cols,
                )
            else:
                removed_cols = []

            X_raw, preproc_meta = _preprocess_fit_transform(
                X_unscaled=X_unscaled,
                train_idx=train_idx,
                feature_names=feature_names,
                feature_types=raw_feature_types,
                categorical_impute_strategy=getattr(
                    preproc_cfg, "categorical_impute_strategy", "mode"
                ) if preproc_cfg else "mode",
                ordinal_impute_strategy=getattr(
                    preproc_cfg, "ordinal_impute_strategy", "median"
                ) if preproc_cfg else "median",
                log1p_categoricals=getattr(
                    preproc_cfg, "log1p_categoricals", False
                ) if preproc_cfg else False,
                log1p_ordinals=getattr(
                    preproc_cfg, "log1p_ordinals", False
                ) if preproc_cfg else False,
            )
            # Record removed target columns in preprocessing metadata
            preproc_meta["removed_target_columns"] = removed_cols

            # Phase 3: Use feature_types_out from _preprocess_fit_transform
            # which already includes "numeric" entries for appended missingness
            # indicators.  Fall back to manual extension for robustness.
            extended_types = preproc_meta.get("feature_types_out", None)
            if extended_types is None or len(extended_types) != len(preproc_meta["feature_names"]):
                n_orig = len(feature_names)
                n_out = len(preproc_meta["feature_names"])
                extended_types = list(raw_feature_types) + ["numeric"] * (n_out - n_orig)
            preproc_meta["feature_types"] = extended_types

            # Phase 2: Type-aware standardization.
            # Categorical columns are NOT scaled (they are integer codes).
            # Ordinal columns are optionally scaled.
            # Numeric columns and missingness indicators are always scaled.
            scale_ordinals = getattr(preproc_cfg, "scale_ordinals", True) if preproc_cfg else True
            scale_categoricals = getattr(preproc_cfg, "scale_categoricals", False) if preproc_cfg else False

            # Build per-column scale mask
            n_total = X_raw.shape[1]
            scale_mask = np.ones(n_total, dtype=bool)
            for j in range(n_total):
                ft = extended_types[j] if j < len(extended_types) else "numeric"
                if ft == "categorical" and not scale_categoricals:
                    scale_mask[j] = False
                elif ft == "ordinal" and not scale_ordinals:
                    scale_mask[j] = False

            preproc_cfg_obj = preproc_cfg  # may be None

            scaler = StandardScaler()
            # Fit scaler on train split, scaleable columns only
            scaleable_idx = np.flatnonzero(scale_mask)
            if scaleable_idx.size > 0 and scaleable_idx.size < n_total:
                # Partial scaling: fit/transform only scaleable columns
                X_scaleable_train = X_raw[np.ix_(train_idx, scaleable_idx)]
                scaler.fit(X_scaleable_train)
                X_scaled = X_raw.copy()
                X_scaled[:, scaleable_idx] = scaler.transform(
                    X_raw[:, scaleable_idx]
                ).astype(np.float32)
                X_scaled = X_scaled.astype(np.float32)
            else:
                # All columns scaleable (legacy path)
                scaler.fit(X_raw[train_idx])
                X_scaled = scaler.transform(X_raw).astype(np.float32)

            # Store scale_mask in metadata for downstream consumers
            preproc_meta["scale_mask"] = scale_mask.tolist()

            # Keep raw (imputed/transformed, unscaled) as float32 for storage
            X_raw = X_raw.astype(np.float32)

        timer.checkpoint("Preprocessing complete", X_scaled_shape=X_scaled.shape)

        # ------------------------------------------------------------
        # (2) Raw-space evaluation set E (|E| fixed, stratified by state)
        # ------------------------------------------------------------
        with timer.section("create_eval_split"):
            eval_size = int(getattr(cfg.eval, "eval_size", 2000))
            eval_size = min(eval_size, N)

            eval_counts = _rounded_counts(
                geo.pi,
                eval_size,
                min_one_per_group=True,
            )
            eval_idx = _sample_by_group(geo.group_ids, eval_counts, rng)

            # ------------------------------------------------------------
            # (3) Evaluation train/test split (within each state)
            # ------------------------------------------------------------
            eval_train_frac = float(getattr(cfg.eval, "eval_train_frac", 0.8))
            eval_train_idx, eval_test_idx = _split_within_groups(
                eval_idx=eval_idx,
                group_ids=geo.group_ids,
                train_frac=eval_train_frac,
                rng=np.random.default_rng(seed + 1),
            )

        # ---- Phase 4.2: Persist splits to splits.npz ----
        try:
            save_splits(
                cfg.files.cache_dir, rep_id,
                train_idx=train_idx, val_idx=val_idx,
                eval_idx=eval_idx,
                eval_train_idx=eval_train_idx,
                eval_test_idx=eval_test_idx,
                seed=seed,
                metadata={"eval_size": np.array([eval_size])},
            )
            timer.checkpoint("Splits persisted to splits.npz")
        except Exception as e:
            # Non-fatal: log and continue (cache itself has the splits)
            print(f"[cache] Warning: could not persist splits.npz: {e}")


        # ------------------------------------------------------------
        # Train VAE (optional, per cfg.vae.epochs)
        # ------------------------------------------------------------
        Z_vae = None
        Z_logvar = None

        # Build column specification for mixed-type VAE (needed for both
        # VAE training and for saving in the cache for later augmentation).
        column_spec = None
        cat_maps = data_manager.category_maps() if hasattr(data_manager, 'category_maps') else {}
        if bool(getattr(cfg.vae, "use_mixed_likelihood", False)):
            column_spec = _build_column_spec(
                feature_types=extended_types,
                scale_mask=preproc_meta.get("scale_mask", [True] * X_scaled.shape[1]),
                X_raw=X_raw,
                feature_names=list(preproc_meta["feature_names"]),
                category_maps=cat_maps,
            )

        if int(cfg.vae.epochs) > 0:
            # Safety guard: ensure no target columns remain before representation learning
            validate_no_leakage(preproc_meta["feature_names"])

            with timer.section("VAE_training", epochs=cfg.vae.epochs, latent_dim=cfg.vae.latent_dim):
                use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
                device = torch.device(cfg.device if use_cuda else "cpu")
                timer.checkpoint("VAE device", device=str(device), cuda_available=torch.cuda.is_available())

                vae_trainer = VAETrainer(cfg.vae, seed, device, column_spec=column_spec)
                vae_trainer.train(X_scaled[train_idx], X_val=X_scaled[val_idx])

            with timer.section("VAE_embedding"):
                Z_vae, Z_logvar = vae_trainer.embed(X_scaled)
                Z_vae = np.asarray(Z_vae, dtype=np.float32)
                Z_logvar = np.asarray(Z_logvar, dtype=np.float32)

            timer.checkpoint("VAE complete", Z_vae_shape=Z_vae.shape)

        # ------------------------------------------------------------
        # PCA representation (optional)
        # ------------------------------------------------------------
        Z_pca = None
        if int(cfg.pca.n_components) > 0:
            # Safety guard: ensure no target columns remain before representation learning
            validate_no_leakage(preproc_meta["feature_names"])

            with timer.section("PCA_fitting", n_components=cfg.pca.n_components):
                pca = PCA(
                    n_components=int(cfg.pca.n_components),
                    whiten=bool(cfg.pca.whiten),
                    random_state=seed,
                )
                pca.fit(X_scaled[train_idx])
                Z_pca = pca.transform(X_scaled).astype(np.float32)

            timer.checkpoint("PCA complete", Z_pca_shape=Z_pca.shape)

        # Create output directory
        cache_dir = os.path.join(cfg.files.cache_dir, f"rep{rep_id:02d}")
        ensure_dir(cache_dir)
        cache_path = os.path.join(cache_dir, "assets.npz")

        with timer.section("save_cache_to_disk"):
            save_dict = {
                "X_raw": X_raw,
                "X_scaled": X_scaled,
                "state_labels": state_labels,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "eval_idx": eval_idx,
                "eval_train_idx": eval_train_idx,
                "eval_test_idx": eval_test_idx,
                # Preprocessing metadata (object arrays allowed in npz)
                "feature_names": np.array(preproc_meta["feature_names"], dtype=object),
                "missing_feature_names": np.array(preproc_meta["missing_feature_names"], dtype=object),
                "log1p_feature_names": np.array(preproc_meta["log1p_feature_names"], dtype=object),
                # Phase 1: Feature type metadata
                "feature_types": np.array(
                    preproc_meta.get("feature_types", []), dtype=object,
                ),
                # Phase 4.3: record which target columns were removed (audit trail)
                "removed_target_columns": np.array(
                    preproc_meta.get("removed_target_columns", []), dtype=object,
                ),
                # Phase 2: scale mask (which columns were standardized)
                "scale_mask": np.array(
                    preproc_meta.get("scale_mask", [True] * X_raw.shape[1]),
                    dtype=bool,
                ),
                # Phase 3: per-type column lists
                "categorical_columns": np.array(
                    preproc_meta.get("categorical_columns", []), dtype=object,
                ),
                "ordinal_columns": np.array(
                    preproc_meta.get("ordinal_columns", []), dtype=object,
                ),
                "numeric_columns": np.array(
                    preproc_meta.get("numeric_columns", []), dtype=object,
                ),
            }

            # Phase 2: Store category_maps as serialized JSON string
            # (cat_maps was already fetched above for column_spec building)
            if cat_maps:
                import json
                # Convert keys to strings for JSON serialization
                serializable_maps = {}
                for col, mapping in cat_maps.items():
                    serializable_maps[col] = {str(k): int(v) for k, v in mapping.items()}
                save_dict["category_maps_json"] = np.array(
                    json.dumps(serializable_maps), dtype=object,
                )

            # Phase 3: Store imputation statistics as serialized JSON string
            impute_vals = preproc_meta.get("impute_values", {})
            if impute_vals:
                import json as _json_imp
                save_dict["impute_values_json"] = np.array(
                    _json_imp.dumps({k: v for k, v in impute_vals.items()}),
                    dtype=object,
                )

            # Phase 2: Detect and store target type
            target_type_cfg = getattr(preproc_cfg, "target_type", "auto") if preproc_cfg else "auto"
            if target_type_cfg == "auto" and y is not None:
                from ..evaluation.classification_metrics import infer_target_type
                cardinality_thr = getattr(preproc_cfg, "classification_cardinality_threshold", 50) if preproc_cfg else 50
                # Check the primary target(s)
                if y.ndim == 2:
                    detected_type = infer_target_type(y[:, 0], cardinality_threshold=cardinality_thr)
                else:
                    detected_type = infer_target_type(y, cardinality_threshold=cardinality_thr)
            elif target_type_cfg != "auto":
                detected_type = target_type_cfg
            else:
                detected_type = "regression"
            save_dict["target_type"] = np.array(detected_type, dtype=object)

            # Mixed-type VAE column specification (for augmentation path)
            if column_spec is not None:
                save_dict["colspec_continuous_idx"] = column_spec.continuous_idx
                save_dict["colspec_binary_idx"] = column_spec.binary_idx
                if column_spec.categorical_specs:
                    save_dict["colspec_cat_specs"] = np.array(
                        column_spec.categorical_specs, dtype=np.int64,
                    )
                save_dict["colspec_n_total"] = np.array(column_spec.n_total, dtype=np.int64)

            if Z_vae is not None:
                save_dict["Z_vae"] = Z_vae
                save_dict["Z_logvar"] = Z_logvar

            if Z_pca is not None:
                save_dict["Z_pca"] = Z_pca

            if y is not None:
                save_dict["y"] = y

            # Save individual targets for Brazil telecom data
            if y_4G is not None:
                save_dict["y_4G"] = y_4G
            if y_5G is not None:
                save_dict["y_5G"] = y_5G

            # Save extra coverage targets for multi-target KRR (Table IV)
            if extra_targets_dict:
                save_dict["extra_target_names"] = np.array(
                    list(extra_targets_dict.keys()), dtype=object,
                )
                for name, arr in extra_targets_dict.items():
                    save_dict[f"y_extra_{name}"] = np.asarray(arr, dtype=np.float64)

            # Save derived extra regression targets (beyond coverage)
            if derived_extra_reg:
                save_dict["extra_reg_target_names"] = np.array(
                    list(derived_extra_reg.keys()), dtype=object,
                )
                for name, arr in derived_extra_reg.items():
                    save_dict[f"y_extreg_{name}"] = np.asarray(arr, dtype=np.float64)

            # Save derived classification targets
            if derived_cls:
                save_dict["cls_target_names"] = np.array(
                    list(derived_cls.keys()), dtype=object,
                )
                for name, arr in derived_cls.items():
                    save_dict[f"y_cls_{name}"] = np.asarray(arr, dtype=np.int64)

            # Save QoS target (qf_mean — Qualidade do Funcionamento)
            if qos_target is not None:
                save_dict["qos_target"] = qos_target

            # Optional plotting metadata
            if lat is not None and lon is not None:
                save_dict["latitude"] = np.asarray(lat)
                save_dict["longitude"] = np.asarray(lon)
            if pop is not None:
                save_dict["population"] = np.asarray(pop)

            atomic_savez(cache_path, **save_dict)

        # Write classification-target metadata JSON alongside the cache
        if derived_cls and hasattr(derived_cls, '_metadata'):
            _meta_json_path = str(
                Path(cache_path).with_suffix(".cls_metadata.json")
            )
            try:
                import json as _json_meta
                with open(_meta_json_path, "w", encoding="utf-8") as _f:
                    _json_meta.dump(derived_cls._metadata, _f, indent=2, ensure_ascii=False)
                timer.checkpoint("Classification metadata saved", path=_meta_json_path)
            except Exception as e:
                print(f"[cache] Warning: could not write cls metadata JSON: {e}")

    return cache_path


def _seed_dim_cache(
    base_cache_dir: str,
    dim_cache_dir: str,
    rep_id: int,
    space_tag: str,
) -> None:
    """Copy base cache into a dimension-specific directory, stripping old representation.

    This allows ensure_replicate_cache()'s augmentation logic to train only
    the VAE/PCA at the new dimension, reusing all preprocessing and splits.

    Parameters
    ----------
    base_cache_dir : str
        Path to base cache directory (e.g. "replicate_cache_seed123")
    dim_cache_dir : str
        Path to dimension-specific directory (e.g. "replicate_cache_seed123_vae_d8")
    rep_id : int
        Replicate index
    space_tag : str
        "vae" or "pca" — determines which representation keys to strip
    """
    src = os.path.join(base_cache_dir, f"rep{rep_id:02d}", "assets.npz")
    dst_dir = os.path.join(dim_cache_dir, f"rep{rep_id:02d}")
    dst = os.path.join(dst_dir, "assets.npz")

    # Skip if destination already exists (resume-safe)
    if os.path.exists(dst):
        return

    if not os.path.exists(src):
        # Base cache doesn't exist yet — fall back to full build
        print(f"[cache] rep{rep_id:02d}: Base cache not found at {src}, "
              f"will build dimension cache from scratch", flush=True)
        return

    ensure_dir(dst_dir)

    # Load base cache, strip old representation keys
    data = dict(np.load(src, allow_pickle=True))
    if space_tag == "vae":
        data.pop("Z_vae", None)
        data.pop("Z_logvar", None)
    elif space_tag == "pca":
        data.pop("Z_pca", None)

    # Save stripped copy — ensure_replicate_cache will augment with new dim
    print(f"[cache] rep{rep_id:02d}: Seeding {dim_cache_dir} from base cache "
          f"(stripped {space_tag} keys)", flush=True)
    atomic_savez(dst, **data)


def ensure_replicate_cache(cfg: ExperimentConfig, rep_id: int) -> str:
    """Ensure a replicate cache exists and includes required representations.

    This function is safe to call from multiple processes: it uses a lock file
    to avoid concurrent writes, and it *augments* an existing cache by adding
    missing representations (VAE/PCA) rather than overwriting and deleting keys.
    """
    # Determine cache path (prefer explicit cfg.files.cache_path)
    cache_dir = os.path.join(cfg.files.cache_dir, f"rep{rep_id:02d}")
    ensure_dir(cache_dir)
    cache_path = cfg.files.cache_path or os.path.join(cache_dir, "assets.npz")
    lock_path = os.path.join(cache_dir, ".assets_build.lock")

    # Determine required keys for this run.
    required: List[str] = [
        "X_raw",
        "X_scaled",
        "state_labels",
        "train_idx",
        "val_idx",
        "eval_idx",
        "eval_train_idx",
        "eval_test_idx",
    ]
    if int(getattr(cfg.vae, "epochs", 0)) > 0:
        required.extend(["Z_vae", "Z_logvar"])
    if int(getattr(cfg.pca, "n_components", 0)) > 0:
        required.append("Z_pca")

    # Fast path: cache exists and is valid.
    valid, missing = validate_cache(cache_path, required)
    if valid:
        # Shape-check: verify representation dimensions match config.
        # Without this, a cache with Z_vae@D=16 would be silently reused
        # even when the config requests D=8 (dimension sweep bug).
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                vae_dim = int(getattr(cfg.vae, "latent_dim", 0))
                if vae_dim > 0 and "Z_vae" in data.files:
                    actual_d = data["Z_vae"].shape[1]
                    if actual_d != vae_dim:
                        print(f"[cache] rep{rep_id:02d}: Z_vae dimension mismatch "
                              f"(cached={actual_d}, requested={vae_dim}), will retrain",
                              flush=True)
                        valid = False
                pca_dim = int(getattr(cfg.pca, "n_components", 0))
                if pca_dim > 0 and "Z_pca" in data.files:
                    actual_d = data["Z_pca"].shape[1]
                    if actual_d != pca_dim:
                        print(f"[cache] rep{rep_id:02d}: Z_pca dimension mismatch "
                              f"(cached={actual_d}, requested={pca_dim}), will retrain",
                              flush=True)
                        valid = False
        except Exception:
            pass  # Fall through to normal validation below

    if valid:
        print(f"[cache] rep{rep_id:02d}: Reusing existing cache (all representations present)", flush=True)
        return cache_path

    fd: Optional[int] = None
    try:
        fd = _acquire_build_lock(lock_path)

        # Re-check after acquiring lock.
        valid, missing = validate_cache(cache_path, required)
        if valid:
            # Same shape check as fast path above
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    vae_dim = int(getattr(cfg.vae, "latent_dim", 0))
                    if vae_dim > 0 and "Z_vae" in data.files:
                        if data["Z_vae"].shape[1] != vae_dim:
                            valid = False
                    pca_dim = int(getattr(cfg.pca, "n_components", 0))
                    if pca_dim > 0 and "Z_pca" in data.files:
                        if data["Z_pca"].shape[1] != pca_dim:
                            valid = False
            except Exception:
                pass
        if valid:
            print(f"[cache] rep{rep_id:02d}: Reusing existing cache (built by another process)", flush=True)
            return cache_path

        # If cache missing entirely, build it from scratch for this run.
        if not os.path.exists(cache_path):
            print(f"[cache] rep{rep_id:02d}: Building new cache (no existing cache found)", flush=True)
            build_replicate_cache(cfg, rep_id)
            valid2, missing2 = validate_cache(cache_path, required)
            if not valid2:
                raise RuntimeError(f"Cache build failed; missing keys: {missing2}")
            return cache_path

        # Otherwise, augment existing cache with missing representations.
        # Handle potential corruption from interrupted writes.
        try:
            existing: Dict[str, np.ndarray] = dict(np.load(cache_path, allow_pickle=True))
        except Exception:
            # Corrupted cache: delete and rebuild
            print(f"[cache] rep{rep_id:02d}: Cache corrupted, rebuilding from scratch", flush=True)
            try:
                os.remove(cache_path)
            except OSError:
                pass
            build_replicate_cache(cfg, rep_id)
            valid2, missing2 = validate_cache(cache_path, required)
            if not valid2:
                raise RuntimeError(f"Cache rebuild after corruption failed; missing keys: {missing2}")
            return cache_path

        # Core arrays must exist; if they don't, rebuild.
        core_missing = [k for k in ["X_scaled", "train_idx", "val_idx"] if k not in existing]
        if core_missing:
            print(f"[cache] rep{rep_id:02d}: Core arrays missing, rebuilding from scratch", flush=True)
            build_replicate_cache(cfg, rep_id)
            try:
                existing = dict(np.load(cache_path, allow_pickle=True))
            except Exception:
                raise RuntimeError(f"Cache reload failed after rebuild; corruption suspected")

        seed = int(cfg.seed + rep_id)
        augmented = False

        # Add VAE embeddings if required and missing OR wrong dimension.
        vae_dim = int(getattr(cfg.vae, "latent_dim", 0))
        need_vae = (
            int(getattr(cfg.vae, "epochs", 0)) > 0
            and (
                "Z_vae" not in existing
                or "Z_logvar" not in existing
                or (vae_dim > 0 and "Z_vae" in existing
                    and existing["Z_vae"].shape[1] != vae_dim)
            )
        )
        if need_vae:
            import torch
            from ..models.vae import VAETrainer

            # Safety guard: verify no target leakage before training on cached data
            if "feature_names" in existing:
                validate_no_leakage(list(existing["feature_names"]))

            print(f"[cache] rep{rep_id:02d}: Augmenting cache with VAE embeddings (missing from existing cache)", flush=True)
            X_scaled = np.asarray(existing["X_scaled"], dtype=np.float32)
            train_idx = np.asarray(existing["train_idx"], dtype=int)
            val_idx = np.asarray(existing.get("val_idx"), dtype=int)
            use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
            device = torch.device(cfg.device if use_cuda else "cpu")

            # Reconstruct ColumnSpec from cached arrays (if available)
            column_spec = None
            if bool(getattr(cfg.vae, "use_mixed_likelihood", False)):
                if "colspec_continuous_idx" in existing and "colspec_binary_idx" in existing:
                    from ..models._vae_networks import ColumnSpec
                    cat_specs_arr = existing.get("colspec_cat_specs", None)
                    cat_specs_list = []
                    if cat_specs_arr is not None and cat_specs_arr.size > 0:
                        cat_specs_list = [(int(row[0]), int(row[1])) for row in cat_specs_arr]
                    column_spec = ColumnSpec(
                        continuous_idx=np.asarray(existing["colspec_continuous_idx"], dtype=np.int64),
                        binary_idx=np.asarray(existing["colspec_binary_idx"], dtype=np.int64),
                        categorical_specs=cat_specs_list,
                        n_total=int(existing.get("colspec_n_total", X_scaled.shape[1])),
                    )
                    print(
                        f"[cache] rep{rep_id:02d}: Reconstructed ColumnSpec from cache "
                        f"({column_spec.n_continuous} cont, {column_spec.n_binary} bin, "
                        f"{column_spec.n_categorical} cat)",
                        flush=True,
                    )
                else:
                    # No saved ColumnSpec — build from feature metadata
                    feat_types = list(existing.get("feature_types", []))
                    scale_mask = list(existing.get("scale_mask", []))
                    feat_names = list(existing.get("feature_names", []))
                    X_raw = np.asarray(existing.get("X_raw", X_scaled), dtype=np.float32)
                    # Try to load category maps from cache
                    cat_maps = {}
                    if "category_maps_json" in existing:
                        import json as _json_aug
                        try:
                            cat_maps = _json_aug.loads(str(existing["category_maps_json"]))
                        except Exception:
                            pass
                    if feat_types and scale_mask:
                        column_spec = _build_column_spec(
                            feature_types=feat_types,
                            scale_mask=scale_mask,
                            X_raw=X_raw,
                            feature_names=feat_names,
                            category_maps=cat_maps,
                        )

            vae_trainer = VAETrainer(cfg.vae, seed, device, column_spec=column_spec)
            vae_trainer.train(X_scaled[train_idx], X_val=X_scaled[val_idx] if val_idx.size else None)
            Z_vae, Z_logvar = vae_trainer.embed(X_scaled)
            existing["Z_vae"] = np.asarray(Z_vae, dtype=np.float32)
            existing["Z_logvar"] = np.asarray(Z_logvar, dtype=np.float32)

            # Persist column_spec in augmented cache for future use
            if column_spec is not None and "colspec_continuous_idx" not in existing:
                existing["colspec_continuous_idx"] = column_spec.continuous_idx
                existing["colspec_binary_idx"] = column_spec.binary_idx
                if column_spec.categorical_specs:
                    existing["colspec_cat_specs"] = np.array(
                        column_spec.categorical_specs, dtype=np.int64,
                    )
                existing["colspec_n_total"] = np.array(column_spec.n_total, dtype=np.int64)

            augmented = True

        # Add PCA embeddings if required and missing OR wrong dimension.
        pca_dim = int(getattr(cfg.pca, "n_components", 0))
        need_pca = (
            pca_dim > 0
            and (
                "Z_pca" not in existing
                or ("Z_pca" in existing
                    and existing["Z_pca"].shape[1] != pca_dim)
            )
        )
        if need_pca:
            # Safety guard: verify no target leakage before training on cached data
            if "feature_names" in existing:
                validate_no_leakage(list(existing["feature_names"]))

            print(f"[cache] rep{rep_id:02d}: Augmenting cache with PCA embeddings (missing from existing cache)", flush=True)
            X_scaled = np.asarray(existing["X_scaled"], dtype=np.float32)
            train_idx = np.asarray(existing["train_idx"], dtype=int)
            pca = PCA(
                n_components=int(cfg.pca.n_components),
                whiten=bool(cfg.pca.whiten),
                random_state=seed,
            )
            pca.fit(X_scaled[train_idx])
            existing["Z_pca"] = pca.transform(X_scaled).astype(np.float32)
            augmented = True

        # Write back merged cache atomically (only if augmented).
        if augmented:
            atomic_savez(cache_path, **existing)

        valid3, missing3 = validate_cache(cache_path, required)
        if not valid3:
            raise RuntimeError(f"Cache augmentation failed; missing keys: {missing3}")

        return cache_path
    finally:
        _release_build_lock(lock_path, fd)


def prebuild_full_cache(
    base_cfg: ExperimentConfig,
    rep_id: int,
    *,
    seed: int = 123,
    force_rebuild: bool = False,
) -> str:
    """
    Pre-build a replicate cache with ALL representations (VAE + PCA).

    This function should be called ONCE per replicate BEFORE running any
    experiment configurations.  It guarantees that the cache contains both
    VAE and PCA embeddings so that all subsequent experiment runs (R1-R11)
    reuse the exact same representations, ensuring consistency and
    comparability across experimental conditions.

    Parameters
    ----------
    base_cfg : ExperimentConfig
        Base configuration (must have valid vae and pca sub-configs with
        the desired hyperparameters, e.g. vae.epochs=1500, pca.n_components=32).
    rep_id : int
        Replicate identifier.
    seed : int
        Base random seed.  The actual seed used for this replicate is
        ``seed + rep_id`` (matching the convention in run_scenario).
    force_rebuild : bool
        If True, delete existing cache and rebuild from scratch.

    Returns
    -------
    str
        Path to the (possibly pre-existing) cache file.
    """
    from dataclasses import replace as dc_replace

    # Build a "full" config that requests BOTH VAE and PCA.
    full_cfg = dc_replace(
        base_cfg,
        rep_id=rep_id,
        seed=int(seed + rep_id),
        vae=dc_replace(base_cfg.vae, epochs=max(int(base_cfg.vae.epochs), 1500)),
        pca=dc_replace(base_cfg.pca, n_components=max(int(base_cfg.pca.n_components), 16)),
    )

    cache_dir = os.path.join(full_cfg.files.cache_dir, f"rep{rep_id:02d}")
    ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, "assets.npz")

    if force_rebuild:
        try:
            os.remove(cache_path)
        except OSError:
            pass

    # Full set of keys that any experiment could need.
    full_required: List[str] = [
        "X_raw", "X_scaled", "state_labels",
        "train_idx", "val_idx",
        "eval_idx", "eval_train_idx", "eval_test_idx",
        "Z_vae", "Z_logvar",
        "Z_pca",
    ]

    valid, missing = validate_cache(cache_path, full_required)
    if valid and not force_rebuild:
        print(
            f"[cache] rep{rep_id:02d}: Full cache already exists "
            f"(VAE + PCA present) — skipping rebuild",
            flush=True,
        )
        return cache_path

    print(
        f"[cache] rep{rep_id:02d}: Pre-building full cache "
        f"(VAE + PCA) with seed={full_cfg.seed}",
        flush=True,
    )

    # Use ensure_replicate_cache which handles locking and augmentation.
    full_cfg_with_path = dc_replace(
        full_cfg,
        files=dc_replace(full_cfg.files, cache_path=cache_path),
    )
    result_path = ensure_replicate_cache(full_cfg_with_path, rep_id)

    # Verify completeness.
    valid2, missing2 = validate_cache(result_path, full_required)
    if not valid2:
        raise RuntimeError(
            f"Pre-build for rep{rep_id:02d} incomplete; missing keys: {missing2}"
        )

    return result_path


def load_replicate_cache(asset_path: str) -> ReplicateAssets:
    """
    Load cached replicate assets from disk.

    Parameters
    ----------
    asset_path : str
        Path to .npz cache file

    Returns
    -------
    ReplicateAssets
        Loaded assets
    """
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Cache file not found: {asset_path}")

    data = np.load(asset_path, allow_pickle=True)

    X_raw = data["X_raw"]
    X_scaled = data["X_scaled"]
    state_labels = data["state_labels"]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"] if "val_idx" in data.files else None

    eval_idx = data["eval_idx"] if "eval_idx" in data.files else None
    eval_train_idx = data["eval_train_idx"] if "eval_train_idx" in data.files else None
    eval_test_idx = data["eval_test_idx"] if "eval_test_idx" in data.files else None

    Z_vae = data["Z_vae"] if "Z_vae" in data.files else None
    Z_logvar = data["Z_logvar"] if "Z_logvar" in data.files else None
    Z_pca = data["Z_pca"] if "Z_pca" in data.files else None
    y = data["y"] if "y" in data.files else None

    metadata = {}
    if "y_4G" in data.files:
        metadata["y_4G"] = data["y_4G"]
    if "y_5G" in data.files:
        metadata["y_5G"] = data["y_5G"]

    # Load extra coverage targets (manuscript Table IV)
    extra_targets: dict = {}
    if "extra_target_names" in data.files:
        names = list(data["extra_target_names"])
        for name in names:
            key = f"y_extra_{name}"
            if key in data.files:
                extra_targets[str(name)] = data[key]
    metadata["extra_targets"] = extra_targets

    # Load derived extra regression targets (beyond coverage)
    extra_reg_targets: dict = {}
    if "extra_reg_target_names" in data.files:
        names = list(data["extra_reg_target_names"])
        for name in names:
            key = f"y_extreg_{name}"
            if key in data.files:
                extra_reg_targets[str(name)] = np.asarray(data[key], dtype=np.float64)
    metadata["extra_regression_targets"] = extra_reg_targets

    # Load derived classification targets
    cls_targets: dict = {}
    if "cls_target_names" in data.files:
        names = list(data["cls_target_names"])
        for name in names:
            key = f"y_cls_{name}"
            if key in data.files:
                cls_targets[str(name)] = np.asarray(data[key], dtype=np.int64)
    metadata["classification_targets"] = cls_targets

    # Load QoS target (qf_mean — Qualidade do Funcionamento)
    if "qos_target" in data.files:
        metadata["qos_target"] = np.asarray(data["qos_target"], dtype=np.float64)

    # Mixed-type VAE column specification
    if "colspec_continuous_idx" in data.files:
        metadata["colspec_continuous_idx"] = np.asarray(data["colspec_continuous_idx"], dtype=np.int64)
    if "colspec_binary_idx" in data.files:
        metadata["colspec_binary_idx"] = np.asarray(data["colspec_binary_idx"], dtype=np.int64)
    if "colspec_cat_specs" in data.files:
        metadata["colspec_cat_specs"] = np.asarray(data["colspec_cat_specs"], dtype=np.int64)
    if "colspec_n_total" in data.files:
        metadata["colspec_n_total"] = int(data["colspec_n_total"])

    # Geo metadata for plotting
    if "latitude" in data.files:
        metadata["latitude"] = data["latitude"]
    if "longitude" in data.files:
        metadata["longitude"] = data["longitude"]
    if "population" in data.files:
        metadata["population"] = data["population"]

    # Preprocessing metadata (manuscript Section 5.7)
    if "feature_names" in data.files:
        metadata["feature_names"] = list(data["feature_names"])
    if "missing_feature_names" in data.files:
        metadata["missing_feature_names"] = list(data["missing_feature_names"])
    if "log1p_feature_names" in data.files:
        metadata["log1p_feature_names"] = list(data["log1p_feature_names"])
    # Phase 4.3: record which target columns were excluded (audit trail)
    if "removed_target_columns" in data.files:
        metadata["removed_target_columns"] = list(data["removed_target_columns"])

    # Phase 2: Feature type metadata
    feature_types_list = []
    if "feature_types" in data.files:
        feature_types_list = list(data["feature_types"])
        metadata["feature_types"] = feature_types_list

    # Phase 2: Category maps (JSON-serialized)
    category_maps = {}
    if "category_maps_json" in data.files:
        import json
        try:
            raw_json = str(data["category_maps_json"])
            category_maps = json.loads(raw_json)
        except (json.JSONDecodeError, ValueError):
            category_maps = {}
    metadata["category_maps"] = category_maps

    # Phase 2: Scale mask
    if "scale_mask" in data.files:
        metadata["scale_mask"] = list(data["scale_mask"])

    # Phase 2: Target type
    target_type = "regression"
    if "target_type" in data.files:
        target_type = str(data["target_type"])
    metadata["target_type"] = target_type

    # Phase 3: Per-type column lists
    if "categorical_columns" in data.files:
        metadata["categorical_columns"] = list(data["categorical_columns"])
    if "ordinal_columns" in data.files:
        metadata["ordinal_columns"] = list(data["ordinal_columns"])
    if "numeric_columns" in data.files:
        metadata["numeric_columns"] = list(data["numeric_columns"])

    # Phase 3: Imputation statistics (JSON-serialized)
    impute_values = {}
    if "impute_values_json" in data.files:
        import json as _json_load
        try:
            raw_json = str(data["impute_values_json"])
            impute_values = _json_load.loads(raw_json)
        except (ValueError, _json_load.JSONDecodeError):
            impute_values = {}
    metadata["impute_values"] = impute_values

    # Extract population weights as a first-class field (Phase 2)
    population = metadata.pop("population", None)
    if population is not None:
        population = np.asarray(population, dtype=np.float64)

    return ReplicateAssets(
        X_raw=X_raw,
        X_scaled=X_scaled,
        Z_vae=Z_vae,
        Z_logvar=Z_logvar,
        Z_pca=Z_pca,
        state_labels=state_labels,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        eval_train_idx=eval_train_idx,
        eval_test_idx=eval_test_idx,
        y=y,
        population=population,
        metadata=metadata,
        feature_types=feature_types_list,
        category_maps=category_maps,
        target_type=target_type,
    )
