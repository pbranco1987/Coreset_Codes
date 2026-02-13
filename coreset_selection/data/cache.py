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
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config.dataclasses import ExperimentConfig, ReplicateAssets
from ..utils.io import ensure_dir
from ..utils.debug_timing import timer
from .target_columns import detect_target_columns, remove_target_columns
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
                derived_cls = derive_classification_targets(raw_df)
                timer.checkpoint(
                    "Derived targets extracted",
                    n_extra_reg=len(derived_extra_reg),
                    n_cls=len(derived_cls),
                )

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

        if int(cfg.vae.epochs) > 0:
            with timer.section("VAE_training", epochs=cfg.vae.epochs, latent_dim=cfg.vae.latent_dim):
                use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
                device = torch.device(cfg.device if use_cuda else "cpu")
                timer.checkpoint("VAE device", device=str(device), cuda_available=torch.cuda.is_available())

                vae_trainer = VAETrainer(cfg.vae, seed, device)
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
            cat_maps = data_manager.category_maps() if hasattr(data_manager, 'category_maps') else {}
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

            # Optional plotting metadata
            if lat is not None and lon is not None:
                save_dict["latitude"] = np.asarray(lat)
                save_dict["longitude"] = np.asarray(lon)
            if pop is not None:
                save_dict["population"] = np.asarray(pop)

            atomic_savez(cache_path, **save_dict)
    return cache_path


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
        print(f"[cache] rep{rep_id:02d}: Reusing existing cache (all representations present)", flush=True)
        return cache_path

    fd: Optional[int] = None
    try:
        fd = _acquire_build_lock(lock_path)

        # Re-check after acquiring lock.
        valid, missing = validate_cache(cache_path, required)
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

        # Add VAE embeddings if required and missing.
        if int(getattr(cfg.vae, "epochs", 0)) > 0 and ("Z_vae" not in existing or "Z_logvar" not in existing):
            import torch
            from ..models.vae import VAETrainer

            print(f"[cache] rep{rep_id:02d}: Augmenting cache with VAE embeddings (missing from existing cache)", flush=True)
            X_scaled = np.asarray(existing["X_scaled"], dtype=np.float32)
            train_idx = np.asarray(existing["train_idx"], dtype=int)
            val_idx = np.asarray(existing.get("val_idx"), dtype=int)
            use_cuda = torch.cuda.is_available() and str(cfg.device).startswith("cuda")
            device = torch.device(cfg.device if use_cuda else "cpu")
            vae_trainer = VAETrainer(cfg.vae, seed, device)
            vae_trainer.train(X_scaled[train_idx], X_val=X_scaled[val_idx] if val_idx.size else None)
            Z_vae, Z_logvar = vae_trainer.embed(X_scaled)
            existing["Z_vae"] = np.asarray(Z_vae, dtype=np.float32)
            existing["Z_logvar"] = np.asarray(Z_logvar, dtype=np.float32)
            augmented = True

        # Add PCA embeddings if required and missing.
        if int(getattr(cfg.pca, "n_components", 0)) > 0 and "Z_pca" not in existing:
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
