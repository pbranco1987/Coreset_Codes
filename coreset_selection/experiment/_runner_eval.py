"""
Evaluation and baseline mixin for ExperimentRunner.

Extracted from runner.py to reduce file size.  Contains:
- _build_multitarget_y  (free function)
- EvalMixin._evaluate_coreset
- EvalMixin._compute_quota_floor
- EvalMixin._run_baselines
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.dataclasses import ExperimentConfig
from ..evaluation.geo_diagnostics import dual_geo_diagnostics
from ..evaluation.raw_space import RawSpaceEvaluator
from ..geo.projector import GeographicConstraintProjector


# ------------------------------------------------------------------
# Free function (was module-level in runner.py)
# ------------------------------------------------------------------

def _build_multitarget_y(assets) -> tuple:
    """Build ``(y_multi, target_names)`` from replicate assets.

    Returns the stacked target array and corresponding list of canonical
    target names matching manuscript Table V.  If extra coverage targets
    are available in ``assets.metadata["extra_targets"]``, they are
    appended after the primary 4G/5G pair so the evaluator computes KRR
    for all indicators (manuscript Table V).

    Returns
    -------
    (y_multi, target_names)
        ``y_multi`` is ``(N, T)`` ndarray; ``target_names`` is a list of T
        strings.  If no targets are available, returns ``(None, None)``.
    """
    y_base = assets.y
    if y_base is None:
        return None, None

    y_base = np.asarray(y_base)
    if y_base.ndim == 1:
        y_base = y_base.reshape(-1, 1)

    # Start with the primary targets
    n_base = y_base.shape[1]
    if n_base == 2:
        base_names = ["cov_area_4G", "cov_area_5G"]
    elif n_base == 1:
        base_names = ["target"]
    else:
        base_names = [f"target_{i}" for i in range(n_base)]

    # Append extra coverage targets from cache metadata
    extra = {}
    if hasattr(assets, "metadata") and isinstance(assets.metadata, dict):
        extra = assets.metadata.get("extra_targets", {})

    if not extra:
        return y_base, base_names

    # Build ordered extension matching Table V order
    from ..config.constants import COVERAGE_TARGET_NAMES, _LEGACY_TARGET_KEY_MAP

    # Normalize extra keys: map legacy names to Table V names
    normalized_extra = {}
    for k, v in extra.items():
        canonical = _LEGACY_TARGET_KEY_MAP.get(k, k)
        normalized_extra[canonical] = v

    extra_names = [
        k for k in COVERAGE_TARGET_NAMES
        if k in normalized_extra and k not in base_names
    ]
    if not extra_names:
        return y_base, base_names

    extra_arrays = [np.asarray(normalized_extra[k], dtype=np.float64) for k in extra_names]
    y_multi = np.column_stack([y_base] + extra_arrays)
    all_names = base_names + extra_names

    return y_multi, all_names


# ------------------------------------------------------------------
# Mixin class
# ------------------------------------------------------------------

class EvalMixin:
    """Mixin providing evaluation and baseline methods for ExperimentRunner."""

    def _evaluate_coreset(
        self,
        *,
        idx_sel: np.ndarray,
        geo,
        k: int,
        raw_evaluator: Optional[RawSpaceEvaluator],
        state_labels: Optional[np.ndarray] = None,
        extra_regression_targets: Optional[Dict[str, np.ndarray]] = None,
        classification_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a coreset selection.

        - Geographic diagnostics: count-based (municipality-share) KL, L1, maxdev
        - Population-share diagnostics (always computed alongside municipality-share)
        - Raw-space metrics (Nyström error, kPCA distortion, KRR RMSE)
        - State-conditioned KPI stability (Kendall's τ, drift) if state_labels given
        - Manuscript Section VII KPI stability: per-state target means
        - Multi-model downstream evaluation (KNN, RF, LR, GBT) if enabled
        """
        from ..evaluation.kpi_stability import state_kpi_stability

        idx_sel = np.asarray(idx_sel, dtype=int)

        row: Dict[str, Any] = {}

        # G9: Always compute BOTH municipality-share and population-share
        # geographic diagnostics.  dual_geo_diagnostics returns legacy unsuffixed
        # keys (geo_kl, geo_l1, geo_maxdev), plus _muni and _pop suffixed keys.
        # Population-share keys are NaN when weights are unavailable.
        geo_all = dual_geo_diagnostics(
            geo, idx_sel, k, alpha=float(self.cfg.geo.alpha_geo)
        )
        row.update(geo_all)

        # Raw-space evaluation (with state-conditioned stability if labels available)
        if raw_evaluator is not None:
            if state_labels is not None:
                row.update(raw_evaluator.all_metrics_with_state_stability(idx_sel, state_labels))
            else:
                row.update(raw_evaluator.all_metrics(idx_sel))

        # Manuscript Section VII: state-conditioned KPI stability via target means
        # (complementary to the split-half RMSE stability already computed above)
        if state_labels is not None and raw_evaluator is not None and raw_evaluator.y is not None:
            try:
                kpi_stab = state_kpi_stability(
                    y=raw_evaluator.y,
                    state_labels=state_labels,
                    S_idx=idx_sel,
                )
                row.update(kpi_stab)
            except Exception:
                pass  # non-critical

            # G8: Export per-state KPI drift CSV for heatmap visualization.
            # Saved alongside other results so that ManuscriptArtifacts can
            # discover the file via glob("**/state_kpi_drift*.csv").
            try:
                from ..evaluation.kpi_stability import export_state_kpi_drift_csv
                drift_csv_path = os.path.join(
                    self.saver.results_dir,
                    f"state_kpi_drift_k{k}.csv",
                )
                export_state_kpi_drift_csv(
                    y=raw_evaluator.y,
                    state_labels=state_labels,
                    S_idx=idx_sel,
                    output_path=drift_csv_path,
                    run_id=str(self.cfg.run_id),
                    k=k,
                )
            except Exception:
                pass  # non-critical; heatmap will fall back to placeholder

        # Multi-model downstream evaluation (KNN, RF, LR, GBT)
        if (
            raw_evaluator is not None
            and getattr(self.cfg.eval, "multi_model_enabled", False)
            and (extra_regression_targets or classification_targets)
        ):
            try:
                reg_targets = extra_regression_targets or {}
                cls_targets = classification_targets or {}
                multi_metrics = raw_evaluator.multi_model_downstream(
                    S_idx=idx_sel,
                    regression_targets=reg_targets,
                    classification_targets=cls_targets,
                    seed=int(self.cfg.seed),
                )
                row.update(multi_metrics)
            except Exception as e:
                print(f"[runner] WARNING: multi-model downstream failed: {e}")

        # QoS downstream evaluation (OLS, Ridge, Elastic Net, PLS, Constrained)
        if (
            raw_evaluator is not None
            and raw_evaluator.y is not None
            and getattr(self.cfg.eval, "qos_enabled", False)
        ):
            try:
                from ..evaluation.qos_tasks import QoSConfig, qos_coreset_evaluation

                qos_cfg = QoSConfig(
                    models=list(getattr(self.cfg.eval, "qos_models", [
                        "ols", "ridge", "elastic_net", "pls",
                        "constrained", "heuristic",
                    ])),
                    run_fixed_effects=bool(getattr(
                        self.cfg.eval, "qos_run_fixed_effects", True,
                    )),
                )

                # Extract entity and time IDs from assets metadata if available
                _assets = getattr(self, "_current_assets", None)
                _meta = (
                    _assets.metadata
                    if _assets is not None and hasattr(_assets, "metadata")
                    and isinstance(_assets.metadata, dict)
                    else {}
                )
                entity_ids = _meta.get("entity_ids")
                time_ids = _meta.get("time_ids")

                qos_metrics = qos_coreset_evaluation(
                    X_full=raw_evaluator.X_raw,
                    y_full=raw_evaluator.y.ravel()
                    if raw_evaluator.y.ndim == 1
                    else raw_evaluator.y[:, 0],
                    S_idx=idx_sel,
                    eval_test_idx=raw_evaluator.eval_test_idx,
                    entity_ids=entity_ids,
                    time_ids=time_ids,
                    state_labels=state_labels,
                    config=qos_cfg,
                )
                row.update(qos_metrics)
            except Exception as e:
                print(f"[runner] WARNING: QoS downstream evaluation failed: {e}")

        return row

    # ------------------------------------------------------------------
    # R0: quota floor computation (KL_min(k) and c*(k))
    # ------------------------------------------------------------------

    def _compute_quota_floor(self, *, geo, projector: GeographicConstraintProjector, k: int) -> Dict[str, Any]:
        from ..geo.kl import kl_pi_hat_from_counts, min_achievable_geo_kl_bounded

        k = int(k)
        kl_min, counts = min_achievable_geo_kl_bounded(
            pi=np.asarray(geo.pi, dtype=np.float64),
            group_sizes=np.asarray(geo.group_sizes, dtype=int),
            k=k,
            alpha_geo=float(self.cfg.geo.alpha_geo),
            min_one_per_group=bool(self.cfg.geo.min_one_per_group),
        )

        # Unsmooothed histogram diagnostics
        pi = np.asarray(geo.pi, dtype=np.float64)
        pi_hat = counts / float(max(1, counts.sum()))
        l1 = float(np.sum(np.abs(pi - pi_hat)))
        maxdev = float(np.max(np.abs(pi - pi_hat)))

        # Smoothed KL should match kl_min but compute explicitly for auditing
        kl_smooth = float(kl_pi_hat_from_counts(pi=pi, counts=counts, k=k, alpha=float(self.cfg.geo.alpha_geo)))

        # Save the quota vector for reuse/debugging
        try:
            np.savez_compressed(
                os.path.join(self.saver.results_dir, f"quota_cstar_k{k}.npz"),
                counts=np.asarray(counts, dtype=np.int32),
            )
        except Exception:
            pass

        return {
            "run_id": self.cfg.run_id,
            "rep_id": self.cfg.rep_id,
            "k": k,
            "kl_min": float(kl_min),
            "geo_kl": kl_smooth,
            "geo_l1": l1,
            "geo_maxdev": maxdev,
            "cstar": counts.tolist(),
        }

    # ------------------------------------------------------------------
    # R6: baselines (all spaces x both regimes)
    # ------------------------------------------------------------------

    def _run_baselines(
        self,
        *,
        assets,
        geo,
        projector: GeographicConstraintProjector,
        raw_evaluator: Optional[RawSpaceEvaluator],
        seed: int,
    ) -> List[Dict[str, Any]]:
        from ..baselines import (
            baseline_uniform,
            baseline_uniform_quota,
            baseline_kmeans_reps,
            baseline_kmeans_reps_quota,
            baseline_kernel_herding,
            baseline_kernel_herding_quota,
            baseline_farthest_first,
            baseline_farthest_first_quota,
            baseline_rls,
            baseline_rls_quota,
            baseline_dpp,
            baseline_dpp_quota,
            baseline_kernel_thinning,
            baseline_kernel_thinning_quota,
            baseline_kkmeans_nystrom,
            baseline_kkmeans_nystrom_quota,
        )
        from ..baselines.utils import rff_features
        from ..utils.math import median_sq_dist

        k = int(self.cfg.solver.k)
        rng = np.random.default_rng(int(seed))

        spaces: Dict[str, np.ndarray] = {
            "raw": np.asarray(assets.X_scaled, dtype=np.float64),
        }
        if assets.Z_vae is not None:
            spaces["vae"] = np.asarray(assets.Z_vae, dtype=np.float64)
        if assets.Z_pca is not None:
            spaces["pca"] = np.asarray(assets.Z_pca, dtype=np.float64)

        print(f"[R6] Running baselines with k={k}", flush=True)
        print(f"[R6] Available spaces: {list(spaces.keys())}", flush=True)
        print(
            f"[R6] Baseline methods: U, KM, KH, FF, RLS, DPP, KT, KKN (exact-k and quota-matched)",
            flush=True,
        )

        # Shared RFF dimension matches the MMD surrogate (manuscript)
        rff_dim = int(getattr(self.cfg.mmd, "rff_dim", 2000))
        bandwidth_mult = float(getattr(self.cfg.mmd, "bandwidth_mult", 1.0))

        rows: List[Dict[str, Any]] = []

        for space_name, Xs in spaces.items():
            # Bandwidth via median heuristic σ^2=median(d^2)/2
            sigma_sq = (median_sq_dist(Xs, sample_size=2048, seed=seed + hash(space_name) % 997) / 2.0)
            sigma_sq = float(max(sigma_sq * bandwidth_mult, 1e-12))

            # Precompute RFF features once per space (for KH/RLS/DPP)
            Phi = rff_features(Xs, m=rff_dim, sigma_sq=sigma_sq, seed=seed + 17)

            # Approximate mean kernel evaluation per point for KT-SWAP.
            # For RBF kernels, k(x_i, x_j) ≈ φ(x_i)^T φ(x_j), so
            #   mean_j k(x_i,x_j) ≈ φ(x_i)^T mean_j φ(x_j)
            mean_phi = Phi.mean(axis=0)
            meanK_approx = Phi @ mean_phi

            # Exact-k baselines
            n_samples = Xs.shape[0]  # baseline_uniform expects int, not array
            exact_methods = {
                "U": lambda n=n_samples: baseline_uniform(n, k=k, seed=seed + 1),
                "KM": lambda: baseline_kmeans_reps(Xs, k=k, seed=seed + 2),
                "KH": lambda: baseline_kernel_herding(Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 3),
                "FF": lambda: baseline_farthest_first(Xs, k=k, seed=seed + 4),
                "RLS": lambda: baseline_rls(Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 5),
                "DPP": lambda: baseline_dpp(Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 6),
                "KT": lambda: baseline_kernel_thinning(
                    Xs,
                    k=k,
                    sigma_sq=sigma_sq,
                    seed=seed + 7,
                    meanK=meanK_approx,
                    meanK_rff_dim=rff_dim,
                    unique=True,
                ),
                "KKN": lambda: baseline_kkmeans_nystrom(
                    Xs, k=k, seed=seed + 8, sigma_sq=sigma_sq,
                ),
            }

            # Quota-matched baselines (KL-optimal c*(k))
            quota_methods = {
                "SU": lambda: baseline_uniform_quota(geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 11),
                "SKM": lambda: baseline_kmeans_reps_quota(
                    Xs, geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 12,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
                "SKH": lambda: baseline_kernel_herding_quota(
                    Xs, Phi=Phi, geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 13,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
                "SFF": lambda: baseline_farthest_first_quota(
                    Xs, geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 14,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
                "SRLS": lambda: baseline_rls_quota(
                    Xs, Phi=Phi, geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 15,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
                "SDPP": lambda: baseline_dpp_quota(
                    Xs, Phi=Phi, geo=geo, k=k, alpha_geo=float(self.cfg.geo.alpha_geo), seed=seed + 16,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
                "SKT": lambda: baseline_kernel_thinning_quota(
                    Xs,
                    geo=geo,
                    k=k,
                    alpha_geo=float(self.cfg.geo.alpha_geo),
                    sigma_sq=sigma_sq,
                    seed=seed + 17,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                    meanK_rff_dim=min(512, rff_dim),
                    unique=True,
                ),
                "SKKN": lambda: baseline_kkmeans_nystrom_quota(
                    Xs,
                    geo=geo,
                    k=k,
                    alpha_geo=float(self.cfg.geo.alpha_geo),
                    seed=seed + 18,
                    sigma_sq=sigma_sq,
                    min_one_per_group=bool(self.cfg.geo.min_one_per_group),
                ),
            }

            # Run both regimes
            for regime, methods in [("exactk", exact_methods), ("quota", quota_methods)]:
                print(f"[R6] Running {regime} baselines in {space_name} space...", flush=True)
                for short_name, fn in methods.items():
                    print(f"[R6]   {short_name}...", end=" ", flush=True)
                    try:
                        sel = np.asarray(fn(), dtype=int)
                        print(f"done (selected {len(sel)} points)", flush=True)
                    except Exception as e:
                        # Log the error instead of silently continuing
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Baseline {short_name} failed in {space_name}/{regime}: {e}"
                        )
                        print(f"FAILED: {e}", flush=True)
                        continue

                    # Enforce feasibility with the same projection operator used by NSGA-II
                    mask = np.zeros(Xs.shape[0], dtype=bool)
                    mask[sel] = True
                    if regime == "quota":
                        mask = projector.project_to_quota_mask(mask, k=k, rng=rng)
                    else:
                        mask = projector.project_to_exact_k_mask(mask, k=k, rng=rng)
                    sel = np.flatnonzero(mask)

                    _bl_extra_reg = assets.metadata.get("extra_regression_targets", {}) if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
                    _bl_cls = assets.metadata.get("classification_targets", {}) if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
                    row = self._evaluate_coreset(idx_sel=sel, geo=geo, k=k, raw_evaluator=raw_evaluator, state_labels=assets.state_labels, extra_regression_targets=_bl_extra_reg, classification_targets=_bl_cls)
                    row.update(
                        {
                            "run_id": self.cfg.run_id,
                            "rep_id": self.cfg.rep_id,
                            "space": space_name,
                            "method": short_name,
                            "rep_name": "baseline",
                            "k": k,
                            "constraint_regime": regime,
                        }
                    )
                    rows.append(row)

                    # Save coreset indices
                    self.saver.save_coreset(
                        name=f"{short_name}_{space_name}_{regime}",
                        indices=sel,
                        metadata=row,
                    )

        return rows
