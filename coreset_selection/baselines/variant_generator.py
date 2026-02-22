"""Structured baseline variant generation (G7, manuscript Sec VII.6 / R10).

Provides :class:`BaselineVariantGenerator`, a higher-level driver that
creates *both* unconstrained (exact-k) and quota-matched variants of every
baseline, tracks which quota vector ``c*(k)`` was used, records per-method
timing, and emits a structured summary CSV.

Usage::

    gen = BaselineVariantGenerator(
        geo=geo, projector=projector, k=200, alpha_geo=1.0,
        rff_dim=2000, seed=42, min_one_per_group=True,
    )
    rows = gen.run_all(
        spaces={"raw": X_raw, "vae": Z_vae},
        evaluator_fn=lambda idx: runner._evaluate_coreset(idx_sel=idx, ...),
    )
    gen.save_summary(rows, output_dir="runs_out/R10/rep00/results")
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import rff_features

# Import registry, pairs, and result container from helpers and re-export
from ._vg_helpers import (
    METHOD_REGISTRY,
    VARIANT_PAIRS,
    POP_QUOTA_PAIRS,
    JOINT_QUOTA_PAIRS,
    BaselineResult,
)


class BaselineVariantGenerator:
    """Structured driver for generating baseline variants (G7).

    Encapsulates the logic in ``ExperimentRunner._run_baselines`` but adds:

    * A *registry* that pairs exact-k methods with their quota counterparts.
    * Per-method **timing** (wall-clock seconds).
    * Records which **quota vector** ``c*(k)`` was applied to each variant.
    * A ``save_summary`` helper that writes a structured CSV with all
      metadata columns, ready for ``tab_baseline_summary`` and downstream
      analysis.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group information (with ``pi``, ``population_weights``).
    projector : GeographicConstraintProjector
        Used to enforce feasibility on the raw selection masks.
    k : int
        Target coreset size.
    alpha_geo : float
        Dirichlet smoothing parameter for quota computation.
    rff_dim : int
        Random Fourier Feature dimension (for kernel baselines).
    seed : int
        Base random seed.
    min_one_per_group : bool
        Whether the quota enforces >=1 per group (lower bound l_g = 1).
    bandwidth_mult : float
        Multiplier on the median-heuristic bandwidth (default 1.0).
    """

    def __init__(
        self,
        *,
        geo,
        projector,
        k: int,
        alpha_geo: float = 1.0,
        rff_dim: int = 2000,
        seed: int = 42,
        min_one_per_group: bool = True,
        bandwidth_mult: float = 1.0,
    ):
        self.geo = geo
        self.projector = projector
        self.k = int(k)
        self.alpha_geo = float(alpha_geo)
        self.rff_dim = int(rff_dim)
        self.seed = int(seed)
        self.min_one = bool(min_one_per_group)
        self.bw_mult = float(bandwidth_mult)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_all(
        self,
        spaces: Dict[str, np.ndarray],
        evaluator_fn: Optional[Callable[[np.ndarray], Dict[str, Any]]] = None,
        save_indices_fn: Optional[Callable[[str, np.ndarray, Dict], None]] = None,
        regimes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run all baseline variants across all supplied spaces.

        Parameters
        ----------
        spaces : dict
            Mapping ``space_name -> X`` (feature matrices).
        evaluator_fn : callable, optional
            ``f(idx_sel) -> metrics_dict``.  Called on every selection.
        save_indices_fn : callable, optional
            ``f(name, indices, metadata)``.  Called to persist each coreset.
        regimes : list of str, optional
            Which regimes to run.  Defaults to ``["exactk", "quota"]`` for
            backward compatibility.  Supported values:
            ``"exactk"``, ``"quota"``, ``"pop_quota"``, ``"joint_quota"``.

        Returns
        -------
        list of dict
            One row per (method, space) combination.
        """
        from .utils import rff_features as _rff
        from ..utils.math import median_sq_dist
        from ..geo.projector import GeographicConstraintProjector

        if regimes is None:
            regimes = ["exactk", "quota"]

        rows: List[Dict[str, Any]] = []
        seed = self.seed
        k = self.k
        geo = self.geo

        # Build per-regime projectors lazily
        _projectors: Dict[str, GeographicConstraintProjector] = {}

        def _get_projector(regime: str) -> GeographicConstraintProjector:
            if regime not in _projectors:
                if regime in ("exactk", "quota"):
                    _projectors[regime] = self.projector
                elif regime == "pop_quota":
                    _projectors[regime] = GeographicConstraintProjector(
                        geo=geo,
                        alpha_geo=self.alpha_geo,
                        min_one_per_group=self.min_one,
                        weight_type="pop",
                    )
                elif regime == "joint_quota":
                    # For joint, we need both projectors; store pop here
                    _projectors[regime] = GeographicConstraintProjector(
                        geo=geo,
                        alpha_geo=self.alpha_geo,
                        min_one_per_group=self.min_one,
                        weight_type="pop",
                    )
                else:
                    _projectors[regime] = self.projector
            return _projectors[regime]

        # Pre-build the muni projector for joint enforcement
        muni_projector = self.projector  # default projector uses muni

        for space_name, Xs in spaces.items():
            Xs = np.asarray(Xs, dtype=np.float64)
            n = Xs.shape[0]

            # Bandwidth via median heuristic sigma^2 = median(d^2) / 2
            sigma_sq = median_sq_dist(
                Xs, sample_size=2048,
                seed=seed + hash(space_name) % 997,
            ) / 2.0
            sigma_sq = float(max(sigma_sq * self.bw_mult, 1e-12))

            # Pre-compute shared RFF features
            Phi = _rff(Xs, m=self.rff_dim, sigma_sq=sigma_sq, seed=seed + 17)
            mean_phi = Phi.mean(axis=0)
            meanK_approx = Phi @ mean_phi

            # Build method factories per regime
            regime_methods: List[Tuple[str, Dict[str, Callable]]] = []

            if "exactk" in regimes:
                exact_methods = self._build_exact_methods(
                    Xs, n, k, sigma_sq, Phi, meanK_approx, seed,
                )
                regime_methods.append(("exactk", exact_methods))

            if "quota" in regimes:
                quota_methods = self._build_quota_methods(
                    Xs, Phi, geo, k, sigma_sq, seed,
                )
                regime_methods.append(("quota", quota_methods))

            if "pop_quota" in regimes:
                pop_methods = self._build_pop_quota_methods(
                    Xs, Phi, geo, k, sigma_sq, seed,
                )
                regime_methods.append(("pop_quota", pop_methods))

            if "joint_quota" in regimes:
                joint_methods = self._build_joint_quota_methods(
                    Xs, Phi, geo, k, sigma_sq, seed,
                )
                regime_methods.append(("joint_quota", joint_methods))

            # Compute quota vectors for metadata
            quota_vector_muni = self._compute_quota_vector(k, weight_type="muni")
            quota_vector_pop = (
                self._compute_quota_vector(k, weight_type="pop")
                if any(r in regimes for r in ("pop_quota", "joint_quota"))
                else None
            )

            # Run all methods
            for regime, methods in regime_methods:
                proj = _get_projector(regime)

                for short_name, fn in methods.items():
                    info = METHOD_REGISTRY.get(short_name, {})
                    full_name = info.get("full_name", short_name)

                    t0 = time.perf_counter()
                    try:
                        sel = np.asarray(fn(), dtype=int)
                    except Exception as exc:
                        print(f"[BaselineVariantGenerator] {short_name} "
                              f"({space_name}/{regime}) FAILED: {exc}")
                        continue
                    wall = time.perf_counter() - t0

                    # Enforce feasibility
                    mask = np.zeros(n, dtype=bool)
                    mask[sel] = True

                    if regime == "exactk":
                        mask = self.projector.project_to_exact_k_mask(
                            mask, k=k, rng=self.rng,
                        )
                    elif regime == "quota":
                        mask = self.projector.project_to_quota_mask(
                            mask, k=k, rng=self.rng,
                        )
                    elif regime == "pop_quota":
                        mask = proj.project_to_quota_mask(
                            mask, k=k, rng=self.rng,
                        )
                    elif regime == "joint_quota":
                        # Two-pass: first enforce pop-share quotas,
                        # then enforce muni-share quotas
                        mask = proj.project_to_quota_mask(
                            mask, k=k, rng=self.rng,
                        )
                        mask = muni_projector.project_to_quota_mask(
                            mask, k=k, rng=self.rng,
                        )

                    sel = np.flatnonzero(mask)

                    # Evaluate
                    metrics: Dict[str, Any] = {}
                    if evaluator_fn is not None:
                        try:
                            metrics = evaluator_fn(sel)
                        except Exception:
                            pass

                    # Choose quota vector for metadata
                    if regime == "pop_quota":
                        qv = quota_vector_pop
                    elif regime in ("quota", "joint_quota"):
                        qv = quota_vector_muni
                    else:
                        qv = None

                    result = BaselineResult(
                        method=short_name,
                        full_name=full_name,
                        regime=regime,
                        space=space_name,
                        k=k,
                        selected_indices=sel,
                        wall_time_s=wall,
                        quota_vector=qv,
                        metrics=metrics,
                    )

                    rows.append(result.to_row())

                    # Persist coreset indices
                    if save_indices_fn is not None:
                        try:
                            save_indices_fn(
                                f"{short_name}_{space_name}_{regime}",
                                sel,
                                result.to_row(),
                            )
                        except Exception:
                            pass

        return rows

    def save_summary(
        self,
        rows: List[Dict[str, Any]],
        output_dir: str,
        filename: str = "baseline_variants_summary.csv",
    ) -> str:
        """Write a structured CSV summarising all baseline variant results.

        The CSV includes: method, full_name, regime, space, k, k_actual,
        wall_time_s, quota_vector_used, plus all evaluation metrics.

        Parameters
        ----------
        rows : list of dict
            Output of :meth:`run_all`.
        output_dir : str
            Directory to write the file.
        filename : str
            CSV filename (default ``baseline_variants_summary.csv``).

        Returns
        -------
        str
            Path to the written CSV.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)

        if not rows:
            return path

        # Collect all keys
        all_keys: List[str] = []
        seen = set()
        # Fixed columns first
        fixed = [
            "method", "full_name", "regime", "space",
            "k", "k_actual", "wall_time_s", "quota_vector_used",
        ]
        for col in fixed:
            if col not in seen:
                all_keys.append(col)
                seen.add(col)
        # Then metric columns (sorted)
        metric_keys = set()
        for row in rows:
            metric_keys.update(k for k in row if k not in seen)
        for col in sorted(metric_keys):
            all_keys.append(col)
            seen.add(col)

        with open(path, "w") as f:
            f.write(",".join(all_keys) + "\n")
            for row in rows:
                vals = [str(row.get(c, "")) for c in all_keys]
                f.write(",".join(vals) + "\n")

        return path

    @staticmethod
    def paired_comparison_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a paired comparison table (exact-k vs quota-matched).

        For each ``(method_pair, space)`` combination, computes the delta
        of key metrics between the exact-k and quota-matched variant.

        Returns
        -------
        list of dict
            One row per pair x space with ``delta_*`` columns.
        """
        # Index rows by (method, space)
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            by_key[(r["method"], r["space"])] = r

        compare_metrics = [
            "nystrom_error", "kpca_distortion", "geo_kl",
            "krr_rmse_4G", "krr_rmse_5G",
        ]

        paired_rows: List[Dict[str, Any]] = []
        for exact_code, quota_code in VARIANT_PAIRS:
            for space in {r["space"] for r in rows}:
                exact = by_key.get((exact_code, space))
                quota = by_key.get((quota_code, space))
                if exact is None or quota is None:
                    continue

                pr: Dict[str, Any] = {
                    "exactk_method": exact_code,
                    "quota_method": quota_code,
                    "space": space,
                    "k": exact.get("k", ""),
                }
                for m in compare_metrics:
                    ev = exact.get(m)
                    qv = quota.get(m)
                    if ev is not None and qv is not None:
                        try:
                            pr[f"exactk_{m}"] = float(ev)
                            pr[f"quota_{m}"] = float(qv)
                            pr[f"delta_{m}"] = float(qv) - float(ev)
                        except (TypeError, ValueError):
                            pass
                paired_rows.append(pr)

        return paired_rows

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_quota_vector(
        self, k: int, weight_type: str = "muni",
    ) -> Optional[np.ndarray]:
        """Return the KL-optimal quota vector c*(k), or None on failure.

        Parameters
        ----------
        k : int
            Coreset size.
        weight_type : str
            ``"muni"`` or ``"pop"`` — selects the target distribution.
        """
        try:
            from ..geo.kl import min_achievable_geo_kl_bounded
            pi = self.geo.get_target_distribution(weight_type)
            _, counts = min_achievable_geo_kl_bounded(
                pi=np.asarray(pi, dtype=np.float64),
                group_sizes=np.asarray(self.geo.group_sizes, dtype=int),
                k=k,
                alpha_geo=self.alpha_geo,
                min_one_per_group=self.min_one,
            )
            return np.asarray(counts, dtype=int)
        except Exception:
            return None

    def _build_exact_methods(
        self, Xs, n, k, sigma_sq, Phi, meanK_approx, seed,
    ) -> Dict[str, Callable]:
        from . import (
            baseline_uniform, baseline_kmeans_reps,
            baseline_kernel_herding, baseline_farthest_first,
            baseline_rls, baseline_dpp, baseline_kernel_thinning,
            baseline_kkmeans_nystrom,
        )
        rff_dim = self.rff_dim
        return {
            "U":   lambda n=n: baseline_uniform(n, k=k, seed=seed + 1),
            "KM":  lambda: baseline_kmeans_reps(Xs, k=k, seed=seed + 2),
            "KH":  lambda: baseline_kernel_herding(
                       Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 3),
            "FF":  lambda: baseline_farthest_first(Xs, k=k, seed=seed + 4),
            "RLS": lambda: baseline_rls(
                       Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 5),
            "DPP": lambda: baseline_dpp(
                       Xs, k=k, sigma_sq=sigma_sq, rff_dim=rff_dim, seed=seed + 6),
            "KT":  lambda: baseline_kernel_thinning(
                       Xs, k=k, sigma_sq=sigma_sq, seed=seed + 7,
                       meanK=meanK_approx, meanK_rff_dim=rff_dim, unique=True),
            "KKN": lambda: baseline_kkmeans_nystrom(
                       Xs, k=k, seed=seed + 8, sigma_sq=sigma_sq),
        }

    def _build_quota_methods(
        self, Xs, Phi, geo, k, sigma_sq, seed,
    ) -> Dict[str, Callable]:
        from . import (
            baseline_uniform_quota, baseline_kmeans_reps_quota,
            baseline_kernel_herding_quota, baseline_farthest_first_quota,
            baseline_rls_quota, baseline_dpp_quota,
            baseline_kernel_thinning_quota,
            baseline_kkmeans_nystrom_quota,
        )
        ag = self.alpha_geo
        m1 = self.min_one
        rff_dim = self.rff_dim
        return {
            "SU":   lambda: baseline_uniform_quota(
                        geo=geo, k=k, alpha_geo=ag, seed=seed + 11),
            "SKM":  lambda: baseline_kmeans_reps_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 12,
                        min_one_per_group=m1),
            "SKH":  lambda: baseline_kernel_herding_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 13,
                        min_one_per_group=m1),
            "SFF":  lambda: baseline_farthest_first_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 14,
                        min_one_per_group=m1),
            "SRLS": lambda: baseline_rls_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 15,
                        min_one_per_group=m1),
            "SDPP": lambda: baseline_dpp_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 16,
                        min_one_per_group=m1),
            "SKT":  lambda: baseline_kernel_thinning_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, sigma_sq=sigma_sq,
                        seed=seed + 17, min_one_per_group=m1,
                        meanK_rff_dim=min(512, rff_dim), unique=True),
            "SKKN": lambda: baseline_kkmeans_nystrom_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 18,
                        sigma_sq=sigma_sq, min_one_per_group=m1),
        }

    def _build_pop_quota_methods(
        self, Xs, Phi, geo, k, sigma_sq, seed,
    ) -> Dict[str, Callable]:
        """Build population-share quota baselines (P-prefix).

        Identical to ``_build_quota_methods`` but the underlying baseline
        functions still receive the *same* ``geo`` object.  The difference
        is enforced at the **projection** step: ``run_all`` uses a
        population-share projector for the ``pop_quota`` regime.
        """
        from . import (
            baseline_uniform_quota, baseline_kmeans_reps_quota,
            baseline_kernel_herding_quota, baseline_farthest_first_quota,
            baseline_rls_quota, baseline_dpp_quota,
            baseline_kernel_thinning_quota,
            baseline_kkmeans_nystrom_quota,
        )
        ag = self.alpha_geo
        m1 = self.min_one
        rff_dim = self.rff_dim
        return {
            "PU":   lambda: baseline_uniform_quota(
                        geo=geo, k=k, alpha_geo=ag, seed=seed + 21),
            "PKM":  lambda: baseline_kmeans_reps_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 22,
                        min_one_per_group=m1),
            "PKH":  lambda: baseline_kernel_herding_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 23,
                        min_one_per_group=m1),
            "PFF":  lambda: baseline_farthest_first_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 24,
                        min_one_per_group=m1),
            "PRLS": lambda: baseline_rls_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 25,
                        min_one_per_group=m1),
            "PDPP": lambda: baseline_dpp_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 26,
                        min_one_per_group=m1),
            "PKT":  lambda: baseline_kernel_thinning_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, sigma_sq=sigma_sq,
                        seed=seed + 27, min_one_per_group=m1,
                        meanK_rff_dim=min(512, rff_dim), unique=True),
            "PKKN": lambda: baseline_kkmeans_nystrom_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 28,
                        sigma_sq=sigma_sq, min_one_per_group=m1),
        }

    def _build_joint_quota_methods(
        self, Xs, Phi, geo, k, sigma_sq, seed,
    ) -> Dict[str, Callable]:
        """Build joint-constrained baselines (J-prefix).

        These baselines must satisfy *both* population-share and
        municipality-share quota constraints.  The initial selection uses
        population-share quotas; the projection step in ``run_all``
        enforces both constraints (see ``joint_quota`` regime handling).
        """
        from . import (
            baseline_uniform_quota, baseline_kmeans_reps_quota,
            baseline_kernel_herding_quota, baseline_farthest_first_quota,
            baseline_rls_quota, baseline_dpp_quota,
            baseline_kernel_thinning_quota,
            baseline_kkmeans_nystrom_quota,
        )
        ag = self.alpha_geo
        m1 = self.min_one
        rff_dim = self.rff_dim
        return {
            "JU":   lambda: baseline_uniform_quota(
                        geo=geo, k=k, alpha_geo=ag, seed=seed + 31),
            "JKM":  lambda: baseline_kmeans_reps_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 32,
                        min_one_per_group=m1),
            "JKH":  lambda: baseline_kernel_herding_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 33,
                        min_one_per_group=m1),
            "JFF":  lambda: baseline_farthest_first_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 34,
                        min_one_per_group=m1),
            "JRLS": lambda: baseline_rls_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 35,
                        min_one_per_group=m1),
            "JDPP": lambda: baseline_dpp_quota(
                        Xs, Phi=Phi, geo=geo, k=k, alpha_geo=ag, seed=seed + 36,
                        min_one_per_group=m1),
            "JKT":  lambda: baseline_kernel_thinning_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, sigma_sq=sigma_sq,
                        seed=seed + 37, min_one_per_group=m1,
                        meanK_rff_dim=min(512, rff_dim), unique=True),
            "JKKN": lambda: baseline_kkmeans_nystrom_quota(
                        Xs, geo=geo, k=k, alpha_geo=ag, seed=seed + 38,
                        sigma_sq=sigma_sq, min_one_per_group=m1),
        }
