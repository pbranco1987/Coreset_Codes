"""
Experiment runner.

This module orchestrates a single run (R#) for a single replicate.

Key responsibilities:
- Build/load replicate cache assets (data splits, standardized features, VAE/PCA reps)
- Construct objective computers in the requested representation space
- Run NSGA-II in the selected space (default: VAE latent space) with exact-k
  or quota constraints
- Evaluate representative solutions in standardized raw space (always!)
- Save Pareto fronts, selected indices, and evaluation summaries
"""

from __future__ import annotations

import os
import time as _time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.dataclasses import ExperimentConfig
from ..data.cache import ensure_replicate_cache, load_replicate_cache
from ..evaluation.geo_diagnostics import dual_geo_diagnostics
from ..evaluation.raw_space import RawSpaceEvaluator
from ..geo.info import build_geo_info
from ..geo.projector import GeographicConstraintProjector
from ..constraints import (
    ProportionalityConstraintSet,
    build_population_share_constraint,
    build_municipality_share_constraint,
)
from ..objectives.computer import build_space_objective_computer
from ..optimization.nsga2_internal import nsga2_optimize
from ..optimization.selection import select_pareto_representatives
from ..utils.random import set_global_seed
from ..utils.debug_timing import timer
from .saver import ParetoFrontData, ResultsSaver

# Mixin imports
from ._runner_eval import EvalMixin, _build_multitarget_y
from ._runner_r0 import R0Mixin
from ._runner_diagnostics import DiagnosticsMixin
from ._runner_effort import EffortMixin


class ExperimentRunner(R0Mixin, DiagnosticsMixin, EffortMixin, EvalMixin):
    """Runs a single experiment (run_id, replicate)."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.saver = ResultsSaver(cfg.files.output_dir, cfg.run_id, cfg.rep_id)
        # Always-on wall-clock tracking (independent of CORESET_DEBUG).
        self._t0: float = 0.0
        self._phases: Dict[str, float] = {}

    def _phase_start(self) -> float:
        """Return a phase start timestamp."""
        return _time.perf_counter()

    def _phase_end(self, name: str, t_start: float) -> float:
        """Record a phase's wall-clock duration and return it."""
        dur = _time.perf_counter() - t_start
        self._phases[name] = self._phases.get(name, 0.0) + dur
        return dur

    def _finalize_with_timing(
        self,
        result: Dict[str, Any],
        *,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Inject wall-clock timing into *result* dict, save ``wall_clock.json``.

        Parameters
        ----------
        result : dict
            The summary dict that ``run()`` will return.
        rows : list of dict, optional
            If provided, ``wall_clock_s`` is injected into every row
            **in-place** so that the timing appears in any CSV that was
            (or will be) written from these rows.

        Returns
        -------
        dict
            *result* with ``wall_clock_s`` and per-phase keys added.
        """
        wall_total = _time.perf_counter() - self._t0

        # Build timing record
        timing: Dict[str, Any] = {
            "wall_clock_total_s": round(wall_total, 4),
            "run_id": self.cfg.run_id,
            "rep_id": self.cfg.rep_id,
        }
        for phase_name, phase_dur in self._phases.items():
            timing[f"wall_clock_{phase_name}_s"] = round(phase_dur, 4)

        # Inject into return dict
        result["wall_clock_s"] = round(wall_total, 4)
        for phase_name, phase_dur in self._phases.items():
            result[f"wall_clock_{phase_name}_s"] = round(phase_dur, 4)

        # Inject into rows (so CSVs carry timing)
        if rows:
            for row in rows:
                row.setdefault("wall_clock_s", round(wall_total, 4))

        # Persist to disk
        try:
            self.saver.save_wall_clock(timing)
        except Exception:
            pass  # never fail a run over bookkeeping

        print(
            f"[timing] {self.cfg.run_id} rep={self.cfg.rep_id}: "
            f"total={wall_total:.2f}s  phases={{"
            + ", ".join(f"{k}={v:.2f}s" for k, v in self._phases.items())
            + "}",
            flush=True,
        )

        return result

    def run(self) -> Dict[str, Any]:
        """Run the experiment and return a small summary dict."""
        cfg = self.cfg
        rep_seed = int(cfg.seed + cfg.rep_id)
        set_global_seed(rep_seed)
        self._t0 = _time.perf_counter()
        self._phases = {}

        # Write run manifest for reproducibility
        try:
            self.saver.save_run_manifest(seed=rep_seed)
        except Exception:
            pass  # non-critical

        with timer.section("ExperimentRunner.run", run_id=cfg.run_id, rep_id=cfg.rep_id):
            # ------------------------------------------------------------
            # R0 can run without a replicate cache (quota computation only).
            # ------------------------------------------------------------
            base_run_id = str(cfg.run_id).split("_")[0]
            if base_run_id == "R0":
                return self._run_r0_quota_only(rep_seed)

            # ------------------------------------------------------------
            # Load or (safely) build/augment replicate cache.
            # This is safe under multi-process parallel execution.
            # ------------------------------------------------------------
            _tp = self._phase_start()
            with timer.section("ensure_replicate_cache"):
                cache_path = ensure_replicate_cache(cfg, cfg.rep_id)

            with timer.section("load_replicate_cache"):
                assets = load_replicate_cache(cache_path)
            self._phase_end("cache", _tp)

            timer.checkpoint("Cache loaded",
                             X_scaled_shape=assets.X_scaled.shape if assets.X_scaled is not None else None,
                             has_Z_vae=assets.Z_vae is not None,
                             has_Z_pca=assets.Z_pca is not None)

            # ------------------------------------------------------------
            # Geographic information + projector
            # Per manuscript Section IV-B, build_geo_info now accepts
            # population_weights to compute π_pop (population-share target).
            # ------------------------------------------------------------
            _tp = self._phase_start()
            with timer.section("build_geo_info"):
                pop_array = getattr(assets, "population", None)
                if pop_array is None:
                    pop_array = assets.metadata.get("population", None)
                geo = build_geo_info(
                    assets.state_labels,
                    population_weights=pop_array,
                )

            with timer.section("build_projector"):
                projector = GeographicConstraintProjector(
                    geo=geo,
                    alpha_geo=float(cfg.geo.alpha_geo),
                    min_one_per_group=bool(cfg.geo.min_one_per_group),
                )
            self._phase_end("setup_geo", _tp)

            # ------------------------------------------------------------
            # Weighted proportionality constraints (population-share / joint / muni)
            # Per manuscript Section IV-B and Table 2:
            #   population_share: w_i = pop_i (primary)
            #   municipality_share_quota: w_i ≡ 1 (count quota mode)
            #   joint: both population-share AND municipality-share quota
            #   none: no proportionality constraints (exact-k only)
            # ------------------------------------------------------------
            constraint_set: Optional[ProportionalityConstraintSet] = None
            with timer.section("build_proportionality_constraints"):
                mode = str(cfg.geo.constraint_mode)
                constraints_list = []

                if mode in {"population_share", "joint"}:
                    if geo.population_weights is not None:
                        pop = geo.population_weights
                    elif getattr(assets, "population", None) is not None:
                        pop = np.asarray(assets.population, dtype=np.float64)
                    else:
                        pop = np.ones(geo.N, dtype=np.float64)
                    c_pop = build_population_share_constraint(
                        geo=geo,
                        population=pop,
                        alpha=float(cfg.geo.alpha_geo),
                        tau=float(cfg.geo.tau_population),
                    )
                    constraints_list.append(c_pop)

                if mode == "joint":
                    c_muni = build_municipality_share_constraint(
                        geo=geo,
                        alpha=float(cfg.geo.alpha_geo),
                        tau=float(cfg.geo.tau_municipality),
                    )
                    constraints_list.append(c_muni)

                if constraints_list:
                    preserve = (mode == "joint")
                    constraint_set = ProportionalityConstraintSet(
                        geo=geo,
                        constraints=constraints_list,
                        min_one_per_group=bool(cfg.geo.min_one_per_group),
                        preserve_group_counts=preserve,
                        max_iters=200,
                    )

            # Save config for reproducibility
            self.saver.save_config(cfg)

            # ------------------------------------------------------------
            # R11: Post-hoc diagnostics
            #   Proxy stability (Table IV) + objective–metric alignment (Fig 4)
            # ------------------------------------------------------------
            if base_run_id == "R11":
                _tp = self._phase_start()
                result = self._run_r7_diagnostics(assets=assets, seed=rep_seed)
                self._phase_end("diagnostics", _tp)
                return self._finalize_with_timing(result)

            # ------------------------------------------------------------
            # R12: Effort sweep (vary NSGA-II pop_size/n_gen, log wall-clock)
            # ------------------------------------------------------------
            if base_run_id == "R12":
                result = self._run_r12_effort_sweep(
                    assets=assets,
                    geo=geo,
                    projector=projector,
                    constraint_set=constraint_set,
                    raw_evaluator=None,  # build below if eval enabled
                    seed=rep_seed,
                )
                return self._finalize_with_timing(result)

            # ------------------------------------------------------------
            # Raw-space evaluator (build once per replicate)
            # ------------------------------------------------------------
            raw_evaluator: Optional[RawSpaceEvaluator] = None
            if cfg.eval.enabled and assets.eval_idx is not None:
                _tp = self._phase_start()
                with timer.section("build_raw_evaluator"):
                    y_multi, tgt_names = _build_multitarget_y(assets)
                    raw_evaluator = RawSpaceEvaluator.build(
                        X_raw=assets.X_scaled,  # standardized raw attribute space
                        y=y_multi,
                        eval_idx=assets.eval_idx,
                        eval_train_idx=assets.eval_train_idx,
                        eval_test_idx=assets.eval_test_idx,
                        seed=rep_seed,
                        target_names=tgt_names,
                    )
                self._phase_end("build_evaluator", _tp)

            # Extract downstream targets from cache assets (for multi-model eval)
            _extra_reg_targets: Dict[str, np.ndarray] = {}
            _cls_targets: Dict[str, np.ndarray] = {}
            if hasattr(assets, "metadata") and isinstance(assets.metadata, dict):
                _extra_reg_targets = assets.metadata.get("extra_regression_targets", {})
                _cls_targets = assets.metadata.get("classification_targets", {})

            rows: List[Dict[str, Any]] = []

            # ------------------------------------------------------------
            # R10 (or legacy R6): baseline heuristic comparison (no NSGA-II)
            # ------------------------------------------------------------
            if cfg.baselines.enabled and str(cfg.run_id).startswith(("R10", "R6")):
                _tp = self._phase_start()
                rows.extend(
                    self._run_baselines(
                        assets=assets,
                        geo=geo,
                        projector=projector,
                        raw_evaluator=raw_evaluator,
                        seed=rep_seed,
                    )
                )
                self._phase_end("baselines", _tp)

                # G7: Also run the structured BaselineVariantGenerator to
                # produce the paired comparison summary CSV.
                try:
                    from ..baselines.variant_generator import BaselineVariantGenerator
                    bvg = BaselineVariantGenerator(
                        geo=geo,
                        projector=projector,
                        k=int(cfg.solver.k),
                        alpha_geo=float(cfg.geo.alpha_geo),
                        rff_dim=int(getattr(cfg.mmd, "rff_dim", 2000)),
                        seed=rep_seed,
                        min_one_per_group=bool(cfg.geo.min_one_per_group),
                        bandwidth_mult=float(getattr(cfg.mmd, "bandwidth_mult", 1.0)),
                    )
                    bvg_spaces: Dict[str, np.ndarray] = {
                        "raw": np.asarray(assets.X_scaled, dtype=np.float64),
                    }
                    if assets.Z_vae is not None:
                        bvg_spaces["vae"] = np.asarray(assets.Z_vae, dtype=np.float64)
                    if assets.Z_pca is not None:
                        bvg_spaces["pca"] = np.asarray(assets.Z_pca, dtype=np.float64)

                    bvg_rows = bvg.run_all(
                        spaces=bvg_spaces,
                        evaluator_fn=lambda idx: self._evaluate_coreset(
                            idx_sel=idx, geo=geo, k=int(cfg.solver.k),
                            raw_evaluator=raw_evaluator,
                            state_labels=assets.state_labels,
                            extra_regression_targets=_extra_reg_targets,
                            classification_targets=_cls_targets,
                        ),
                        save_indices_fn=lambda name, idx, meta: self.saver.save_coreset(
                            name=name, indices=idx, metadata=meta,
                        ),
                    )
                    # Save the structured summary CSV
                    bvg.save_summary(bvg_rows, output_dir=self.saver.results_dir)

                    # Save paired comparison
                    paired = bvg.paired_comparison_table(bvg_rows)
                    if paired:
                        bvg.save_summary(
                            paired,
                            output_dir=self.saver.results_dir,
                            filename="baseline_paired_comparison.csv",
                        )
                    print(f"[G7] Structured baseline summary saved ({len(bvg_rows)} rows)")
                except Exception as exc:
                    print(f"[G7] BaselineVariantGenerator fallback: {exc}")

                result = self._finalize_with_timing(
                    {
                        "run_id": cfg.run_id,
                        "rep_id": cfg.rep_id,
                        "space": cfg.space,
                        "n_rows": len(rows),
                        "output_dir": self.saver.run_dir,
                    },
                    rows=rows,
                )
                if rows:
                    self.saver.save_rows(rows, name="all_results")

                    # Build the cross-method comparison report (effect isolation,
                    # rank tables, dominance, stability) from the saved result
                    # rows.  This implements the evaluation protocol from the
                    # kernel k-means vs MMD+Sinkhorn analysis.
                    try:
                        from ..evaluation.method_comparison import build_comparison_report
                        comparison_dir = os.path.join(self.saver.results_dir, "comparison")
                        build_comparison_report(rows, output_dir=comparison_dir)
                        print(f"[R10] Comparison report saved to {comparison_dir}")
                    except Exception as exc:
                        print(f"[R10] Comparison report fallback: {exc}")

                return result

            # ------------------------------------------------------------
            # Objective computers (spaces)
            # ------------------------------------------------------------
            _tp = self._phase_start()
            with timer.section("build_objective_computers", space=cfg.space):
                computers = self._build_objective_computers(assets, rep_seed)
            self._phase_end("build_objectives", _tp)

            if cfg.space not in computers:
                raise ValueError(
                    f"Unknown space '{cfg.space}'. Available: {sorted(computers.keys())}"
                )

            timer.checkpoint("Ready to start NSGA-II",
                             k=cfg.solver.k,
                             pop_size=cfg.solver.pop_size,
                             n_gen=cfg.solver.n_gen,
                             objectives=cfg.solver.objectives)

            # ------------------------------------------------------------
            # NSGA-II (requested space)
            # ------------------------------------------------------------
            if cfg.solver.enabled:
                space = cfg.space
                algo = cfg.solver.get_algorithm()
                _tp = self._phase_start()
                with timer.section(f"{algo.upper()} optimization", space=space, k=cfg.solver.k):
                    pareto_data = self._run_solver_single_space(
                        computer=computers[space],
                        projector=projector,
                        constraint_set=constraint_set,
                        space=space,
                        seed=rep_seed,
                    )
                self._phase_end("solver", _tp)

            self.saver.save_pareto_front(space, pareto_data)

            # Evaluate the entire final front in raw space to enable
            # objective--metric alignment diagnostics.
            _tp = self._phase_start()
            if raw_evaluator is not None:
                method_label = cfg.solver.get_algorithm()
                front_rows: List[Dict[str, Any]] = []
                for i_pf in range(pareto_data.X.shape[0]):
                    idx_i = np.flatnonzero(pareto_data.X[i_pf])
                    r = self._evaluate_coreset(
                        idx_sel=idx_i,
                        geo=geo,
                        k=int(cfg.solver.k),
                        raw_evaluator=raw_evaluator,
                        state_labels=assets.state_labels,
                        extra_regression_targets=_extra_reg_targets,
                        classification_targets=_cls_targets,
                    )
                    fvals = pareto_data.F[i_pf]
                    for j, obj in enumerate(pareto_data.objectives):
                        r[f"f_{obj}"] = float(fvals[j])
                    r.update(
                        {
                            "run_id": cfg.run_id,
                            "rep_id": cfg.rep_id,
                            "space": space,
                            "method": method_label,
                            "rep_name": "front",
                            "pareto_index": int(i_pf),
                            "k": int(cfg.solver.k),
                            "constraint_regime": self._constraint_regime(cfg),
                        }
                    )
                    front_rows.append(r)
                self.saver.save_rows(front_rows, name=f"front_metrics_{space}")

            # Evaluate and save representative solutions
            method_label = cfg.solver.get_algorithm()
            for rep_name, rep_pf_idx in pareto_data.representatives.items():
                idx_sel = pareto_data.selected_indices[rep_name]

                row = self._evaluate_coreset(
                    idx_sel=idx_sel,
                    geo=geo,
                    k=int(cfg.solver.k),
                    raw_evaluator=raw_evaluator,
                    state_labels=assets.state_labels,
                    extra_regression_targets=_extra_reg_targets,
                    classification_targets=_cls_targets,
                )
                # Also record the optimization-space objective values for this representative
                fvals = pareto_data.F[int(rep_pf_idx)]
                for j, obj in enumerate(pareto_data.objectives):
                    row[f"f_{obj}"] = float(fvals[j])
                row.update(
                    {
                        "run_id": cfg.run_id,
                        "rep_id": cfg.rep_id,
                        "space": space,
                        "method": method_label,
                        "rep_name": rep_name,
                        "k": int(cfg.solver.k),
                        "pareto_index": int(rep_pf_idx),
                        "constraint_regime": self._constraint_regime(cfg),
                    }
                )
                rows.append(row)

                self.saver.save_coreset(
                    name=f"{space}_{rep_name}",
                    indices=idx_sel,
                    metadata=row,
                )
            self._phase_end("eval", _tp)

        # Finalize timing (injects wall_clock_s into rows) then save
        result = self._finalize_with_timing(
            {
                "run_id": cfg.run_id,
                "rep_id": cfg.rep_id,
                "space": cfg.space,
                "n_rows": len(rows),
                "output_dir": self.saver.run_dir,
            },
            rows=rows,
        )

        # Save summary rows (now with wall_clock_s injected)
        if rows:
            self.saver.save_rows(rows, name="all_results")

        return result

    # ------------------------------------------------------------------
    # Objective computer construction
    # ------------------------------------------------------------------

    def _build_objective_computers(self, assets, seed: int):
        """
        Build objective computer only for the requested representation space.

        This avoids expensive computation for spaces that won't be used.
        The default is VAE latent space; raw and PCA are ablation-only (R8/R9).

        Spaces:
        - "raw" : standardized raw features (X_scaled)
        - "vae" : VAE mean embedding (Z_vae) with logvars
        - "pca" : PCA embedding (Z_pca)
        """
        computers = {}
        requested_space = self.cfg.space

        # Only build the computer for the requested space
        if requested_space == "raw":
            computers["raw"] = build_space_objective_computer(
                X=assets.X_scaled,
                logvars=None,
                mmd_cfg=self.cfg.mmd,
                sinkhorn_cfg=self.cfg.sinkhorn,
                seed=seed,
            )
        elif requested_space == "vae":
            if assets.Z_vae is None:
                raise ValueError("VAE space requested but Z_vae is None in cache")
            computers["vae"] = build_space_objective_computer(
                X=assets.Z_vae,
                logvars=assets.Z_logvar,
                mmd_cfg=self.cfg.mmd,
                sinkhorn_cfg=self.cfg.sinkhorn,
                seed=seed,
            )
        elif requested_space == "pca":
            if assets.Z_pca is None:
                raise ValueError("PCA space requested but Z_pca is None in cache")
            computers["pca"] = build_space_objective_computer(
                X=assets.Z_pca,
                logvars=None,
                mmd_cfg=self.cfg.mmd,
                sinkhorn_cfg=self.cfg.sinkhorn,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown space: {requested_space}. Expected 'raw', 'vae', or 'pca'.")

        return computers

    # ------------------------------------------------------------------
    # NSGA-II
    # ------------------------------------------------------------------

    def _constraint_regime(self, cfg: ExperimentConfig) -> str:
        """Human-readable constraint regime label for logging/artifacts.

        Per manuscript Table II, the canonical switch is
        ``GeoConfig.constraint_mode``.
        """
        mode = str(cfg.geo.constraint_mode)
        exactk = bool(cfg.solver.enforce_exact_k)
        if mode == "joint":
            return "joint" if exactk else "joint_noexactk"
        if mode == "municipality_share_quota":
            return "quota+exactk" if exactk else "quota_only"
        if mode == "population_share":
            return "popshare+exactk" if exactk else "popshare_only"
        if mode == "none":
            return "exactk_only" if exactk else "unconstrained"
        return f"{mode}+{'exactk' if exactk else 'noexactk'}"

    def _run_solver_single_space(
        self,
        *,
        computer,
        projector: GeographicConstraintProjector,
        constraint_set: Optional[ProportionalityConstraintSet],
        space: str,
        seed: int,
    ) -> ParetoFrontData:
        cfg = self.cfg

        X_pareto, F_pareto, repair_stats = nsga2_optimize(
            computer=computer,
            projector=projector,
            constraint_set=constraint_set,
            k=int(cfg.solver.k),
            objectives=cfg.solver.objectives,
            pop_size=int(cfg.solver.pop_size),
            n_gen=int(cfg.solver.n_gen),
            crossover_prob=float(cfg.solver.crossover_prob),
            mutation_prob=float(cfg.solver.mutation_prob),
            use_quota=bool(cfg.geo.use_quota_constraints),
            enforce_exact_k=bool(cfg.solver.enforce_exact_k),
            seed=seed,
            verbose=bool(cfg.solver.verbose),
        )

        objectives = [str(o) for o in cfg.solver.objectives]

        # Persist repair activity logs for R6 diagnostics
        try:
            repair_needed = np.asarray(repair_stats.get("repair_needed", []), dtype=bool)
            repair_magnitude = np.asarray(repair_stats.get("repair_magnitude", []), dtype=np.int32)
            # Save raw arrays (compact)
            np.savez_compressed(
                os.path.join(self.saver.results_dir, f"{space}_repair_log.npz"),
                repair_needed=repair_needed,
                repair_magnitude=repair_magnitude,
            )
            # Save summary scalars
            if repair_needed.size > 0:
                rate = float(np.mean(repair_needed))
                mag_mean = float(np.mean(repair_magnitude))
                mag_std = float(np.std(repair_magnitude))
                qs = np.quantile(repair_magnitude, [0.25, 0.5, 0.75]).tolist()
            else:
                rate, mag_mean, mag_std, qs = 0.0, 0.0, 0.0, [0.0, 0.0, 0.0]
            self.saver.save_metrics(
                {
                    "repair_rate": rate,
                    "repair_magnitude_mean": mag_mean,
                    "repair_magnitude_std": mag_std,
                    "repair_magnitude_q25": qs[0],
                    "repair_magnitude_median": qs[1],
                    "repair_magnitude_q75": qs[2],
                },
                name=f"{space}_repair_summary",
            )
        except Exception:
            # Do not fail the run if logging fails
            pass

        # Select representative solutions from the Pareto set
        reps = select_pareto_representatives(F_pareto, objectives, add_pairwise_knees=True)

        selected_indices = {name: np.flatnonzero(X_pareto[i]) for name, i in reps.items()}

        return ParetoFrontData(
            F=F_pareto,
            X=X_pareto,
            objectives=objectives,
            representatives=reps,
            selected_indices=selected_indices,
        )


# ----------------------------------------------------------------------
# Convenience entrypoints (kept for backward compatibility)
# ----------------------------------------------------------------------

def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run one (run_id, rep_id) experiment."""
    return ExperimentRunner(cfg).run()


def run_sweep(cfg: ExperimentConfig, sweep_k: List[int]) -> List[Dict[str, Any]]:
    """
    Run a sweep over k values.

    Note: run_specs.py already encodes k-sweeps for the manuscript runs. This
    helper is kept for CLI convenience.
    """
    from dataclasses import replace

    out = []
    for k in sweep_k:
        cfg_k = replace(cfg, solver=replace(cfg.solver, k=int(k)), run_id=f"{cfg.run_id}_k{int(k)}")
        out.append(run_single_experiment(cfg_k))
    return out
