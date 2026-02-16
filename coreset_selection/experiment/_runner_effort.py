"""
R12 effort sweep handler mixin for ExperimentRunner.

Extracted from runner.py to reduce file size.  Contains:
- EffortMixin._run_r12_effort_sweep
"""

from __future__ import annotations

import os
import time as _time
from typing import Any, Dict, List, Optional

import numpy as np

from ..evaluation.raw_space import RawSpaceEvaluator
from ..utils.debug_timing import timer

from ._runner_eval import _build_multitarget_y


class EffortMixin:
    """Mixin providing the R12 effort sweep handler for ExperimentRunner."""

    def _run_r12_effort_sweep(
        self,
        *,
        assets,
        geo,
        projector,
        constraint_set,
        raw_evaluator,
        seed: int,
    ) -> Dict[str, Any]:
        """R12: Run NSGA-II at multiple effort levels and log wall-clock time.

        The effort grid is determined in the following priority order:
          1. ``SolverConfig.effort_grid`` (structured config — preferred)
          2. Environment variables ``CORESET_R12_POP_SIZES`` /
             ``CORESET_R12_N_GENS`` (legacy, Cartesian product)

        For each ``(P, T)`` pair the method:
          - Runs NSGA-II with pop_size=P, n_gen=T (all other config identical)
          - Records wall-clock selection time
          - Evaluates all downstream metrics for the knee representative
          - Saves per-run results and a combined ``effort_sweep_results.csv``
        """
        cfg = self.cfg

        # ---- Resolve effort grid ----
        env_pop = os.environ.get("CORESET_R12_POP_SIZES")
        env_gen = os.environ.get("CORESET_R12_N_GENS")
        if env_pop or env_gen:
            # Legacy env-var path (Cartesian product for backward compat)
            pop_sizes = [int(x.strip()) for x in (env_pop or "20,50,100,150,200,300").split(",") if x.strip()]
            n_gens = [int(x.strip()) for x in (env_gen or "100,300,500,700,1000,1500").split(",") if x.strip()]
            grid = [(p, t) for p in pop_sizes for t in n_gens]
            print(f"[R12] Effort grid from env vars (Cartesian): {len(grid)} combos", flush=True)
        else:
            grid = cfg.solver.effort_grid.grid()
            print(f"[R12] Effort grid from config: {grid}", flush=True)

        # Build objective computer
        with timer.section("build_objective_computers", space=cfg.space):
            computers = self._build_objective_computers(assets, seed)
        computer = computers[cfg.space]

        # Build raw evaluator for downstream metrics
        if cfg.eval.enabled and assets.eval_idx is not None and raw_evaluator is None:
            y_multi, tgt_names = _build_multitarget_y(assets)
            raw_evaluator = RawSpaceEvaluator.build(
                X_raw=assets.X_scaled,
                y=y_multi,
                eval_idx=assets.eval_idx,
                eval_train_idx=assets.eval_train_idx,
                eval_test_idx=assets.eval_test_idx,
                seed=seed,
                target_names=tgt_names,
            )

        rows: List[Dict[str, Any]] = []
        _tp_sweep = self._phase_start()
        for P, T in grid:
            print(f"[R12]   Running P={P}, T={T} ...", flush=True)

            # ---- Phase timing ----
            t_start = _time.perf_counter()

            # Temporarily override solver hyperparameters
            from dataclasses import replace as dc_replace
            solver_override = dc_replace(cfg.solver, pop_size=P, n_gen=T)
            cfg_effort = dc_replace(cfg, solver=solver_override)
            self.cfg = cfg_effort  # swap temporarily

            try:
                pareto_data = self._run_solver_single_space(
                    computer=computer,
                    projector=projector,
                    constraint_set=constraint_set,
                    space=cfg.space,
                    seed=seed,
                )
            finally:
                self.cfg = cfg  # restore original

            wall_clock_s = _time.perf_counter() - t_start

            # Evaluate the knee (or median-objective) representative
            for rep_name, rep_pf_idx in pareto_data.representatives.items():
                idx_sel = pareto_data.selected_indices[rep_name]
                _r12_extra_reg = assets.metadata.get("extra_regression_targets", {}) if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
                _r12_cls = assets.metadata.get("classification_targets", {}) if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
                row = self._evaluate_coreset(
                    idx_sel=idx_sel,
                    geo=geo,
                    k=int(cfg.solver.k),
                    raw_evaluator=raw_evaluator,
                    state_labels=assets.state_labels if hasattr(assets, 'state_labels') else None,
                    extra_regression_targets=_r12_extra_reg,
                    classification_targets=_r12_cls,
                )
                fvals = pareto_data.F[int(rep_pf_idx)]
                for j, obj in enumerate(pareto_data.objectives):
                    row[f"f_{obj}"] = float(fvals[j])
                row.update({
                    "run_id": cfg.run_id,
                    "rep_id": cfg.rep_id,
                    "space": cfg.space,
                    "method": "nsga2",
                    "rep_name": rep_name,
                    "k": int(cfg.solver.k),
                    "pop_size": P,
                    "n_gen": T,
                    "effort_P_x_T": P * T,
                    "wall_clock_s": float(wall_clock_s),
                    "front_size": int(pareto_data.X.shape[0]),
                    "constraint_regime": self._constraint_regime(cfg),
                })
                rows.append(row)

            print(f"[R12]   P={P}, T={T}: wall_clock={wall_clock_s:.1f}s, "
                  f"front_size={pareto_data.X.shape[0]}", flush=True)
        self._phase_end("effort_sweep", _tp_sweep)

        # Save both the standard all_results and a dedicated effort_sweep CSV
        if rows:
            self.saver.save_rows(rows, name="all_results")
            self.saver.save_rows(rows, name="effort_sweep_results")

            # G10: Structured effort-sweep outputs.
            # 1. Parameter grid config CSV (documents the grid itself)
            self.saver.save_effort_grid_csv(
                grid,
                k=int(cfg.solver.k),
                objectives=tuple(str(o) for o in cfg.solver.objectives),
                constraint_regime=self._constraint_regime(cfg),
                space=cfg.space,
            )
            # 2. Filtered summary CSV (knee representative per effort level)
            self.saver.save_effort_sweep_summary(rows)

        return {
            "run_id": cfg.run_id,
            "rep_id": cfg.rep_id,
            "space": cfg.space,
            "n_rows": len(rows),
            "output_dir": self.saver.run_dir,
        }
