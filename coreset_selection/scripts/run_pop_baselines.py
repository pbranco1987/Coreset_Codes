#!/usr/bin/env python3
"""
Standalone driver for population-share and joint-constrained baselines.

Loads cached replicate assets, builds evaluator, and runs baseline methods
under any of the 4 constraint regimes.  Results are saved to CSV in the
same format as the existing R10 pipeline.

Supported regimes:
  - pop_quota:    population-share hard quotas (P-prefix methods)
  - muni_quota:   municipality-share hard quotas (S-prefix methods)
  - joint_quota:  both pop + muni quotas (J-prefix methods)
  - exactk:       no geographic constraint, |S|=k only (unprefixed methods)

Usage:
    python -m coreset_selection.scripts.run_pop_baselines \
        --k 100 --rep-id 0 --regime pop_quota \
        --cache-dir replicate_cache --output-dir runs_out_pop_baselines \
        --seed 123 --job-num 5 --total-jobs 42

Arguments:
    --k INT             Coreset size (from K_GRID: 30,50,100,200,300,400,500)
    --rep-id INT        Replicate index (0-4)
    --regime STR        Constraint regime: 'pop_quota', 'muni_quota', 'joint_quota', or 'exactk'
    --spaces STR        Comma-separated spaces to run (default: 'raw,vae,pca')
    --cache-dir STR     Path to replicate cache directory
    --output-dir STR    Output directory for results
    --seed INT          Base random seed (default: 123)
    --alpha-geo FLOAT   Dirichlet smoothing (default: 1.0)
    --rff-dim INT       RFF dimension (default: 2000)
    --job-num INT       Job number within the experiment (1-based, for display)
    --total-jobs INT    Total number of jobs in the experiment (for display)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Elapsed-time helper ────────────────────────────────────────────────
_T0_GLOBAL: float = 0.0  # set once in main()


def _elapsed() -> str:
    """Return '[HH:MM:SS]' elapsed since _T0_GLOBAL."""
    dt = time.perf_counter() - _T0_GLOBAL
    m, s = divmod(int(dt), 60)
    h, m = divmod(m, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def _fmt_duration(seconds: float) -> str:
    """Format seconds as 'Xh YYm ZZs' or 'Ym ZZs' or 'ZZs'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(
        description="Run population-share / joint baselines on cached assets."
    )
    parser.add_argument("--k", type=int, required=True, help="Coreset size")
    parser.add_argument("--rep-id", type=int, required=True, help="Replicate index (0-4)")
    parser.add_argument(
        "--regime", type=str, required=True,
        choices=["pop_quota", "muni_quota", "joint_quota", "exactk"],
        help="Constraint regime: pop_quota, muni_quota, joint_quota, or exactk",
    )
    parser.add_argument(
        "--spaces", type=str, default="raw,vae,pca",
        help="Comma-separated spaces (default: raw,vae,pca)",
    )
    parser.add_argument("--cache-dir", type=str, default="replicate_cache")
    parser.add_argument("--output-dir", type=str, default="runs_out_pop_baselines")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--alpha-geo", type=float, default=1.0)
    parser.add_argument("--rff-dim", type=int, default=2000)
    parser.add_argument(
        "--job-num", type=int, default=0,
        help="Job number in the experiment (1-based, for display)",
    )
    parser.add_argument(
        "--total-jobs", type=int, default=0,
        help="Total jobs in the experiment (for display)",
    )
    args = parser.parse_args()

    k = args.k
    rep_id = args.rep_id
    regime = args.regime
    requested_spaces = [s.strip() for s in args.spaces.split(",")]
    seed_base = args.seed
    rep_seed = seed_base + rep_id
    alpha_geo = args.alpha_geo
    rff_dim = args.rff_dim
    job_num = args.job_num
    total_jobs = args.total_jobs

    # Build job label for progress output
    job_label = ""
    if job_num > 0 and total_jobs > 0:
        job_label = f"[Job {job_num}/{total_jobs}]"
    elif job_num > 0:
        job_label = f"[Job {job_num}]"

    # Output directory structure: output_dir/<regime>/k<k>/rep<rep_id>/results/
    run_label = f"{regime}_k{k}"
    rep_label = f"rep{rep_id:02d}"
    results_dir = os.path.join(args.output_dir, regime, f"k{k}", rep_label, "results")
    os.makedirs(results_dir, exist_ok=True)

    global _T0_GLOBAL
    t_global_start = time.perf_counter()
    _T0_GLOBAL = t_global_start

    n_methods = 8  # U, KM, KH, FF, RLS, DPP, KT, KKN
    n_spaces_req = len(requested_spaces)
    total_combos = n_methods * n_spaces_req
    total_stages_per_combo = 5  # c, d, e, f, g
    total_operations = total_combos * (2 + total_stages_per_combo)  # +2 for a,b

    print(f"{_elapsed()} {'=' * 64}")
    print(f"{_elapsed()}   Pop/Joint Baseline Driver")
    if job_label:
        print(f"{_elapsed()}   {job_label}")
    print(f"{_elapsed()}   regime={regime}  k={k}  rep={rep_id}  seed={rep_seed}")
    print(f"{_elapsed()}   spaces={requested_spaces}")
    print(f"{_elapsed()}   combos={n_methods} methods x {n_spaces_req} spaces = {total_combos}")
    print(f"{_elapsed()}   eval stages per combo: {total_stages_per_combo} "
          f"({total_combos * total_stages_per_combo} total evaluations)")
    print(f"{_elapsed()}   cache={args.cache_dir}")
    print(f"{_elapsed()}   output={results_dir}")
    print(f"{_elapsed()} {'=' * 64}")

    # ----------------------------------------------------------------
    # 1. Load cached assets
    # ----------------------------------------------------------------
    cache_path = os.path.join(args.cache_dir, f"rep{rep_id:02d}", "assets.npz")
    if not os.path.exists(cache_path):
        print(f"{_elapsed()} [ERROR] Cache not found: {cache_path}")
        sys.exit(1)

    print(f"\n{_elapsed()} [1/5] Loading cache: {cache_path}")
    t0 = time.perf_counter()

    from coreset_selection.data.cache import load_replicate_cache
    assets = load_replicate_cache(cache_path)
    dt = time.perf_counter() - t0
    print(f"{_elapsed()}   X_scaled={assets.X_scaled.shape}, "
          f"Z_vae={'yes' if assets.Z_vae is not None else 'no'}, "
          f"Z_pca={'yes' if assets.Z_pca is not None else 'no'} ({dt:.1f}s)")

    # ----------------------------------------------------------------
    # 2. Build geographic info + projectors
    # ----------------------------------------------------------------
    print(f"\n{_elapsed()} [2/5] Building geographic info & projectors...")
    t0 = time.perf_counter()

    from coreset_selection.geo.info import build_geo_info
    from coreset_selection.geo.projector import GeographicConstraintProjector

    pop_array = getattr(assets, "population", None)
    if pop_array is None and hasattr(assets, "metadata") and isinstance(assets.metadata, dict):
        pop_array = assets.metadata.get("population", None)

    geo = build_geo_info(
        assets.state_labels,
        population_weights=pop_array,
    )

    # The default (muni) projector — used by the BaselineVariantGenerator
    # as self.projector for exact-k and muni-quota regimes.
    projector_muni = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=True,
        weight_type="muni",
    )

    dt = time.perf_counter() - t0
    print(f"{_elapsed()}   G={geo.G} groups, N={geo.N} points, "
          f"pi_pop={'yes' if geo.pi_pop is not None else 'no'} ({dt:.1f}s)")

    if geo.pi_pop is None:
        print(f"{_elapsed()} [ERROR] Population weights not available in cache — cannot run pop-quota baselines.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 3. Build raw-space evaluator
    # ----------------------------------------------------------------
    print(f"\n{_elapsed()} [3/5] Building raw-space evaluator...")
    t0 = time.perf_counter()

    from coreset_selection.experiment._runner_eval import _build_multitarget_y
    from coreset_selection.evaluation.raw_space import RawSpaceEvaluator
    from coreset_selection.evaluation.geo_diagnostics import dual_geo_diagnostics
    from coreset_selection.evaluation.kpi_stability import state_kpi_stability

    y_multi, tgt_names = _build_multitarget_y(assets)
    raw_evaluator = RawSpaceEvaluator.build(
        X_raw=assets.X_scaled,
        y=y_multi,
        eval_idx=assets.eval_idx,
        eval_train_idx=assets.eval_train_idx,
        eval_test_idx=assets.eval_test_idx,
        seed=rep_seed,
        target_names=tgt_names,
    )
    dt = time.perf_counter() - t0
    print(f"{_elapsed()}   Evaluator built ({dt:.1f}s)")

    # Extract downstream targets from cache metadata
    _meta = assets.metadata if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
    extra_reg_targets: Dict[str, np.ndarray] = _meta.get("extra_regression_targets", {})
    cls_targets: Dict[str, np.ndarray] = _meta.get("classification_targets", {})
    state_labels = assets.state_labels

    # ----------------------------------------------------------------
    # 4. Build evaluation function (mirrors _evaluate_coreset)
    # ----------------------------------------------------------------
    n_reg_targets = len(extra_reg_targets)
    n_cls_targets = len(cls_targets)
    n_reg_models = 3   # KNN, RF, GBT
    n_cls_models = 4   # KNN, RF, LR, GBT
    total_model_fits = n_reg_models * n_reg_targets + n_cls_models * n_cls_targets

    print(f"\n{_elapsed()}   Evaluation pipeline per combo ({total_stages_per_combo} stages):")
    print(f"{_elapsed()}     [1/5] Geo diagnostics")
    print(f"{_elapsed()}     [2/5] Nystrom + KRR + state-stability")
    print(f"{_elapsed()}     [3/5] KPI stability")
    print(f"{_elapsed()}     [4/5] Multi-model downstream: "
          f"{n_reg_models} reg x {n_reg_targets} tgts + "
          f"{n_cls_models} cls x {n_cls_targets} tgts "
          f"= {total_model_fits} fits")
    print(f"{_elapsed()}     [5/5] QoS downstream: 3 models (OLS, Ridge, ElasticNet)")

    # ── Stage timing tracker for running averages ──
    _stage_times: Dict[str, List[float]] = {
        "geo": [], "nystrom": [], "kpi": [], "multi": [], "qos": [],
    }
    _stages_order = ["geo", "nystrom", "kpi", "multi", "qos"]
    _eval_count = [0]  # mutable counter in closure
    _combo_t0 = [0.0]  # start time of current combo's evaluation

    def _stage_est(key: str) -> str:
        """Return estimated time string from running average, or empty."""
        times = _stage_times.get(key, [])
        if times:
            avg = sum(times) / len(times)
            return f" (est ~{_fmt_duration(avg)})"
        return ""

    def _stage_record(key: str, dt: float) -> str:
        """Record stage time, return summary with running average."""
        _stage_times[key].append(dt)
        avg = sum(_stage_times[key]) / len(_stage_times[key])
        n = len(_stage_times[key])
        if n > 1:
            return f"done ({dt:.1f}s, avg {avg:.1f}s over {n} combos)"
        return f"done ({dt:.1f}s)"

    def _show_eta(stage_idx: int):
        """Print ETA after every stage — the main progress indicator."""
        combo_num = _eval_count[0]
        combo_elapsed = time.perf_counter() - _combo_t0[0]
        stages_done = stage_idx + 1
        remaining_combos = total_combos - combo_num

        # Estimate remaining time for this combo's unfinished stages
        remaining_this_combo = 0.0
        can_estimate = True
        for i in range(stages_done, len(_stages_order)):
            key = _stages_order[i]
            times = _stage_times.get(key, [])
            if times:
                remaining_this_combo += sum(times) / len(times)
            else:
                can_estimate = False
                break

        if not can_estimate:
            # First combo, stages we haven't seen yet — extrapolate
            if stages_done > 0:
                est_full_combo = combo_elapsed * len(_stages_order) / stages_done
                remaining_this_combo = est_full_combo - combo_elapsed
            else:
                return

        # Best per-combo estimate for remaining combos
        completed_combos = combo_num - 1
        if completed_combos > 0:
            avg_full_combo = sum(
                sum(times) / len(times)
                for times in _stage_times.values() if times
            )
        else:
            avg_full_combo = combo_elapsed + remaining_this_combo

        job_remaining = remaining_this_combo + remaining_combos * avg_full_combo
        job_total_est = (time.perf_counter() - _T0_GLOBAL) + job_remaining
        pct = combo_num * 100 // total_combos if stages_done == len(_stages_order) else (
            ((combo_num - 1) * len(_stages_order) + stages_done) * 100
            // (total_combos * len(_stages_order))
        )

        jl = f" {job_label}" if job_label else ""
        print(f"{_elapsed()}          "
              f">>>>{jl} ETA: ~{_fmt_duration(job_remaining)} remaining | "
              f"combo {combo_num}/{total_combos} stage {stages_done}/5 | "
              f"~{_fmt_duration(job_total_est)} total | {pct}% complete",
              flush=True)

    def evaluate_coreset(idx_sel: np.ndarray) -> Dict[str, Any]:
        """Full evaluation pipeline matching R10's _evaluate_coreset."""
        _eval_count[0] += 1
        combo_num = _eval_count[0]
        _combo_t0[0] = time.perf_counter()
        idx_sel = np.asarray(idx_sel, dtype=int)
        row: Dict[str, Any] = {}

        # [1/5] Geographic diagnostics
        print(f"{_elapsed()}          [1/5] Geo diagnostics{_stage_est('geo')}...",
              end=" ", flush=True)
        t_stage = time.perf_counter()
        geo_all = dual_geo_diagnostics(geo, idx_sel, k, alpha=alpha_geo)
        row.update(geo_all)
        dt_stage = time.perf_counter() - t_stage
        print(f"{_stage_record('geo', dt_stage)} {_elapsed()}", flush=True)
        _show_eta(0)

        # [2/5] Nystrom + KRR + stability
        print(f"{_elapsed()}          [2/5] Nystrom + KRR + stability{_stage_est('nystrom')}...",
              end=" ", flush=True)
        t_stage = time.perf_counter()
        if state_labels is not None:
            row.update(raw_evaluator.all_metrics_with_state_stability(
                idx_sel, state_labels))
        else:
            row.update(raw_evaluator.all_metrics(idx_sel))
        dt_stage = time.perf_counter() - t_stage
        print(f"{_stage_record('nystrom', dt_stage)} {_elapsed()}", flush=True)
        _show_eta(1)

        # [3/5] KPI stability
        print(f"{_elapsed()}          [3/5] KPI stability{_stage_est('kpi')}...",
              end=" ", flush=True)
        t_stage = time.perf_counter()
        if state_labels is not None and raw_evaluator.y is not None:
            try:
                kpi_stab = state_kpi_stability(
                    y=raw_evaluator.y,
                    state_labels=state_labels,
                    S_idx=idx_sel,
                )
                row.update(kpi_stab)
                dt_stage = time.perf_counter() - t_stage
                print(f"{_stage_record('kpi', dt_stage)} {_elapsed()}", flush=True)
            except Exception as e:
                dt_stage = time.perf_counter() - t_stage
                print(f"skipped ({dt_stage:.1f}s): {e} {_elapsed()}", flush=True)
        else:
            print(f"skipped (no state labels) {_elapsed()}", flush=True)
        _show_eta(2)

        # [4/5] Multi-model downstream
        print(f"{_elapsed()}          [4/5] Multi-model ({total_model_fits} fits)"
              f"{_stage_est('multi')}...",
              end=" ", flush=True)
        t_stage = time.perf_counter()
        if extra_reg_targets or cls_targets:
            try:
                multi_metrics = raw_evaluator.multi_model_downstream(
                    S_idx=idx_sel,
                    regression_targets=extra_reg_targets,
                    classification_targets=cls_targets,
                    seed=rep_seed,
                )
                row.update(multi_metrics)
                dt_stage = time.perf_counter() - t_stage
                print(f"{_stage_record('multi', dt_stage)} {_elapsed()}", flush=True)
            except Exception as e:
                dt_stage = time.perf_counter() - t_stage
                print(f"FAILED ({dt_stage:.1f}s): {e} {_elapsed()}", flush=True)
        else:
            print(f"skipped (no targets) {_elapsed()}", flush=True)
        _show_eta(3)

        # [5/5] QoS downstream
        print(f"{_elapsed()}          [5/5] QoS downstream (3 models)"
              f"{_stage_est('qos')}...",
              end=" ", flush=True)
        t_stage = time.perf_counter()
        try:
            from coreset_selection.evaluation.qos_tasks import QoSConfig, qos_coreset_evaluation

            qos_cfg = QoSConfig(
                models=["ols", "ridge", "elastic_net"],
                run_fixed_effects=True,
            )
            _qos_y = _meta.get("qos_target")
            if _qos_y is None:
                _qos_y = (
                    raw_evaluator.y.ravel()
                    if raw_evaluator.y.ndim == 1
                    else raw_evaluator.y[:, 0]
                )
            entity_ids = _meta.get("entity_ids")
            time_ids = _meta.get("time_ids")

            qos_metrics = qos_coreset_evaluation(
                X_full=raw_evaluator.X_raw,
                y_full=_qos_y,
                S_idx=idx_sel,
                eval_test_idx=raw_evaluator.eval_test_idx,
                entity_ids=entity_ids,
                time_ids=time_ids,
                state_labels=state_labels,
                config=qos_cfg,
            )
            row.update(qos_metrics)
            dt_stage = time.perf_counter() - t_stage
            print(f"{_stage_record('qos', dt_stage)} {_elapsed()}", flush=True)
        except Exception as e:
            dt_stage = time.perf_counter() - t_stage
            print(f"skipped ({dt_stage:.1f}s): {e} {_elapsed()}", flush=True)
        _show_eta(4)

        return row

    # ----------------------------------------------------------------
    # 5. Build spaces and run baselines
    # ----------------------------------------------------------------
    print(f"\n{_elapsed()} [4/5] Building feature spaces...")
    spaces: Dict[str, np.ndarray] = {}
    if "raw" in requested_spaces:
        spaces["raw"] = np.asarray(assets.X_scaled, dtype=np.float64)
    if "vae" in requested_spaces and assets.Z_vae is not None:
        spaces["vae"] = np.asarray(assets.Z_vae, dtype=np.float64)
    if "pca" in requested_spaces and assets.Z_pca is not None:
        spaces["pca"] = np.asarray(assets.Z_pca, dtype=np.float64)

    print(f"{_elapsed()}   Active spaces: {list(spaces.keys())}")

    from coreset_selection.baselines.variant_generator import BaselineVariantGenerator

    gen = BaselineVariantGenerator(
        geo=geo,
        projector=projector_muni,
        k=k,
        alpha_geo=alpha_geo,
        rff_dim=rff_dim,
        seed=rep_seed,
        min_one_per_group=True,
    )

    print(f"\n{_elapsed()} [5/5] Running {regime} baselines (k={k})...")
    print(f"{_elapsed()}   Total: {total_combos} method-space combos, each with "
          f"{total_stages_per_combo} eval stages")
    if job_label:
        print(f"{_elapsed()}   {job_label}")
    t0_run = time.perf_counter()

    rows = gen.run_all(
        spaces=spaces,
        evaluator_fn=evaluate_coreset,
        regimes=[regime],
        verbose=True,
        job_label=job_label,
    )

    dt_run = time.perf_counter() - t0_run
    print(f"\n{_elapsed()}   Completed {len(rows)} method-space combos in "
          f"{_fmt_duration(dt_run)}")

    # Print stage timing summary
    print(f"{_elapsed()}   Stage timing summary (avg over {_eval_count[0]} combos):")
    for key, label in [("geo", "Geo diagnostics"),
                       ("nystrom", "Nystrom+KRR"),
                       ("kpi", "KPI stability"),
                       ("multi", "Multi-model"),
                       ("qos", "QoS downstream")]:
        times = _stage_times.get(key, [])
        if times:
            avg = sum(times) / len(times)
            total = sum(times)
            print(f"{_elapsed()}     {label}: avg {avg:.1f}s, "
                  f"total {_fmt_duration(total)} ({len(times)} runs)")

    # ----------------------------------------------------------------
    # 6. Save results
    # ----------------------------------------------------------------
    if rows:
        # Add metadata columns
        for row in rows:
            row["run_id"] = f"R10_{regime}"
            row["rep_id"] = rep_id
            row["seed"] = rep_seed
            row["scenario"] = "brazil_telecom"

        # Save as CSV (same format as all_results.csv)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(results_dir, "all_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{_elapsed()}   Saved {len(rows)} rows to: {csv_path}")

        # Also save the summary via BaselineVariantGenerator
        summary_path = gen.save_summary(rows, results_dir, filename="baseline_variants_summary.csv")
        print(f"{_elapsed()}   Summary: {summary_path}")
    else:
        print(f"\n{_elapsed()}   [WARN] No rows produced!")

    total_time = time.perf_counter() - t_global_start
    print(f"\n{_elapsed()} {'=' * 64}")
    print(f"{_elapsed()}   DONE  regime={regime} k={k} rep={rep_id}")
    if job_label:
        print(f"{_elapsed()}   {job_label}")
    print(f"{_elapsed()}   Total time: {_fmt_duration(total_time)} ({total_time:.0f}s)")
    print(f"{_elapsed()} {'=' * 64}")


if __name__ == "__main__":
    main()
