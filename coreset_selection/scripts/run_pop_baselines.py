#!/usr/bin/env python3
"""
Standalone driver for population-share and joint-constrained baselines.

Loads cached replicate assets, builds evaluator, and runs baseline methods
under pop-quota or joint-quota regimes.  Results are saved to CSV in the
same format as the existing R10 pipeline.

Usage:
    python -m coreset_selection.scripts.run_pop_baselines \
        --k 100 --rep-id 0 --regime pop_quota \
        --cache-dir replicate_cache --output-dir runs_out_pop_baselines \
        --seed 123

Arguments:
    --k INT             Coreset size (from K_GRID: 30,50,100,200,300,400,500)
    --rep-id INT        Replicate index (0-4)
    --regime STR        Constraint regime: 'pop_quota' or 'joint_quota'
    --spaces STR        Comma-separated spaces to run (default: 'raw,vae,pca')
    --cache-dir STR     Path to replicate cache directory
    --output-dir STR    Output directory for results
    --seed INT          Base random seed (default: 123)
    --alpha-geo FLOAT   Dirichlet smoothing (default: 1.0)
    --rff-dim INT       RFF dimension (default: 2000)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Run population-share / joint baselines on cached assets."
    )
    parser.add_argument("--k", type=int, required=True, help="Coreset size")
    parser.add_argument("--rep-id", type=int, required=True, help="Replicate index (0-4)")
    parser.add_argument(
        "--regime", type=str, required=True,
        choices=["pop_quota", "joint_quota"],
        help="Constraint regime",
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
    args = parser.parse_args()

    k = args.k
    rep_id = args.rep_id
    regime = args.regime
    requested_spaces = [s.strip() for s in args.spaces.split(",")]
    seed_base = args.seed
    rep_seed = seed_base + rep_id
    alpha_geo = args.alpha_geo
    rff_dim = args.rff_dim

    # Output directory structure: output_dir/<regime>/k<k>/rep<rep_id>/results/
    run_label = f"{regime}_k{k}"
    rep_label = f"rep{rep_id:02d}"
    results_dir = os.path.join(args.output_dir, regime, f"k{k}", rep_label, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Pop/Joint Baseline Driver")
    print(f"  regime={regime}  k={k}  rep={rep_id}  seed={rep_seed}")
    print(f"  spaces={requested_spaces}")
    print(f"  cache={args.cache_dir}")
    print(f"  output={results_dir}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Load cached assets
    # ----------------------------------------------------------------
    cache_path = os.path.join(args.cache_dir, f"rep{rep_id:02d}", "assets.npz")
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache not found: {cache_path}")
        sys.exit(1)

    print(f"\n[1/5] Loading cache: {cache_path}")
    t0 = time.perf_counter()

    from coreset_selection.data.cache import load_replicate_cache
    assets = load_replicate_cache(cache_path)
    dt = time.perf_counter() - t0
    print(f"  X_scaled={assets.X_scaled.shape}, "
          f"Z_vae={'yes' if assets.Z_vae is not None else 'no'}, "
          f"Z_pca={'yes' if assets.Z_pca is not None else 'no'} ({dt:.1f}s)")

    # ----------------------------------------------------------------
    # 2. Build geographic info + projectors
    # ----------------------------------------------------------------
    print(f"\n[2/5] Building geographic info & projectors...")
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
    print(f"  G={geo.G} groups, N={geo.N} points, "
          f"pi_pop={'yes' if geo.pi_pop is not None else 'no'} ({dt:.1f}s)")

    if geo.pi_pop is None:
        print("[ERROR] Population weights not available in cache — cannot run pop-quota baselines.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 3. Build raw-space evaluator
    # ----------------------------------------------------------------
    print(f"\n[3/5] Building raw-space evaluator...")
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
    print(f"  Evaluator built ({dt:.1f}s)")

    # Extract downstream targets from cache metadata
    _meta = assets.metadata if hasattr(assets, "metadata") and isinstance(assets.metadata, dict) else {}
    extra_reg_targets: Dict[str, np.ndarray] = _meta.get("extra_regression_targets", {})
    cls_targets: Dict[str, np.ndarray] = _meta.get("classification_targets", {})
    state_labels = assets.state_labels

    # ----------------------------------------------------------------
    # 4. Build evaluation function (mirrors _evaluate_coreset)
    # ----------------------------------------------------------------
    def evaluate_coreset(idx_sel: np.ndarray) -> Dict[str, Any]:
        """Full evaluation pipeline matching R10's _evaluate_coreset."""
        idx_sel = np.asarray(idx_sel, dtype=int)
        row: Dict[str, Any] = {}

        # [1/5] Geographic diagnostics
        geo_all = dual_geo_diagnostics(geo, idx_sel, k, alpha=alpha_geo)
        row.update(geo_all)

        # [2/5] Nystrom + KRR + stability
        if state_labels is not None:
            row.update(raw_evaluator.all_metrics_with_state_stability(idx_sel, state_labels))
        else:
            row.update(raw_evaluator.all_metrics(idx_sel))

        # [3/5] KPI stability
        if state_labels is not None and raw_evaluator.y is not None:
            try:
                kpi_stab = state_kpi_stability(
                    y=raw_evaluator.y,
                    state_labels=state_labels,
                    S_idx=idx_sel,
                )
                row.update(kpi_stab)
            except Exception:
                pass

        # [4/5] Multi-model downstream
        if extra_reg_targets or cls_targets:
            try:
                multi_metrics = raw_evaluator.multi_model_downstream(
                    S_idx=idx_sel,
                    regression_targets=extra_reg_targets,
                    classification_targets=cls_targets,
                    seed=rep_seed,
                )
                row.update(multi_metrics)
            except Exception as e:
                print(f"      [warn] multi-model failed: {e}")

        # [5/5] QoS downstream
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
        except Exception as e:
            print(f"      [warn] QoS failed: {e}")

        return row

    # ----------------------------------------------------------------
    # 5. Build spaces and run baselines
    # ----------------------------------------------------------------
    print(f"\n[4/5] Building feature spaces...")
    spaces: Dict[str, np.ndarray] = {}
    if "raw" in requested_spaces:
        spaces["raw"] = np.asarray(assets.X_scaled, dtype=np.float64)
    if "vae" in requested_spaces and assets.Z_vae is not None:
        spaces["vae"] = np.asarray(assets.Z_vae, dtype=np.float64)
    if "pca" in requested_spaces and assets.Z_pca is not None:
        spaces["pca"] = np.asarray(assets.Z_pca, dtype=np.float64)

    print(f"  Active spaces: {list(spaces.keys())}")

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

    print(f"\n[5/5] Running {regime} baselines (k={k})...")
    t0_run = time.perf_counter()

    rows = gen.run_all(
        spaces=spaces,
        evaluator_fn=evaluate_coreset,
        regimes=[regime],
    )

    dt_run = time.perf_counter() - t0_run
    print(f"\n  Completed {len(rows)} method×space combinations ({dt_run:.1f}s)")

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
        print(f"\n  Saved {len(rows)} rows to: {csv_path}")

        # Also save the summary via BaselineVariantGenerator
        summary_path = gen.save_summary(rows, results_dir, filename="baseline_variants_summary.csv")
        print(f"  Summary: {summary_path}")
    else:
        print("\n  [WARN] No rows produced!")

    total_time = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE  regime={regime} k={k} rep={rep_id} ({total_time:.0f}s total)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
