"""
R7/R11 diagnostics handler mixin for ExperimentRunner.

Extracted from runner.py to reduce file size.  Contains:
- DiagnosticsMixin._run_r7_diagnostics
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np


class DiagnosticsMixin:
    """Mixin providing the R7/R11 diagnostics handler for ExperimentRunner."""

    def _run_r7_diagnostics(self, *, assets, seed: int) -> Dict[str, Any]:
        """R11: Post-hoc diagnostics (proxy stability + objective–metric alignment).

        Per manuscript Section VIII.K and Phase 7 of the upgrade plan:
          - Reads Pareto front + front-metrics from R1 at k=300 in VAE space.
          - Optionally loads R10 baseline coreset indices to expand the
            candidate pool for objective–metric alignment (Fig 4).
          - Produces structured CSV deliverables:
              proxy_stability.csv                   (Table IV data)
              objective_metric_alignment.csv          (Fig 4 long-form)
              objective_metric_alignment_heatmap.csv  (Fig 4 pivot-table)

        Overrides supported via environment variables:
            CORESET_R11_SOURCE_RUN   (default: R1)
            CORESET_R11_K            (default: 300)
            CORESET_R11_SOURCE_SPACE (default: vae)
        """
        import csv
        import json
        from dataclasses import asdict

        from ..evaluation.r7_diagnostics import (
            run_r7_diagnostics,
            load_baseline_indices_from_dir,
            find_r10_results_dir,
        )
        from .saver import load_pareto_front

        cfg = self.cfg

        # Support both old R6 and new R11 env var names
        source_run = (
            os.environ.get("CORESET_R11_SOURCE_RUN")
            or os.environ.get("CORESET_R6_SOURCE_RUN", "R1")
        )
        k = int(
            os.environ.get("CORESET_R11_K")
            or os.environ.get("CORESET_R6_K", "300")
        )
        source_space = (
            os.environ.get("CORESET_R11_SOURCE_SPACE")
            or os.environ.get("CORESET_R6_SOURCE_SPACE", "vae")
        )

        print(f"[R11] Starting post-hoc diagnostics", flush=True)
        print(f"[R11]   Source run: {source_run}", flush=True)
        print(f"[R11]   Source space: {source_space}", flush=True)
        print(f"[R11]   k: {k}", flush=True)

        # Resolve source run directory name
        candidate = f"{source_run}_k{k}"
        run_dir_name = candidate if os.path.exists(os.path.join(cfg.files.output_dir, candidate)) else source_run
        src_results_dir = os.path.join(
            cfg.files.output_dir,
            run_dir_name,
            f"rep{cfg.rep_id:02d}",
            "results",
        )

        print(f"[R11]   Looking for source results in: {src_results_dir}", flush=True)

        pareto_path = os.path.join(src_results_dir, f"{source_space}_pareto.npz")
        if not os.path.exists(pareto_path):
            raise FileNotFoundError(
                f"R11 missing source Pareto front: {pareto_path}. "
                f"Run {run_dir_name} for rep{cfg.rep_id:02d} first."
            )

        print(f"[R11]   Loading Pareto front from: {pareto_path}", flush=True)
        pareto = load_pareto_front(pareto_path)
        print(f"[R11]   Loaded {len(pareto.X)} Pareto solutions", flush=True)

        # Load per-solution raw-space metrics if available
        metrics_path = os.path.join(src_results_dir, f"front_metrics_{source_space}.csv")
        metrics_dict: Dict[str, np.ndarray] = {}
        if os.path.exists(metrics_path):
            print(f"[R11]   Loading front metrics from: {metrics_path}", flush=True)
            rows = []
            with open(metrics_path, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            # Sort by pareto_index when present
            if rows and "pareto_index" in rows[0]:
                try:
                    rows.sort(key=lambda rr: int(rr.get("pareto_index", 0)))
                except Exception:
                    pass

            # Pick numeric metric columns (exclude identifiers / objective columns)
            skip_prefixes = ("f_",)
            skip_names = {
                "run_id",
                "rep_id",
                "space",
                "method",
                "rep_name",
                "pareto_index",
                "constraint_regime",
                "k",
            }
            keys = [k_ for k_ in rows[0].keys() if k_ not in skip_names and not any(k_.startswith(p) for p in skip_prefixes)]
            for key in keys:
                vals: List[float] = []
                for rr in rows:
                    v = rr.get(key, "")
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(np.nan)
                metrics_dict[key] = np.asarray(vals, dtype=np.float64)
            print(f"[R11]   Loaded {len(metrics_dict)} metric columns", flush=True)
        else:
            print(f"[R11]   No front metrics file found at: {metrics_path}", flush=True)

        # ------------------------------------------------------------------
        # Phase 7: Load R10 baseline coreset indices to expand the candidate
        # pool for objective–metric alignment (Fig 4).
        # ------------------------------------------------------------------
        baseline_indices: List[tuple] = []
        r10_dir = find_r10_results_dir(cfg.files.output_dir, rep_id=cfg.rep_id)
        if r10_dir is not None:
            print(f"[R11]   Loading R10 baseline indices from: {r10_dir}", flush=True)
            baseline_indices = load_baseline_indices_from_dir(r10_dir)
            print(f"[R11]   Loaded {len(baseline_indices)} baseline coresets from R10", flush=True)
        else:
            print(f"[R11]   R10 results directory not found — skipping baseline pool expansion", flush=True)

        # Validate required representations are available
        if assets.Z_vae is None or assets.Z_pca is None:
            raise RuntimeError(
                "R11 requires both Z_vae and Z_pca in the replicate cache. "
                "Re-run with a config that enables both (e.g., R11 spec)."
            )

        print(f"[R11] Running diagnostics...", flush=True)
        print(f"[R11]   - Surrogate sensitivity analysis (Table IV §1–2)", flush=True)
        print(f"[R11]   - Cross-space objective re-evaluation (Table IV §3)", flush=True)
        print(f"[R11]   - Objective–metric alignment (Fig 4)", flush=True)
        if baseline_indices:
            print(f"[R11]   - Including {len(baseline_indices)} R10 baselines in candidate pool", flush=True)

        r11 = run_r7_diagnostics(
            X_vae=np.asarray(assets.Z_vae, dtype=np.float64),
            X_pca=np.asarray(assets.Z_pca, dtype=np.float64),
            X_raw=np.asarray(assets.X_scaled, dtype=np.float64),
            pareto_F=np.asarray(pareto.F, dtype=np.float64),
            pareto_X=np.asarray(pareto.X),
            objective_names=tuple(pareto.objectives),
            metrics_dict=metrics_dict,
            seed=int(seed),
            output_dir=self.saver.results_dir,
            baseline_indices=baseline_indices if baseline_indices else None,
        )

        out = {
            "run_id": cfg.run_id,
            "rep_id": cfg.rep_id,
            "source_run": run_dir_name,
            "source_space": source_space,
            "k": k,
            "n_pareto_solutions": int(pareto.F.shape[0]),
            "n_baseline_coresets": len(baseline_indices),
            "n_total_candidates": int(pareto.F.shape[0]) + len(baseline_indices),
            "results": asdict(r11),
        }

        # Save JSON bundle
        out_path = os.path.join(self.saver.results_dir, f"r11_diagnostics_{run_dir_name}_{source_space}_k{k}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

        # Also save the legacy filename for backward compatibility
        legacy_path = os.path.join(self.saver.results_dir, f"r7_diagnostics_{run_dir_name}_{source_space}_k{k}.json")
        if legacy_path != out_path:
            try:
                with open(legacy_path, "w") as f:
                    json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            except Exception:
                pass

        print(f"[R11] Diagnostics complete!", flush=True)
        print(f"[R11]   JSON results saved to: {out_path}", flush=True)
        if r11.proxy_stability_csv:
            print(f"[R11]   proxy_stability.csv: {r11.proxy_stability_csv}", flush=True)
        if r11.objective_metric_alignment_csv:
            print(f"[R11]   objective_metric_alignment.csv: {r11.objective_metric_alignment_csv}", flush=True)
        if r11.alignment_heatmap_csv:
            print(f"[R11]   alignment heatmap CSV: {r11.alignment_heatmap_csv}", flush=True)

        return {
            "run_id": cfg.run_id,
            "rep_id": cfg.rep_id,
            "output_dir": self.saver.run_dir,
            "diagnostics_path": out_path,
            "proxy_stability_csv": r11.proxy_stability_csv,
            "objective_metric_alignment_csv": r11.objective_metric_alignment_csv,
            "alignment_heatmap_csv": r11.alignment_heatmap_csv,
        }
