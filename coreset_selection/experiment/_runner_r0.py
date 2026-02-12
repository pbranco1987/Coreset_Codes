"""
R0 quota-only handler mixin for ExperimentRunner.

Extracted from runner.py to reduce file size.  Contains:
- R0Mixin._run_r0_quota_only
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ..geo.info import build_geo_info
from ..geo.projector import GeographicConstraintProjector


class R0Mixin:
    """Mixin providing the R0 quota-only handler for ExperimentRunner."""

    def _run_r0_quota_only(self, seed: int) -> Dict[str, Any]:
        """R0: Quota computation over manuscript cardinality grid K.

        Computes c*(k) and KL_min(k) for every k in K_GRID (default 50..500)
        using the **incremental quota path** algorithm (Phase 6 §6.1).  This
        builds the allocation lazily from k_min to k_max in a single pass,
        producing O(k_max log G) total work instead of O(|K| · k · G).

        Outputs
        -------
        - ``all_results.csv`` : one row per k with summary metrics.
        - ``quota_path.json`` : full detail including c*(k) vectors.
        - ``kl_floor.csv`` : compact k → KL_min(k) table.
        - ``quota_cstar_k{k}.npz`` : per-k count vector for downstream reuse.
        """
        from ..data.manager import DataManager
        from ..config.run_specs import K_GRID
        from ..geo.kl import compute_quota_path, save_quota_path, kl_pi_hat_from_counts

        cfg = self.cfg

        _tp = self._phase_start()
        data_manager = DataManager(
            cfg.files, int(seed),
            preprocessing_cfg=getattr(cfg, 'preprocessing', None),
        )
        data_manager.load()

        state_labels = np.asarray(data_manager.state_labels())

        # Build geo with optional population weights for population-share quota
        pop = data_manager.population()
        geo = build_geo_info(state_labels, population_weights=pop)
        projector = GeographicConstraintProjector(
            geo=geo,
            alpha_geo=float(cfg.geo.alpha_geo),
            min_one_per_group=bool(cfg.geo.min_one_per_group),
        )
        self._phase_end("setup", _tp)

        # Save config for reproducibility
        self.saver.save_config(cfg)

        # Determine the set of k values to sweep
        from ..config.run_specs import get_run_specs
        spec = get_run_specs().get("R0")
        k_values = list(spec.sweep_k) if (spec and spec.sweep_k) else list(K_GRID)
        # Always include cfg.solver.k if not already present
        if int(cfg.solver.k) not in k_values:
            k_values.append(int(cfg.solver.k))
        k_values = sorted(set(k_values))

        # Filter out infeasible k values
        feasible_k = [k_val for k_val in k_values if k_val <= geo.N]
        skipped = [k_val for k_val in k_values if k_val > geo.N]
        for k_val in skipped:
            print(f"[R0] Skipping k={k_val} > N={geo.N}", flush=True)

        if not feasible_k:
            return self._finalize_with_timing({
                "run_id": cfg.run_id, "rep_id": cfg.rep_id,
                "space": cfg.space, "n_rows": 0,
                "output_dir": self.saver.run_dir,
            })

        # ---- Incremental quota path (Phase 6 §6.1) ----
        _tp = self._phase_start()
        path_rows = compute_quota_path(
            pi=np.asarray(geo.pi, dtype=np.float64),
            group_sizes=np.asarray(geo.group_sizes, dtype=int),
            k_grid=feasible_k,
            alpha_geo=float(cfg.geo.alpha_geo),
            min_one_per_group=bool(cfg.geo.min_one_per_group),
        )
        self._phase_end("quota_computation", _tp)

        # Enrich rows with run metadata and auditing KL
        rows: List[Dict[str, Any]] = []
        for pr in path_rows:
            k_val = pr["k"]
            counts = np.array(pr["cstar"], dtype=int)

            # Smoothed KL for auditing (should ≈ kl_min)
            kl_smooth = float(kl_pi_hat_from_counts(
                pi=geo.pi, counts=counts, k=k_val,
                alpha=float(cfg.geo.alpha_geo),
            ))

            row = {
                "run_id": cfg.run_id,
                "rep_id": cfg.rep_id,
                "k": k_val,
                "kl_min": pr["kl_min"],
                "geo_kl": kl_smooth,
                "geo_l1": pr["geo_l1"],
                "geo_maxdev": pr["geo_maxdev"],
                "cstar": pr["cstar"],
            }
            rows.append(row)

            # Save per-k npz for downstream reuse
            try:
                np.savez_compressed(
                    os.path.join(self.saver.results_dir, f"quota_cstar_k{k_val}.npz"),
                    counts=counts,
                )
            except Exception:
                pass

            print(
                f"[R0] k={k_val:4d}: KL_min={pr['kl_min']:.6f}, "
                f"L1={pr['geo_l1']:.4f}",
                flush=True,
            )

        # Save quota_path.json and kl_floor.csv
        json_path, csv_path = save_quota_path(
            path_rows, self.saver.results_dir,
        )
        print(f"[R0] Saved {json_path}", flush=True)
        print(f"[R0] Saved {csv_path}", flush=True)

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

        # ---- Consolidated outputs (Phase 6 §6.1) ----
        if rows:
            self.saver.save_rows(rows, name="all_results")

        return result
