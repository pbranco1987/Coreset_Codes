"""Generate-all orchestrator and data-loading helpers for ManuscriptArtifacts (mixin)."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._ma_helpers import _set_style


class GenerateAllMixin:
    """Mixin providing generate_all() and data-loading helpers."""

    def _load_df(self) -> pd.DataFrame:
        """Load concatenated results from all runs."""
        dfs = []
        for path in glob.glob(os.path.join(self.runs_root, "**/all_results.csv"),
                              recursive=True):
            try:
                dfs.append(pd.read_csv(path))
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        # Normalize k column
        if "k" not in df.columns:
            df["k"] = df["run_id"].astype(str).str.extract(r"_k(\d+)").astype(float)
        return df

    def _load_pareto(self, run_id: str, space: str = "raw") -> Optional[dict]:
        """Load a single Pareto front .npz file."""
        for pat in [
            os.path.join(self.runs_root, run_id, "rep*", "results", f"{space}_pareto.npz"),
            os.path.join(self.runs_root, f"{run_id}_k300", "rep*", "results", f"{space}_pareto.npz"),
        ]:
            for p in sorted(glob.glob(pat)):
                try:
                    return dict(np.load(p, allow_pickle=True))
                except Exception:
                    pass
        return None

    def generate_all(self) -> Dict[str, List[str]]:
        """Generate all manuscript and complementary artifacts.

        Returns
        -------
        Dict[str, List[str]]
            ``{"figures": [...], "tables": [...]}`` with paths to every
            generated artefact.  Manuscript-referenced artefacts are
            generated first so that failures are immediately visible.
        """
        _set_style()
        df = self._load_df()
        gen: Dict[str, List[str]] = {"figures": [], "tables": []}

        # ---- Manuscript figures (Figs 1-4, Section VIII) ----
        manuscript_figs = [
            ("Fig 1 (geo ablation scatter)",        lambda: self.fig_geo_ablation_scatter(df)),
            ("Fig 2 (distortion cardinality R1)",   lambda: self.fig_distortion_cardinality_r1(df)),
            ("Fig 3 (regional validity k=300)",     lambda: self.fig_regional_validity_k300(df)),
            ("Fig 4 (obj-metric alignment heatmap)",lambda: self.fig_objective_metric_alignment(df)),
        ]
        for name, fn in manuscript_figs:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["figures"].extend(out)
                elif isinstance(out, str) and out:
                    gen["figures"].append(out)
                print(f"[ManuscriptArtifacts] OK  {name}")
            except Exception as e:
                print(f"[ManuscriptArtifacts] FAIL {name}: {e}")

        # ---- Complementary / narrative-strengthening figures ----
        # Phase 10a (Figs N1-N6): Enhanced figures for strengthened narrative
        phase10a_figs = [
            ("Fig N1 (KL floor vs k)",             lambda: self.fig_kl_floor_vs_k()),
            ("Fig N2 (Pareto front k=300)",        lambda: self.fig_pareto_front_k300(df)),
            ("Fig N3 (Objective ablation bars)",    lambda: self.fig_objective_ablation_bars(df)),
            ("Fig N4 (Constraint comparison bars)", lambda: self.fig_constraint_comparison(df)),
            ("Fig N5 (Effort-quality trade-off)",   lambda: self.fig_effort_quality(df)),
            ("Fig N6 (Baseline comparison)",        lambda: self.fig_baseline_comparison(df)),
        ]
        for name, fn in phase10a_figs:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["figures"].extend(out)
                elif isinstance(out, str) and out:
                    gen["figures"].append(out)
                print(f"[ManuscriptArtifacts] OK  {name}")
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING Phase 10a figure '{name}': {e}")

        # Additional complementary figures (Phase 10b scope + legacy extras)
        # Phase 10b (Figs N7-N12): New/enhanced narrative-strengthening figures
        phase10b_figs = [
            ("Fig N7 (Multi-seed boxplot)",          lambda: self.fig_multi_seed_boxplot(df)),
            ("Fig N8 (State KPI heatmap)",           lambda: self.fig_state_kpi_heatmap(df)),
            ("Fig N9 (Composition shift)",           lambda: self.fig_composition_shift(df)),
            ("Fig N10 (Pareto front evolution)",     lambda: self.fig_pareto_front_evolution(df)),
            ("Fig N11 (Nystrom error distribution)", lambda: self.fig_nystrom_error_distribution(df)),
            ("Fig N12 (Worst-state RMSE vs k)",      lambda: self.fig_krr_worst_state_rmse_vs_k(df)),
        ]
        for name, fn in phase10b_figs:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["figures"].extend(out)
                elif isinstance(out, str) and out:
                    gen["figures"].append(out)
                print(f"[ManuscriptArtifacts] OK  {name}")
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING Phase 10b figure '{name}': {e}")

        # Legacy / additional complementary figures
        complementary_figs = [
            ("Representation transfer bars",   lambda: self.fig_representation_transfer(df)),
            ("Cumulative Pareto improvement",  lambda: self.fig_cumulative_pareto_improvement(df)),
            ("Constraint tightness vs fidelity",lambda: self.fig_constraint_tightness_vs_fidelity(df)),
            ("SKL ablation comparison",        lambda: self.fig_skl_ablation_comparison(df)),
        ]
        for name, fn in complementary_figs:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["figures"].extend(out)
                elif isinstance(out, str) and out:
                    gen["figures"].append(out)
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING complementary figure '{name}': {e}")

        # ---- Phase 12: Geographic choropleth map figures (N13-N17) ----
        geo_map_figs = [
            ("Fig N13 (coreset map k-sweep)",
             lambda: self.fig_coreset_map_k_sweep(df)),
            ("Fig N14 (coreset map representatives)",
             lambda: self.fig_coreset_map_representatives(df)),
            ("Fig N15 (coreset map baselines)",
             lambda: self.fig_coreset_map_baselines(df)),
            ("Fig N16 (coreset map constraints)",
             lambda: self.fig_coreset_map_constraint_comparison(df)),
            ("Fig N17 (coreset map representations)",
             lambda: self.fig_coreset_map_representation_comparison(df)),
        ]
        for name, fn in geo_map_figs:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["figures"].extend(out)
                elif isinstance(out, str) and out:
                    gen["figures"].append(out)
                print(f"[ManuscriptArtifacts] OK  {name}")
            except ImportError as e:
                print(f"[ManuscriptArtifacts] SKIP {name} (missing dep): {e}")
            except FileNotFoundError as e:
                print(f"[ManuscriptArtifacts] SKIP {name} (no shapefile): {e}")
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING Phase 12 figure '{name}': {e}")

        # ---- Tables ----
        # Manuscript tables first (Tables I-V), then complementary.
        table_fns = [
            ("Table I (exp settings)",         lambda: self.tab_exp_settings()),
            ("Table II (run matrix)",          lambda: self.tab_run_matrix()),
            ("Table III (R1 by k)",            lambda: self.tab_r1_by_k(df)),
            ("Table IV (proxy stability)",     lambda: self.tab_proxy_stability(df)),
            ("Table V (KRR multi-task)",       lambda: self.tab_krr_multitask(df)),
            ("Baseline summary",               lambda: self.tab_baseline_summary(df)),
            ("Constraint diagnostics",         lambda: self.tab_constraint_diagnostics(df)),
        ]
        for name, fn in table_fns:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["tables"].extend(out)
                elif isinstance(out, str) and out:
                    gen["tables"].append(out)
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING table '{name}': {e}")

        # ---- Phase 11 tables: Narrative-strengthening tables (N1-N7) ----
        phase11_fns = [
            ("Table N1 (constraint diag cross-config)",
             lambda: self.tab_constraint_diagnostics_cross_config(df)),
            ("Table N2 (objective ablation summary)",
             lambda: self.tab_objective_ablation_summary(df)),
            ("Table N3 (representation transfer summary)",
             lambda: self.tab_representation_transfer_summary(df)),
            ("Table N4 (SKL ablation summary)",
             lambda: self.tab_skl_ablation_summary(df)),
            ("Table N5 (multi-seed statistics)",
             lambda: self.tab_multi_seed_statistics(df)),
            ("Table N6 (worst-state RMSE by k)",
             lambda: self.tab_worst_state_rmse_by_k(df)),
            ("Table N7 (baseline paired unconstrained vs quota)",
             lambda: self.tab_baseline_paired_unconstrained_vs_quota(df)),
            ("Table N8 (downstream model comparison)",
             lambda: self.tab_downstream_model_comparison(df)),
            ("Table N9 (dimensionality sweep)",
             lambda: self.tab_dimensionality_sweep(df)),
        ]
        for name, fn in phase11_fns:
            try:
                out = fn()
                if isinstance(out, list):
                    gen["tables"].extend(out)
                elif isinstance(out, str) and out:
                    gen["tables"].append(out)
                print(f"[ManuscriptArtifacts] OK  {name}")
            except Exception as e:
                print(f"[ManuscriptArtifacts] WARNING Phase 11 table '{name}': {e}")

        print(f"[ManuscriptArtifacts] Generated {len(gen['figures'])} figures, "
              f"{len(gen['tables'])} tables")
        return gen
