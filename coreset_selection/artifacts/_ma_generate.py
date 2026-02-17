"""Generate-all orchestrator and data-loading helpers for ManuscriptArtifacts (mixin).

Reorganised to match the manuscript's actual figure/table references:

Manuscript figures (directly referenced via \\includegraphics):
  Fig 1: kl_floor_vs_k.pdf               — Section VIII.A
  Fig 2: geo_ablation_tradeoff_scatter.pdf — Section VIII.B
  Fig 3: distortion_cardinality_R1.pdf    — Section VIII.C
  Fig 4: krr_worst_state_rmse_vs_k.pdf    — Section VIII.D
  Fig 5: regional_validity_k300.pdf       — Section VIII.D
  Fig 6: baseline_comparison_grouped.pdf  — Section VIII.E
  Fig 7: representation_transfer_bars.pdf — Section VIII.G
  Fig 8: objective_metric_alignment_heatmap.pdf — Section VIII.K
  Fig 9: downstream_model_heatmap.pdf     — Section VIII.C

Manuscript tables (directly referenced via \\label):
  Table I:   exp_settings.tex             (tab:exp-settings)
  Table II:  run_matrix.tex               (tab:run-matrix)
  Table III: r1_by_k.tex                  (tab:r1-by-k)
  Table IV:  repr_timing.tex              (tab:repr-timing)      — NEW
  Table V:   proxy_stability.tex          (tab:proxy-stability)
  Table VI:  krr_multitask_k300.tex       (tab:krr-multitask-k300)
"""
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

    def _load_front_metrics(self, run_pattern: Optional[str] = None,
                            space: Optional[str] = None) -> pd.DataFrame:
        """Load concatenated front_metrics CSVs (full Pareto front evaluations).

        Parameters
        ----------
        run_pattern : str, optional
            If provided, only load CSVs whose path contains this substring
            (e.g. ``"R1"``).
        space : str, optional
            Representation space filter (e.g. ``"raw"``, ``"vae"``, ``"pca"``).
            If provided, only loads ``front_metrics_{space}.csv``.

        Returns
        -------
        pd.DataFrame
            Concatenated front metrics with one row per evaluated Pareto
            solution.  Empty DataFrame if no files found.
        """
        pattern = f"front_metrics_{space}.csv" if space else "front_metrics_*.csv"
        dfs: List[pd.DataFrame] = []
        for path in glob.glob(
            os.path.join(self.runs_root, "**", pattern), recursive=True
        ):
            if run_pattern and run_pattern not in path:
                continue
            try:
                dfs.append(pd.read_csv(path))
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        if "k" not in df.columns and "run_id" in df.columns:
            df["k"] = df["run_id"].astype(str).str.extract(r"_k(\d+)").astype(float)
        return df

    @staticmethod
    def _run_artifact(gen: Dict[str, List[str]], key: str,
                      name: str, fn, *, critical: bool = False) -> None:
        """Run a single artifact generator and record the result.

        Parameters
        ----------
        gen : dict
            The accumulation dict ``{"figures": [...], "tables": [...]}``.
        key : str
            Either ``"figures"`` or ``"tables"``.
        name : str
            Human-readable label for log messages.
        fn : callable
            Zero-argument callable that returns a path (str) or list of paths.
        critical : bool
            If True, failures are printed as FAIL; otherwise WARNING.
        """
        try:
            out = fn()
            if isinstance(out, list):
                gen[key].extend(out)
            elif isinstance(out, str) and out:
                gen[key].append(out)
            print(f"[ManuscriptArtifacts] OK  {name}")
        except (ImportError, FileNotFoundError) as e:
            tag = "FAIL" if critical else "SKIP"
            print(f"[ManuscriptArtifacts] {tag} {name}: {e}")
        except Exception as e:
            tag = "FAIL" if critical else "WARNING"
            print(f"[ManuscriptArtifacts] {tag} {name}: {e}")

    def generate_all(self) -> Dict[str, List[str]]:
        """Generate all manuscript and complementary artifacts.

        Artefacts are generated in priority order:
        1. Manuscript figures (8 figures directly referenced in the paper)
        2. Manuscript tables (6 tables directly referenced in the paper)
        3. Complementary / narrative-strengthening figures (N1–N12, geo maps)
        4. Complementary / narrative-strengthening tables (N1–N9)

        Returns
        -------
        Dict[str, List[str]]
            ``{"figures": [...], "tables": [...]}`` with paths to every
            generated artefact.
        """
        _set_style()
        df = self._load_df()
        gen: Dict[str, List[str]] = {"figures": [], "tables": []}
        _fig = lambda name, fn: self._run_artifact(gen, "figures", name, fn, critical=True)
        _tab = lambda name, fn: self._run_artifact(gen, "tables", name, fn, critical=True)
        _cfig = lambda name, fn: self._run_artifact(gen, "figures", name, fn, critical=False)
        _ctab = lambda name, fn: self._run_artifact(gen, "tables", name, fn, critical=False)

        # ==============================================================
        # MANUSCRIPT FIGURES — directly referenced via \includegraphics
        # These are the 8 figures that appear in the manuscript body.
        # Failures here are critical and should be immediately visible.
        # ==============================================================
        print(f"\n{'='*60}")
        print("MANUSCRIPT FIGURES (8 paper-referenced)")
        print(f"{'='*60}\n")

        _fig("Fig 1 — KL feasibility floor vs k (Sec. VIII.A)",
             lambda: self.fig_kl_floor_vs_k())

        _fig("Fig 2 — Geo ablation scatter (Sec. VIII.B)",
             lambda: self.fig_geo_ablation_scatter(df))

        _fig("Fig 3 — Distortion vs cardinality R1 (Sec. VIII.C)",
             lambda: self.fig_distortion_cardinality_r1(df))

        _fig("Fig 4 — Worst-state RMSE vs k (Sec. VIII.D)",
             lambda: self.fig_krr_worst_state_rmse_vs_k(df))

        _fig("Fig 5 — Regional validity at k=300 (Sec. VIII.D)",
             lambda: self.fig_regional_validity_k300(df))

        _fig("Fig 6 — Baseline comparison grouped (Sec. VIII.E)",
             lambda: self.fig_baseline_comparison(df))

        _fig("Fig 7 — Representation transfer bars (Sec. VIII.G)",
             lambda: self.fig_representation_transfer(df))

        _fig("Fig 8 — Objective-metric alignment heatmap (Sec. VIII.K)",
             lambda: self.fig_objective_metric_alignment(df))

        _fig("Fig 9 — Downstream model heatmap (Sec. VIII.C)",
             lambda: self.fig_downstream_model_heatmap(df))

        # ==============================================================
        # MANUSCRIPT TABLES — directly referenced via \label{tab:...}
        # These are the 6 tables that appear in the manuscript body.
        # ==============================================================
        print(f"\n{'='*60}")
        print("MANUSCRIPT TABLES (6 paper-referenced)")
        print(f"{'='*60}\n")

        _tab("Table I — Experiment settings (tab:exp-settings)",
             lambda: self.tab_exp_settings())

        _tab("Table II — Run matrix (tab:run-matrix)",
             lambda: self.tab_run_matrix())

        _tab("Table III — R1 by k (tab:r1-by-k)",
             lambda: self.tab_r1_by_k(df))

        _tab("Table IV — Repr timing (tab:repr-timing)",
             lambda: self.tab_repr_timing(df))

        _tab("Table V — Proxy stability (tab:proxy-stability)",
             lambda: self.tab_proxy_stability(df))

        _tab("Table VI — KRR multi-task k=300 (tab:krr-multitask-k300)",
             lambda: self.tab_krr_multitask(df))

        # ==============================================================
        # COMPLEMENTARY FIGURES — narrative-strengthening (Phase 10a/b)
        # These support the manuscript narrative and pre-empt reviewer
        # concerns but are not directly \includegraphics'd in the paper.
        # ==============================================================
        print(f"\n{'='*60}")
        print("COMPLEMENTARY FIGURES (narrative-strengthening)")
        print(f"{'='*60}\n")

        # Phase 10a (Figs N1-N6)
        _cfig("Fig N1 — Pareto front MMD vs SD at k=300",
              lambda: self.fig_pareto_front_k300(df))
        _cfig("Fig N2 — Objective ablation bars (R1/R2/R3)",
              lambda: self.fig_objective_ablation_bars(df))
        _cfig("Fig N3 — Constraint comparison bars (R1/R4/R5/R6)",
              lambda: self.fig_constraint_comparison(df))
        _cfig("Fig N4 — Effort-quality trade-off (R12)",
              lambda: self.fig_effort_quality(df))

        # Phase 10b (Figs N7-N12)
        _cfig("Fig N5 — Multi-seed stability boxplot",
              lambda: self.fig_multi_seed_boxplot(df))
        _cfig("Fig N6 — State KPI drift heatmap",
              lambda: self.fig_state_kpi_heatmap(df))
        _cfig("Fig N7 — Composition shift (R6 vs R1)",
              lambda: self.fig_composition_shift(df))
        _cfig("Fig N8 — Pareto front evolution",
              lambda: self.fig_pareto_front_evolution(df))
        _cfig("Fig N9 — Nystrom error distribution",
              lambda: self.fig_nystrom_error_distribution(df))

        # Legacy complementary
        _cfig("Cumulative Pareto improvement",
              lambda: self.fig_cumulative_pareto_improvement(df))
        _cfig("Constraint tightness vs fidelity",
              lambda: self.fig_constraint_tightness_vs_fidelity(df))
        _cfig("SKL ablation comparison (R7 vs R9)",
              lambda: self.fig_skl_ablation_comparison(df))

        # Phase 12: Geographic choropleth maps (N13-N17)
        _cfig("Fig N10 — Coreset map k-sweep",
              lambda: self.fig_coreset_map_k_sweep(df))
        _cfig("Fig N11 — Coreset map representatives",
              lambda: self.fig_coreset_map_representatives(df))
        _cfig("Fig N12 — Coreset map baselines",
              lambda: self.fig_coreset_map_baselines(df))
        _cfig("Fig N13 — Coreset map constraint comparison",
              lambda: self.fig_coreset_map_constraint_comparison(df))
        _cfig("Fig N14 — Coreset map representation comparison",
              lambda: self.fig_coreset_map_representation_comparison(df))

        # ==============================================================
        # COMPLEMENTARY TABLES — narrative-strengthening (Phase 11)
        # ==============================================================
        print(f"\n{'='*60}")
        print("COMPLEMENTARY TABLES (narrative-strengthening)")
        print(f"{'='*60}\n")

        _ctab("Baseline summary CSV",
              lambda: self.tab_baseline_summary(df))
        _ctab("Constraint diagnostics CSV",
              lambda: self.tab_constraint_diagnostics(df))

        # Phase 11 tables (N1-N9)
        _ctab("Table N1 — Constraint diagnostics cross-config",
              lambda: self.tab_constraint_diagnostics_cross_config(df))
        _ctab("Table N2 — Objective ablation summary",
              lambda: self.tab_objective_ablation_summary(df))
        _ctab("Table N3 — Representation transfer summary",
              lambda: self.tab_representation_transfer_summary(df))
        _ctab("Table N4 — SKL ablation summary",
              lambda: self.tab_skl_ablation_summary(df))
        _ctab("Table N5 — Multi-seed statistics",
              lambda: self.tab_multi_seed_statistics(df))
        _ctab("Table N6 — Worst-state RMSE by k",
              lambda: self.tab_worst_state_rmse_by_k(df))
        _ctab("Table N7 — Baseline paired (unconstrained vs quota)",
              lambda: self.tab_baseline_paired_unconstrained_vs_quota(df))
        _ctab("Table N8 — Downstream model comparison",
              lambda: self.tab_downstream_model_comparison(df))
        _ctab("Table N9 — Dimensionality sweep",
              lambda: self.tab_dimensionality_sweep(df))

        # ==============================================================
        # SUMMARY
        # ==============================================================
        print(f"\n{'='*60}")
        print(f"[ManuscriptArtifacts] Generated {len(gen['figures'])} figures, "
              f"{len(gen['tables'])} tables")
        print(f"{'='*60}")
        return gen
