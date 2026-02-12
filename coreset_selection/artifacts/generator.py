"""
Manuscript artifact generator.

Contains:
- ManuscriptArtifactGenerator: Generate all figures and tables for the paper
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.cache import load_replicate_cache
from ..geo import build_geo_info
from ..geo.kl import min_achievable_geo_kl_bounded
from ..experiment.saver import load_pareto_front
from ..utils.io import ensure_dir
from ..utils.plotting import set_manuscript_style

from .plots import (
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_metric_vs_k,
    plot_geographic_distribution,
    save_pareto_front_plots,
)
from .tables import (
    generate_latex_table,
    generate_csv_table,
    results_summary_table,
    baseline_comparison_table,
    run_specs_table,
    geographic_distribution_table,
)

from ._gen_figures import GenFiguresMixin
from ._gen_batches import GenBatchesMixin


class ManuscriptArtifactGenerator(GenFiguresMixin, GenBatchesMixin):
    """
    Generate manuscript figures and tables from experiment results.

    Produces:
    - R0: KL minimum achievability curves
    - R1: Cardinality curves and Pareto front plots
    - R2-R6: Ablation study figures
    - R6: Diagnostic plots
    - R7-R8: Representation transfer analysis

    Attributes
    ----------
    runs_root : str
        Root directory containing run outputs
    cache_root : str
        Root directory containing replicate caches
    out_dir : str
        Output directory for generated artifacts
    """

    def __init__(
        self,
        runs_root: str,
        cache_root: str,
        out_dir: str,
    ):
        """
        Initialize the artifact generator.

        Parameters
        ----------
        runs_root : str
            Root directory with run results (e.g., "runs_out/")
        cache_root : str
            Root directory with replicate caches
        out_dir : str
            Output directory for artifacts
        """
        self.runs_root = runs_root
        self.cache_root = cache_root
        self.out_dir = ensure_dir(out_dir)

        # Subdirectories
        self.figures_dir = ensure_dir(os.path.join(out_dir, "figures"))
        self.tables_dir = ensure_dir(os.path.join(out_dir, "tables"))

    def generate_all(self) -> Dict[str, List[str]]:
        """
        Generate all manuscript artifacts.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from artifact type to list of generated file paths
        """
        set_manuscript_style()

        generated = {
            "figures": [],
            "tables": [],
        }

        # Load all results
        df_all = self._load_all_results()

        if len(df_all) == 0:
            print("[ArtifactGenerator] No results found!")
            return generated

        print(f"[ArtifactGenerator] Loaded {len(df_all)} result rows")

        # Generate manuscript-facing artifacts with the exact filenames
        # referenced in main.tex (figures/<name>.pdf).
        for fn in [
            self.figure_klmin_vs_k,
            lambda: self.figure_pareto_r1_k300(),
            lambda: self.figure_distortion_cardinality_r1(df_all),
            lambda: self.figure_baseline_comparison_k300(df_all),
            lambda: self.figure_geo_ablation_scatter(df_all),
            lambda: self.figure_objective_ablation_fronts_k300(),
            lambda: self.figure_representation_transfer_k300(df_all),
            lambda: self.figure_pareto_biobjective_grid_k300(),
            lambda: self.figure_r7_surrogate_sensitivity(),
            lambda: self.figure_r7_repair_histograms(),
            lambda: self.figure_r7_objective_metric_alignment(),
        ]:
            try:
                out = fn()
                if isinstance(out, list):
                    generated["figures"].extend(out)
                elif isinstance(out, str) and out:
                    generated["figures"].append(out)
            except Exception as e:
                print(f"[ArtifactGenerator] WARNING: {getattr(fn, '__name__', 'artifact')} failed: {e}")

        # Tables (optional external .tex snippets mirroring the LaTeX tables in main.tex)
        try:
            generated["tables"].extend(self.write_table_snippets(df_all))
        except Exception as e:
            print(f"[ArtifactGenerator] WARNING: table snippets failed: {e}")

        return generated

    def _load_all_results(self) -> pd.DataFrame:
        """Load and concatenate all result CSV files."""
        all_dfs = []

        # Find all results files
        pattern = os.path.join(self.runs_root, "**/all_results.csv")
        result_files = glob.glob(pattern, recursive=True)

        for path in result_files:
            try:
                df = pd.read_csv(path)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Manuscript-facing figures (exact filenames referenced in main.tex)
    # ------------------------------------------------------------------

    def _save_pdf(self, fig, filename: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
