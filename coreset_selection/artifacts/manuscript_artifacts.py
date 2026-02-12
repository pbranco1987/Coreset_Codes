r"""Manuscript-aligned artifact generator.

Produces ALL figures and tables referenced in the manuscript plus
complementary analyses for Section VIII (Results and Analysis).

Manuscript Figures (8, directly referenced via \includegraphics):
  Fig 1: kl_floor_vs_k.pdf               — KL feasibility floor (Sec. VIII.A)
  Fig 2: geo_ablation_tradeoff_scatter.pdf — Geographic ablation (Sec. VIII.B)
  Fig 3: distortion_cardinality_R1.pdf    — Raw-space metrics vs k (Sec. VIII.C)
  Fig 4: krr_worst_state_rmse_vs_k.pdf    — Worst-state RMSE equity (Sec. VIII.D)
  Fig 5: regional_validity_k300.pdf       — Regional KPI validity (Sec. VIII.D)
  Fig 6: baseline_comparison_grouped.pdf  — Baseline comparison (Sec. VIII.E)
  Fig 7: representation_transfer_bars.pdf — Representation transfer (Sec. VIII.G)
  Fig 8: objective_metric_alignment_heatmap.pdf — Obj-metric Spearman (Sec. VIII.K)

Manuscript Tables (6, directly referenced via \label):
  Table I:   exp_settings.tex             (tab:exp-settings)
  Table II:  run_matrix.tex               (tab:run-matrix)
  Table III: r1_by_k.tex                  (tab:r1-by-k)
  Table IV:  repr_timing.tex              (tab:repr-timing)
  Table V:   proxy_stability.tex          (tab:proxy-stability)
  Table VI:  krr_multitask_k300.tex       (tab:krr-multitask-k300)

Plus ~15 complementary figures (Pareto fronts, boxplots, heatmaps,
geographic choropleth maps) and ~12 complementary tables for
narrative-strengthening and reviewer defense.

IEEE compliance:
- Column widths: single-col ~3.5 in, double-col ~7 in.
- Font sizes: >= 8 pt for annotations, >= 10 pt for axis labels.
- Consistent grid, spine removal, and publication-quality style.
- 300 dpi saved output.

Implementation split across mixin modules for maintainability:
- _ma_helpers.py:          Shared helpers (_set_style, _save)
- _ma_generate.py:         generate_all() orchestrator + data loading
- _ma_geo_maps.py:         Geographic choropleth map methods
- _ma_pareto_figs.py:      Pareto front figure methods
- _ma_metric_figs.py:      Metric/distortion figure methods
- _ma_comparison_figs.py:  Comparison/diagnostic figure methods
- _ma_tables.py:           Table generation methods (incl. repr_timing)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Re-export helpers at module level for backward compatibility
from ._ma_helpers import _set_style, _save  # noqa: F401

# Import all mixin classes
from ._ma_generate import GenerateAllMixin
from ._ma_geo_maps import GeoMapsMixin
from ._ma_pareto_figs import ParetoFigsMixin
from ._ma_metric_figs import MetricFigsMixin
from ._ma_comparison_figs import ComparisonFigsMixin
from ._ma_tables import TablesMixin


class ManuscriptArtifacts(
    GenerateAllMixin,
    GeoMapsMixin,
    ParetoFigsMixin,
    MetricFigsMixin,
    ComparisonFigsMixin,
    TablesMixin,
):
    """Generate all manuscript figures and tables from collected results.

    Methods are organized across mixin modules by theme:
    - GenerateAllMixin:    generate_all(), _load_df(), _load_pareto()
    - GeoMapsMixin:        fig_coreset_map_* (N13-N17), geographic helpers
    - ParetoFigsMixin:     fig_pareto_front_*, fig_cumulative_pareto_*
    - MetricFigsMixin:     fig_distortion_*, fig_effort_*, fig_kl_floor_*,
                           fig_nystrom_*, fig_krr_*, fig_multi_seed_*
    - ComparisonFigsMixin: fig_geo_ablation_*, fig_regional_*, fig_objective_*,
                           fig_constraint_*, fig_baseline_*, fig_representation_*,
                           fig_state_kpi_*, fig_composition_*, fig_skl_*
    - TablesMixin:         tab_* (all table generation methods)
    """

    def __init__(self, runs_root: str, cache_root: str, out_dir: str,
                 data_dir: Optional[str] = None):
        self.runs_root = runs_root
        self.cache_root = cache_root
        self.data_dir = data_dir
        self.fig_dir = os.path.join(out_dir, "figures")
        self.tab_dir = os.path.join(out_dir, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.tab_dir, exist_ok=True)
        # Lazy caches for geo-map data
        self._ibge_codes: Optional[np.ndarray] = None
        self._gdf = None   # gpd.GeoDataFrame (cached)
        self._border = None  # gpd.GeoSeries   (cached)
