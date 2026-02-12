r"""Manuscript-aligned artifact generator (Phase 8 + Phase 9 + Phase 10 + Phase 11 compliant).

Produces ALL figures and tables referenced in the manuscript plus
complementary analyses for Section VIII (Results and Analysis).

Phase 8 ensures the four manuscript-referenced figures comply with:
- IEEE column width conventions (single-col ~3.5 in, double-col ~7 in).
- Font sizes: >= 8 pt for annotations, >= 10 pt for axis labels.
- Consistent grid, spine removal, and plot style.
- Correct data loading paths and fallback logic.

Phase 9 ensures the five manuscript-referenced tables comply with:
- Table I  (exp_settings.tex):       Component | Setting layout; all values
                                     sourced from config/constants.py.
- Table II (run_matrix.tex):         ID | k | Opt. repr | Constraints |
                                     Objectives | Seeds | Purpose; sourced
                                     from config/run_specs.py.
- Table III (r1_by_k.tex):           k | e_Nys | e_kPCA | RMSE(4G) | RMSE(5G).
                                     Multi-seed: mean envelope across seeds.
- Table IV (proxy_stability.tex):    Three sections (RFF sweep, anchor sweep,
                                     cross-representation); Spearman rho values.
- Table V  (krr_multitask_k300.tex): 10 coverage targets x 3 columns
                                     (R1 knee, R9 knee, Best pool); best
                                     values bolded.  R9 uses VAE-mean
                                     representation.

Phase 10a introduces enhanced narrative-strengthening figures (Figs N1-N6)
that go beyond the 4 manuscript-referenced figures to pre-empt likely
reviewer concerns and strengthen the empirical narrative.

Phase 10b (Figs N7-N12): New/enhanced narrative-strengthening figures.

Phase 11 (Tables N1-N9): New/enhanced narrative-strengthening tables.

Phase 12 (Figs N13-N17): Geographic choropleth map figures.

Implementation split across mixin modules for maintainability:
- _ma_helpers.py:          Shared helpers (_set_style, _save)
- _ma_generate.py:         generate_all() orchestrator + data loading
- _ma_geo_maps.py:         Geographic choropleth map methods
- _ma_pareto_figs.py:      Pareto front figure methods
- _ma_metric_figs.py:      Metric/distortion figure methods
- _ma_comparison_figs.py:  Comparison/diagnostic figure methods
- _ma_tables.py:           Table generation methods
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
