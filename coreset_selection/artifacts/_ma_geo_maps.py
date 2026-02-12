"""Geographic choropleth map mixin for ManuscriptArtifacts.

Provides the ``GeoMapsMixin`` class containing all geographic helper
methods (IBGE code loading, shapefile loading, selection scanning) and
the choropleth figure generators (Figs N13--N17).

These methods rely on instance attributes initialised by the main
``ManuscriptArtifacts.__init__``:
    self.runs_root, self.data_dir, self.fig_dir,
    self._ibge_codes, self._gdf, self._border
"""

from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np
import pandas as pd

from ._ma_helpers import _save


# ---------------------------------------------------------------------------
# Mixin: geographic choropleth maps
# ---------------------------------------------------------------------------

class GeoMapsMixin:
    """Geographic choropleth map helpers and figure methods."""

    # ── Geographic helpers ─────────────────────────────────────────────

    def _load_ibge_codes(self) -> np.ndarray:
        """Load IBGE municipality codes from the raw data files.

        Returns
        -------
        np.ndarray
            Shape (N,) with 7-digit IBGE codes aligned to experiment indices.
        """
        if self._ibge_codes is not None:
            return self._ibge_codes
        if self.data_dir is None:
            raise ValueError(
                "data_dir must be set on ManuscriptArtifacts to generate "
                "choropleth maps.  Pass data_dir= when constructing the object."
            )
        from ..data.brazil_telecom_loader import BrazilTelecomDataLoader
        loader = BrazilTelecomDataLoader(self.data_dir)
        data = loader.load()
        self._ibge_codes = data.ibge_codes
        return self._ibge_codes

    def _load_shapefile(self):
        """Load IBGE municipality shapefile and dissolved border.

        Returns
        -------
        Tuple[gpd.GeoDataFrame, gpd.GeoSeries]
        """
        if self._gdf is not None:
            return self._gdf, self._border
        if self.data_dir is None:
            raise ValueError(
                "data_dir must be set on ManuscriptArtifacts to generate "
                "choropleth maps."
            )
        from ..geo.shapefile import load_brazil_municipalities, get_brazil_border

        # Look for shapefile directory candidates
        shapefile_dir = None
        for candidate in [
            os.path.join(self.data_dir, "BR_Municipios_2024"),
            os.path.join(self.data_dir, "shapefiles", "BR_Municipios_2024"),
            os.path.join(self.data_dir, "shapefiles"),
            self.data_dir,
        ]:
            if os.path.isdir(candidate):
                shapefile_dir = candidate
                break

        if shapefile_dir is None:
            shapefile_dir = os.path.join(self.data_dir, "BR_Municipios_2024")

        self._gdf = load_brazil_municipalities(shapefile_dir)
        self._border = get_brazil_border(self._gdf)
        return self._gdf, self._border

    def _load_pareto_selections(
        self, run_id: str, space: str = "raw",
    ) -> Dict[str, np.ndarray]:
        """Load Pareto front selections for a given run.

        Returns
        -------
        Dict[str, np.ndarray]
            Maps representative name -> selected indices (int array).
            Empty dict if no Pareto front found.
        """
        from ..experiment.saver import load_pareto_front

        selections: Dict[str, np.ndarray] = {}
        patterns = [
            os.path.join(self.runs_root, run_id, "rep*", "results",
                         f"{space}_pareto.npz"),
            os.path.join(self.runs_root, f"{run_id}_k300", "rep*", "results",
                         f"{space}_pareto.npz"),
        ]
        for pat in patterns:
            for p in sorted(glob.glob(pat)):
                try:
                    pf = load_pareto_front(p)
                    for name in pf.representatives:
                        try:
                            selections[name] = pf.get_indices(name)
                        except Exception:
                            pass
                    return selections
                except Exception:
                    continue
        return selections

    def _load_baseline_selections(
        self, run_id: str = "R10",
    ) -> Dict[str, np.ndarray]:
        """Load baseline coreset index files.

        Returns
        -------
        Dict[str, np.ndarray]
            Maps label (derived from filename) -> selected indices.
        """
        selections: Dict[str, np.ndarray] = {}
        patterns = [
            os.path.join(self.runs_root, f"{run_id}*", "rep*",
                         "coresets", "*.npz"),
        ]
        for pat in patterns:
            for fpath in sorted(glob.glob(pat)):
                bname = os.path.splitext(os.path.basename(fpath))[0]
                try:
                    data = np.load(fpath, allow_pickle=True)
                    if "indices" in data:
                        selections[bname] = np.asarray(data["indices"],
                                                       dtype=int)
                except Exception:
                    continue
        return selections

    def _scan_k_sweep_selections(
        self,
        run_id: str,
        space: str = "raw",
        rep_name: str = "knee",
    ) -> Dict[int, np.ndarray]:
        """Load selections across a k-sweep for one representative.

        Returns
        -------
        Dict[int, np.ndarray]
            Maps k value -> selected indices.
        """
        import re
        from ..experiment.saver import load_pareto_front

        selections: Dict[int, np.ndarray] = {}

        # Pattern 1: Pareto front NPZ
        pareto_pat = os.path.join(
            self.runs_root, f"{run_id}_k*", "rep*", "results",
            f"{space}_pareto.npz",
        )
        for p in sorted(glob.glob(pareto_pat)):
            m = re.search(rf"{run_id}_k(\d+)", p)
            if not m:
                continue
            k = int(m.group(1))
            try:
                pf = load_pareto_front(p)
                if rep_name in pf.representatives:
                    selections[k] = pf.get_indices(rep_name)
            except Exception:
                continue

        # Pattern 2: Direct coreset NPZ fallback
        coreset_pat = os.path.join(
            self.runs_root, f"{run_id}_k*", "rep*", "coresets",
            f"{space}_{rep_name}.npz",
        )
        for fpath in sorted(glob.glob(coreset_pat)):
            m = re.search(rf"{run_id}_k(\d+)", fpath)
            if not m:
                continue
            k = int(m.group(1))
            if k in selections:
                continue  # already have from Pareto front
            try:
                data = np.load(fpath, allow_pickle=True)
                if "indices" in data:
                    selections[k] = np.asarray(data["indices"], dtype=int)
            except Exception:
                continue

        return dict(sorted(selections.items()))

    # ── Geographic figure methods ──────────────────────────────────────

    def _indices_to_ibge(self, indices: np.ndarray) -> np.ndarray:
        """Convert 0-based selection indices to 7-digit IBGE codes."""
        ibge = self._load_ibge_codes()
        return ibge[indices]

    # ── Fig N13: k-sweep map ──────────────────────────────────────────

    def fig_coreset_map_k_sweep(self, df: pd.DataFrame) -> str:
        r"""Fig N13: Geographic coverage across cardinality sweep (R1 knee).

        Composite 2 |times| 3 panel — one panel per k in K_GRID.
        Selected municipalities highlighted; country border visible.
        """
        import matplotlib.pyplot as plt
        from ..geo.shapefile import plot_choropleth_panel

        gdf, border = self._load_shapefile()
        k_selections = self._scan_k_sweep_selections("R1", "raw", "knee")

        if not k_selections:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No R1 k-sweep data found",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return _save(fig, os.path.join(self.fig_dir,
                                           "coreset_map_k_sweep_R1.pdf"))

        k_values = sorted(k_selections.keys())
        n_panels = len(k_values)
        ncols = min(n_panels, 3)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7.0, 2.6 * nrows))
        if n_panels == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, k_val in enumerate(k_values):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            indices = k_selections[k_val]
            codes = self._indices_to_ibge(indices)
            title = f"$k = {k_val}$ ({len(indices)} sel.)"
            plot_choropleth_panel(ax, gdf, border, codes, title=title)

        # Hide unused panels
        for idx in range(n_panels, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle("R1 knee-point selections across cardinality $k$",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(self.fig_dir,
                                       "coreset_map_k_sweep_R1.pdf"))

    # ── Fig N14: Pareto representative map ────────────────────────────

    def fig_coreset_map_representatives(self, df: pd.DataFrame) -> str:
        r"""Fig N14: Knee vs best-MMD vs best-Sinkhorn at k = 300 (R1).

        Composite 1 |times| 3 panel, each representative in a distinct colour.
        """
        import matplotlib.pyplot as plt
        from ..geo.shapefile import plot_choropleth_panel

        gdf, border = self._load_shapefile()
        sels = self._load_pareto_selections("R1", "raw")

        # Canonical representative ordering and colours
        rep_display = [
            ("knee",           "Knee",         "#1f77b4"),
            ("best_mmd",       "Best MMD",     "#2ca02c"),
            ("best_sinkhorn",  "Best Sinkhorn","#d62728"),
        ]
        # Filter to those actually found
        available = [(key, label, col) for key, label, col in rep_display
                     if key in sels]

        if not available:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No R1 Pareto selections found",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return _save(fig, os.path.join(
                self.fig_dir, "coreset_map_representatives_R1_k300.pdf"))

        ncols = len(available)
        fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 2.8))
        if ncols == 1:
            axes = [axes]

        for ax, (key, label, col) in zip(axes, available):
            indices = sels[key]
            codes = self._indices_to_ibge(indices)
            plot_choropleth_panel(ax, gdf, border, codes,
                                 title=f"{label} ({len(indices)})",
                                 color=col)

        fig.suptitle(r"R1 Pareto representatives ($k{=}300$)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.fig_dir, "coreset_map_representatives_R1_k300.pdf"))

    # ── Fig N15: Baseline comparison map ──────────────────────────────

    def fig_coreset_map_baselines(self, df: pd.DataFrame) -> str:
        r"""Fig N15: Baseline method selections at k = 300 (R10, raw, exact-k).

        Composite 2 |times| 4 panel — one per baseline method.
        """
        import matplotlib.pyplot as plt
        from ..geo.shapefile import plot_choropleth_panel

        gdf, border = self._load_shapefile()
        all_baselines = self._load_baseline_selections("R10")

        # Filter to raw-space, exact-k baselines
        display_order = [
            ("U",    "Uniform"),
            ("KM",   "K-means"),
            ("KH",   "Herding"),
            ("FF",   "Farthest-first"),
            ("RLS",  "RLS"),
            ("DPP",  "DPP"),
            ("KT",   "Kernel Thin."),
            ("KKN",  "KKM-Nys."),
            ("SKKN", "S-KKM-Nys."),
        ]

        panels = []
        for prefix, label in display_order:
            # Match {prefix}_raw_exactk
            for bname, indices in all_baselines.items():
                if bname.startswith(prefix + "_") and "raw" in bname and "exactk" in bname:
                    panels.append((label, indices))
                    break

        if not panels:
            # Fallback: show whatever raw baselines we have
            for bname, indices in sorted(all_baselines.items()):
                if "raw" in bname:
                    panels.append((bname, indices))
            if not panels:
                fig, ax = plt.subplots(figsize=(3.5, 2.8))
                ax.text(0.5, 0.5, "No R10 baseline data found",
                        ha="center", va="center", fontsize=10,
                        transform=ax.transAxes)
                ax.set_axis_off()
                return _save(fig, os.path.join(
                    self.fig_dir, "coreset_map_baselines_k300.pdf"))

        ncols = min(len(panels), 4)
        nrows = (len(panels) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7.0, 2.6 * nrows))
        if len(panels) == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        palette = [
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "#ff7f0e", "#2ca02c", "#d62728",
        ]

        for idx, (label, indices) in enumerate(panels):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            codes = self._indices_to_ibge(indices)
            col = palette[idx % len(palette)]
            plot_choropleth_panel(ax, gdf, border, codes,
                                 title=f"{label} ({len(indices)})",
                                 color=col)

        for idx in range(len(panels), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle(r"Baseline selections ($k{=}300$, raw space, exact-$k$)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.fig_dir, "coreset_map_baselines_k300.pdf"))

    # ── Fig N16: Constraint comparison map ────────────────────────────

    def fig_coreset_map_constraint_comparison(self, df: pd.DataFrame) -> str:
        r"""Fig N16: Effect of constraint regime on geographic coverage.

        Composite 2 |times| 2: R1 (pop-share), R4 (muni-quota), R5 (joint),
        R6 (no constraint).  All at k = 300, knee representative.
        """
        import matplotlib.pyplot as plt
        from ..geo.shapefile import plot_choropleth_panel

        gdf, border = self._load_shapefile()

        configs = [
            ("R1",  "raw", "Pop-share (R1)"),
            ("R4",  "raw", "Muni-quota (R4)"),
            ("R5",  "raw", "Joint (R5)"),
            ("R6",  "raw", "No constraint (R6)"),
        ]

        panels = []
        for run_id, space, label in configs:
            sels = self._load_pareto_selections(run_id, space)
            if "knee" in sels:
                panels.append((label, sels["knee"]))
            elif sels:
                # Take whatever representative is available
                first_key = next(iter(sels))
                panels.append((label, sels[first_key]))

        if not panels:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No constraint comparison data found",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return _save(fig, os.path.join(
                self.fig_dir,
                "coreset_map_constraint_comparison_k300.pdf"))

        ncols = 2
        nrows = (len(panels) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7.0, 3.0 * nrows))
        if len(panels) == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, (label, indices) in enumerate(panels):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            codes = self._indices_to_ibge(indices)
            col = colours[idx % len(colours)]
            plot_choropleth_panel(ax, gdf, border, codes,
                                 title=f"{label} ({len(indices)})",
                                 color=col)

        for idx in range(len(panels), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle(r"Constraint regime comparison ($k{=}300$, knee)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.fig_dir, "coreset_map_constraint_comparison_k300.pdf"))

    # ── Fig N17: Representation comparison map ────────────────────────

    def fig_coreset_map_representation_comparison(
        self, df: pd.DataFrame,
    ) -> str:
        r"""Fig N17: Raw vs PCA vs VAE representation effect on coverage.

        1 |times| 3 panel: R1 (raw), R8 (PCA), R9 (VAE), all k = 300 knee.
        """
        import matplotlib.pyplot as plt
        from ..geo.shapefile import plot_choropleth_panel

        gdf, border = self._load_shapefile()

        configs = [
            ("R1",  "raw", "Raw (R1)"),
            ("R8",  "pca", "PCA (R8)"),
            ("R9",  "vae", "VAE (R9)"),
        ]

        panels = []
        for run_id, space, label in configs:
            sels = self._load_pareto_selections(run_id, space)
            if "knee" in sels:
                panels.append((label, sels["knee"]))
            elif sels:
                first_key = next(iter(sels))
                panels.append((label, sels[first_key]))

        if not panels:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.text(0.5, 0.5, "No representation comparison data found",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return _save(fig, os.path.join(
                self.fig_dir,
                "coreset_map_representation_comparison_k300.pdf"))

        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 2.8))
        if ncols == 1:
            axes = [axes]

        colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for ax, (label, indices), col in zip(axes, panels, colours):
            codes = self._indices_to_ibge(indices)
            plot_choropleth_panel(ax, gdf, border, codes,
                                 title=f"{label} ({len(indices)})",
                                 color=col)

        fig.suptitle(
            r"Representation space comparison ($k{=}300$, knee)",
            fontsize=11, y=1.02)
        fig.tight_layout()
        return _save(fig, os.path.join(
            self.fig_dir,
            "coreset_map_representation_comparison_k300.pdf"))
