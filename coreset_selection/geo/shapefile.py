"""Brazil municipality shapefile loading and choropleth panel rendering.

Provides utilities for loading IBGE shapefiles and rendering coreset
selections as filled municipality polygons on matplotlib axes.

Requires ``geopandas >= 0.12``.  Install via::

    pip install geopandas

or::

    pip install -e ".[geo]"
"""

from __future__ import annotations

import os
from typing import Optional, Set, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Shapefile loading
# ---------------------------------------------------------------------------

def load_brazil_municipalities(shapefile_dir: str) -> "gpd.GeoDataFrame":
    """Load the IBGE municipality shapefile.

    Parameters
    ----------
    shapefile_dir : str
        Directory containing the shapefile (e.g., ``data/BR_Municipios_2024``).
        Accepts ``.shp`` or ``.gpkg`` formats.

    Returns
    -------
    gpd.GeoDataFrame
        Municipality polygons with an ``ibge_code`` column (int64).

    Raises
    ------
    ImportError
        If *geopandas* is not installed.
    FileNotFoundError
        If no recognised shapefile is found in *shapefile_dir*.
    """
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for choropleth map generation. "
            "Install it with:  pip install geopandas"
        )

    # Candidate file names (IBGE releases vary across years)
    shp_candidates = [
        "BR_Municipios_2024.shp",
        "BR_Municipios_2022.shp",
        "BR_Municipios_2021.shp",
        "BR_Municipios_2020.shp",
        "municipios.shp",
    ]
    gpkg_candidates = [
        "BR_Municipios_2024.gpkg",
        "BR_Municipios_2022.gpkg",
    ]

    path: Optional[str] = None
    for name in shp_candidates + gpkg_candidates:
        candidate = os.path.join(shapefile_dir, name)
        if os.path.isfile(candidate):
            path = candidate
            break

    # Also try a recursive glob for any .shp in the directory
    if path is None:
        import glob as _glob
        for shp in sorted(_glob.glob(os.path.join(shapefile_dir, "**", "*.shp"), recursive=True)):
            path = shp
            break

    if path is None:
        raise FileNotFoundError(
            f"No IBGE municipality shapefile found in '{shapefile_dir}'. "
            f"Please download the BR_Municipios_2024 shapefile from "
            f"https://www.ibge.gov.br/geociencias/organizacao-do-territorio/"
            f"malhas-territoriais/15774-malhas.html and place it in "
            f"'{shapefile_dir}'."
        )

    gdf = gpd.read_file(path)

    # Detect the IBGE code column (varies across shapefile releases)
    code_col: Optional[str] = None
    for candidate_col in ("CD_MUN", "CD_GEOCODM", "CD_GEOCMU", "CD_GEOCOD",
                          "codigo_ibge", "COD_MUN", "GEOCODIGO"):
        if candidate_col in gdf.columns:
            code_col = candidate_col
            break

    if code_col is None:
        raise KeyError(
            f"Could not find IBGE municipality code column in shapefile. "
            f"Available columns: {list(gdf.columns)}. "
            f"Expected one of: CD_MUN, CD_GEOCODM, CD_GEOCMU, CD_GEOCOD."
        )

    # Normalise to int64 for join with ibge_codes
    gdf["ibge_code"] = gdf[code_col].astype(str).str.strip().astype(np.int64)

    return gdf


def get_brazil_border(municipalities: "gpd.GeoDataFrame") -> "gpd.GeoSeries":
    """Dissolve municipality polygons into a single country boundary.

    Parameters
    ----------
    municipalities : gpd.GeoDataFrame
        Output of :func:`load_brazil_municipalities`.

    Returns
    -------
    gpd.GeoSeries
        Single-row GeoSeries with the dissolved Brazil outline.
    """
    dissolved = municipalities.dissolve()
    return dissolved.geometry


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def plot_choropleth_panel(
    ax,
    municipalities: "gpd.GeoDataFrame",
    border: "gpd.GeoSeries",
    selected_codes: Union[np.ndarray, Set[int], Sequence[int]],
    title: str = "",
    color: str = "#1f77b4",
    unselected_color: str = "#e0e0e0",
    border_color: str = "black",
    border_linewidth: float = 0.6,
    alpha: float = 0.85,
    fontsize: int = 9,
) -> None:
    """Render a single choropleth panel on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    municipalities : gpd.GeoDataFrame
        Municipality polygons (must have ``ibge_code`` column).
    border : gpd.GeoSeries
        Dissolved country border (from :func:`get_brazil_border`).
    selected_codes : array-like of int
        IBGE municipality codes to highlight.
    title : str
        Panel title.
    color : str
        Fill colour for selected municipalities.
    unselected_color : str
        Fill colour for unselected municipalities.
    border_color : str
        Stroke colour for country border.
    border_linewidth : float
        Linewidth for country border.
    alpha : float
        Fill opacity for selected municipalities.
    fontsize : int
        Title font size.
    """
    selected_set = set(int(c) for c in selected_codes)

    # Classify municipalities
    is_selected = municipalities["ibge_code"].isin(selected_set)

    # Plot unselected first (background)
    municipalities[~is_selected].plot(
        ax=ax, color=unselected_color, edgecolor="#cccccc",
        linewidth=0.1,
    )

    # Plot selected (foreground)
    if is_selected.any():
        municipalities[is_selected].plot(
            ax=ax, color=color, edgecolor=color,
            linewidth=0.2, alpha=alpha,
        )

    # Overlay country border
    border.boundary.plot(
        ax=ax, color=border_color, linewidth=border_linewidth,
    )

    # Styling
    ax.set_title(title, fontsize=fontsize, pad=4)
    ax.set_axis_off()
    ax.set_aspect("equal")
