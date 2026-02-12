"""
Plotting utilities and matplotlib defaults.

Contains:
- set_manuscript_style: Configure matplotlib for publication-quality figures
- Color palettes and marker styles
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Color palettes
PALETTE_METHODS = {
    "pareto_knee": "#1f77b4",      # Blue
    "pareto_best_skl": "#ff7f0e",  # Orange
    "pareto_best_mmd": "#2ca02c",  # Green
    "pareto_best_sinkhorn": "#d62728",  # Red
    "baseline_uniform": "#9467bd",  # Purple
    "baseline_kmeans": "#8c564b",   # Brown
    "baseline_herding": "#e377c2",  # Pink
    "baseline_farthest_first": "#7f7f7f",  # Gray
    "baseline_rls": "#bcbd22",      # Yellow-green
    "baseline_dpp": "#17becf",      # Cyan
}

PALETTE_OBJECTIVES = {
    "skl": "#1f77b4",
    "mmd": "#ff7f0e", 
    "sinkhorn": "#2ca02c",
    "geo_kl": "#d62728",
}

MARKERS_METHODS = {
    "pareto_knee": "o",
    "pareto_best_skl": "s",
    "pareto_best_mmd": "^",
    "pareto_best_sinkhorn": "v",
    "baseline_uniform": "x",
    "baseline_kmeans": "+",
    "baseline_herding": "d",
    "baseline_farthest_first": "*",
    "baseline_rls": "p",
    "baseline_dpp": "h",
}


def set_manuscript_style(
    font_size: int = 10,
    font_family: str = "serif",
    use_latex: bool = False,
) -> None:
    """
    Configure matplotlib for publication-quality figures.
    
    Parameters
    ----------
    font_size : int
        Base font size
    font_family : str
        Font family ('serif', 'sans-serif', 'monospace')
    use_latex : bool
        Whether to use LaTeX for text rendering
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        # Font settings
        'font.size': font_size,
        'font.family': font_family,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        
        # Figure settings
        'figure.figsize': (6.4, 4.8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Legend settings
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
        
        # LaTeX settings
        'text.usetex': use_latex,
    })
    
    if use_latex:
        plt.rcParams.update({
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })


def get_method_style(method: str) -> Tuple[str, str]:
    """
    Get color and marker for a method.
    
    Parameters
    ----------
    method : str
        Method name
        
    Returns
    -------
    Tuple[str, str]
        (color, marker)
    """
    color = PALETTE_METHODS.get(method, "#333333")
    marker = MARKERS_METHODS.get(method, "o")
    return color, marker


def get_objective_color(objective: str) -> str:
    """
    Get color for an objective.
    
    Parameters
    ----------
    objective : str
        Objective name
        
    Returns
    -------
    str
        Color hex code
    """
    return PALETTE_OBJECTIVES.get(objective, "#333333")


def objective_label(obj: str) -> str:
    """
    Get display label for an objective.
    
    Parameters
    ----------
    obj : str
        Objective name
        
    Returns
    -------
    str
        Display label (potentially with LaTeX)
    """
    labels = {
        "skl": r"SKL",
        "mmd": r"MMD$^2$",
        "mmd2": r"MMD$^2$",
        "sinkhorn": r"Sinkhorn",
        "geo_kl": r"Geo-KL",
    }
    return labels.get(obj, obj)


def method_label(method: str) -> str:
    """
    Get display label for a method.
    
    Parameters
    ----------
    method : str
        Method name
        
    Returns
    -------
    str
        Display label
    """
    labels = {
        "pareto_knee": "Pareto (knee)",
        "pareto_best_skl": "Pareto (best SKL)",
        "pareto_best_mmd": "Pareto (best MMD)",
        "pareto_best_sinkhorn": "Pareto (best Sinkhorn)",
        "baseline_uniform": "Uniform",
        "baseline_kmeans": "K-means",
        "baseline_herding": "Herding",
        "baseline_farthest_first": "Farthest-first",
        "baseline_rls": "RLS",
        "baseline_dpp": "k-DPP",
    }
    return labels.get(method, method)


def figure_size(width: str = "single", aspect: float = 0.75) -> Tuple[float, float]:
    """
    Get figure size for publication.
    
    Parameters
    ----------
    width : str
        'single' for single-column, 'double' for double-column
    aspect : float
        Height/width aspect ratio
        
    Returns
    -------
    Tuple[float, float]
        (width, height) in inches
    """
    widths = {
        "single": 3.5,   # Single column (typical journal)
        "double": 7.0,   # Double column
        "half": 1.75,    # Half single column
    }
    w = widths.get(width, 3.5)
    return (w, w * aspect)


def add_panel_label(
    ax,
    label: str,
    loc: str = "upper left",
    fontsize: int = 12,
    fontweight: str = "bold",
) -> None:
    """
    Add a panel label (a), (b), etc. to an axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to label
    label : str
        Label text (e.g., "(a)")
    loc : str
        Location ('upper left', 'upper right', etc.)
    fontsize : int
        Font size
    fontweight : str
        Font weight
    """
    loc_coords = {
        "upper left": (0.02, 0.98),
        "upper right": (0.98, 0.98),
        "lower left": (0.02, 0.02),
        "lower right": (0.98, 0.02),
    }
    
    x, y = loc_coords.get(loc, (0.02, 0.98))
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha, va=va,
    )


def truncate_colormap(cmap_name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """
    Truncate a colormap to a subset of its range.
    
    Parameters
    ----------
    cmap_name : str
        Name of the colormap
    minval : float
        Minimum value (0-1)
    maxval : float
        Maximum value (0-1)
    n : int
        Number of colors
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
