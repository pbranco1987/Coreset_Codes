"""Shared helpers for the ManuscriptArtifacts mixin modules.

Provides common utilities (style setup, figure saving) used across the
various manuscript-artifact generation sub-modules.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# IEEE publication-quality style
# ---------------------------------------------------------------------------

def _set_style():
    """Set matplotlib style for IEEE-compliant manuscript figures.

    Enforces:
    - Base font size >= 10 pt (axes labels), >= 8 pt (annotations/legend).
    - Top/right spines removed for clean appearance.
    - Transparent grid at 25 % alpha.
    - 300 dpi saved output.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # Font sizes (IEEE requires >= 8 pt at final column width)
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        # Figure quality
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # Grid & spines
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })


def _save(fig, path: str) -> str:
    """Save figure to *path* at publication quality (300 dpi, tight bbox)."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    return path
