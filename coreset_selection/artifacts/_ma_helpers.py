"""Shared helpers for the ManuscriptArtifacts mixin modules.

Provides common utilities (style setup, figure saving) used across the
various manuscript-artifact generation sub-modules.

R/ggplot2 integration:
    ``use_r()`` returns True when Rscript is available and not disabled.
    ``_save_r()`` renders a figure via R/ggplot2 with automatic fallback to
    matplotlib if R is unavailable or the script fails.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# IEEE publication-quality style (matplotlib)
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


# ---------------------------------------------------------------------------
# R / ggplot2 integration
# ---------------------------------------------------------------------------

def use_r() -> bool:
    """Return True if R/ggplot2 rendering should be used.

    Conditions for returning True:
    1. ``CORESET_FORCE_MATPLOTLIB`` env var is **not** set to ``1``.
    2. ``Rscript`` is found on PATH.

    The result is cached after the first call.
    """
    from ._ma_r_bridge import r_is_available
    return r_is_available()


def _save_r(
    script_name: str,
    data: pd.DataFrame,
    output_path: str,
    fallback_fn: Callable[[], str],
    extra_args: Optional[Dict[str, str]] = None,
) -> str:
    """Try rendering a figure via R/ggplot2; fall back to matplotlib.

    Parameters
    ----------
    script_name : str
        R script filename inside ``r_scripts/``, e.g. ``"fig_kl_floor_vs_k.R"``.
    data : pd.DataFrame
        Tidy-format data to pass to the R script as CSV.
    output_path : str
        Absolute path for the output PDF.
    fallback_fn : callable
        Zero-argument callable that renders the matplotlib version and returns
        the output path.  Called when R is unavailable or fails.
    extra_args : dict, optional
        Additional ``--key=value`` arguments forwarded to the R script.

    Returns
    -------
    str
        Path to the generated PDF.
    """
    if not use_r():
        return fallback_fn()

    from ._ma_r_bridge import run_r_figure, RScriptError

    try:
        return run_r_figure(
            script_name=script_name,
            data=data,
            output_pdf=output_path,
            extra_args=extra_args,
        )
    except (RScriptError, FileNotFoundError, TimeoutError, OSError) as exc:
        print(f"[ManuscriptArtifacts] R fallback for {script_name}: {exc}")
        return fallback_fn()
