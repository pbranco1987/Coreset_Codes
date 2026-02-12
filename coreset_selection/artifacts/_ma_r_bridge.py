"""Python-to-R bridge for ggplot2 figure rendering.

Provides utilities to call Rscript subprocesses that render publication-quality
figures via ggplot2, while keeping all data preparation in Python.

Workflow:
    1. Python prepares a tidy-format pandas DataFrame.
    2. ``run_r_figure()`` writes it to a temporary CSV, invokes Rscript, and
       cleans up the temp file on success.
    3. If Rscript fails or is unavailable, callers fall back to matplotlib.

Environment control:
    - ``CORESET_FORCE_MATPLOTLIB=1`` disables R rendering globally.
    - ``CORESET_RSCRIPT_TIMEOUT`` sets subprocess timeout (default 120 s).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RScriptError(RuntimeError):
    """Raised when an Rscript subprocess fails."""

    def __init__(self, script: str, returncode: int, stderr: str):
        self.script = script
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Rscript '{script}' failed (exit {returncode}):\n{stderr[:2000]}"
        )


# ---------------------------------------------------------------------------
# R availability check (cached)
# ---------------------------------------------------------------------------

_R_AVAILABLE: Optional[bool] = None


def r_is_available() -> bool:
    """Return True if Rscript is on PATH and not globally disabled."""
    global _R_AVAILABLE
    if _R_AVAILABLE is not None:
        return _R_AVAILABLE

    # Global override via environment variable
    if os.environ.get("CORESET_FORCE_MATPLOTLIB", "").strip() == "1":
        _R_AVAILABLE = False
        return False

    _R_AVAILABLE = shutil.which("Rscript") is not None
    return _R_AVAILABLE


# ---------------------------------------------------------------------------
# Locate r_scripts/ directory
# ---------------------------------------------------------------------------

_R_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "r_scripts")


def _script_path(script_name: str) -> str:
    """Resolve the full path to an R script inside ``r_scripts/``."""
    path = os.path.join(_R_SCRIPTS_DIR, script_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"R script not found: {path}"
        )
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_r_figure(
    script_name: str,
    data: pd.DataFrame,
    output_pdf: str,
    extra_args: Optional[Dict[str, str]] = None,
    timeout_s: Optional[int] = None,
) -> str:
    """Render a figure by calling an Rscript with tidy CSV data.

    Parameters
    ----------
    script_name : str
        Filename of the R script inside ``r_scripts/``, e.g.
        ``"fig_kl_floor_vs_k.R"``.
    data : pd.DataFrame
        Tidy-format DataFrame that will be written to a temporary CSV.
    output_pdf : str
        Absolute path where the PDF should be saved by the R script.
    extra_args : dict, optional
        Additional ``--key=value`` arguments forwarded to the R script.
    timeout_s : int, optional
        Subprocess timeout in seconds.  Defaults to ``CORESET_RSCRIPT_TIMEOUT``
        env var or 120.

    Returns
    -------
    str
        The *output_pdf* path (unchanged, for chaining).

    Raises
    ------
    RScriptError
        If the Rscript exits with a non-zero code.
    FileNotFoundError
        If the R script does not exist.
    TimeoutError
        If the subprocess exceeds *timeout_s*.
    """
    if not r_is_available():
        raise FileNotFoundError("Rscript is not available on PATH")

    script_path = _script_path(script_name)

    if timeout_s is None:
        timeout_s = int(os.environ.get("CORESET_RSCRIPT_TIMEOUT", "120"))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    # Write data to temp CSV
    tmp_csv = None
    try:
        fd, tmp_csv = tempfile.mkstemp(suffix=".csv", prefix="coreset_r_")
        os.close(fd)
        data.to_csv(tmp_csv, index=False)

        # Build command
        cmd: List[str] = [
            "Rscript", "--vanilla",
            script_path,
            tmp_csv,
            output_pdf,
        ]
        if extra_args:
            for k, v in extra_args.items():
                cmd.append(f"--{k}={v}")

        # Run
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if result.returncode != 0:
            raise RScriptError(script_name, result.returncode, result.stderr)

        # Verify output was created
        if not os.path.isfile(output_pdf):
            raise RScriptError(
                script_name, -1,
                f"R script completed but output file not found: {output_pdf}\n"
                f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
            )

        return output_pdf

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"Rscript '{script_name}' timed out after {timeout_s}s"
        )
    finally:
        # Clean up temp CSV
        if tmp_csv and os.path.exists(tmp_csv):
            try:
                os.unlink(tmp_csv)
            except OSError:
                pass
