"""Core table formatting utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def format_number(
    value: float,
    precision: int = 3,
    scientific: bool = False,
    bold_if_best: bool = False,
    is_best: bool = False,
) -> str:
    """
    Format a number for table display.

    Parameters
    ----------
    value : float
        Value to format
    precision : int
        Number of decimal places
    scientific : bool
        Use scientific notation
    bold_if_best : bool
        Whether to bold best values
    is_best : bool
        Whether this is the best value

    Returns
    -------
    str
        Formatted string
    """
    if np.isnan(value):
        return "--"

    if scientific:
        formatted = f"{value:.{precision}e}"
    else:
        formatted = f"{value:.{precision}f}"

    if bold_if_best and is_best:
        formatted = f"\\textbf{{{formatted}}}"

    return formatted


def format_mean_std(
    mean: float,
    std: float,
    precision: int = 3,
    show_pm: bool = True,
) -> str:
    """
    Format mean +/- std for table display.

    Parameters
    ----------
    mean : float
        Mean value
    std : float
        Standard deviation
    precision : int
        Number of decimal places
    show_pm : bool
        Whether to show +/- symbol (LaTeX)

    Returns
    -------
    str
        Formatted string
    """
    if np.isnan(mean):
        return "--"

    mean_str = f"{mean:.{precision}f}"

    if np.isnan(std) or std == 0:
        return mean_str

    std_str = f"{std:.{precision}f}"

    if show_pm:
        return f"{mean_str} $\\pm$ {std_str}"
    else:
        return f"{mean_str} ({std_str})"


def generate_latex_table(
    data: List[List[str]],
    headers: List[str],
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None,
    highlight_best: Optional[Dict[int, str]] = None,
) -> str:
    """
    Generate a LaTeX table.

    Parameters
    ----------
    data : List[List[str]]
        Table data (rows of cells)
    headers : List[str]
        Column headers
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    column_format : Optional[str]
        LaTeX column format (e.g., "l|ccc"). If None, auto-generated.
    highlight_best : Optional[Dict[int, str]]
        Dict mapping column index to 'min' or 'max' for highlighting

    Returns
    -------
    str
        LaTeX table code
    """
    n_cols = len(headers)

    if column_format is None:
        column_format = "l" + "c" * (n_cols - 1)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    lines.append("\\toprule")

    # Header
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for row in data:
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_csv_table(
    data: List[List[Any]],
    headers: List[str],
    output_path: str,
) -> str:
    """
    Generate and save a CSV table.

    Parameters
    ----------
    data : List[List[Any]]
        Table data
    headers : List[str]
        Column headers
    output_path : str
        Path to save CSV file

    Returns
    -------
    str
        Path to saved file
    """
    with open(output_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in data:
            f.write(",".join(str(x) for x in row) + "\n")

    return output_path


def bold_best_in_column(
    values: List[float],
    formatted: List[str],
    direction: str = "min",
) -> List[str]:
    """Return *formatted* strings with the best value wrapped in ``\\textbf``.

    Parameters
    ----------
    values : List[float]
        Numeric values for comparison (may contain ``nan``).
    formatted : List[str]
        Already-formatted string representations.
    direction : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).

    Returns
    -------
    List[str]
        Copy of *formatted* with the best entry bolded.
    """
    finite = [(i, v) for i, v in enumerate(values) if np.isfinite(v)]
    if not finite:
        return list(formatted)
    best_idx = min(finite, key=lambda t: t[1])[0] if direction == "min" \
        else max(finite, key=lambda t: t[1])[0]
    out = list(formatted)
    out[best_idx] = f"\\textbf{{{out[best_idx]}}}"
    return out


def generate_wrapped_latex_table(
    body_lines: List[str],
    caption: str,
    label: str,
    column_format: str,
    headers: List[str],
    double_column: bool = False,
) -> str:
    r"""Build a full ``\begin{table}...\end{table}`` environment.

    This is a higher-level wrapper than :func:`generate_latex_table` that
    accepts pre-formatted body lines (already containing ``\\``) and
    wraps them with caption, label, and booktabs rules.

    Parameters
    ----------
    body_lines : List[str]
        Data rows (each ending with ``\\``).
    caption, label, column_format, headers : str / List[str]
        Standard LaTeX table metadata.
    double_column : bool
        If *True*, use ``table*`` for two-column spanning.

    Returns
    -------
    str
        Complete LaTeX table code.
    """
    env = "table*" if double_column else "table"
    lines = [
        f"\\begin{{{env}}}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(body_lines)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\end{{{env}}}",
    ])
    return "\n".join(lines)


def generate_sectioned_latex_table(
    sections: List[Tuple[str, List[List[str]]]],
    caption: str,
    label: str,
    column_format: str,
    headers: List[str],
) -> str:
    r"""Build a LaTeX table with ``\midrule``-separated sections.

    Useful for tables like manuscript Table IV (proxy stability) that
    group rows into visually distinct sections.

    Parameters
    ----------
    sections : List[Tuple[str, List[List[str]]]]
        Each element is ``(section_name, rows)`` where *rows* is a list
        of cell-value lists.  *section_name* is informational only (not
        rendered as a row) --- use an empty string to suppress.
    caption, label, column_format, headers
        Standard LaTeX table metadata.

    Returns
    -------
    str
        Complete LaTeX table code.
    """
    body: List[str] = []
    for i, (_, rows) in enumerate(sections):
        if i > 0:
            body.append("\\midrule")
        for row in rows:
            body.append(" & ".join(row) + " \\\\")
    return generate_wrapped_latex_table(
        body, caption, label, column_format, headers,
    )
