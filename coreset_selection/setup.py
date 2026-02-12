"""
Install script for coreset_selection.

This setup.py lives INSIDE the package directory (``coreset_selection``).
All internal imports use ``coreset_selection``.

Usage
-----
From INSIDE this directory::

    cd /path/to/coreset_selection
    pip install -e .

This creates an editable install mapping ``import coreset_selection`` to
this directory.
"""

from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# Discover all sub-packages (directories containing __init__.py)
packages = ["coreset_selection"]
for dirpath, dirnames, filenames in os.walk(here):
    # Skip hidden, __pycache__, .git, etc.
    dirnames[:] = [
        d for d in dirnames
        if not d.startswith((".", "__pycache__"))
    ]
    if "__init__.py" in filenames and dirpath != here:
        rel = os.path.relpath(dirpath, here).replace(os.sep, ".")
        packages.append(f"coreset_selection.{rel}")

setup(
    name="coreset-selection",
    version="0.1.0",
    description=(
        "Geographically-constrained Pareto coreset selection "
        "for multi-objective optimization"
    ),
    package_dir={"coreset_selection": "."},
    packages=packages,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "torch>=1.10",
        "matplotlib>=3.4",
        "seaborn>=0.11",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "geo": ["geopandas>=0.12"],
    },
    entry_points={
        "console_scripts": [
            "coreset-selection=coreset_selection.cli:cli",
        ],
    },
)
