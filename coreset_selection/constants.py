"""Backward-compatibility stub — real implementation in config/constants_common.py.

This module re-exports everything from ``config.constants_common`` so that
existing imports such as ``from coreset_selection.constants import X``
continue to work.
"""
from .config.constants_common import *  # noqa: F401,F403
from .config.constants_common import (
    _MONTH_ABBR,
    _MONTH_ABBR_EN,
    BRAZILIAN_STATES,
    BRAZILIAN_REGIONS,
)
