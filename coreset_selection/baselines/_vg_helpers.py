"""Internal helpers for the BaselineVariantGenerator.

Extracted from variant_generator.py. Contains the method registry data
structures, variant pair definitions, and the BaselineResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------
# Registry: canonical name -> (exact-k factory, quota factory | None)
# -----------------------------------------------------------------------

# Mapping from short method code to (human-readable name, variant_type)
METHOD_REGISTRY: Dict[str, Dict[str, str]] = {
    # exact-k baselines
    "U":    {"full_name": "Uniform",              "regime": "exactk"},
    "KM":   {"full_name": "K-means reps",         "regime": "exactk"},
    "KH":   {"full_name": "Kernel herding",       "regime": "exactk"},
    "FF":   {"full_name": "Farthest-first",       "regime": "exactk"},
    "RLS":  {"full_name": "Ridge leverage",        "regime": "exactk"},
    "DPP":  {"full_name": "k-DPP",                "regime": "exactk"},
    "KT":   {"full_name": "Kernel thinning",      "regime": "exactk"},
    "KKN":  {"full_name": "KKM-Nystrom",          "regime": "exactk"},
    # municipality-share quota baselines (S-prefix)
    "SU":   {"full_name": "Uniform (muni-quota)",       "regime": "muni_quota"},
    "SKM":  {"full_name": "K-means reps (muni-quota)",  "regime": "muni_quota"},
    "SKH":  {"full_name": "Kernel herding (muni-quota)","regime": "muni_quota"},
    "SFF":  {"full_name": "Farthest-first (muni-quota)","regime": "muni_quota"},
    "SRLS": {"full_name": "Ridge leverage (muni-quota)","regime": "muni_quota"},
    "SDPP": {"full_name": "k-DPP (muni-quota)",         "regime": "muni_quota"},
    "SKT":  {"full_name": "Kernel thinning (muni-quota)","regime": "muni_quota"},
    "SKKN": {"full_name": "KKM-Nystrom (muni-quota)",   "regime": "muni_quota"},
    # Population-share quota baselines (P-prefix)
    "PU":   {"full_name": "Uniform (pop-quota)",        "regime": "pop_quota"},
    "PKM":  {"full_name": "K-means reps (pop-quota)",   "regime": "pop_quota"},
    "PKH":  {"full_name": "Kernel herding (pop-quota)",  "regime": "pop_quota"},
    "PFF":  {"full_name": "Farthest-first (pop-quota)",  "regime": "pop_quota"},
    "PRLS": {"full_name": "Ridge leverage (pop-quota)",  "regime": "pop_quota"},
    "PDPP": {"full_name": "k-DPP (pop-quota)",           "regime": "pop_quota"},
    "PKT":  {"full_name": "Kernel thinning (pop-quota)", "regime": "pop_quota"},
    "PKKN": {"full_name": "KKM-Nystrom (pop-quota)",     "regime": "pop_quota"},
    # Joint-constrained baselines (J-prefix)
    "JU":   {"full_name": "Uniform (joint)",        "regime": "joint_quota"},
    "JKM":  {"full_name": "K-means reps (joint)",   "regime": "joint_quota"},
    "JKH":  {"full_name": "Kernel herding (joint)",  "regime": "joint_quota"},
    "JFF":  {"full_name": "Farthest-first (joint)",  "regime": "joint_quota"},
    "JRLS": {"full_name": "Ridge leverage (joint)",  "regime": "joint_quota"},
    "JDPP": {"full_name": "k-DPP (joint)",           "regime": "joint_quota"},
    "JKT":  {"full_name": "Kernel thinning (joint)", "regime": "joint_quota"},
    "JKKN": {"full_name": "KKM-Nystrom (joint)",     "regime": "joint_quota"},
}

# Pairs of (exact-k code, quota-matched code) for structured comparison
VARIANT_PAIRS: List[Tuple[str, str]] = [
    ("U",   "SU"),
    ("KM",  "SKM"),
    ("KH",  "SKH"),
    ("FF",  "SFF"),
    ("RLS", "SRLS"),
    ("DPP", "SDPP"),
    ("KT",  "SKT"),
    ("KKN", "SKKN"),
]

# Population-share quota pairs: (exact-k code, pop-quota code)
POP_QUOTA_PAIRS: List[Tuple[str, str]] = [
    ("U",   "PU"),
    ("KM",  "PKM"),
    ("KH",  "PKH"),
    ("FF",  "PFF"),
    ("RLS", "PRLS"),
    ("DPP", "PDPP"),
    ("KT",  "PKT"),
    ("KKN", "PKKN"),
]

# Joint-constrained pairs: (exact-k code, joint code)
JOINT_QUOTA_PAIRS: List[Tuple[str, str]] = [
    ("U",   "JU"),
    ("KM",  "JKM"),
    ("KH",  "JKH"),
    ("FF",  "JFF"),
    ("RLS", "JRLS"),
    ("DPP", "JDPP"),
    ("KT",  "JKT"),
    ("KKN", "JKKN"),
]


@dataclass
class BaselineResult:
    """Container for a single baseline run result."""

    method: str
    full_name: str
    regime: str          # "exactk", "muni_quota", "pop_quota", or "joint_quota"
    space: str           # "raw", "vae", "pca"
    k: int
    selected_indices: np.ndarray
    wall_time_s: float
    quota_vector: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for CSV serialization."""
        row: Dict[str, Any] = {
            "method": self.method,
            "full_name": self.full_name,
            "regime": self.regime,
            "space": self.space,
            "k": self.k,
            "k_actual": len(self.selected_indices),
            "wall_time_s": round(self.wall_time_s, 4),
            "quota_vector_used": self.quota_vector is not None,
        }
        row.update(self.metrics)
        return row
