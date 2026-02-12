r"""Split persistence utilities (Phase 4 — Milestone 4.2).

Per the manuscript (Section VII), each replicate must produce a *fixed*
three-tier split:

1. **Representation-learning split** ``(I_train, I_val)`` — 80/20 stratified
   by state.  Used for preprocessing stats, PCA fitting, and VAE training.
2. **Evaluation set** ``E`` — |E| = 2000, stratified by state.
3. **Supervised split inside E** ``(E_train, E_test)`` — 80/20 stratified by
   state, used for KRR hyperparameter selection and final evaluation.

This module provides helpers to:

- **save** a ``splits.npz`` file alongside the replicate cache,
- **load** it back and verify integrity,
- **validate** that saved splits are consistent (disjoint, correct sizes,
  stratification preserved within tolerance).

These are consumed by ``data/cache.py`` (build path) and by the unit-test
``tests/test_preprocessing.py``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_splits(
    cache_dir: str,
    rep_id: int,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    seed: int,
    metadata: Optional[Dict] = None,
) -> str:
    """Persist split indices for a single replicate.

    Parameters
    ----------
    cache_dir : str
        Root cache directory (e.g. ``replicate_cache``).
    rep_id : int
        Replicate identifier.
    train_idx, val_idx, eval_idx, eval_train_idx, eval_test_idx : np.ndarray
        Integer index arrays from the cache builder.
    seed : int
        Seed used to generate these splits (stored for audit).
    metadata : dict, optional
        Additional metadata (e.g. group proportions).

    Returns
    -------
    str
        Path to the written ``splits.npz`` file.
    """
    rep_dir = os.path.join(cache_dir, f"rep{rep_id:02d}")
    ensure_dir(rep_dir)
    path = os.path.join(rep_dir, "splits.npz")

    save_dict = {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "eval_idx": np.asarray(eval_idx, dtype=np.int64),
        "eval_train_idx": np.asarray(eval_train_idx, dtype=np.int64),
        "eval_test_idx": np.asarray(eval_test_idx, dtype=np.int64),
        "seed": np.array([seed], dtype=np.int64),
    }

    if metadata is not None:
        for key, val in metadata.items():
            save_dict[f"meta_{key}"] = np.asarray(val)

    np.savez_compressed(path, **save_dict)
    return path


def load_splits(cache_dir: str, rep_id: int) -> Dict[str, np.ndarray]:
    """Load previously saved splits.

    Parameters
    ----------
    cache_dir : str
        Root cache directory.
    rep_id : int
        Replicate identifier.

    Returns
    -------
    dict
        Mapping with keys ``train_idx``, ``val_idx``, ``eval_idx``,
        ``eval_train_idx``, ``eval_test_idx``, ``seed``.

    Raises
    ------
    FileNotFoundError
        If the splits file does not exist.
    """
    path = os.path.join(cache_dir, f"rep{rep_id:02d}", "splits.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Splits file not found: {path}")

    data = dict(np.load(path, allow_pickle=False))
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    N: int,
    *,
    expected_eval_size: int = 2000,
    expected_eval_train_frac: float = 0.8,
    group_ids: Optional[np.ndarray] = None,
    stratification_tol: float = 0.10,
) -> Dict[str, bool]:
    """Validate split integrity.

    Checks performed:

    1. **Disjointness**: train and val are disjoint.
    2. **Coverage**: train ∪ val covers all N indices.
    3. **Eval size**: |E| matches expected_eval_size (or N if smaller).
    4. **Eval sub-split**: eval_train ∪ eval_test = eval (as sets).
    5. **Eval train fraction**: |E_train| / |E| ≈ expected_eval_train_frac.
    6. **Index bounds**: all indices in [0, N).
    7. **Stratification** (if group_ids provided): per-group proportions in each
       split are within *stratification_tol* of the population proportions.

    Parameters
    ----------
    N : int
        Total dataset size.
    group_ids : np.ndarray, optional
        Integer group labels, shape ``(N,)``.
    stratification_tol : float
        Maximum allowed deviation in group proportions.

    Returns
    -------
    dict
        Mapping of check names to pass/fail booleans.
    """
    results: Dict[str, bool] = {}

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    eval_set = set(eval_idx.tolist())
    eval_train_set = set(eval_train_idx.tolist())
    eval_test_set = set(eval_test_idx.tolist())

    # 1. Disjointness of train / val
    results["train_val_disjoint"] = len(train_set & val_set) == 0

    # 2. Coverage
    results["train_val_cover_all"] = (train_set | val_set) == set(range(N))

    # 3. Eval size
    eff_eval = min(expected_eval_size, N)
    results["eval_size_correct"] = len(eval_set) == eff_eval

    # 4. Eval sub-split covers eval
    results["eval_subsplit_covers"] = (eval_train_set | eval_test_set) == eval_set
    results["eval_subsplit_disjoint"] = len(eval_train_set & eval_test_set) == 0

    # 5. Eval train fraction
    if len(eval_set) > 0:
        actual_frac = len(eval_train_set) / len(eval_set)
        results["eval_train_frac_ok"] = abs(actual_frac - expected_eval_train_frac) < 0.05
    else:
        results["eval_train_frac_ok"] = False

    # 6. Bounds
    all_idx = np.concatenate([train_idx, val_idx, eval_idx, eval_train_idx, eval_test_idx])
    results["indices_in_bounds"] = bool(np.all(all_idx >= 0) and np.all(all_idx < N))

    # 7. Stratification (optional)
    if group_ids is not None:
        group_ids = np.asarray(group_ids, dtype=int)
        G = int(group_ids.max()) + 1
        pop_props = np.bincount(group_ids, minlength=G) / N

        def _check_strat(idx_arr, label):
            if len(idx_arr) == 0:
                return True
            sub_groups = group_ids[idx_arr]
            sub_props = np.bincount(sub_groups, minlength=G) / len(idx_arr)
            max_dev = float(np.max(np.abs(sub_props - pop_props)))
            return max_dev <= stratification_tol

        results["train_stratified"] = _check_strat(train_idx, "train")
        results["val_stratified"] = _check_strat(val_idx, "val")
        results["eval_stratified"] = _check_strat(eval_idx, "eval")

    return results
