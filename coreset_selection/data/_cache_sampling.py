"""Sampling and splitting utilities for the replicate cache."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _rounded_counts(pi: np.ndarray, m: int, min_one_per_group: bool = True) -> np.ndarray:
    """
    Construct integer group counts \\tilde{c} that approximate m*pi and sum to m.

    Uses a largest-remainder style rule:
    - start from floor(m*pi)
    - distribute remaining counts to the largest fractional remainders
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    G = pi.size

    frac = pi * m
    counts = np.floor(frac).astype(int)
    remainders = frac - counts

    if min_one_per_group and m >= G:
        # Ensure at least one per group when possible
        zero_mask = counts == 0
        n_add = int(np.sum(zero_mask))
        if n_add > 0:
            counts[zero_mask] = 1
            # Remove the added mass from the remainder budget
            # (we will correct total afterwards)

    # Fix total to exactly m
    total = int(counts.sum())
    if total < m:
        # Add to largest remainders
        order = np.argsort(-remainders)
        for g in order:
            if total >= m:
                break
            counts[g] += 1
            total += 1
    elif total > m:
        # Remove from smallest remainders (but keep non-negative)
        order = np.argsort(remainders)
        for g in order:
            if total <= m:
                break
            if counts[g] > 0:
                counts[g] -= 1
                total -= 1

    # Final sanity
    if counts.sum() != m:
        # As a last resort, adjust the first group
        counts[0] += (m - int(counts.sum()))
    counts = np.maximum(counts, 0)
    return counts.astype(int)


def _sample_by_group(group_ids: np.ndarray, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample exact per-group counts uniformly without replacement."""
    group_ids = np.asarray(group_ids, dtype=int)
    counts = np.asarray(counts, dtype=int)

    sel = []
    for g in range(counts.size):
        c = int(counts[g])
        if c <= 0:
            continue
        idx_g = np.where(group_ids == g)[0]
        if c > idx_g.size:
            raise ValueError(f"Cannot sample {c} from group {g} with size {idx_g.size}")
        sel.append(rng.choice(idx_g, size=c, replace=False))
    if not sel:
        return np.array([], dtype=int)
    out = np.concatenate(sel)
    rng.shuffle(out)
    return out.astype(int)


def _split_within_groups(
    *,
    eval_idx: np.ndarray,
    group_ids: np.ndarray,
    train_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split eval_idx into train/test by applying the same proportion within each group.

    This implements the manuscript description: within each state, split using a
    fixed proportion (default 80/20), then concatenate across states.
    """
    eval_idx = np.asarray(eval_idx, dtype=int)
    group_ids = np.asarray(group_ids, dtype=int)

    train_parts = []
    test_parts = []

    for g in np.unique(group_ids[eval_idx]):
        idx_g = eval_idx[group_ids[eval_idx] == g].copy()
        rng.shuffle(idx_g)
        n = idx_g.size
        if n == 0:
            continue

        if n == 1:
            # Cannot split; place the single point in train
            n_train = 1
        else:
            n_train = int(np.floor(train_frac * n))
            n_train = max(1, min(n_train, n - 1))

        train_parts.append(idx_g[:n_train])
        test_parts.append(idx_g[n_train:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.sort(train_idx), np.sort(test_idx)
