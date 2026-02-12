"""Kernel Thinning baseline (adapted from *goodpoints*).

This module adds **Kernel Thinning** as an additional baseline coreset selection
method for experiment configuration **R6**.

The implementation is adapted from the `goodpoints` project (MIT License):

  - Raaz Dwivedi and Lester Mackey.
    *Kernel Thinning.* https://arxiv.org/pdf/2105.05842.pdf

We use the **O(nd)** memory variant (no kernel matrix storage) and implement the
core pieces of the Goodpoints KT pipeline:

  - KT-SPLIT (produces 2^m candidate coresets of size floor(n/2^m))
  - KT-SWAP (select best candidate and refine by greedy swaps)

Important implementation detail
-------------------------------
The original Goodpoints reference implementation defines kernels as
`kernel_eval(x, y)` that return *row-wise* evaluations:

  * If `x` has one row, it is broadcast against all rows of `y`.
  * Otherwise, `x` and `y` are expected to have the same shape and the kernel is
    evaluated between corresponding rows.

This is slightly different from the common "full pairwise matrix" convention.
We preserve that behaviour here because the KT-SPLIT / KT-SWAP code relies on it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..geo import GeoInfo, GeographicConstraintProjector
from .utils import rff_features

# Import internal helpers and re-export for backward compatibility
from ._kt_helpers import (
    _make_rbf_kernel_by_row,
    TWO_LOG_2,
    _largest_power_of_two,
    _kt_thin_X,
    _kt_split_X,
    _kt_swap_X,
    _kt_kernel_matrix_row_mean,
    _kt_best_X,
    _kt_squared_emp_rel_mmd_X,
    _kt_refine_X,
)


# -----------------------------------------------------------------------------
# Public baseline API
# -----------------------------------------------------------------------------

def baseline_kernel_thinning(
    X: np.ndarray,
    *,
    k: int,
    sigma_sq: float,
    seed: int,
    delta: float = 0.5,
    m: Optional[int] = None,
    meanK: Optional[np.ndarray] = None,
    meanK_rff_dim: int = 512,
    unique: bool = True,
) -> np.ndarray:
    """Kernel Thinning baseline (unconstrained).

    Parameters
    ----------
    X:
        Data matrix, shape (n, d).
    k:
        Desired coreset size.
    sigma_sq:
        RBF bandwidth (variance). Typically from a median heuristic.
    seed:
        Random seed.
    delta:
        KT-SPLIT failure probability parameter.
    m:
        Number of halving rounds. If None, choose m so that
        floor(n / 2^m) >= k and is close to k (i.e., m=floor(log2(n/k))).
    meanK:
        Optional precomputed mean kernel evaluations. If provided, it should be
        an array of shape (n,) with meanK[i] ≈ mean_j k(x_i, x_j).
        Passing this avoids an O(n^2) computation.
    meanK_rff_dim:
        If meanK is not provided, we approximate it via random Fourier features
        of this dimension (O(n * meanK_rff_dim)).
    unique:
        Enforce uniqueness during KT-SWAP refinement.

    Returns
    -------
    np.ndarray
        Selected indices into X, shape (k,).
    """
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    k = int(k)
    sigma_sq = float(sigma_sq)

    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k == 0:
        return np.array([], dtype=int)
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if not np.isfinite(sigma_sq) or sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive and finite, got {sigma_sq}")

    # Choose halving rounds so that output size s = floor(n / 2^m) is in [k, 2k)
    if m is None:
        ratio = n / float(k)
        m = int(np.floor(np.log2(ratio))) if ratio > 1.0 else 0
        m = max(m, 0)

    # Build kernel closures (row-wise convention as in goodpoints)
    split_kernel = _make_rbf_kernel_by_row(sigma_sq)
    swap_kernel = split_kernel

    # Approximate meanK (mean kernel evaluation per point) if not supplied.
    # This keeps KT-SWAP near O(nk) rather than O(n^2).
    if meanK is None:
        meanK_rff_dim = int(meanK_rff_dim)
        meanK_rff_dim = max(32, meanK_rff_dim)
        # Cap to avoid extreme memory in pathological configs.
        meanK_rff_dim = min(meanK_rff_dim, 4096)
        Phi = rff_features(X, m=meanK_rff_dim, sigma_sq=sigma_sq, seed=seed + 991)
        mean_phi = Phi.mean(axis=0)
        meanK = Phi @ mean_phi

    sel = _kt_thin_X(
        X,
        m=m,
        split_kernel=split_kernel,
        swap_kernel=swap_kernel,
        delta=float(delta),
        seed=int(seed),
        meanK=np.asarray(meanK, dtype=np.float64),
        unique=bool(unique),
    )

    # KT returns size floor(n/2^m); adjust to exactly k if needed.
    sel = np.asarray(sel, dtype=int)
    rng = np.random.default_rng(int(seed) + 123)

    if sel.size > k:
        sel = rng.choice(sel, size=k, replace=False)
    elif sel.size < k:
        # Rare (only if user forces m too large). Fill from remaining points.
        remaining = np.setdiff1d(np.arange(n, dtype=int), sel, assume_unique=False)
        need = k - sel.size
        if remaining.size >= need:
            extra = rng.choice(remaining, size=need, replace=False)
        else:
            # Fallback: allow repeats (will be repaired by exact-k projection later).
            extra = rng.choice(np.arange(n, dtype=int), size=need, replace=True)
        sel = np.concatenate([sel, extra])

    return np.asarray(sel, dtype=int)


def baseline_kernel_thinning_quota(
    X: np.ndarray,
    *,
    geo: GeoInfo,
    k: int,
    alpha_geo: float,
    sigma_sq: float,
    seed: int,
    delta: float = 0.5,
    min_one_per_group: bool = True,
    meanK_rff_dim: int = 512,
    unique: bool = True,
) -> np.ndarray:
    """Kernel Thinning baseline with geographic quota constraints.

    Strategy:
      1) Compute KL-optimal target counts c*(k) via GeographicConstraintProjector.
      2) For each geographic group g, run kernel thinning on X_g to select
         exactly c*_g points.

    This mirrors how other quota baselines are implemented: the method tries to
    satisfy quotas directly, and the experiment runner applies the same quota
    projection operator afterwards for guaranteed feasibility.
    """
    X = np.asarray(X, dtype=np.float64)
    k = int(k)
    alpha_geo = float(alpha_geo)
    sigma_sq = float(sigma_sq)

    if k == 0:
        return np.array([], dtype=int)

    projector = GeographicConstraintProjector(
        geo=geo,
        alpha_geo=alpha_geo,
        min_one_per_group=bool(min_one_per_group),
    )
    target_counts = projector.target_counts(k)

    selected_global: list[np.ndarray] = []

    # Run independent KT inside each group (group-local MMD fidelity)
    for g in range(geo.G):
        t = int(target_counts[g])
        if t <= 0:
            continue

        idx_g = np.asarray(geo.group_to_indices[g], dtype=int)
        if idx_g.size < t:
            raise RuntimeError(
                f"Group {geo.groups[g]}: need {t} but only {idx_g.size} available"
            )

        Xg = X[idx_g]
        sel_local = baseline_kernel_thinning(
            Xg,
            k=t,
            sigma_sq=sigma_sq,
            seed=int(seed) + 31 * g,
            delta=float(delta),
            m=None,
            meanK=None,
            meanK_rff_dim=int(meanK_rff_dim),
            unique=bool(unique),
        )
        selected_global.append(idx_g[np.asarray(sel_local, dtype=int)])

    sel = np.concatenate(selected_global) if selected_global else np.array([], dtype=int)
    if sel.size != k:
        # Safety check – should not happen unless quotas changed mid-run.
        raise RuntimeError(f"Quota KT produced {sel.size} points, expected {k}")
    return np.asarray(sel, dtype=int)
