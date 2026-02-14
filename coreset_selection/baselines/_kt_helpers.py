"""Internal helpers for the Kernel Thinning baseline.

Extracted from kernel_thinning.py. Contains the row-wise RBF kernel helper,
the KT-SPLIT and KT-SWAP internals (adapted from goodpoints.kt, pure NumPy).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# -----------------------------------------------------------------------------
# Kernel helper (row-wise RBF)
# -----------------------------------------------------------------------------

def _make_rbf_kernel_by_row(sigma_sq: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a row-wise Gaussian (RBF) kernel evaluation function.

    The returned callable matches Goodpoints' `kernel_eval` convention:
      - If `x` has one row, evaluate k(x, y_j) for all rows in y.
      - Else, x and y must have the same shape and we evaluate k(x_i, y_i).
    """
    sigma_sq = float(sigma_sq)

    def _k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Broadcast single row against the other argument
        if x.shape[0] == 1 and y.shape[0] >= 1:
            diff = y - x  # (n, d)
        elif y.shape[0] == 1 and x.shape[0] >= 1:
            diff = x - y
        else:
            if x.shape != y.shape:
                raise ValueError(
                    "Kernel called with incompatible shapes: "
                    f"x={x.shape}, y={y.shape}. Expected one row broadcast "
                    "or equal shapes."
                )
            diff = x - y

        sq = np.sum(diff * diff, axis=1)
        return np.exp(-0.5 * sq / sigma_sq)

    return _k


# -----------------------------------------------------------------------------
# Kernel thinning internals (adapted from goodpoints.kt, pure NumPy)
# -----------------------------------------------------------------------------


TWO_LOG_2 = 2.0 * np.log(2.0)


def _largest_power_of_two(n: int) -> int:
    """Largest j such that 2**j divides n (n>0)."""
    return (n & (~(n - 1))).bit_length() - 1


def _kt_thin_X(
    X: np.ndarray,
    *,
    m: int,
    split_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    swap_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    delta: float = 0.5,
    seed: Optional[int] = None,
    meanK: Optional[np.ndarray] = None,
    unique: bool = False,
) -> np.ndarray:
    """Return kernel thinning coreset indices of size floor(n/2^m)."""
    if m == 0:
        return np.arange(X.shape[0], dtype=int)

    coresets = _kt_split_X(X, m=m, kernel=split_kernel, delta=delta, seed=seed)
    coreset = _kt_swap_X(
        X,
        coresets=coresets,
        kernel=swap_kernel,
        meanK=meanK,
        unique=unique,
    )
    return np.asarray(coreset, dtype=int)


def _kt_split_X(
    X: np.ndarray,
    *,
    m: int,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    delta: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """KT-SPLIT (O(nd) memory) – returns 2^m candidate coresets."""
    import time as _time

    if m == 0:
        return np.arange(X.shape[0], dtype=int)[np.newaxis, :]

    # Helper: kernel between indexed rows of X.
    def k(ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
        return kernel(X[ii], X[jj])

    rng = np.random.default_rng(seed)
    n = int(X.shape[0])

    coresets: dict[int, np.ndarray] = {}
    KC: dict[int, np.ndarray] = {}
    sig_sqd: dict[int, np.ndarray] = {}

    log_multiplier = 2.0 * np.log(2.0 * n * m / float(delta))

    for j in range(m + 1):
        num_coresets = int(2**j)
        num_points_in_coreset = n // num_coresets
        coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
        KC[j] = np.empty((num_coresets, num_points_in_coreset), dtype=np.float64)
        sig_sqd[j] = np.zeros(num_coresets, dtype=np.float64)

    diagK = np.empty(n, dtype=np.float64)

    _t_split_start = _time.perf_counter()
    _pct_step = max(1, n // 10)  # print every 10%
    for i in range(n):
        if i > 0 and i % _pct_step == 0:
            _elapsed = _time.perf_counter() - _t_split_start
            print(
                f"              KT-SPLIT {i}/{n} ({100*i//n}%, {_elapsed:.0f}s)",
                flush=True,
            )
        # Add each datapoint to coreset[0][0]
        coreset0 = coresets[0][0]
        coreset0[i] = i

        # Capture index i as 1D array so that X[i_array] is 2D
        i_array = coreset0[i, np.newaxis]

        # Store kernel evaluation with all points <= i
        ki = k(i_array, coreset0[: (i + 1)])
        KC[0][0, i] = float(np.sum(ki[:i]))
        diagK[i] = float(ki[i])

        # If 2^(j+1) divides (i+1), halve parent coresets into children
        for j in range(min(m, _largest_power_of_two(i + 1))):
            parent_coresets = coresets[j]
            child_coresets = coresets[j + 1]
            parent_KC = KC[j]
            child_KC = KC[j + 1]
            num_parent_coresets = int(parent_coresets.shape[0])

            # See goodpoints.kt.split_X for derivation
            j_log_multiplier = log_multiplier - j * TWO_LOG_2

            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                parent_idx = (i + 1) // num_parent_coresets

                # Last two points from parent
                # Use scalar indices for diagK lookups; wrap in arrays for k()
                p1_idx = int(parent_coreset[parent_idx - 2])
                p2_idx = int(parent_coreset[parent_idx - 1])
                point1 = np.array([p1_idx])
                point2 = np.array([p2_idx])
                K12_val = float(k(point1, point2).ravel()[0])

                b_sqd = float(diagK[p2_idx]) + float(diagK[p1_idx]) - 2.0 * K12_val
                thresh = max(
                    np.sqrt(sig_sqd[j][j2] * b_sqd * j_log_multiplier),
                    b_sqd,
                )

                if sig_sqd[j][j2] == 0:
                    sig_sqd[j][j2] = b_sqd
                elif thresh != 0:
                    sig_sqd_update = 0.5 + (b_sqd / (2.0 * thresh) - 1.0) * sig_sqd[j][j2] / thresh
                    if sig_sqd_update > 0:
                        sig_sqd[j][j2] += 2.0 * b_sqd * sig_sqd_update

                if thresh == 0:
                    thresh = 1.0

                if parent_idx > 2:
                    alpha = parent_KC[j2, parent_idx - 2] - parent_KC[j2, parent_idx - 1] + K12_val
                else:
                    alpha = 0.0

                left_child_coreset = child_coresets[2 * j2]
                right_child_coreset = child_coresets[2 * j2 + 1]
                child_idx = (parent_idx // 2) - 1

                if child_idx > 0:
                    child_points = left_child_coreset[:child_idx]
                    point1_kernel_sum = float(np.sum(k(point1, child_points)))
                    point2_kernel_sum = float(np.sum(k(point2, child_points)))
                    alpha -= 2.0 * (point1_kernel_sum - point2_kernel_sum)
                else:
                    point1_kernel_sum = 0.0
                    point2_kernel_sum = 0.0

                prob_point2 = 0.5 * (1.0 - alpha / thresh)
                if rng.random() <= prob_point2:
                    left_child_coreset[child_idx] = p2_idx
                    right_child_coreset[child_idx] = p1_idx
                    child_KC[2 * j2, child_idx] = point2_kernel_sum
                    child_KC[2 * j2 + 1, child_idx] = point1_kernel_sum
                else:
                    left_child_coreset[child_idx] = p1_idx
                    right_child_coreset[child_idx] = p2_idx
                    child_KC[2 * j2, child_idx] = point1_kernel_sum
                    child_KC[2 * j2 + 1, child_idx] = point2_kernel_sum

    return coresets[m]


def _kt_swap_X(
    X: np.ndarray,
    *,
    coresets: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    meanK: Optional[np.ndarray] = None,
    unique: bool = False,
) -> np.ndarray:
    """KT-SWAP: choose best candidate coreset and refine by greedy swaps."""
    if meanK is None:
        # This is O(n^2). We keep it as a fallback but callers should
        # prefer passing an approximation (e.g. via RFFs).
        meanK = _kt_kernel_matrix_row_mean(X, kernel)
    meanK = np.asarray(meanK, dtype=np.float64)

    best = _kt_best_X(X, coresets=coresets, kernel=kernel, meanK=meanK)
    refined = _kt_refine_X(X, coreset=best, kernel=kernel, meanK=meanK, unique=unique)
    return refined


def _kt_kernel_matrix_row_mean(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    n = int(X.shape[0])
    meanK = np.empty(n, dtype=np.float64)
    for ii in range(n):
        meanK[ii] = float(np.mean(kernel(X[ii, np.newaxis], X)))
    return meanK


def _kt_best_X(
    X: np.ndarray,
    *,
    coresets: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    meanK: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select the candidate coreset with smallest empirical relative MMD."""
    n = int(X.shape[0])
    coreset_size = int(coresets.shape[1])

    # Standard thinning coreset from the end as initial reference
    step = max(1, n // coreset_size)
    best_coreset = np.array(range(n - 1, -1, -step), dtype=int)[::-1]
    best_rel_mmd2 = _kt_squared_emp_rel_mmd_X(X, best_coreset, kernel, meanK=meanK)

    for coreset in coresets:
        rel_mmd2 = _kt_squared_emp_rel_mmd_X(X, coreset, kernel, meanK=meanK)
        if rel_mmd2 < best_rel_mmd2:
            best_rel_mmd2 = rel_mmd2
            best_coreset = np.asarray(coreset, dtype=int)

    return np.asarray(best_coreset, dtype=int)


def _kt_squared_emp_rel_mmd_X(
    X: np.ndarray,
    coreset: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    meanK: Optional[np.ndarray] = None,
) -> float:
    """Squared empirical relative MMD as in goodpoints.kt."""
    coreset = np.asarray(coreset, dtype=int)
    coreset_size = int(coreset.size)
    if coreset_size == 0:
        return float("inf")

    k_core_core = 0.0

    if meanK is None:
        k_core_all = 0.0
        for ii in coreset:
            kii = kernel(X[ii, np.newaxis], X)
            k_core_core += float(np.mean(kii[coreset]))
            k_core_all += float(np.mean(kii))
    else:
        meanK = np.asarray(meanK, dtype=np.float64)
        k_core_all = float(np.sum(meanK[coreset]))
        for ii in coreset:
            k_core_core += float(np.mean(kernel(X[ii, np.newaxis], X[coreset])))

    return (k_core_core - 2.0 * k_core_all) / float(coreset_size)


def _kt_refine_X(
    X: np.ndarray,
    *,
    coreset: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    meanK: Optional[np.ndarray] = None,
    unique: bool = False,
) -> np.ndarray:
    """Greedy coreset refinement (swap each element to reduce MMD)."""
    import time as _time

    n = int(X.shape[0])
    coreset = np.asarray(coreset, dtype=int).copy()
    coreset_size = int(coreset.size)
    if coreset_size == 0:
        return coreset

    two_over_coreset_size = 2.0 / float(coreset_size)

    coreset_indicator = np.zeros(n, dtype=bool)
    coreset_indicator[coreset] = True

    _t_refine = _time.perf_counter()
    _pct_step_n = max(1, n // 10)
    if meanK is None:
        sufficient_stat = np.empty(n, dtype=np.float64)
        for ii in range(n):
            if ii > 0 and ii % _pct_step_n == 0:
                print(
                    f"              KT-REFINE init {ii}/{n} ({100*ii//n}%)",
                    flush=True,
                )
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                kii = kernel(X[ii, np.newaxis], X)
                sufficient_stat[ii] = (
                    2.0 * (float(np.mean(kii[coreset])) - float(np.mean(kii)))
                    + float(kii[ii]) / float(coreset_size)
                )
    else:
        meanK = np.asarray(meanK, dtype=np.float64)
        # kernel(X, X) returns diagonal evaluations (row-wise convention)
        sufficient_stat = kernel(X, X) / float(coreset_size) - 2.0 * meanK
        for ii in range(n):
            if ii > 0 and ii % _pct_step_n == 0:
                print(
                    f"              KT-REFINE init {ii}/{n} ({100*ii//n}%)",
                    flush=True,
                )
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                kiicore = kernel(X[ii, np.newaxis], X[coreset])
                sufficient_stat[ii] += 2.0 * float(np.mean(kiicore))

    _elapsed_init = _time.perf_counter() - _t_refine
    print(f"              KT-REFINE init done ({_elapsed_init:.0f}s), swapping {coreset_size} points...", flush=True)

    _pct_step_cs = max(1, coreset_size // 10)
    for coreset_idx in range(coreset_size):
        if coreset_idx > 0 and coreset_idx % _pct_step_cs == 0:
            _elapsed = _time.perf_counter() - _t_refine
            print(
                f"              KT-SWAP {coreset_idx}/{coreset_size} ({100*coreset_idx//coreset_size}%, {_elapsed:.0f}s)",
                flush=True,
            )
        if unique:
            cidx = int(coreset[coreset_idx])
            if meanK is None:
                kcidx = kernel(X[cidx, np.newaxis], X)
                sufficient_stat[cidx] = (
                    2.0 * (float(np.mean(kcidx[coreset])) - float(np.mean(kcidx)))
                    + float(kcidx[cidx]) / float(coreset_size)
                )
            else:
                kcidxcore = kernel(X[cidx, np.newaxis], X[coreset])
                sufficient_stat[cidx] = (
                    float(kernel(X[cidx, np.newaxis], X[cidx, np.newaxis]).ravel()[0]) / float(coreset_size)
                    - 2.0 * float(meanK[cidx])
                    + 2.0 * float(np.mean(kcidxcore))
                )

        sufficient_stat -= kernel(X[coreset[coreset_idx], np.newaxis], X) * two_over_coreset_size
        best_point = int(np.argmin(sufficient_stat))

        coreset[coreset_idx] = best_point
        sufficient_stat += kernel(X[best_point, np.newaxis], X) * two_over_coreset_size

        if unique:
            sufficient_stat[best_point] = np.inf

    return coreset
