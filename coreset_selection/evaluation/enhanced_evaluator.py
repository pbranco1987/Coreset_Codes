r"""Enhanced raw-space evaluator with incremental downstream metrics.

Wraps :class:`evaluation.raw_space.RawSpaceEvaluator` to additionally
compute the metrics recommended by the kernel k-means comparison analysis:

    - Tail absolute errors (P90, P95, P99)
    - Per-state RMSE / MAE / R²
    - Macro-averaged and worst-group RMSE
    - Overall MAE and R²

Usage: drop-in replacement for ``RawSpaceEvaluator.all_metrics()``::

    from evaluation.enhanced_evaluator import EnhancedRawSpaceEvaluator

    evaluator = EnhancedRawSpaceEvaluator.from_base(base_evaluator)
    metrics = evaluator.all_metrics_enhanced(S_idx, state_labels)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .raw_space import (
    RawSpaceEvaluator,
    _nystrom_components,
    _nystrom_features,
    _safe_cholesky_solve,
    _select_lambda_ridge,
    _rmse,
)
from .downstream_metrics import (
    full_downstream_evaluation,
    multitarget_downstream_evaluation,
)


class EnhancedRawSpaceEvaluator:
    """Wraps a RawSpaceEvaluator with extended downstream metrics."""

    def __init__(self, base: RawSpaceEvaluator):
        self.base = base

    @staticmethod
    def from_base(base: RawSpaceEvaluator) -> "EnhancedRawSpaceEvaluator":
        return EnhancedRawSpaceEvaluator(base)

    def all_metrics_enhanced(
        self,
        S_idx: np.ndarray,
        state_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        r"""Compute all standard + incremental metrics.

        Returns a dict combining:
          - nystrom_error, kpca_distortion, krr_rmse_* (from base)
          - overall_rmse_*, overall_mae_*, overall_r2_* (new)
          - abs_err_p90_*, abs_err_p95_*, abs_err_p99_* (new)
          - macro_rmse_*, worst_group_rmse_*, rmse_dispersion_* (new)
          - macro_r2_*, worst_group_r2_* (new)
          - state-conditioned stability (from base)
        """
        S_idx = np.asarray(S_idx, dtype=int)

        # Standard metrics from base evaluator
        out = self.base.all_metrics(S_idx)

        # If state labels available, add base stability too
        if state_labels is not None:
            try:
                stab = self.base._state_conditioned_stability(S_idx, state_labels)
                out.update(stab)
            except Exception:
                pass

        # Now compute incremental downstream metrics via KRR predictions
        if self.base.y is None:
            return out

        # Reproduce the KRR prediction pipeline to get y_pred
        y_pred_te, y_true_te, te_states = self._predict_krr(S_idx, state_labels)
        if y_pred_te is None:
            return out

        y = np.asarray(self.base.y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        T = y.shape[1]

        if self.base.target_names is not None and len(self.base.target_names) == T:
            target_names = [f"_{n}" if not n.startswith("_") else n
                            for n in self.base.target_names]
        elif T == 2:
            target_names = ["_4G", "_5G"]
        elif T == 1:
            target_names = [""]
        else:
            target_names = [f"_{i}" for i in range(T)]

        incremental = multitarget_downstream_evaluation(
            y_true=y_true_te,
            y_pred=y_pred_te,
            state_labels=te_states,
            target_names=target_names,
        )

        # Remove internal per-state detail dicts (not serializable)
        incremental = {k: v for k, v in incremental.items()
                       if not k.startswith("_per_state_detail")}

        out.update(incremental)
        return out

    def _predict_krr(
        self,
        S_idx: np.ndarray,
        state_labels: Optional[np.ndarray],
    ):
        """Internal: fit KRR on E_train, predict on E_test."""
        if self.base.y is None:
            return None, None, None

        N = self.base.X_raw.shape[0]
        pos = np.full(N, -1, dtype=int)
        pos[self.base.eval_idx] = np.arange(self.base.eval_idx.size, dtype=int)

        tr_pos = pos[self.base.eval_train_idx]; tr_pos = tr_pos[tr_pos >= 0]
        te_pos = pos[self.base.eval_test_idx];  te_pos = te_pos[te_pos >= 0]

        X_E = self.base.X_raw[self.base.eval_idx]
        X_S = self.base.X_raw[S_idx]

        C, W, lambda_nys = _nystrom_components(X_E, X_S, self.base.sigma_sq)
        Phi = _nystrom_features(C, W, lambda_nys)

        Phi_tr = Phi[tr_pos]
        Phi_te = Phi[te_pos]

        y = np.asarray(self.base.y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_tr = y[self.base.eval_train_idx]
        y_te = y[self.base.eval_test_idx]

        lambdas = np.logspace(-6, 6, 13)
        T = y.shape[1]
        y_pred_te = np.zeros_like(y_te)

        for t in range(T):
            yt_tr = y_tr[:, t]
            best_lam = _select_lambda_ridge(
                Phi_tr, yt_tr, lambdas=lambdas, n_folds=5, seed=12345 + t,
            )
            A = Phi_tr.T @ Phi_tr + float(best_lam) * np.eye(Phi_tr.shape[1])
            b = Phi_tr.T @ yt_tr
            w, _ = _safe_cholesky_solve(A, b)
            y_pred_te[:, t] = Phi_te @ w

        te_states = state_labels[self.base.eval_test_idx] if state_labels is not None else None
        return y_pred_te, y_te, te_states

    def per_state_detail(
        self,
        S_idx: np.ndarray,
        state_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Get the full per-state breakdown (for CSV export / heatmaps)."""
        from .downstream_metrics import per_state_downstream_metrics

        y_pred_te, y_te, te_states = self._predict_krr(S_idx, state_labels)
        if y_pred_te is None or te_states is None:
            return {}

        y = np.asarray(self.base.y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        T = y.shape[1]

        result = {}
        for t in range(T):
            suffix = f"_t{t}" if T > 2 else ("_4G" if t == 0 else "_5G") if T == 2 else ""
            detail = per_state_downstream_metrics(y_te[:, t], y_pred_te[:, t], te_states)
            result[f"target{suffix}"] = detail
        return result
