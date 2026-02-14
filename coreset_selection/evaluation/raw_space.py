"""
Raw-space evaluation metrics for coreset landmark sets.

Implements the manuscript's raw-space evaluation protocol (Section 5.9):
- Nyström Gram-matrix approximation error e_Nys(S)
- Kernel PCA spectral distortion e_kPCA(S)
- Nyström-feature KRR test RMSE for 4G and 5G targets

All metrics are computed in *standardized raw attribute space* on a fixed
evaluation index set E of size |E|=2000 (per replicate).

Notation (manuscript):
- E : evaluation index set
- S : selected landmark index set, |S| = k
- C = K_{E,S}
- W = K_{S,S}
- λ_nys = 1e-6 * tr(W) / k
- K_hat_{E,E} = C (W + λ_nys I)^{-1} C^T
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# --- Kernel / math utilities (extracted to _raw_kernels) ---
from ._raw_kernels import (
    _rbf_kernel,
    _median_sq_dist,
    _center_gram,
    _safe_cholesky_solve,
)

# --- Nyström-specific utilities (extracted to _raw_nystrom) ---
from ._raw_nystrom import (
    _nystrom_components,
    _nystrom_approx_gram,
    _nystrom_features,
    _rmse,
    _kfold_indices,
    _select_lambda_ridge,
)


@dataclass
class _NystromCache:
    """Cached Nyström decomposition for a single landmark set S."""
    S_key: int           # hash of S_idx for cache invalidation
    C: np.ndarray        # (|E|, k)
    W: np.ndarray        # (k, k)
    lambda_nys: float
    K_hat: np.ndarray    # (|E|, |E|)  — only built when needed
    Phi: np.ndarray      # (|E|, k)
    Phi_tr: np.ndarray   # (n_train, k)
    Phi_te: np.ndarray   # (n_test, k)
    tr_pos: np.ndarray
    te_pos: np.ndarray


@dataclass
class RawSpaceEvaluator:
    """
    Raw-space evaluator bound to a replicate.

    Parameters
    ----------
    X_raw : np.ndarray
        Standardized raw feature matrix (N, D). (Name kept for backward compatibility.)
    y : Optional[np.ndarray]
        Targets (N,) or (N, T) — columns match ``target_names`` order.
    target_names : Optional[list]
        Human-readable names for each target column.  When *None* the legacy
        naming ``krr_rmse_4G`` / ``krr_rmse_5G`` is used for a 2-column y.
    eval_idx : np.ndarray
        Evaluation index set E (absolute indices into X_raw)
    eval_train_idx : np.ndarray
        Training indices within E (absolute indices)
    eval_test_idx : np.ndarray
        Test indices within E (absolute indices)
    sigma_sq : float
        Kernel bandwidth squared (sigma_raw^2)
    _K_EE : Optional[np.ndarray]
        Full raw-space Gram matrix on E (computed lazily)
    _seed : int
        Random seed for reproducibility
    """
    X_raw: np.ndarray
    y: Optional[np.ndarray]
    target_names: Optional[list]
    eval_idx: np.ndarray
    eval_train_idx: np.ndarray
    eval_test_idx: np.ndarray
    sigma_sq: float
    _K_EE: Optional[np.ndarray] = None
    _seed: int = 0
    _nys_cache: Optional[_NystromCache] = field(default=None, repr=False)

    @property
    def K_EE(self) -> np.ndarray:
        """Lazily compute and cache the full Gram matrix on E."""
        if self._K_EE is None:
            X_E = self.X_raw[self.eval_idx]
            # This is the expensive operation - only done when actually needed
            self._K_EE = _rbf_kernel(X_E, X_E, self.sigma_sq)
        return self._K_EE

    @staticmethod
    def build(
        *,
        X_raw: np.ndarray,
        y: Optional[np.ndarray],
        eval_idx: np.ndarray,
        eval_train_idx: np.ndarray,
        eval_test_idx: np.ndarray,
        seed: int,
        target_names: Optional[list] = None,
    ) -> "RawSpaceEvaluator":
        r"""Construct a RawSpaceEvaluator with bandwidth from the median heuristic.

        Per manuscript Section VII.C, the raw kernel bandwidth σ²_raw is
        computed via the median heuristic on E_train only:
            σ² = median(‖x_i − x_j‖²) / 2,  i,j ∈ E_train.
        The full Gram matrix K_EE is computed lazily on first access.

        Parameters
        ----------
        X_raw : (N, D) ndarray
            Standardized raw feature matrix.
        y : (N, T) ndarray or None
            Coverage targets for KRR evaluation.
        eval_idx : (|E|,) ndarray
            Evaluation set indices.
        eval_train_idx, eval_test_idx : ndarray
            Train/test split within E (80/20 stratified).
        seed : int
            Random seed for reproducibility.
        target_names : list of str or None
            Names for each target column (Table V labels).
        """
        X_raw = np.asarray(X_raw, dtype=np.float64)
        eval_idx = np.asarray(eval_idx, dtype=int)
        X_E = X_raw[eval_idx]

        # Median heuristic bandwidth on E (this is fast - just samples pairs)
        med_d2 = _median_sq_dist(X_E, seed=seed)
        sigma_sq = med_d2 / 2.0  # median heuristic for RBF often uses sigma^2 = median(d^2)/2
        sigma_sq = float(max(sigma_sq, 1e-12))

        # NOTE: K_EE is NOT computed here anymore - it's computed lazily on first access
        # This avoids the expensive 2000x2000 kernel computation at startup

        return RawSpaceEvaluator(
            X_raw=X_raw,
            y=y,
            target_names=target_names,
            eval_idx=eval_idx,
            eval_train_idx=np.asarray(eval_train_idx, dtype=int),
            eval_test_idx=np.asarray(eval_test_idx, dtype=int),
            sigma_sq=sigma_sq,
            _K_EE=None,  # Will be computed lazily
            _seed=seed,
        )

    # ------------------------------------------------------------------
    # Nyström cache management
    # ------------------------------------------------------------------

    def _get_nystrom(self, S_idx: np.ndarray) -> _NystromCache:
        """Return cached Nyström decomposition for S, recomputing only if S changed."""
        S_idx = np.asarray(S_idx, dtype=int)
        s_key = hash(S_idx.tobytes())

        if self._nys_cache is not None and self._nys_cache.S_key == s_key:
            return self._nys_cache

        X_E = self.X_raw[self.eval_idx]
        X_S = self.X_raw[S_idx]

        C, W, lambda_nys = _nystrom_components(X_E, X_S, self.sigma_sq)
        K_hat = _nystrom_approx_gram(C, W, lambda_nys)
        Phi = _nystrom_features(C, W, lambda_nys)

        # Map absolute indices -> positions within eval_idx
        N = self.X_raw.shape[0]
        pos = np.full(N, -1, dtype=int)
        pos[self.eval_idx] = np.arange(self.eval_idx.size, dtype=int)

        tr_pos = pos[self.eval_train_idx]
        te_pos = pos[self.eval_test_idx]
        tr_pos = tr_pos[tr_pos >= 0]
        te_pos = te_pos[te_pos >= 0]

        cache = _NystromCache(
            S_key=s_key,
            C=C, W=W, lambda_nys=lambda_nys,
            K_hat=K_hat, Phi=Phi,
            Phi_tr=Phi[tr_pos], Phi_te=Phi[te_pos],
            tr_pos=tr_pos, te_pos=te_pos,
        )
        self._nys_cache = cache
        return cache

    # ------------------------------------------------------------------
    # Metrics for a landmark set S
    # ------------------------------------------------------------------

    def nystrom_error(self, S_idx: np.ndarray) -> float:
        """Relative Frobenius error e_Nys(S)."""
        nys = self._get_nystrom(S_idx)
        num = np.linalg.norm(self.K_EE - nys.K_hat, ord="fro")
        den = np.linalg.norm(self.K_EE, ord="fro") + 1e-30
        return float(num / den)

    def kpca_distortion(self, S_idx: np.ndarray, r: int = 20) -> float:
        """Kernel PCA top-r spectral distortion e_kPCA(S)."""
        nys = self._get_nystrom(S_idx)

        Kc = _center_gram(self.K_EE)
        Kc_hat = _center_gram(nys.K_hat)

        # Eigenvalues (ascending), then reverse to descending
        lam = np.linalg.eigvalsh(Kc)[::-1]
        lam_hat = np.linalg.eigvalsh(Kc_hat)[::-1]

        # Clip small negative eigenvalues from numerical noise
        lam = np.maximum(lam, 0.0)
        lam_hat = np.maximum(lam_hat, 0.0)

        r = int(min(r, lam.size, lam_hat.size))
        if r <= 0:
            return 0.0

        v = lam[:r]
        v_hat = lam_hat[:r]
        num = np.linalg.norm(v - v_hat)
        den = np.linalg.norm(v) + 1e-30
        return float(num / den)

    def _krr_fit_targets(
        self,
        Phi_tr: np.ndarray,
        Phi_te: np.ndarray,
    ) -> Dict[str, float]:
        """Fit KRR on all targets using precomputed Phi splits.

        Shared by krr_rmse() and _state_conditioned_stability() to avoid
        duplicating the 5-fold ridge CV.

        Returns per-target RMSE, lambda, and weight vectors.
        """
        if self.y is None:
            return {}, {}

        y = np.asarray(self.y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_tr = y[self.eval_train_idx]
        y_te = y[self.eval_test_idx]

        lambdas = np.logspace(-6, 6, 13)
        n_folds = 5

        n_targets = y.shape[1]

        # Resolve output key names
        if self.target_names is not None and len(self.target_names) == n_targets:
            names = list(self.target_names)
        elif n_targets == 1:
            names = [""]
        elif n_targets == 2:
            names = ["4G", "5G"]
        else:
            names = [str(t) for t in range(n_targets)]

        out: Dict[str, float] = {}
        # Store predictions for reuse by stability metrics
        predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # t -> (y_hat_te, best_lam)

        for t in range(n_targets):
            yt_tr = np.asarray(y_tr[:, t], dtype=np.float64)
            yt_te = np.asarray(y_te[:, t], dtype=np.float64)

            best_lam = _select_lambda_ridge(
                Phi_tr, yt_tr, lambdas=lambdas, n_folds=n_folds,
                seed=12345 + t,
            )

            A = Phi_tr.T @ Phi_tr + float(best_lam) * np.eye(Phi_tr.shape[1])
            b = Phi_tr.T @ yt_tr
            w, _ = _safe_cholesky_solve(A, b)
            y_hat = Phi_te @ w
            rmse = _rmse(yt_te, y_hat)

            suffix = names[t]
            if suffix == "":
                key_rmse = "krr_rmse"
                key_lam = "krr_lambda"
            else:
                key_rmse = f"krr_rmse_{suffix}"
                key_lam = f"krr_lambda_{suffix}"

            out[key_rmse] = float(rmse)
            out[key_lam] = float(best_lam)
            predictions[t] = (y_hat, best_lam)

        # Aggregated RMSE across all targets
        rmse_vals = [v for k, v in out.items() if k.startswith("krr_rmse") and not k.endswith("mean")]
        if len(rmse_vals) >= 2:
            out["krr_rmse_mean"] = float(np.mean(rmse_vals))

        # Backward-compatible aliases when named targets include the primary pair
        if self.target_names is not None:
            if "cov_area_4G" in self.target_names and "krr_rmse_cov_area_4G" in out:
                out.setdefault("krr_rmse_4G", out["krr_rmse_cov_area_4G"])
                out.setdefault("krr_lambda_4G", out["krr_lambda_cov_area_4G"])
            if "cov_area_5G" in self.target_names and "krr_rmse_cov_area_5G" in out:
                out.setdefault("krr_rmse_5G", out["krr_rmse_cov_area_5G"])
                out.setdefault("krr_lambda_5G", out["krr_lambda_cov_area_5G"])

        return out, predictions

    def krr_rmse(self, S_idx: np.ndarray) -> Dict[str, float]:
        """
        Nyström-feature KRR evaluation.

        When ``target_names`` is set (multi-target mode), output keys are
        ``krr_rmse_{name}`` and ``krr_lambda_{name}`` for each named target.
        Backward-compatible ``krr_rmse_4G`` / ``krr_rmse_5G`` keys are
        emitted when ``target_names`` is *None* and y has 2 columns.

        Returns per-target RMSE and the selected ridge lambda (per target).
        """
        if self.y is None:
            return {}
        nys = self._get_nystrom(S_idx)
        out, _predictions = self._krr_fit_targets(nys.Phi_tr, nys.Phi_te)
        return out

    def all_metrics(self, S_idx: np.ndarray) -> Dict[str, float]:
        """Compute all enabled raw-space metrics for S."""
        S_idx = np.asarray(S_idx, dtype=int)
        out: Dict[str, float] = {}

        out["nystrom_error"] = self.nystrom_error(S_idx)
        out["kpca_distortion"] = self.kpca_distortion(S_idx, r=20)

        out.update(self.krr_rmse(S_idx))

        # Report sigma (for debugging)
        out["sigma_sq_raw"] = float(self.sigma_sq)
        return out

    def all_metrics_with_state_stability(
        self,
        S_idx: np.ndarray,
        state_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all raw-space metrics plus state-conditioned KPI stability.

        For each target (4G, 5G), computes per-state RMSE and then:
          - Kendall's tau between state-wise RMSE rankings across two
            independent half-splits of E_test
          - max_drift: max |RMSE_state(half1) - RMSE_state(half2)| over states
          - avg_drift: mean |RMSE_state(half1) - RMSE_state(half2)| over states

        These metrics assess whether landmark quality is consistent across
        geographic regions (manuscript Section V-E).

        Optimized: computes Nyström features and KRR once, reusing predictions
        for both krr_rmse and stability metrics.

        Parameters
        ----------
        S_idx : np.ndarray
            Selected landmark indices
        state_labels : np.ndarray
            Group labels (N,) for computing per-state metrics

        Returns
        -------
        Dict[str, float]
            All base metrics plus stability metrics
        """
        S_idx = np.asarray(S_idx, dtype=int)
        nys = self._get_nystrom(S_idx)

        out: Dict[str, float] = {}

        # Nyström error + kPCA distortion (share cached K_hat)
        out["nystrom_error"] = self.nystrom_error(S_idx)
        out["kpca_distortion"] = self.kpca_distortion(S_idx, r=20)

        # KRR fit once — get both metrics and predictions
        krr_out, predictions = self._krr_fit_targets(nys.Phi_tr, nys.Phi_te)
        out.update(krr_out)
        out["sigma_sq_raw"] = float(self.sigma_sq)

        # State-conditioned stability reusing predictions from KRR
        stability = self._state_conditioned_stability_from_predictions(
            predictions, state_labels,
        )
        out.update(stability)

        return out

    def _state_conditioned_stability_from_predictions(
        self,
        predictions: Dict[int, Tuple[np.ndarray, float]],
        state_labels: np.ndarray,
    ) -> Dict[str, float]:
        """State-conditioned KPI stability using precomputed KRR predictions.

        Splits E_test into two halves (stratified by state) and computes
        per-state RMSE independently on each half. Then measures ranking
        agreement (Kendall's tau) and absolute drift.

        Also computes the *incremental downstream metrics*:
          - Tail absolute errors: P90, P95, P99 of |y - ŷ|
          - Overall MAE and R²
          - Per-state MAE and R² (alongside existing per-state RMSE)
          - Macro-averaged RMSE / MAE / R² (mean over states)
          - Worst-group RMSE / MAE / R²
          - RMSE dispersion (std) and IQR across states
        """
        from scipy.stats import kendalltau

        if self.y is None or not predictions:
            return {}

        state_labels = np.asarray(state_labels)

        y = np.asarray(self.y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_te = y[self.eval_test_idx]

        te_states = state_labels[self.eval_test_idx]
        unique_states = np.unique(te_states)
        if unique_states.size < 2:
            return {}

        out: Dict[str, float] = {}

        for t, (y_hat_te, best_lam) in predictions.items():
            yt_te = np.asarray(y_te[:, t], dtype=np.float64)

            suffix = "" if y.shape[1] == 1 else ("_4G" if t == 0 else "_5G")

            # --- Global metrics: tail errors, MAE, R² ---
            abs_errs = np.abs(yt_te - y_hat_te)
            out[f"abs_err_p90{suffix}"] = float(np.quantile(abs_errs, 0.90))
            out[f"abs_err_p95{suffix}"] = float(np.quantile(abs_errs, 0.95))
            out[f"abs_err_p99{suffix}"] = float(np.quantile(abs_errs, 0.99))
            out[f"abs_err_max{suffix}"] = float(abs_errs.max())
            out[f"overall_mae{suffix}"] = float(np.mean(abs_errs))

            ss_res = float(np.sum((yt_te - y_hat_te) ** 2))
            ss_tot = float(np.sum((yt_te - np.mean(yt_te)) ** 2))
            out[f"overall_r2{suffix}"] = 1.0 - ss_res / max(ss_tot, 1e-30)

            # --- Per-state RMSE, MAE, R² ---
            state_rmses: Dict[Any, float] = {}
            state_maes: Dict[Any, float] = {}
            state_r2s: Dict[Any, float] = {}
            for g in unique_states:
                mask_g = te_states == g
                n_g = int(mask_g.sum())
                if n_g < 2:
                    continue
                yt_g = yt_te[mask_g]
                yh_g = y_hat_te[mask_g]
                state_rmses[g] = _rmse(yt_g, yh_g)
                state_maes[g] = float(np.mean(np.abs(yt_g - yh_g)))
                ss_res_g = float(np.sum((yt_g - yh_g) ** 2))
                ss_tot_g = float(np.sum((yt_g - np.mean(yt_g)) ** 2))
                state_r2s[g] = 1.0 - ss_res_g / max(ss_tot_g, 1e-30)

            # --- Aggregate per-state metrics ---
            if state_rmses:
                rmse_vals = list(state_rmses.values())
                mae_vals = list(state_maes.values())
                r2_vals = list(state_r2s.values())

                out[f"macro_rmse{suffix}"] = float(np.mean(rmse_vals))
                out[f"worst_group_rmse{suffix}"] = float(np.max(rmse_vals))
                out[f"best_group_rmse{suffix}"] = float(np.min(rmse_vals))
                out[f"rmse_dispersion{suffix}"] = float(np.std(rmse_vals))
                out[f"rmse_iqr{suffix}"] = float(
                    np.quantile(rmse_vals, 0.75) - np.quantile(rmse_vals, 0.25)
                )
                out[f"macro_mae{suffix}"] = float(np.mean(mae_vals))
                out[f"worst_group_mae{suffix}"] = float(np.max(mae_vals))
                out[f"macro_r2{suffix}"] = float(np.mean(r2_vals))
                out[f"worst_group_r2{suffix}"] = float(np.min(r2_vals))
                out[f"best_group_r2{suffix}"] = float(np.max(r2_vals))
                out[f"n_groups_evaluated{suffix}"] = len(rmse_vals)

            # --- Split-half stability ---
            rng = np.random.default_rng(self._seed + 9999 + t)
            half1_rmses = {}
            half2_rmses = {}
            for g in sorted(state_rmses.keys()):
                idx_g = np.flatnonzero(te_states == g)
                if idx_g.size < 4:
                    continue
                rng.shuffle(idx_g)
                mid = idx_g.size // 2
                h1, h2 = idx_g[:mid], idx_g[mid:]
                half1_rmses[g] = _rmse(yt_te[h1], y_hat_te[h1])
                half2_rmses[g] = _rmse(yt_te[h2], y_hat_te[h2])

            common_states = sorted(set(half1_rmses) & set(half2_rmses))
            if len(common_states) < 3:
                continue

            r1 = np.array([half1_rmses[g] for g in common_states])
            r2 = np.array([half2_rmses[g] for g in common_states])

            tau, _ = kendalltau(r1, r2)
            drifts = np.abs(r1 - r2)

            out[f"kendall_tau{suffix}"] = float(tau) if np.isfinite(tau) else 0.0
            out[f"max_drift{suffix}"] = float(drifts.max())
            out[f"avg_drift{suffix}"] = float(drifts.mean())
            out[f"n_states_stability{suffix}"] = len(common_states)

            # Per-state RMSE for full test set (for diagnostics CSV)
            out[f"state_rmse_max{suffix}"] = float(max(state_rmses.values()))
            out[f"state_rmse_min{suffix}"] = float(min(state_rmses.values()))
            out[f"state_rmse_std{suffix}"] = float(np.std(list(state_rmses.values())))

        return out

    def _state_conditioned_stability(
        self,
        S_idx: np.ndarray,
        state_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Legacy wrapper — computes Nyström and KRR, then delegates.

        Prefer all_metrics_with_state_stability() for the optimized path.
        """
        if self.y is None:
            return {}
        nys = self._get_nystrom(S_idx)
        _krr_out, predictions = self._krr_fit_targets(nys.Phi_tr, nys.Phi_te)
        return self._state_conditioned_stability_from_predictions(
            predictions, state_labels,
        )

    def multi_model_downstream(
        self,
        S_idx: np.ndarray,
        regression_targets: Dict[str, np.ndarray],
        classification_targets: Dict[str, np.ndarray],
        seed: int = 123,
    ) -> Dict[str, float]:
        """Extended downstream evaluation using multiple models on Nyström features.

        Computes Nyström features once (cached) for landmark set S, then trains
        KNN / RF / LR / GBT on the train portion of E for each regression
        and classification target.

        Parameters
        ----------
        S_idx : np.ndarray
            Selected landmark indices (absolute).
        regression_targets : Dict[str, np.ndarray]
            ``{target_name: (N,) float64 array}`` for regression.
        classification_targets : Dict[str, np.ndarray]
            ``{target_name: (N,) int64 array}`` for classification.
        seed : int
            Random seed.

        Returns
        -------
        Dict[str, float]
            Flat metrics dict keyed as ``{model}_{metric}_{target}``.
        """
        from .multi_model_evaluator import evaluate_all_downstream_models

        nys = self._get_nystrom(S_idx)

        return evaluate_all_downstream_models(
            Phi_train=nys.Phi_tr,
            Phi_test=nys.Phi_te,
            eval_train_idx=self.eval_train_idx,
            eval_test_idx=self.eval_test_idx,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            seed=seed,
        )
