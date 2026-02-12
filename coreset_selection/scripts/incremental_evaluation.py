#!/usr/bin/env python3
r"""Incremental evaluation runner (standalone script).

Produces the additional metrics and comparisons recommended by the
kernel k-means vs MMD+Sinkhorn+NSGA-II analysis:

    1. Tail absolute errors (P90, P95, P99) per target.
    2. Per-state KRR RMSE/MAE/R² breakdown.
    3. Macro-averaged and worst-group RMSE.
    4. Effect isolation: constraint vs. objective benefit.
    5. Rank tables across all methods × all metrics.
    6. Stability comparison across replicates.
    7. Comprehensive cross-method comparison report.

Usage
-----
As a standalone script (requires the project to be importable)::

    python -m scripts.incremental_evaluation \
        --results-dir runs_out/ \
        --output-dir incremental_eval/

Or imported as a library::

    from scripts.incremental_evaluation import (
        run_incremental_evaluation_from_cache,
        run_synthetic_benchmark,
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Direct module imports to avoid triggering the full evaluation.__init__
# which has relative imports requiring the full package hierarchy.
import importlib.util as _ilu

def _load_module(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, str(_PROJECT_ROOT / path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dm = _load_module("downstream_metrics", "evaluation/downstream_metrics.py")
_mc = _load_module("method_comparison", "evaluation/method_comparison.py")

# Re-export what we need
evaluate_nystrom_landmarks_downstream = _dm.evaluate_nystrom_landmarks_downstream
full_downstream_evaluation = _dm.full_downstream_evaluation
multitarget_downstream_evaluation = _dm.multitarget_downstream_evaluation
tail_absolute_errors = _dm.tail_absolute_errors
per_state_downstream_metrics = _dm.per_state_downstream_metrics
aggregate_group_metrics = _dm.aggregate_group_metrics

effect_isolation_table = _mc.effect_isolation_table
rank_table = _mc.rank_table
stability_comparison = _mc.stability_comparison
comprehensive_comparison = _mc.comprehensive_comparison
pairwise_dominance_matrix = _mc.pairwise_dominance_matrix
CANONICAL_LOWER_IS_BETTER = _mc.CANONICAL_LOWER_IS_BETTER
CANONICAL_HIGHER_IS_BETTER = _mc.CANONICAL_HIGHER_IS_BETTER


# =====================================================================
# CSV Writers
# =====================================================================

def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            # Convert non-serializable values
            clean = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    clean[k] = json.dumps(v)
                elif isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            w.writerow(clean)
    print(f"  ✓ Wrote {path}  ({len(rows)} rows)")


# =====================================================================
# Synthetic benchmark (for development / demonstration)
# =====================================================================

def _generate_synthetic_data(
    N: int = 5569,
    D: int = 50,
    G: int = 27,
    T: int = 2,
    seed: int = 42,
):
    """Generate synthetic data mimicking the Brazil telecom structure."""
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((N, D))
    y = np.zeros((N, T))
    state_labels = np.zeros(N, dtype=int)

    # Assign municipalities to states with realistic imbalance
    state_sizes = rng.dirichlet(np.ones(G) * 2.0) * N
    state_sizes = np.round(state_sizes).astype(int)
    state_sizes[-1] = N - state_sizes[:-1].sum()  # Fix rounding

    idx = 0
    for g in range(G):
        n_g = max(1, state_sizes[g])
        end = min(idx + n_g, N)
        state_labels[idx:end] = g

        # State-specific signal
        beta_g = rng.standard_normal((D, T)) * 0.3
        offset_g = rng.standard_normal(T) * 2
        y[idx:end] = X[idx:end] @ beta_g + offset_g + rng.standard_normal((end - idx, T)) * 0.2
        idx = end

    return X, y, state_labels


def _evaluate_landmarks_standalone(
    *,
    X_raw: np.ndarray,
    S_idx: np.ndarray,
    y: np.ndarray,
    eval_train_idx: np.ndarray,
    eval_test_idx: np.ndarray,
    eval_idx: np.ndarray,
    sigma_sq: float,
    state_labels: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Standalone Nyström KRR evaluation (no relative imports needed)."""
    X_raw = np.asarray(X_raw, dtype=np.float64)
    S_idx = np.asarray(S_idx, dtype=int)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    eval_idx = np.asarray(eval_idx, dtype=int)
    eval_train_idx = np.asarray(eval_train_idx, dtype=int)
    eval_test_idx = np.asarray(eval_test_idx, dtype=int)

    # RBF kernel helper
    def _rbf(X1, X2, ssq):
        n1 = np.sum(X1 * X1, axis=1, keepdims=True)
        n2 = np.sum(X2 * X2, axis=1, keepdims=True).T
        D2 = np.maximum(n1 + n2 - 2.0 * (X1 @ X2.T), 0.0)
        return np.exp(-D2 / (2.0 * ssq))

    # Map absolute indices → positions within eval_idx
    N = X_raw.shape[0]
    pos = np.full(N, -1, dtype=int)
    pos[eval_idx] = np.arange(eval_idx.size, dtype=int)
    tr_pos = pos[eval_train_idx]; tr_pos = tr_pos[tr_pos >= 0]
    te_pos = pos[eval_test_idx];  te_pos = te_pos[te_pos >= 0]

    X_E = X_raw[eval_idx]
    X_S = X_raw[S_idx]
    C = _rbf(X_E, X_S, sigma_sq)
    W = _rbf(X_S, X_S, sigma_sq)
    k = max(1, X_S.shape[0])
    lam_nys = 1e-6 * float(np.trace(W)) / float(k)
    W_reg = W + lam_nys * np.eye(k)

    # Nyström features via Cholesky
    try:
        L = np.linalg.cholesky(W_reg)
        Phi = np.linalg.solve(L, C.T).T
    except np.linalg.LinAlgError:
        w_eig, V = np.linalg.eigh(W_reg)
        w_eig = np.maximum(w_eig, 1e-12)
        Winv_sqrt = (V * (1.0 / np.sqrt(w_eig))) @ V.T
        Phi = C @ Winv_sqrt

    Phi_tr = Phi[tr_pos]
    Phi_te = Phi[te_pos]
    y_tr = y[eval_train_idx]
    y_te = y[eval_test_idx]

    T = y.shape[1]
    if target_names is None:
        if T == 2:
            target_names = ["_4G", "_5G"]
        elif T == 1:
            target_names = [""]
        else:
            target_names = [f"_{i}" for i in range(T)]

    y_pred_all = np.zeros_like(y_te)
    lambdas = np.logspace(-6, 6, 13)

    for t in range(T):
        yt_tr = y_tr[:, t]
        # Simple 3-fold CV for lambda selection
        best_lam, best_score = lambdas[0], np.inf
        n_tr = len(yt_tr)
        rng_cv = np.random.default_rng(12345 + t)
        perm = rng_cv.permutation(n_tr)
        folds = np.array_split(perm, 3)
        for lam_val in lambdas:
            scores = []
            for fi in range(3):
                va = folds[fi]
                tr = np.concatenate([folds[j] for j in range(3) if j != fi])
                A = Phi_tr[tr].T @ Phi_tr[tr] + float(lam_val) * np.eye(Phi_tr.shape[1])
                b = Phi_tr[tr].T @ yt_tr[tr]
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                pred = Phi_tr[va] @ w
                scores.append(float(np.sqrt(np.mean((yt_tr[va] - pred) ** 2))))
            if scores:
                mean_s = np.mean(scores)
                if mean_s < best_score:
                    best_score = mean_s
                    best_lam = lam_val

        A = Phi_tr.T @ Phi_tr + float(best_lam) * np.eye(Phi_tr.shape[1])
        b = Phi_tr.T @ yt_tr
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w_eig, V = np.linalg.eigh(A)
            w_eig = np.maximum(w_eig, 1e-12)
            w = (V / w_eig) @ V.T @ b
        y_pred_all[:, t] = Phi_te @ w

    te_states = state_labels[eval_test_idx] if state_labels is not None else None

    return multitarget_downstream_evaluation(
        y_true=y_te,
        y_pred=y_pred_all,
        state_labels=te_states,
        target_names=target_names,
    )


def _generate_coreset_indices(
    N: int,
    k: int,
    state_labels: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    """Simulate coreset selection for different methods."""
    rng = np.random.default_rng(seed)
    G = len(np.unique(state_labels))

    if method == "uniform":
        return rng.choice(N, size=k, replace=False)

    elif method == "kmeans":
        # Simulate cluster-biased selection (dense regions)
        # Bias towards larger states (mimicking k-means density focus)
        weights = np.ones(N)
        for g in np.unique(state_labels):
            mask = state_labels == g
            weights[mask] = mask.sum()
        weights /= weights.sum()
        idx = rng.choice(N, size=k, replace=False, p=weights)
        return idx

    elif method == "kkmeans":
        # Simulate kernel k-means: biased towards dense regions
        weights = np.ones(N)
        for g in np.unique(state_labels):
            mask = state_labels == g
            weights[mask] = mask.sum() ** 1.5  # stronger density bias
        weights /= weights.sum()
        return rng.choice(N, size=k, replace=False, p=weights)

    elif method == "kkmeans_quota":
        # Kernel k-means with proportionality constraint
        idx = []
        counts = np.bincount(state_labels, minlength=G)
        quotas = np.round(counts / counts.sum() * k).astype(int)
        quotas = np.maximum(quotas, 1)
        # Adjust to sum to k
        while quotas.sum() > k:
            quotas[quotas.argmax()] -= 1
        while quotas.sum() < k:
            quotas[quotas.argmin()] += 1
        for g in range(G):
            g_idx = np.flatnonzero(state_labels == g)
            n_sel = min(quotas[g], len(g_idx))
            idx.extend(rng.choice(g_idx, size=n_sel, replace=False).tolist())
        return np.array(idx[:k], dtype=int)

    elif method == "nsga2":
        # Simulate MMD+Sinkhorn selection: better coverage + proportional
        idx = []
        counts = np.bincount(state_labels, minlength=G)
        quotas = np.round(counts / counts.sum() * k).astype(int)
        quotas = np.maximum(quotas, 1)
        while quotas.sum() > k:
            quotas[quotas.argmax()] -= 1
        while quotas.sum() < k:
            quotas[quotas.argmin()] += 1
        for g in range(G):
            g_idx = np.flatnonzero(state_labels == g)
            n_sel = min(quotas[g], len(g_idx))
            # Use quasi-random selection for better coverage
            if n_sel > 0 and len(g_idx) > 0:
                step = max(1, len(g_idx) // n_sel)
                start = rng.integers(0, max(1, step))
                chosen = g_idx[start::step][:n_sel]
                if len(chosen) < n_sel:
                    remaining = np.setdiff1d(g_idx, chosen)
                    extra = rng.choice(remaining, size=n_sel - len(chosen), replace=False)
                    chosen = np.concatenate([chosen, extra])
                idx.extend(chosen.tolist())
        return np.array(idx[:k], dtype=int)

    else:
        return rng.choice(N, size=k, replace=False)


def run_synthetic_benchmark(
    output_dir: str = "incremental_eval",
    k: int = 300,
    n_replicates: int = 3,
    seed: int = 42,
):
    r"""Run the full incremental evaluation on synthetic data.

    This is useful for:
      - Validating that all metrics compute correctly.
      - Demonstrating the output format.
      - Development / CI testing.
    """
    print("=" * 72)
    print("INCREMENTAL EVALUATION — Synthetic Benchmark")
    print("=" * 72)

    os.makedirs(output_dir, exist_ok=True)

    # Method registry matching the project conventions
    methods = {
        "U":     ("uniform",        "exactk"),
        "KM":    ("kmeans",         "exactk"),
        "KKN":   ("kkmeans",        "exactk"),
        "SKKN":  ("kkmeans_quota",  "quota"),
        "Pareto":("nsga2",          "quota"),
    }

    target_names = ["_4G", "_5G"]
    all_replicate_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    for rep in range(n_replicates):
        rep_seed = seed + rep * 1000
        print(f"\n--- Replicate {rep} (seed={rep_seed}) ---")

        X, y, state_labels = _generate_synthetic_data(seed=rep_seed)
        N, D = X.shape
        T = y.shape[1]

        # Build eval set (stratified 2000-point subset)
        rng = np.random.default_rng(rep_seed + 1)
        eval_idx = rng.choice(N, size=min(2000, N), replace=False)
        eval_idx.sort()

        # Train/test split (80/20)
        n_eval = len(eval_idx)
        perm = rng.permutation(n_eval)
        n_tr = int(0.8 * n_eval)
        eval_train_idx = eval_idx[perm[:n_tr]]
        eval_test_idx = eval_idx[perm[n_tr:]]

        # Sigma via median heuristic
        X_E = X[eval_idx]
        pairs_i = rng.integers(0, n_eval, size=5000)
        pairs_j = rng.integers(0, n_eval, size=5000)
        d2 = np.sum((X_E[pairs_i] - X_E[pairs_j]) ** 2, axis=1)
        sigma_sq = float(np.median(d2) / 2.0)

        for mname, (sim_type, regime) in methods.items():
            print(f"  Method: {mname} ({sim_type}, {regime})")
            S_idx = _generate_coreset_indices(N, k, state_labels, sim_type, rep_seed + hash(mname) % 9999)

            res = _evaluate_landmarks_standalone(
                X_raw=X,
                S_idx=S_idx,
                y=y,
                eval_train_idx=eval_train_idx,
                eval_test_idx=eval_test_idx,
                eval_idx=eval_idx,
                sigma_sq=sigma_sq,
                state_labels=state_labels,
                target_names=target_names,
            )

            # Add method metadata
            res["method"] = mname
            res["regime"] = regime
            res["rep_id"] = rep
            res["k"] = k

            # Remove non-serializable per-state detail for CSV
            per_state_4g = res.pop("_per_state_detail_4G", {})
            per_state_5g = res.pop("_per_state_detail_5G", {})

            all_replicate_results[mname].append(res)

            # Write per-state detail CSV for this rep/method
            if per_state_4g:
                _write_per_state_csv(
                    per_state_4g,
                    os.path.join(output_dir, f"per_state_4G_{mname}_rep{rep}.csv"),
                    method=mname, rep=rep, target="4G",
                )
            if per_state_5g:
                _write_per_state_csv(
                    per_state_5g,
                    os.path.join(output_dir, f"per_state_5G_{mname}_rep{rep}.csv"),
                    method=mname, rep=rep, target="5G",
                )

    # ------------------------------------------------------------------
    # Aggregate outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("AGGREGATION & COMPARISON")
    print("=" * 72)

    # 1. All individual results (flat CSV)
    all_rows = []
    for mname, reps in all_replicate_results.items():
        all_rows.extend(reps)
    if all_rows:
        fields = sorted(set().union(*(r.keys() for r in all_rows)))
        _write_csv(os.path.join(output_dir, "all_downstream_metrics.csv"), all_rows, fields)

    # 2. Stability comparison
    metrics_for_stability = [
        "overall_rmse_4G", "overall_rmse_5G",
        "macro_rmse_4G", "macro_rmse_5G",
        "worst_group_rmse_4G", "worst_group_rmse_5G",
        "abs_err_p90_4G", "abs_err_p95_4G",
        "abs_err_p90_5G", "abs_err_p95_5G",
        "overall_r2_4G", "overall_r2_5G",
        "macro_r2_4G", "macro_r2_5G",
    ]
    stab = stability_comparison(all_replicate_results, metrics_for_stability)
    stab_rows = [{"method": m, **v} for m, v in stab.items()]
    if stab_rows:
        fields = ["method"] + sorted(set().union(*(r.keys() for r in stab_rows)) - {"method"})
        _write_csv(os.path.join(output_dir, "stability_comparison.csv"), stab_rows, fields)

    # 3. Mean results per method (for rank table and comparisons)
    mean_results: Dict[str, Dict[str, float]] = {}
    for mname, reps in all_replicate_results.items():
        combined: Dict[str, List[float]] = {}
        for r in reps:
            for k_m, v in r.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    combined.setdefault(k_m, []).append(v)
        mean_results[mname] = {k_m: float(np.mean(vals)) for k_m, vals in combined.items()}

    # 4. Comprehensive comparison
    report = comprehensive_comparison(mean_results)

    # 4a. Rank table
    all_metrics_present = [
        m for m in CANONICAL_LOWER_IS_BETTER + CANONICAL_HIGHER_IS_BETTER
        if any(m in r for r in mean_results.values())
    ]
    # Use the more granular per-suffix metrics
    lower_present = [m for m in mean_results.get("U", {}).keys()
                     if any(m.startswith(prefix) for prefix in
                            ["overall_rmse", "overall_mae", "macro_rmse", "worst_group_rmse",
                             "rmse_dispersion", "macro_mae", "worst_group_mae",
                             "abs_err_p"])]
    higher_present = [m for m in mean_results.get("U", {}).keys()
                      if any(m.startswith(prefix) for prefix in
                             ["overall_r2", "macro_r2", "worst_group_r2", "best_group_r2"])]

    if lower_present:
        rt_lower = rank_table(mean_results, lower_present, lower_is_better=True)
        rt_rows = [{"method": m, **v} for m, v in rt_lower.items()]
        if rt_rows:
            fields = ["method"] + sorted(set().union(*(r.keys() for r in rt_rows)) - {"method"})
            _write_csv(os.path.join(output_dir, "rank_table_lower_is_better.csv"), rt_rows, fields)

    if higher_present:
        rt_higher = rank_table(mean_results, higher_present, lower_is_better=False)
        rt_rows = [{"method": m, **v} for m, v in rt_higher.items()]
        if rt_rows:
            fields = ["method"] + sorted(set().union(*(r.keys() for r in rt_rows)) - {"method"})
            _write_csv(os.path.join(output_dir, "rank_table_higher_is_better.csv"), rt_rows, fields)

    # 5. Effect isolation table
    if "KKN" in mean_results and "SKKN" in mean_results and "Pareto" in mean_results:
        iso_metrics = lower_present + higher_present
        iso = effect_isolation_table(mean_results, iso_metrics)
        iso_rows = list(iso.values())
        if iso_rows:
            fields = list(iso_rows[0].keys())
            _write_csv(os.path.join(output_dir, "effect_isolation.csv"), iso_rows, fields)

    # 6. Pairwise dominance
    if lower_present:
        dom = pairwise_dominance_matrix(mean_results, lower_present, lower_is_better=True)
        dom_rows = []
        for a, inner in dom.items():
            for b, dominates in inner.items():
                if a != b:
                    dom_rows.append({"method_a": a, "method_b": b, "a_dominates_b": dominates})
        if dom_rows:
            _write_csv(os.path.join(output_dir, "pairwise_dominance.csv"), dom_rows)

    # 7. Summary JSON
    summary = {
        "n_methods": len(methods),
        "n_replicates": n_replicates,
        "k": k,
        "methods": list(methods.keys()),
        "summary_avg_rank": report.get("summary_avg_rank", {}),
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ Wrote {summary_path}")

    print("\n" + "=" * 72)
    print("DONE — All incremental evaluation outputs in:", output_dir)
    print("=" * 72)

    return report


def _write_per_state_csv(
    per_state: Dict[str, Dict[str, float]],
    path: str,
    method: str = "",
    rep: int = 0,
    target: str = "",
):
    """Write per-state breakdown to CSV."""
    rows = []
    for state, metrics in sorted(per_state.items()):
        rows.append({
            "state": state,
            "method": method,
            "rep_id": rep,
            "target": target,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })
    if rows:
        fields = list(rows[0].keys())
        _write_csv(path, rows, fields)


# =====================================================================
# Process existing results directory
# =====================================================================

def run_incremental_evaluation_from_results(
    results_dir: str,
    output_dir: str,
):
    r"""Re-evaluate existing result NPZ/CSV files with incremental metrics.

    Scans *results_dir* for saved coreset indices (``*_indices.npz``) and
    evaluation results (``*_results.csv``, ``*_summary.json``), then
    computes the additional downstream metrics.

    This is the offline/post-hoc path for when experiments have already
    been run and you want to add metrics without re-running NSGA-II.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_dir} for existing results...")

    # Collect all CSV result files
    csv_files = sorted(results_dir.rglob("*.csv"))
    npz_files = sorted(results_dir.rglob("*.npz"))
    json_files = sorted(results_dir.rglob("*.json"))

    print(f"  Found {len(csv_files)} CSV, {len(npz_files)} NPZ, {len(json_files)} JSON files")

    # If we find baseline summary CSVs, load them and augment
    baseline_csvs = [f for f in csv_files if "baseline" in f.name.lower()
                     or "summary" in f.name.lower()]

    if baseline_csvs:
        print(f"  Processing {len(baseline_csvs)} baseline/summary CSV files")
        for csv_path in baseline_csvs:
            _augment_csv_with_incremental(csv_path, output_dir)
    else:
        print("  No existing result CSVs found — running synthetic benchmark instead.")
        run_synthetic_benchmark(output_dir=str(output_dir))


def _augment_csv_with_incremental(csv_path: Path, output_dir: Path):
    """Read an existing results CSV and compute rank tables / comparisons."""
    import csv as csv_mod

    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            # Convert numeric strings
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)

    if not rows:
        return

    # Group by method
    by_method: Dict[str, List[Dict]] = {}
    for row in rows:
        mname = str(row.get("method", "unknown"))
        by_method.setdefault(mname, []).append(row)

    # Compute mean per method
    mean_results: Dict[str, Dict[str, float]] = {}
    for mname, reps in by_method.items():
        combined: Dict[str, List[float]] = {}
        for r in reps:
            for k, v in r.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    combined.setdefault(k, []).append(v)
        mean_results[mname] = {k: float(np.mean(vals)) for k, vals in combined.items()}

    # Produce comparison outputs
    stem = csv_path.stem
    report = comprehensive_comparison(mean_results)

    # Write summary
    summary_path = output_dir / f"{stem}_incremental_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "source_file": str(csv_path),
            "n_methods": len(mean_results),
            "methods": sorted(mean_results.keys()),
            "summary_avg_rank": report.get("summary_avg_rank", {}),
        }, f, indent=2, default=str)
    print(f"  ✓ {summary_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Incremental evaluation: downstream metrics & method comparison",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to existing results directory (optional)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="incremental_eval",
        help="Directory for incremental evaluation outputs",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run synthetic benchmark (default if no results-dir)",
    )
    parser.add_argument(
        "--k", type=int, default=300,
        help="Coreset size for synthetic benchmark",
    )
    parser.add_argument(
        "--replicates", type=int, default=3,
        help="Number of replicates for synthetic benchmark",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.results_dir and not args.synthetic:
        run_incremental_evaluation_from_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
        )
    else:
        run_synthetic_benchmark(
            output_dir=args.output_dir,
            k=args.k,
            n_replicates=args.replicates,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
