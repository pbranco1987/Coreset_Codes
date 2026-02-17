"""Table generation methods for ManuscriptArtifacts (mixin)."""
from __future__ import annotations
import glob
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TablesMixin:
    """Mixin providing LaTeX/CSV table generation methods."""

    def tab_r1_by_k(self, df: pd.DataFrame) -> str:
        r"""Table III: R1 envelope + knee values across k (``\label{tab:r1-by-k}``).

        Per the manuscript (Section VIII.C), this table reports, for each k
        in the cardinality grid :math:`\mathcal{K}`:

        * **Envelope (best)**: metric-wise minimum over the Pareto front.
        * **Knee**: value at the balanced (knee) solution.

        The gap between the two quantifies the cost of the balanced
        compromise.  When multiple seeds are present (R1 has 5 seeds),
        values are averaged across seeds.

        Columns (manuscript Table III):
          k | e_Nys(best) | e_Nys(knee) | RMSE_4G(best) | RMSE_4G(knee) | ...
        """
        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            return ""

        d["k"] = d["k"].astype(int)

        # Manuscript-canonical column mapping: internal name -> short label
        col_map = {
            "nystrom_error":        "$e_{\\mathrm{Nys}}$",
            "kpca_distortion":      "$e_{\\mathrm{kPCA}}$",
            "krr_rmse_cov_area_4G": "$\\mathrm{RMSE}_{\\mathrm{4G}}$",
            "krr_rmse_cov_area_5G": "$\\mathrm{RMSE}_{\\mathrm{5G}}$",
        }
        # Fallback: try legacy column names if canonical ones missing
        _legacy_aliases = {
            "krr_rmse_cov_area_4G": ["krr_rmse_4G", "krr_rmse_area_4G"],
            "krr_rmse_cov_area_5G": ["krr_rmse_5G", "krr_rmse_area_5G"],
        }
        resolved_cols = {}
        for canonical, label in col_map.items():
            if canonical in d.columns:
                resolved_cols[canonical] = label
            else:
                for alias in _legacy_aliases.get(canonical, []):
                    if alias in d.columns:
                        d[canonical] = d[alias]
                        resolved_cols[canonical] = label
                        break

        # Must have at least the Nystrom error column
        metrics = [c for c in col_map if c in resolved_cols]
        if not metrics:
            return ""

        # Detect seed column (rep_id or seed)
        seed_col = None
        for cand in ["rep_id", "seed", "replicate"]:
            if cand in d.columns:
                seed_col = cand
                break

        # --- Envelope (metric-wise best) ---
        if seed_col is not None and d[seed_col].nunique() > 1:
            per_seed_env = d.groupby(["k", seed_col])[metrics].min().reset_index()
            env = per_seed_env.groupby("k")[metrics].mean().reset_index()
        else:
            env = d.groupby("k")[metrics].min().reset_index()
        env = env.sort_values("k")

        # --- Knee-point values ---
        has_rep_name = "rep_name" in d.columns
        if has_rep_name:
            d_knee = d[d["rep_name"].astype(str) == "knee"]
        else:
            d_knee = pd.DataFrame()

        if not d_knee.empty:
            if seed_col is not None and d_knee[seed_col].nunique() > 1:
                knee_agg = d_knee.groupby("k")[metrics].mean().reset_index()
            else:
                knee_agg = d_knee.groupby("k")[metrics].mean().reset_index()
        else:
            # Fallback: knee unavailable, repeat envelope values
            knee_agg = env.copy()

        knee_agg = knee_agg.sort_values("k")
        # Align k values with envelope
        knee_dict = {}
        for _, row in knee_agg.iterrows():
            knee_dict[int(row["k"])] = row

        # --- LaTeX (paired best/knee columns per metric) ---
        # Column spec: k | best_m1 | knee_m1 | best_m2 | knee_m2 | ...
        ncols = len(metrics) * 2
        headers_row = "$k$"
        subheaders_row = ""
        col_spec = "c"  # for k
        for m in metrics:
            short = resolved_cols[m]
            headers_row += f" & \\multicolumn{{2}}{{c}}{{{short}}}"
            subheaders_row += " & best & knee"
            col_spec += " cc"

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{R1 envelope (best) and knee-point values across coreset "
            r"sizes $k$. The gap quantifies the cost of the balanced compromise.}",
            r"\label{tab:r1-by-k}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            headers_row + r" \\",
            r"\cmidrule(lr){2-3}" if len(metrics) >= 1 else "",
        ]
        # Add cmidrule for each metric pair
        cmidrules = []
        for i in range(len(metrics)):
            start = 2 + i * 2
            end = start + 1
            cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        lines[-1] = " ".join(cmidrules)
        lines.append(subheaders_row.lstrip(" &") if not subheaders_row.startswith(" &") else
                      "& " + subheaders_row.lstrip(" & ") if subheaders_row.strip() else "")
        # Fix subheader row: prepend empty cell for k column
        lines[-1] = " " + subheaders_row + r" \\"

        lines.append(r"\midrule")

        for _, row in env.iterrows():
            k_val = int(row["k"])
            knee_row = knee_dict.get(k_val, None)
            vals_parts = []
            for m in metrics:
                best_val = f"{row[m]:.4f}"
                if knee_row is not None and m in knee_row.index and np.isfinite(knee_row[m]):
                    knee_val = f"{knee_row[m]:.4f}"
                else:
                    knee_val = "---"
                vals_parts.append(f"{best_val} & {knee_val}")
            lines.append(f"{k_val} & " + " & ".join(vals_parts) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "r1_by_k.tex")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))

        # Companion CSV
        csv_rows = []
        for _, row in env.iterrows():
            k_val = int(row["k"])
            knee_row = knee_dict.get(k_val, None)
            csv_row = {"k": k_val}
            for m in metrics:
                m_label = m.replace("_", " ")
                csv_row[f"{m_label} (best)"] = row[m]
                if knee_row is not None and m in knee_row.index:
                    csv_row[f"{m_label} (knee)"] = knee_row[m]
                else:
                    csv_row[f"{m_label} (knee)"] = np.nan
            csv_rows.append(csv_row)
        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(self.tab_dir, "r1_by_k.csv")
        csv_df.to_csv(csv_path, index=False, float_format="%.6f")

        return path

    def tab_proxy_stability(self, df: pd.DataFrame) -> str:
        r"""Table IV: Proxy stability diagnostics --- R11 (``\label{tab:proxy-stability}``).

        Per the manuscript (Section VIII.K), Table IV has three sections:
          1. MMD RFF dimension sweep (Spearman rho vs reference m=2000)
          2. Sinkhorn anchor count sweep (Spearman rho vs reference A=200)
          3. Cross-representation correlations (VAE vs raw, PCA vs raw, etc.)

        Data is loaded from ``proxy_stability*.csv`` produced by R11.
        If not available, a template matching the manuscript structure is
        emitted so that the table shell is compilable.
        """
        # Look for proxy stability results
        pf = glob.glob(os.path.join(self.runs_root, "**/proxy_stability*.csv"),
                       recursive=True)
        if pf:
            data = pd.read_csv(pf[0])
        else:
            # Template matching manuscript Table IV structure exactly
            data = pd.DataFrame({
                "Section": [
                    "RFF sweep", "RFF sweep", "RFF sweep",
                    "Anchor sweep", "Anchor sweep", "Anchor sweep",
                    "Cross-repr.", "Cross-repr.", "Cross-repr.",
                    "Cross-repr.", "Cross-repr.", "Cross-repr.",
                ],
                "Objective": [
                    "MMD", "MMD", "MMD",
                    "Sinkhorn", "Sinkhorn", "Sinkhorn",
                    "MMD", "Sinkhorn", "MMD",
                    "Sinkhorn", "MMD", "Sinkhorn",
                ],
                "Diagnostic": [
                    "$m = 500$", "$m = 1{,}000$", "$m = 4{,}000$",
                    "$A = 50$", "$A = 100$", "$A = 400$",
                    "VAE vs raw", "VAE vs raw", "VAE vs PCA",
                    "VAE vs PCA", "PCA vs raw", "PCA vs raw",
                ],
                "Spearman_rho": [np.nan] * 12,
            })

        # --- Build sectioned LaTeX ---
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Proxy stability diagnostics (R11): Spearman rank "
            r"correlations of surrogate objectives against reference settings.}",
            r"\label{tab:proxy-stability}",
            r"\begin{tabular}{l l l c}",
            r"\toprule",
            r"Section & Objective & Diagnostic & Spearman $\rho$ \\",
            r"\midrule",
        ]

        # Track sections for midrule separators
        prev_section = None
        for _, row in data.iterrows():
            section = row.get("Section", "")
            if prev_section is not None and section != prev_section:
                lines.append(r"\midrule")
            prev_section = section

            obj = row.get("Objective", "")
            diag = row.get("Diagnostic", "")
            rho_val = row.get("Spearman_rho", np.nan)
            rho = f"{rho_val:.3f}" if np.isfinite(rho_val) else "---"
            lines.append(f"{section} & {obj} & {diag} & {rho}" + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "proxy_stability.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "proxy_stability.csv")
        data.to_csv(csv_path, index=False)
        return path

    def tab_krr_multitask(self, df: pd.DataFrame) -> str:
        r"""Table V: Multi-target KRR test RMSE at k=300 (``\label{tab:krr-multitask-k300}``).

        Per the manuscript (Section VIII.E / Table V), this table reports
        KRR prediction RMSE for **10** coverage targets:

          1. Area (4G)           2. Area (5G)
          3. Households (4G)     4. Residents (4G)
          5. Area (4G+5G)        6. Area (All)
          7. Households (4G+5G)  8. Households (All)
          9. Residents (4G+5G)  10. Residents (All)

        Columns:
          Coverage Target | R1 knee | R9 knee | Best (pool)

        The "R9 knee" column uses the knee-point from R9 (VAE-mean
        representation).  "Best (pool)" scans all candidates across
        **all** runs to find the globally best RMSE per target.

        The code discovers ``krr_rmse_*`` columns and maps them to the
        canonical Table V labels using ``COVERAGE_TARGETS_TABLE_V``.
        """
        d = df[df["k"].fillna(300).astype(int) == 300].copy()

        # ----------------------------------------------------------
        # Resolve the 10 canonical target columns from Table V
        # ----------------------------------------------------------
        try:
            from ..config.constants import COVERAGE_TARGETS_TABLE_V
        except ImportError:
            COVERAGE_TARGETS_TABLE_V = {}

        # Discover available krr_rmse_* columns
        rmse_cols_available = sorted([
            c for c in d.columns
            if c.startswith("krr_rmse") and c != "krr_rmse_mean"
        ])
        if not rmse_cols_available:
            return ""

        # Build ordered list of (internal_col, human_label) matching Table V
        target_order = []
        for canon_key, label in COVERAGE_TARGETS_TABLE_V.items():
            col = f"krr_rmse_{canon_key}"
            if col in d.columns:
                target_order.append((col, label))
        # Also add any discovered columns not yet in the ordered list
        ordered_internal = {t[0] for t in target_order}
        for col in rmse_cols_available:
            if col not in ordered_internal:
                suffix = col.replace("krr_rmse_", "")
                label = COVERAGE_TARGETS_TABLE_V.get(
                    suffix, suffix.replace("_", " ").title()
                )
                target_order.append((col, label))

        if not target_order:
            return ""

        rmse_cols = [t[0] for t in target_order]
        labels = [t[1] for t in target_order]

        # ----------------------------------------------------------
        # Helper: extract knee-point rows for a given run_id pattern
        # ----------------------------------------------------------
        def _get_knee_rmse(run_pattern: str) -> dict:
            sub = d[d["run_id"].astype(str).str.contains(run_pattern)]
            # Prefer explicit knee-point rows
            knee = sub[sub.get("rep_name", pd.Series(dtype=str)).astype(str) == "knee"]
            if knee.empty:
                # Fallback: use all rows and take min per target
                knee = sub
            if knee.empty:
                return {}
            return {c: knee[c].min() for c in rmse_cols if c in knee.columns}

        summary = {}
        summary["R1 knee"] = _get_knee_rmse("R1")
        summary["R9 knee"] = _get_knee_rmse("R9")
        # Best (pool): global best RMSE across ALL runs for each target
        summary["Best (pool)"] = {
            c: d[c].min() for c in rmse_cols if c in d.columns
        }

        col_keys = ["R1 knee", "R9 knee", "Best (pool)"]

        # ----------------------------------------------------------
        # LaTeX
        # ----------------------------------------------------------
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Multi-target KRR test RMSE at $k=300$ "
            r"(10~coverage targets, manuscript Table~V).}",
            r"\label{tab:krr-multitask-k300}",
            r"\begin{tabular}{l " + " ".join(["c"] * len(col_keys)) + "}",
            r"\toprule",
            "Coverage Target & " + " & ".join(col_keys) + r" \\",
            r"\midrule",
        ]

        # Identify best (lowest) per target across columns for bolding
        for col, label in target_order:
            vals_numeric = []
            val_strs = []
            for ck in col_keys:
                v = summary.get(ck, {}).get(col, np.nan)
                vals_numeric.append(v)
            best_idx = int(np.nanargmin(vals_numeric)) if any(np.isfinite(v) for v in vals_numeric) else -1
            for i, v in enumerate(vals_numeric):
                if np.isfinite(v):
                    s = f"{v:.4f}"
                    if i == best_idx:
                        s = r"\textbf{" + s + "}"
                    val_strs.append(s)
                else:
                    val_strs.append("---")
            lines.append(f"{label} & " + " & ".join(val_strs) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "krr_multitask_k300.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        # Companion CSV
        csv_rows = []
        for col, label in target_order:
            row = {"target": label}
            for ck in col_keys:
                row[ck] = summary.get(ck, {}).get(col, np.nan)
            csv_rows.append(row)
        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(self.tab_dir, "krr_multitask_k300.csv")
        csv_df.to_csv(csv_path, index=False, float_format="%.6f")

        return path

    def tab_baseline_summary(self, df: pd.DataFrame) -> str:
        """Baseline method comparison table at k=300."""
        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        if "method" not in d.columns:
            return ""

        metrics = ["nystrom_error", "kpca_distortion", "krr_rmse_4G", "geo_kl"]
        metrics = [m for m in metrics if m in d.columns]
        if not metrics:
            return ""

        summary = d.groupby("method")[metrics].agg(["mean", "std"])
        path = os.path.join(self.tab_dir, "baseline_summary_k300.csv")
        summary.to_csv(path, float_format="%.6f")
        return path

    def tab_constraint_diagnostics(self, df: pd.DataFrame) -> str:
        """Constraint diagnostics summary across configurations."""
        d = df[df["k"].fillna(300).astype(int) == 300].copy()
        geo_cols = [c for c in d.columns if c.startswith("geo_")]
        if not geo_cols or "run_id" not in d.columns:
            return ""

        summary = d.groupby("run_id")[geo_cols].mean()
        path = os.path.join(self.tab_dir, "constraint_diagnostics_k300.csv")
        summary.to_csv(path, float_format="%.6f")
        return path

    # ------------------------------------------------------------------
    # G6: Auto-generated LaTeX tables for tab:exp-settings and tab:run-matrix
    # ------------------------------------------------------------------

    def tab_exp_settings(self) -> str:
        r"""Table I: Experiment settings (``\label{tab:exp-settings}``).

        Auto-generates a LaTeX table from the canonical constants in
        ``config.constants``, reproducing the manuscript Table I layout:

            Component | Setting

        Sources all values from ``config/constants.py``.  Also writes a
        companion CSV for machine consumption.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        try:
            from ..config.constants import (
                N_MUNICIPALITIES, G_STATES, D_FEATURES,
                NSGA2_POP_SIZE, NSGA2_N_GENERATIONS,
                NSGA2_CROSSOVER_PROB, NSGA2_MUTATION_PROB,
                RFF_DIM_DEFAULT, SINKHORN_N_ANCHORS, SINKHORN_ETA,
                SINKHORN_MAX_ITER, ALPHA_GEO,
                VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EPOCHS,
                VAE_BATCH_SIZE, VAE_LR, VAE_KL_WEIGHT,
                VAE_EARLY_STOPPING_PATIENCE,
                EVAL_SIZE, EVAL_TRAIN_FRAC, KPCA_COMPONENTS,
                NYSTROM_LAMBDA, N_REPLICATES_PRIMARY,
                K_GRID,
            )
        except ImportError:
            return ""

        # Rows: (Component, Setting)
        # Grouped by manuscript subsection for readability
        entries = [
            # --- Dataset ---
            ("Dataset",              "Brazil telecom municipalities"),
            (r"$N$ (municipalities)", str(N_MUNICIPALITIES)),
            (r"$G$ (states)",         str(G_STATES)),
            (r"$D$ (covariates)",     str(D_FEATURES)),
            (r"$\mathcal{K}$ (cardinality grid)",
                r"$\{" + ", ".join(str(k) for k in K_GRID) + r"\}$"),
            # --- NSGA-II ---
            (r"NSGA-II pop.\ size $P$",     str(NSGA2_POP_SIZE)),
            (r"NSGA-II generations $T$",     str(NSGA2_N_GENERATIONS)),
            (r"Crossover prob.\ $p_c$",      str(NSGA2_CROSSOVER_PROB)),
            (r"Mutation prob.\ $p_m$",        str(NSGA2_MUTATION_PROB)),
            # --- Objectives ---
            (r"RFF dimension $m$",            str(RFF_DIM_DEFAULT)),
            (r"Sinkhorn anchors $A$",         str(SINKHORN_N_ANCHORS)),
            (r"Sinkhorn $\eta$",              str(SINKHORN_ETA)),
            (r"Sinkhorn iterations",          str(SINKHORN_MAX_ITER)),
            # --- Constraints ---
            (r"Dirichlet smoothing $\alpha$", str(ALPHA_GEO)),
            # --- Representation ---
            (r"VAE latent dim.\ $d_z$",       str(VAE_LATENT_DIM)),
            (r"VAE hidden dim.",              str(VAE_HIDDEN_DIM)),
            (r"VAE epochs",                   str(VAE_EPOCHS)),
            (r"VAE early stopping patience",  str(VAE_EARLY_STOPPING_PATIENCE)),
            (r"VAE batch size",               str(VAE_BATCH_SIZE)),
            (r"VAE learning rate",            str(VAE_LR)),
            (r"VAE KL weight $\beta$",        str(VAE_KL_WEIGHT)),
            # --- Evaluation ---
            (r"$|E|$ (eval.\ set size)",      str(EVAL_SIZE)),
            (r"KRR train fraction",           str(EVAL_TRAIN_FRAC)),
            (r"kPCA components $r$",          str(KPCA_COMPONENTS)),
            (r"Nystr{\"o}m $\lambda$",        f"{NYSTROM_LAMBDA:.1e}"),
            (r"Replicates (R1 $k{=}300$)",   str(N_REPLICATES_PRIMARY)),
        ]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Experimental settings and hyperparameters.}",
            r"\label{tab:exp-settings}",
            r"\begin{tabular}{l c}",
            r"\toprule",
            r"Component & Setting \\",
            r"\midrule",
        ]
        for comp, setting in entries:
            lines.append(f"{comp} & {setting}" + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "exp_settings.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        # Companion CSV
        csv_path = os.path.join(self.tab_dir, "exp_settings.csv")
        with open(csv_path, "w") as f:
            f.write("component,setting\n")
            for comp, setting in entries:
                # Strip LaTeX for CSV
                plain_comp = (comp
                              .replace("$", "")
                              .replace(r"\mathcal{K}", "K")
                              .replace(r"\\", "")
                              .replace(r"\,", ",")
                              .replace(r"\ ", " "))
                plain_setting = (setting
                                 .replace("$", "")
                                 .replace(r"\{", "{")
                                 .replace(r"\}", "}"))
                f.write(f'"{plain_comp}","{plain_setting}"\n')

        return path

    def tab_run_matrix(self) -> str:
        r"""Table II (run matrix): ``\label{tab:run-matrix}``.

        Auto-generates the run-matrix LaTeX table from the run-spec
        registry in ``config.run_specs``, matching the manuscript Table II
        column layout exactly:

            ID | k | Opt. repr | Constraints | Objectives | Seeds | Purpose

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        try:
            from ..config.run_specs import get_run_specs
        except ImportError:
            return ""

        specs = get_run_specs()

        # Constraint mode -> human-readable abbreviation for table
        _cmode_labels = {
            "population_share":         "pop-share",
            "municipality_share_quota": "muni-quota",
            "joint":                    "joint",
            "none":                     "none",
        }

        # Column: ID | k | Opt. repr | Constraints | Objectives | Seeds | Purpose
        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Run matrix: experimental configurations R0--R12.}",
            r"\label{tab:run-matrix}",
            r"\begin{tabular}{l c l l l c l}",
            r"\toprule",
            r"ID & $k$ & Opt.\ repr & Constraints & Objectives & Seeds & Purpose \\",
            r"\midrule",
        ]

        for rid in sorted(specs.keys(), key=lambda r: int(r[1:])):
            s = specs[rid]
            # k column: if sweep, show grid; else show single value
            if s.sweep_k is not None:
                k_str = r"$\mathcal{K}$"
            else:
                k_str = str(s.k)

            space = s.space.upper() if s.space != "raw" else "Raw"
            cmode = _cmode_labels.get(str(s.constraint_mode),
                                      str(s.constraint_mode).replace("_", r"\_"))
            objs = ", ".join(o.upper() if o != "sinkhorn" else "Sink"
                            for o in s.objectives) if s.objectives else "---"
            seeds = str(s.n_reps)
            # Truncate description for table
            desc = s.description
            if len(desc) > 52:
                desc = desc[:49] + r"\ldots"
            lines.append(
                f"{rid} & {k_str} & {space} & {cmode} & {objs} "
                f"& {seeds} & {desc}" + r" \\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

        path = os.path.join(self.tab_dir, "run_matrix.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        # Companion CSV
        csv_path = os.path.join(self.tab_dir, "run_matrix.csv")
        with open(csv_path, "w") as f:
            f.write("run_id,k,opt_repr,constraints,objectives,seeds,purpose\n")
            for rid in sorted(specs.keys(), key=lambda r: int(r[1:])):
                s = specs[rid]
                space = s.space.upper() if s.space != "raw" else "Raw"
                cmode = _cmode_labels.get(str(s.constraint_mode),
                                          str(s.constraint_mode))
                objs = ";".join(s.objectives) if s.objectives else ""
                k_str = (";".join(str(x) for x in s.sweep_k)
                         if s.sweep_k else str(s.k))
                f.write(f'{rid},"{k_str}",{space},{cmode},"{objs}",'
                        f'{s.n_reps},"{s.description}"\n')

        return path

    # ==================================================================
    # PHASE 11 -- New/Enhanced Tables for Strengthened Narrative (N1-N7)
    # ==================================================================

    # ------------------------------------------------------------------
    # Shared helpers for Phase 11 tables
    # ------------------------------------------------------------------

    def _resolve_metric_cols(self, df: pd.DataFrame) -> dict:
        """Resolve canonical metric column names with fallback aliases.

        Returns a dict mapping canonical names to the actual column names
        present in *df*.
        """
        canonical = {
            "nystrom_error":           ["nystrom_error", "e_nys", "e_Nys"],
            "kpca_distortion":         ["kpca_distortion", "e_kpca", "e_kPCA"],
            "krr_rmse_cov_area_4G":    ["krr_rmse_cov_area_4G", "krr_rmse_4G",
                                        "krr_rmse_area_4G", "RMSE_4G"],
            "krr_rmse_cov_area_5G":    ["krr_rmse_cov_area_5G", "krr_rmse_5G",
                                        "krr_rmse_area_5G", "RMSE_5G"],
            "geo_kl":                  ["geo_kl"],
            "geo_l1":                  ["geo_l1"],
            "geo_maxdev":              ["geo_maxdev", "geo_max_dev"],
            "pop_geo_kl":              ["pop_geo_kl"],
            "pop_geo_l1":              ["pop_geo_l1"],
            "pop_geo_maxdev":          ["pop_geo_maxdev", "pop_geo_max_dev"],
            "skl_drift":              ["skl_drift", "skl_obj", "f_skl"],
        }
        resolved = {}
        for canon, aliases in canonical.items():
            for alias in aliases:
                if alias in df.columns:
                    resolved[canon] = alias
                    break
        return resolved

    @staticmethod
    def _fmt(v: float, prec: int = 4) -> str:
        """Format a numeric value; return ``---`` for non-finite."""
        if np.isfinite(v):
            return f"{v:.{prec}f}"
        return "---"

    @staticmethod
    def _fmt_bold_best(vals: list, prec: int = 4, direction: str = "min") -> list:
        """Format a list of values; bold the best one.

        Parameters
        ----------
        vals : list of float
            Numeric values (may contain ``nan``).
        prec : int
            Decimal precision.
        direction : str
            ``"min"`` (lower is better) or ``"max"`` (higher is better).

        Returns
        -------
        list of str
            Formatted strings with the best entry wrapped in ``\\textbf``.
        """
        formatted = []
        finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        if direction == "min":
            best_idx = min(finite, key=lambda t: t[1])[0] if finite else -1
        else:
            best_idx = max(finite, key=lambda t: t[1])[0] if finite else -1
        for i, v in enumerate(vals):
            s = f"{v:.{prec}f}" if np.isfinite(v) else "---"
            if i == best_idx and np.isfinite(v):
                s = r"\textbf{" + s + "}"
            formatted.append(s)
        return formatted

    def _filter_k300(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows where k == 300."""
        if df.empty:
            return df
        d = df.copy()
        d["k"] = d["k"].fillna(300).astype(int)
        return d[d["k"] == 300]

    def _run_mean(self, df: pd.DataFrame, run_pattern: str, col: str) -> float:
        """Mean of *col* for rows whose run_id matches *run_pattern*."""
        sub = df[df["run_id"].astype(str).str.contains(run_pattern)]
        if sub.empty or col not in sub.columns:
            return np.nan
        return sub[col].mean()

    # ------------------------------------------------------------------
    # Table N1: constraint_diagnostics_cross_config.tex
    # ------------------------------------------------------------------

    def tab_constraint_diagnostics_cross_config(self, df: pd.DataFrame) -> str:
        r"""Table N1: Cross-configuration constraint diagnostics at k=300.

        Compares proportionality diagnostics across multiple constraint
        configurations and representative baselines (Section VIII.F).

        Rows: R1 (pop-share), R4 (muni-quota), R5 (joint), R6 (none),
        plus any identifiable baselines.

        Columns (manuscript Phase 11 spec):
          Run ID | Constraint Mode | geo_kl (muni) | geo_l1 (muni) |
          geo_maxdev (muni) | pop_geo_kl | pop_geo_l1 | pop_geo_maxdev

        Both ``.tex`` and companion ``.csv`` files are emitted.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty or "run_id" not in d.columns:
            return ""

        rcols = self._resolve_metric_cols(d)

        # Identify columns to report
        geo_cols_canon = [
            "geo_kl", "geo_l1", "geo_maxdev",
            "pop_geo_kl", "pop_geo_l1", "pop_geo_maxdev",
        ]
        geo_cols = [(c, rcols.get(c, c)) for c in geo_cols_canon]

        # Constraint mode labels for known run IDs
        _cmode_map = {
            "R1": "pop-share", "R4": "muni-quota", "R5": "joint",
            "R6": "none",
        }

        # Collect rows: R1, R4, R5, R6, then any baselines
        run_ids_ordered = []
        for rp in ["R1", "R4", "R5", "R6"]:
            matches = d[d["run_id"].astype(str).str.contains(rp)]["run_id"].unique()
            for m in sorted(matches):
                if m not in run_ids_ordered:
                    run_ids_ordered.append(m)
        # Add any baseline rows
        baseline_mask = d["run_id"].astype(str).str.contains(
            "R10|baseline|Baseline", na=False)
        for rid in sorted(d.loc[baseline_mask, "run_id"].unique()):
            if rid not in run_ids_ordered:
                run_ids_ordered.append(rid)

        if not run_ids_ordered:
            return ""

        # LaTeX column headers
        col_headers = [
            "Run", "Constraint",
            r"$D_\mathrm{KL}^{(\mathrm{m})}$",
            r"$\ell_1^{(\mathrm{m})}$",
            r"$\Delta_\mathrm{max}^{(\mathrm{m})}$",
            r"$D_\mathrm{KL}^{(\mathrm{p})}$",
            r"$\ell_1^{(\mathrm{p})}$",
            r"$\Delta_\mathrm{max}^{(\mathrm{p})}$",
        ]

        body_lines = []
        csv_rows = []
        for rid in run_ids_ordered:
            sub = d[d["run_id"] == rid]
            # Determine constraint mode label
            short = rid.split("_")[0] if "_" in str(rid) else str(rid)
            cmode = _cmode_map.get(short, "---")
            vals = []
            csv_row = {"run_id": rid, "constraint_mode": cmode}
            for canon, actual in geo_cols:
                v = sub[actual].mean() if actual in sub.columns else np.nan
                vals.append(v)
                csv_row[canon] = v
            csv_rows.append(csv_row)
            val_strs = [self._fmt(v) for v in vals]
            body_lines.append(
                f"{rid} & {cmode} & " + " & ".join(val_strs) + r" \\"
            )

        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Cross-configuration proportionality diagnostics "
            r"at $k=300$. Superscripts (m) and (p) denote municipality-share "
            r"and population-share metrics respectively.}",
            r"\label{tab:constraint-diag-crossconfig}",
            r"\begin{tabular}{l l " + " ".join(["c"] * 6) + "}",
            r"\toprule",
            " & ".join(col_headers) + r" \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

        path = os.path.join(self.tab_dir,
                            "constraint_diagnostics_cross_config.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir,
                                "constraint_diagnostics_cross_config.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N2: objective_ablation_summary.tex
    # ------------------------------------------------------------------

    def tab_objective_ablation_summary(self, df: pd.DataFrame) -> str:
        r"""Table N2: Objective ablation --- R1 (bi-obj) vs R2 (MMD-only) vs R3 (SD-only).

        Quantifies the benefit of bi-objective optimisation at k = 300
        (Section VIII.C).

        Columns: Method | :math:`e_\mathrm{Nys}` | :math:`e_\mathrm{kPCA}` |
        RMSE(4G) | RMSE(5G) | :math:`D_\mathrm{KL}^{(\mathrm{geo})}`.

        Best (lowest) values per column are bolded.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty:
            return ""

        rcols = self._resolve_metric_cols(d)
        metric_canon = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_cov_area_4G", "krr_rmse_cov_area_5G", "geo_kl",
        ]
        metrics = [(c, rcols[c]) for c in metric_canon if c in rcols]
        if not metrics:
            return ""

        # Run configurations
        configs = [
            ("R1", "R1 (MMD + Sink)"),
            ("R2", "R2 (MMD only)"),
            ("R3", "R3 (Sink only)"),
        ]

        # Collect values
        rows_data = []  # list of (label, [vals...])
        for pattern, label in configs:
            vals = []
            for canon, actual in metrics:
                vals.append(self._run_mean(d, pattern, actual))
            rows_data.append((label, vals))

        # Bold best per column
        n_metrics = len(metrics)
        col_formatted = []
        for j in range(n_metrics):
            col_vals = [r[1][j] for r in rows_data]
            col_formatted.append(self._fmt_bold_best(col_vals))

        # LaTeX headers
        latex_headers = {
            "nystrom_error":        r"$e_{\mathrm{Nys}}$",
            "kpca_distortion":      r"$e_{\mathrm{kPCA}}$",
            "krr_rmse_cov_area_4G": r"RMSE$_{4\mathrm{G}}$",
            "krr_rmse_cov_area_5G": r"RMSE$_{5\mathrm{G}}$",
            "geo_kl":               r"$D_\mathrm{KL}^\mathrm{geo}$",
        }
        hdrs = ["Method"] + [latex_headers.get(c, c) for c, _ in metrics]

        body_lines = []
        csv_rows = []
        for i, (label, vals) in enumerate(rows_data):
            val_strs = [col_formatted[j][i] for j in range(n_metrics)]
            body_lines.append(f"{label} & " + " & ".join(val_strs) + r" \\")
            csv_row = {"method": label}
            for j, (canon, _) in enumerate(metrics):
                csv_row[canon] = vals[j]
            csv_rows.append(csv_row)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Objective ablation at $k=300$: bi-objective (R1) "
            r"vs single-objective (R2, R3). Best per column in bold.}",
            r"\label{tab:objective-ablation-summary}",
            r"\begin{tabular}{l " + " ".join(["c"] * n_metrics) + "}",
            r"\toprule",
            " & ".join(hdrs) + r" \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "objective_ablation_summary.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "objective_ablation_summary.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N3: representation_transfer_summary.tex
    # ------------------------------------------------------------------

    def tab_representation_transfer_summary(self, df: pd.DataFrame) -> str:
        r"""Table N3: Representation transfer --- R1 (raw) vs R8 (PCA) vs R9 (VAE).

        Quantifies the representation-mismatch effect at k = 300 when
        evaluated in raw space (Section VIII.G/H).

        Columns: Opt.\ Space | :math:`e_\mathrm{Nys}` |
        :math:`e_\mathrm{kPCA}` | RMSE(4G) | RMSE(5G).

        Best (lowest) values per column are bolded.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty:
            return ""

        rcols = self._resolve_metric_cols(d)
        metric_canon = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_cov_area_4G", "krr_rmse_cov_area_5G",
        ]
        metrics = [(c, rcols[c]) for c in metric_canon if c in rcols]
        if not metrics:
            return ""

        configs = [
            ("R1", "Raw (R1)"),
            ("R8", "PCA (R8)"),
            ("R9", "VAE (R9)"),
        ]

        rows_data = []
        for pattern, label in configs:
            vals = [self._run_mean(d, pattern, actual) for _, actual in metrics]
            rows_data.append((label, vals))

        n_metrics = len(metrics)
        col_formatted = []
        for j in range(n_metrics):
            col_vals = [r[1][j] for r in rows_data]
            col_formatted.append(self._fmt_bold_best(col_vals))

        latex_headers = {
            "nystrom_error":        r"$e_{\mathrm{Nys}}$",
            "kpca_distortion":      r"$e_{\mathrm{kPCA}}$",
            "krr_rmse_cov_area_4G": r"RMSE$_{4\mathrm{G}}$",
            "krr_rmse_cov_area_5G": r"RMSE$_{5\mathrm{G}}$",
        }
        hdrs = [r"Opt.\ Space"] + [latex_headers.get(c, c) for c, _ in metrics]

        body_lines = []
        csv_rows = []
        for i, (label, vals) in enumerate(rows_data):
            val_strs = [col_formatted[j][i] for j in range(n_metrics)]
            body_lines.append(f"{label} & " + " & ".join(val_strs) + r" \\")
            csv_row = {"opt_space": label}
            for j, (canon, _) in enumerate(metrics):
                csv_row[canon] = vals[j]
            csv_rows.append(csv_row)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Representation transfer at $k=300$: raw-space "
            r"evaluation metrics for optimisation in raw, PCA, and VAE "
            r"spaces. Best per column in bold.}",
            r"\label{tab:repr-transfer-summary}",
            r"\begin{tabular}{l " + " ".join(["c"] * n_metrics) + "}",
            r"\toprule",
            " & ".join(hdrs) + r" \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "representation_transfer_summary.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir,
                                "representation_transfer_summary.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N4: skl_ablation_summary.tex
    # ------------------------------------------------------------------

    def tab_skl_ablation_summary(self, df: pd.DataFrame) -> str:
        r"""Table N4: SKL ablation --- R1 (bi-obj, raw) vs R7 (tri-obj, VAE).

        Shows whether adding the SKL objective provides non-redundant
        signal (Section VIII.I).

        Columns: Objectives | :math:`e_\mathrm{Nys}` |
        :math:`e_\mathrm{kPCA}` | RMSE(4G) | RMSE(5G) | SKL drift.

        Best (lowest) per column are bolded.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty:
            return ""

        rcols = self._resolve_metric_cols(d)
        metric_canon = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_cov_area_4G", "krr_rmse_cov_area_5G",
            "skl_drift",
        ]
        metrics = [(c, rcols[c]) for c in metric_canon if c in rcols]
        # Must have at least basic metrics
        if len(metrics) < 2:
            return ""

        configs = [
            ("R1", "R1: MMD + Sink (raw)"),
            ("R7", "R7: MMD + Sink + SKL (VAE)"),
        ]

        rows_data = []
        for pattern, label in configs:
            vals = [self._run_mean(d, pattern, actual) for _, actual in metrics]
            rows_data.append((label, vals))

        n_metrics = len(metrics)
        col_formatted = []
        for j in range(n_metrics):
            col_vals = [r[1][j] for r in rows_data]
            col_formatted.append(self._fmt_bold_best(col_vals))

        latex_headers = {
            "nystrom_error":        r"$e_{\mathrm{Nys}}$",
            "kpca_distortion":      r"$e_{\mathrm{kPCA}}$",
            "krr_rmse_cov_area_4G": r"RMSE$_{4\mathrm{G}}$",
            "krr_rmse_cov_area_5G": r"RMSE$_{5\mathrm{G}}$",
            "skl_drift":            r"SKL drift",
        }
        hdrs = ["Objectives"] + [latex_headers.get(c, c) for c, _ in metrics]

        body_lines = []
        csv_rows = []
        for i, (label, vals) in enumerate(rows_data):
            val_strs = [col_formatted[j][i] for j in range(n_metrics)]
            body_lines.append(f"{label} & " + " & ".join(val_strs) + r" \\")
            csv_row = {"objectives": label}
            for j, (canon, _) in enumerate(metrics):
                csv_row[canon] = vals[j]
            csv_rows.append(csv_row)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{SKL ablation at $k=300$: bi-objective (R1, raw) "
            r"vs tri-objective with SKL (R7, VAE). Best per column in bold.}",
            r"\label{tab:skl-ablation-summary}",
            r"\begin{tabular}{l " + " ".join(["c"] * n_metrics) + "}",
            r"\toprule",
            " & ".join(hdrs) + r" \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "skl_ablation_summary.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "skl_ablation_summary.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N5: multi_seed_statistics.tex
    # ------------------------------------------------------------------

    def tab_multi_seed_statistics(self, df: pd.DataFrame) -> str:
        r"""Table N5: Multi-seed descriptive statistics for R1 and R5.

        Reports mean, std, min, max of key metrics across the 5 seeds
        at k = 300 to quantify robustness (Section VIII.D).

        Columns: Run | Metric | Mean | Std | Min | Max.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty:
            return ""

        rcols = self._resolve_metric_cols(d)
        metric_canon = [
            "nystrom_error", "kpca_distortion",
            "krr_rmse_cov_area_4G", "krr_rmse_cov_area_5G", "geo_kl",
        ]
        metrics = [(c, rcols[c]) for c in metric_canon if c in rcols]
        if not metrics:
            return ""

        latex_names = {
            "nystrom_error":        r"$e_{\mathrm{Nys}}$",
            "kpca_distortion":      r"$e_{\mathrm{kPCA}}$",
            "krr_rmse_cov_area_4G": r"RMSE$_{4\mathrm{G}}$",
            "krr_rmse_cov_area_5G": r"RMSE$_{5\mathrm{G}}$",
            "geo_kl":               r"$D_\mathrm{KL}^\mathrm{geo}$",
        }

        body_lines = []
        csv_rows = []
        first_run = True
        for run_pattern, run_label in [("R1", "R1"), ("R5", "R5")]:
            sub = d[d["run_id"].astype(str).str.contains(run_pattern)]
            if sub.empty:
                continue
            if not first_run:
                body_lines.append(r"\midrule")
            first_run = False
            for j, (canon, actual) in enumerate(metrics):
                if actual not in sub.columns:
                    continue
                vals = sub[actual].dropna()
                if vals.empty:
                    continue
                m, s, mn, mx = vals.mean(), vals.std(), vals.min(), vals.max()
                run_cell = run_label if j == 0 else ""
                metric_label = latex_names.get(canon, canon)
                body_lines.append(
                    f"{run_cell} & {metric_label} & "
                    f"{self._fmt(m)} & {self._fmt(s)} & "
                    f"{self._fmt(mn)} & {self._fmt(mx)}" + r" \\"
                )
                csv_rows.append({
                    "run": run_label, "metric": canon,
                    "mean": m, "std": s, "min": mn, "max": mx,
                })

        if not body_lines:
            return ""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Multi-seed statistics at $k=300$: mean, standard "
            r"deviation, min, and max across 5~seeds for R1 and R5.}",
            r"\label{tab:multi-seed-stats}",
            r"\begin{tabular}{l l c c c c}",
            r"\toprule",
            r"Run & Metric & Mean & Std & Min & Max \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "multi_seed_statistics.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "multi_seed_statistics.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N6: worst_state_rmse_by_k.tex
    # ------------------------------------------------------------------

    def tab_worst_state_rmse_by_k(self, df: pd.DataFrame) -> str:
        r"""Table N6: Worst-state KRR RMSE and state dispersion by k (R1).

        For each k in the cardinality grid, reports:
        * Avg RMSE (4G/5G) --- overall average across states
        * Worst-state RMSE (4G/5G) --- maximum per-state RMSE
        * State RMSE dispersion (std across states)

        Supports equity analysis for Section VIII.D.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            return ""

        d["k"] = d["k"].fillna(300).astype(int)

        # Discover worst-state and dispersion columns
        ws_4g = next((c for c in d.columns
                      if "worst" in c.lower() and "rmse" in c.lower()
                      and "4G" in c), None)
        ws_5g = next((c for c in d.columns
                      if "worst" in c.lower() and "rmse" in c.lower()
                      and "5G" in c), None)
        disp_4g = next((c for c in d.columns
                        if "disp" in c.lower() and "rmse" in c.lower()
                        and "4G" in c), None)
        disp_5g = next((c for c in d.columns
                        if "disp" in c.lower() and "rmse" in c.lower()
                        and "5G" in c), None)

        # Fallback: try to use avg KRR RMSE columns
        rcols = self._resolve_metric_cols(d)
        avg_4g = rcols.get("krr_rmse_cov_area_4G")
        avg_5g = rcols.get("krr_rmse_cov_area_5G")

        # Must have at least average RMSE columns
        if avg_4g is None and ws_4g is None:
            return ""

        # Aggregate per k
        k_vals = sorted(d["k"].unique())

        body_lines = []
        csv_rows = []
        for k in k_vals:
            dk = d[d["k"] == k]

            def _agg(col):
                if col and col in dk.columns:
                    v = dk[col].dropna()
                    return v.mean() if len(v) > 0 else np.nan
                return np.nan

            vals = [
                _agg(avg_4g), _agg(ws_4g), _agg(disp_4g),
                _agg(avg_5g), _agg(ws_5g), _agg(disp_5g),
            ]
            val_strs = [self._fmt(v) for v in vals]
            body_lines.append(f"{k} & " + " & ".join(val_strs) + r" \\")
            csv_rows.append({
                "k": k,
                "avg_rmse_4G": vals[0], "worst_rmse_4G": vals[1],
                "disp_rmse_4G": vals[2],
                "avg_rmse_5G": vals[3], "worst_rmse_5G": vals[4],
                "disp_rmse_5G": vals[5],
            })

        if not body_lines:
            return ""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{R1 state-level KRR RMSE equity analysis across "
            r"coreset sizes $k$. ``Worst'' = maximum per-state RMSE; "
            r"``Disp.'' = standard deviation across states.}",
            r"\label{tab:worst-state-rmse-by-k}",
            r"\begin{tabular}{c cc c cc c}",
            r"\toprule",
            r" & \multicolumn{3}{c}{4G Target} "
            r"& \multicolumn{3}{c}{5G Target} \\",
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            r"$k$ & Avg & Worst & Disp. & Avg & Worst & Disp. \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "worst_state_rmse_by_k.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "worst_state_rmse_by_k.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # ------------------------------------------------------------------
    # Table N7: baseline_paired_unconstrained_vs_quota.tex
    # ------------------------------------------------------------------

    def tab_baseline_paired_unconstrained_vs_quota(
        self, df: pd.DataFrame,
    ) -> str:
        r"""Table N7: Baseline paired comparison --- unconstrained vs quota-matched.

        For each of the 7 baseline methods, shows evaluation metrics for
        both the unconstrained and the quota-matched variant at k = 300.
        This reveals the cost of imposing proportionality on baselines
        (Section VIII.E, R10).

        Columns: Method | Variant | :math:`e_\mathrm{Nys}` |
        RMSE(4G) | :math:`D_\mathrm{KL}^\mathrm{geo}`.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty or "method" not in d.columns:
            return ""

        rcols = self._resolve_metric_cols(d)
        metric_canon = ["nystrom_error", "krr_rmse_cov_area_4G", "geo_kl"]
        metrics = [(c, rcols[c]) for c in metric_canon if c in rcols]
        if not metrics:
            return ""

        latex_headers = {
            "nystrom_error":        r"$e_{\mathrm{Nys}}$",
            "krr_rmse_cov_area_4G": r"RMSE$_{4\mathrm{G}}$",
            "geo_kl":               r"$D_\mathrm{KL}^\mathrm{geo}$",
        }

        # Identify baselines and their variants
        baseline_methods = [
            "uniform", "kmeans", "herding", "farthest_first",
            "kernel_thinning", "leverage", "dpp",
        ]

        body_lines = []
        csv_rows = []
        first = True
        for bm in baseline_methods:
            # Find unconstrained and quota-matched variants
            variants = []
            for variant_label, variant_pattern in [
                ("Unconstrained", f"{bm}"),
                ("Quota-matched", f"{bm}_quota"),
            ]:
                # Try to match rows by method name containing the pattern
                mask = d["method"].astype(str).str.lower().str.contains(
                    variant_pattern.replace("_", ".*"), regex=True, na=False,
                )
                sub = d[mask]
                if sub.empty:
                    # Fallback: look for variant column
                    if "variant" in d.columns:
                        is_quota = "quota" in variant_label.lower()
                        base_mask = d["method"].astype(str).str.lower().str.contains(
                            bm, na=False)
                        if is_quota:
                            sub = d[base_mask
                                    & d["variant"].astype(str).str.contains(
                                        "quota", case=False, na=False)]
                        else:
                            sub = d[base_mask
                                    & ~d["variant"].astype(str).str.contains(
                                        "quota", case=False, na=False)]
                if not sub.empty:
                    vals = [
                        sub[actual].mean() if actual in sub.columns else np.nan
                        for _, actual in metrics
                    ]
                    variants.append((variant_label, vals))

            if not variants:
                continue

            if not first:
                body_lines.append(r"\midrule")
            first = False

            for vi, (vlabel, vals) in enumerate(variants):
                method_cell = bm.replace("_", " ").title() if vi == 0 else ""
                val_strs = [self._fmt(v) for v in vals]
                body_lines.append(
                    f"{method_cell} & {vlabel} & "
                    + " & ".join(val_strs) + r" \\"
                )
                csv_rows.append({
                    "method": bm, "variant": vlabel,
                    **{canon: v for (canon, _), v in zip(metrics, vals)},
                })

        if not body_lines:
            return ""

        n_metrics = len(metrics)
        hdrs = (["Method", "Variant"]
                + [latex_headers.get(c, c) for c, _ in metrics])

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Baseline methods: unconstrained vs quota-matched "
            r"variants at $k=300$. Shows the cost of imposing "
            r"proportionality on baselines.}",
            r"\label{tab:baseline-paired-unconstrained-vs-quota}",
            r"\begin{tabular}{l l " + " ".join(["c"] * n_metrics) + "}",
            r"\toprule",
            " & ".join(hdrs) + r" \\",
            r"\midrule",
        ]
        lines.extend(body_lines)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir,
                            "baseline_paired_unconstrained_vs_quota.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir,
                                "baseline_paired_unconstrained_vs_quota.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.6f")

        return path

    # -- Downstream model comparison table --------------------------------

    def tab_downstream_model_comparison(self, df: pd.DataFrame) -> str:
        r"""Table: downstream model comparison across regression and classification targets.

        Compares KRR, KNN, RF, LR, and GBT downstream models on Nystrom features.
        Reports mean metric values across all R1 results at k=300.

        Regression sub-table: rows = models, columns = RMSE per target.
        Classification sub-table: rows = models, columns = accuracy per target.
        Emits both ``.tex`` and ``.csv``.
        """
        d = df[df["run_id"].astype(str).str.contains("R1")].copy()
        if d.empty:
            return ""
        d["k"] = d["k"].astype(int)
        d = d[d["k"] == 300]
        if d.empty:
            # Fall back to any k
            d = df[df["run_id"].astype(str).str.contains("R1")].copy()
            d["k"] = d["k"].astype(int)

        # -- Discover regression metric columns --
        # Pattern: {model}_rmse_{target}
        reg_models = {"krr": "KRR", "knn": "KNN", "rf": "RF", "gbt": "GBT"}
        cls_models = {"knn": "KNN", "rf": "RF", "lr": "LR", "gbt": "GBT"}

        import re as _re

        # Find all regression targets that have RMSE columns
        reg_targets = set()
        for col in d.columns:
            m = _re.match(r"^(krr|knn|rf|gbt)_rmse_(.+)$", col)
            if m:
                reg_targets.add(m.group(2))
        reg_targets = sorted(reg_targets)

        # Find all classification targets that have accuracy columns
        cls_targets = set()
        for col in d.columns:
            m = _re.match(r"^(knn|rf|lr|gbt)_accuracy_(.+)$", col)
            if m:
                cls_targets.add(m.group(2))
        cls_targets = sorted(cls_targets)

        if not reg_targets and not cls_targets:
            return ""

        paths = []

        # -- Regression sub-table --
        if reg_targets:
            reg_rows = []
            for model_key, model_label in reg_models.items():
                row = {"Model": model_label}
                for tgt in reg_targets:
                    col_name = f"{model_key}_rmse_{tgt}"
                    if col_name in d.columns and d[col_name].notna().any():
                        row[tgt] = float(d[col_name].mean())
                    else:
                        row[tgt] = np.nan
                reg_rows.append(row)

            reg_df = pd.DataFrame(reg_rows).set_index("Model")
            # Drop targets where all models have NaN
            reg_df = reg_df.dropna(axis=1, how="all")

            if not reg_df.empty and reg_df.shape[1] > 0:
                # Shorten target names for display
                short = {c: c.replace("cov_area_", "").replace("cov_hh_", "hh_")
                         .replace("cov_res_", "res_") for c in reg_df.columns}
                reg_display = reg_df.rename(columns=short)

                # LaTeX
                ncols = len(reg_display.columns)
                lines = [
                    r"\begin{table}[htbp]",
                    r"\centering",
                    r"\small",
                    r"\caption{Downstream regression model comparison (RMSE, R1 $k{=}300$).}",
                    r"\label{tab:downstream-reg-comparison}",
                    r"\begin{tabular}{l " + " ".join(["c"] * ncols) + "}",
                    r"\toprule",
                    "Model & " + " & ".join(
                        [c.replace("_", r"\_") for c in reg_display.columns]) + r" \\",
                    r"\midrule",
                ]
                for model_name, row in reg_display.iterrows():
                    vals = []
                    for v in row:
                        if np.isfinite(v):
                            vals.append(f"{v:.4f}")
                        else:
                            vals.append("--")
                    lines.append(f"{model_name} & " + " & ".join(vals) + r" \\")
                lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

                tex_path = os.path.join(self.tab_dir,
                                        "downstream_regression_comparison.tex")
                with open(tex_path, "w") as f:
                    f.write("\n".join(lines))

                csv_path = os.path.join(self.tab_dir,
                                        "downstream_regression_comparison.csv")
                reg_df.to_csv(csv_path, float_format="%.6f")
                paths.append(tex_path)

        # -- Classification sub-table --
        if cls_targets:
            cls_rows = []
            for model_key, model_label in cls_models.items():
                row = {"Model": model_label}
                for tgt in cls_targets:
                    acc_col = f"{model_key}_accuracy_{tgt}"
                    bal_col = f"{model_key}_bal_accuracy_{tgt}"
                    f1_col = f"{model_key}_macro_f1_{tgt}"
                    # Use balanced accuracy as the primary metric
                    if bal_col in d.columns and d[bal_col].notna().any():
                        row[tgt] = float(d[bal_col].mean())
                    elif acc_col in d.columns and d[acc_col].notna().any():
                        row[tgt] = float(d[acc_col].mean())
                    else:
                        row[tgt] = np.nan
                cls_rows.append(row)

            cls_df = pd.DataFrame(cls_rows).set_index("Model")
            cls_df = cls_df.dropna(axis=1, how="all")

            if not cls_df.empty and cls_df.shape[1] > 0:
                ncols = len(cls_df.columns)
                lines = [
                    r"\begin{table}[htbp]",
                    r"\centering",
                    r"\small",
                    r"\caption{Downstream classification model comparison "
                    r"(balanced accuracy, R1 $k{=}300$).}",
                    r"\label{tab:downstream-cls-comparison}",
                    r"\begin{tabular}{l " + " ".join(["c"] * ncols) + "}",
                    r"\toprule",
                    "Model & " + " & ".join(
                        [c.replace("_", r"\_") for c in cls_df.columns]) + r" \\",
                    r"\midrule",
                ]
                for model_name, row in cls_df.iterrows():
                    vals = []
                    for v in row:
                        if np.isfinite(v):
                            vals.append(f"{v:.4f}")
                        else:
                            vals.append("--")
                    lines.append(f"{model_name} & " + " & ".join(vals) + r" \\")
                lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

                tex_path = os.path.join(self.tab_dir,
                                        "downstream_classification_comparison.tex")
                with open(tex_path, "w") as f:
                    f.write("\n".join(lines))

                csv_path = os.path.join(self.tab_dir,
                                        "downstream_classification_comparison.csv")
                cls_df.to_csv(csv_path, float_format="%.6f")
                paths.append(tex_path)

        return paths if len(paths) > 1 else (paths[0] if paths else "")

    # -- Dimensionality sweep table ---------------------------------------

    def tab_dimensionality_sweep(self, df: pd.DataFrame) -> str:
        r"""Table N9: Dimensionality sweep --- VAE (R13) and PCA (R14).

        Reports key evaluation metrics as a function of representation
        dimension D for both the VAE latent space (R13) and PCA component
        space (R14).  Each sub-table has rows indexed by D and columns
        for core metrics: :math:`e_{\mathrm{Nys}}`, RMSE 4G, RMSE 5G,
        :math:`f_{\mathrm{MMD}}`, and :math:`f_{\mathrm{SD}}`.

        Emits both ``.tex`` and ``.csv``.
        """
        import re as _re

        paths = []

        for run_prefix, space_label in [("R13", "VAE"), ("R14", "PCA")]:
            d_sub = df[df["run_id"].astype(str).str.startswith(run_prefix)].copy()
            if d_sub.empty:
                continue

            # Extract D from run_id (e.g. "R13_d16" -> 16)
            d_sub["dim"] = d_sub["run_id"].astype(str).str.extract(
                r"_d(\d+)", expand=False
            ).astype(float)
            d_sub = d_sub.dropna(subset=["dim"])
            if d_sub.empty:
                continue
            d_sub["dim"] = d_sub["dim"].astype(int)

            # Metric columns to report
            metric_map = {
                "nystrom_error":        r"$e_{\mathrm{Nys}}$",
                "kpca_distortion":      r"$e_{\mathrm{kPCA}}$",
                "krr_rmse_cov_area_4G": r"RMSE$_{\mathrm{4G}}$",
                "krr_rmse_cov_area_5G": r"RMSE$_{\mathrm{5G}}$",
            }

            # Resolve legacy aliases
            _legacy = {
                "krr_rmse_cov_area_4G": ["krr_rmse_4G", "krr_rmse_area_4G"],
                "krr_rmse_cov_area_5G": ["krr_rmse_5G", "krr_rmse_area_5G"],
            }
            for canonical in list(metric_map.keys()):
                if canonical not in d_sub.columns:
                    for alias in _legacy.get(canonical, []):
                        if alias in d_sub.columns:
                            d_sub[canonical] = d_sub[alias]
                            break

            # Also try to include objective values
            for obj_col, obj_label in [
                ("f_mmd", r"$f_{\mathrm{MMD}}$"),
                ("f_sinkhorn", r"$f_{\mathrm{SD}}$"),
            ]:
                if obj_col in d_sub.columns and d_sub[obj_col].notna().any():
                    metric_map[obj_col] = obj_label

            # Keep only available metrics
            available = [m for m in metric_map if m in d_sub.columns and d_sub[m].notna().any()]
            if not available:
                continue

            # Aggregate by dim (mean across replicates / representatives)
            env = d_sub.groupby("dim")[available].mean().reset_index()
            env = env.sort_values("dim")

            headers = [metric_map[m] for m in available]

            # --- LaTeX ---
            ncols = len(available)
            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\small",
                rf"\caption{{{space_label} dimensionality sweep ({run_prefix}, $k{{=}}300$).}}",
                rf"\label{{tab:dim-sweep-{space_label.lower()}}}",
                r"\begin{tabular}{c " + " ".join(["c"] * ncols) + "}",
                r"\toprule",
                "$D$ & " + " & ".join(headers) + r" \\",
                r"\midrule",
            ]
            for _, row in env.iterrows():
                vals = " & ".join([f"{row[m]:.4f}" for m in available])
                lines.append(f"{int(row['dim'])} & {vals}" + r" \\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

            tex_name = f"dimensionality_sweep_{space_label.lower()}.tex"
            tex_path = os.path.join(self.tab_dir, tex_name)
            with open(tex_path, "w") as f:
                f.write("\n".join(lines))

            csv_name = f"dimensionality_sweep_{space_label.lower()}.csv"
            csv_path = os.path.join(self.tab_dir, csv_name)
            env.to_csv(csv_path, index=False, float_format="%.6f")
            paths.append(tex_path)

        return paths if len(paths) > 1 else (paths[0] if paths else "")

    # ------------------------------------------------------------------
    # Table: repr_timing.tex  (tab:repr-timing)
    # ------------------------------------------------------------------

    def tab_repr_timing(self, df: pd.DataFrame) -> str:
        r"""Representation transfer timing table (``\label{tab:repr-timing}``).

        Per the manuscript (Section VIII.G), this table reports the
        computational cost of optimising in different representation
        spaces: raw, PCA, and VAE (mean).

        Columns:
          Opt. space | dim. $p$ | solver time (s) | total time (s)

        Data is extracted from the ``wall_clock_*_s`` columns in the
        results CSV for R1 (raw), R8 (PCA), and R9 (VAE) at k=300.
        If wall-clock columns are unavailable, the table is populated
        from ``wall_clock.json`` files if present.

        Returns
        -------
        str
            Path to the generated ``.tex`` file.
        """
        d = self._filter_k300(df)
        if d.empty:
            # Emit template so LaTeX compiles
            return self._emit_repr_timing_template()

        # Run configurations and their representation dimensions
        configs = [
            ("R1", "Raw"),
            ("R8", "PCA"),
            ("R9", "VAE-mean"),
        ]

        # Try to resolve dimensions from constants
        try:
            from ..config.constants import D_FEATURES, VAE_LATENT_DIM, KPCA_COMPONENTS
            dim_map = {"R1": D_FEATURES, "R8": KPCA_COMPONENTS, "R9": VAE_LATENT_DIM}
        except ImportError:
            dim_map = {"R1": "---", "R8": "---", "R9": "---"}

        # Discover timing columns
        solver_col = None
        for cand in ["wall_clock_solver_s", "wall_clock_nsga2_s",
                      "wall_clock_optimization_s"]:
            if cand in d.columns:
                solver_col = cand
                break

        total_col = "wall_clock_s" if "wall_clock_s" in d.columns else None

        # If no timing columns in the DF, try loading from wall_clock.json files
        if total_col is None:
            json_timing = self._load_wall_clock_jsons()
            if json_timing:
                return self._emit_repr_timing_from_json(json_timing, dim_map)

        if total_col is None:
            return self._emit_repr_timing_template()

        # ---- Gather per-config timing ----
        rows_data = []
        for run_pattern, space_label in configs:
            sub = d[d["run_id"].astype(str).str.contains(run_pattern)]
            if sub.empty:
                continue

            dim_val = dim_map.get(run_pattern, "---")

            # Solver time: use phase-specific column if available
            if solver_col and solver_col in sub.columns:
                solver_s = float(sub[solver_col].mean())
            else:
                solver_s = np.nan

            # Total time
            if total_col in sub.columns:
                total_s = float(sub[total_col].mean())
            else:
                total_s = np.nan

            rows_data.append({
                "space": space_label,
                "dim": str(dim_val),
                "solver_s": solver_s,
                "total_s": total_s,
            })

        if not rows_data:
            return self._emit_repr_timing_template()

        return self._emit_repr_timing_latex(rows_data)

    def _load_wall_clock_jsons(self) -> dict:
        """Load wall_clock.json files from runs for R1, R8, R9."""
        import json as _json
        timing = {}
        for run_pattern in ["R1", "R8", "R9"]:
            json_files = glob.glob(
                os.path.join(self.runs_root, f"{run_pattern}*",
                             "rep*", "results", "wall_clock.json"),
            )
            if json_files:
                try:
                    with open(json_files[0]) as f:
                        timing[run_pattern] = _json.load(f)
                except Exception:
                    pass
        return timing

    def _emit_repr_timing_from_json(self, json_timing: dict,
                                     dim_map: dict) -> str:
        """Build repr-timing table from wall_clock.json data."""
        label_map = {"R1": "Raw", "R8": "PCA", "R9": "VAE-mean"}
        rows_data = []
        for rid in ["R1", "R8", "R9"]:
            if rid not in json_timing:
                continue
            t = json_timing[rid]
            solver_s = t.get("wall_clock_solver_s",
                             t.get("wall_clock_nsga2_s",
                                   t.get("wall_clock_optimization_s", np.nan)))
            total_s = t.get("wall_clock_total_s", np.nan)
            rows_data.append({
                "space": label_map[rid],
                "dim": str(dim_map.get(rid, "---")),
                "solver_s": float(solver_s) if solver_s is not None else np.nan,
                "total_s": float(total_s) if total_s is not None else np.nan,
            })
        if not rows_data:
            return self._emit_repr_timing_template()
        return self._emit_repr_timing_latex(rows_data)

    def _emit_repr_timing_latex(self, rows_data: list) -> str:
        """Emit the repr-timing LaTeX table from collected row data."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Representation transfer timing at $k=300$: "
            r"wall-clock cost of optimising in different spaces.}",
            r"\label{tab:repr-timing}",
            r"\begin{tabular}{l c c c}",
            r"\toprule",
            r"Opt.\ space & dim.\ $p$ & solver time (s) & total time (s) \\",
            r"\midrule",
        ]
        csv_rows = []
        for row in rows_data:
            solver_str = self._fmt(row["solver_s"], 1) if np.isfinite(
                row["solver_s"]) else "---"
            total_str = self._fmt(row["total_s"], 1) if np.isfinite(
                row["total_s"]) else "---"
            lines.append(
                f"{row['space']} & {row['dim']} & "
                f"{solver_str} & {total_str}" + r" \\"
            )
            csv_rows.append(row)
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = os.path.join(self.tab_dir, "repr_timing.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))

        csv_path = os.path.join(self.tab_dir, "repr_timing.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, float_format="%.1f")
        return path

    def _emit_repr_timing_template(self) -> str:
        """Emit a compilable template for repr-timing when no data is available."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Representation transfer timing at $k=300$: "
            r"wall-clock cost of optimising in different spaces.}",
            r"\label{tab:repr-timing}",
            r"\begin{tabular}{l c c c}",
            r"\toprule",
            r"Opt.\ space & dim.\ $p$ & solver time (s) & total time (s) \\",
            r"\midrule",
            r"Raw & --- & --- & --- \\",
            r"PCA & --- & --- & --- \\",
            r"VAE-mean & --- & --- & --- \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        path = os.path.join(self.tab_dir, "repr_timing.tex")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
