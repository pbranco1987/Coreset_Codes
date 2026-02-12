"""
Results saving utilities for experiment outputs.

Contains:
- ParetoFrontData: Data structure for Pareto front results
- ResultsSaver: Unified saving of experiment outputs
- claim_next_rep_id: Atomically claim the next available replicate ID
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Concurrency-safe replicate ID allocation
# ---------------------------------------------------------------------------

def _scan_max_rep_id(run_dir: str) -> int:
    """Return the highest existing repNN index, or -1 if none exist."""
    pattern = re.compile(r"^rep(\d+)$")
    max_id = -1
    try:
        for entry in os.listdir(run_dir):
            m = pattern.match(entry)
            if m and os.path.isdir(os.path.join(run_dir, entry)):
                max_id = max(max_id, int(m.group(1)))
    except FileNotFoundError:
        pass
    return max_id


def claim_next_rep_id(output_dir: str, run_name: str) -> int:
    """
    Atomically claim the next available replicate ID for a given run.

    Uses ``os.mkdir`` as a filesystem-level compare-and-swap: the call
    raises ``FileExistsError`` if another process already created the
    same ``repNN`` directory, so we simply retry with the next candidate.
    This is safe for fully concurrent execution (multiple processes
    launching the same experiment at the same time).

    The created directory is ``{output_dir}/{run_name}/repNN/`` where
    ``NN`` is the claimed ID.  Sub-directories (results/, plots/, …)
    are **not** created here — ``ResultsSaver`` handles that.

    Parameters
    ----------
    output_dir : str
        Base output directory (e.g. ``"runs_out"``).
    run_name : str
        Run name as it appears in the output tree (e.g. ``"R1"`` or
        ``"R1_k300"``).

    Returns
    -------
    int
        The claimed (newly created) replicate ID.

    Examples
    --------
    >>> # Process A and Process B call simultaneously:
    >>> id_a = claim_next_rep_id("runs_out", "R10")  # → 0
    >>> id_b = claim_next_rep_id("runs_out", "R10")  # → 1  (0 was taken)
    """
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Start search from one past the current maximum
    candidate = _scan_max_rep_id(run_dir) + 1

    while True:
        rep_dir = os.path.join(run_dir, f"rep{candidate:02d}")
        try:
            os.mkdir(rep_dir)          # atomic on POSIX
            return candidate
        except FileExistsError:
            candidate += 1            # someone else got it, try next


@dataclass
class ParetoFrontData:
    """
    Container for Pareto front optimization results.
    
    Attributes
    ----------
    F : np.ndarray
        Objective values, shape (n_solutions, n_objectives)
    X : np.ndarray
        Decision variables (boolean masks), shape (n_solutions, n_var)
    objectives : Tuple[str, ...]
        Names of objectives
    representatives : Dict[str, int]
        Mapping from representative names to indices
    selected_indices : Dict[str, np.ndarray]
        Mapping from representative names to actual data indices
    """
    F: np.ndarray
    X: np.ndarray
    objectives: Tuple[str, ...]
    representatives: Dict[str, int]
    selected_indices: Dict[str, np.ndarray] = field(default_factory=dict)

    def get_solution(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a representative solution by name.
        
        Parameters
        ----------
        name : str
            Representative name (e.g., "knee", "best_skl")
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (objective_values, selection_mask)
        """
        idx = self.representatives[name]
        return self.F[idx], self.X[idx]

    def get_indices(self, name: str) -> np.ndarray:
        """
        Get data indices for a representative.
        
        Parameters
        ----------
        name : str
            Representative name
            
        Returns
        -------
        np.ndarray
            Indices of selected points
        """
        if name in self.selected_indices:
            return self.selected_indices[name]
        
        idx = self.representatives[name]
        return np.where(self.X[idx])[0]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "n_solutions": len(self.F),
            "n_objectives": len(self.objectives),
            "objectives": self.objectives,
            "representatives": list(self.representatives.keys()),
        }


class ResultsSaver:
    """
    Unified results saver for experiment outputs.
    
    Handles saving of:
    - Pareto fronts (npz)
    - Selected indices (npz)
    - Metrics (JSON, CSV)
    - Plots (PNG, PDF)
    - Configuration (JSON)
    
    Attributes
    ----------
    base_dir : str
        Base output directory
    run_id : str
        Run identifier
    rep_id : int
        Replicate identifier
    """
    
    def __init__(self, base_dir: str, run_id: str, rep_id: int):
        """
        Initialize the ResultsSaver.
        
        Parameters
        ----------
        base_dir : str
            Base output directory
        run_id : str
            Run identifier (e.g., "R1")
        rep_id : int
            Replicate identifier
        """
        self.base_dir = base_dir
        self.run_id = run_id
        self.rep_id = rep_id
        
        # Create directory structure
        self.run_dir = os.path.join(base_dir, run_id, f"rep{rep_id:02d}")
        self.results_dir = ensure_dir(os.path.join(self.run_dir, "results"))
        self.plots_dir = ensure_dir(os.path.join(self.run_dir, "plots"))
        self.coresets_dir = ensure_dir(os.path.join(self.run_dir, "coresets"))

    def save_run_manifest(
        self,
        *,
        cli_args: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write run_manifest.json capturing full reproducibility metadata.

        Parameters
        ----------
        cli_args : dict, optional
            CLI arguments that launched this run.
        seed : int
            Base random seed.
        extra : dict, optional
            Any additional key/value pairs to record.

        Returns
        -------
        str
            Path to the saved manifest file.
        """
        import datetime
        import platform
        import socket
        import subprocess as _sp
        import sys

        manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "rep_id": self.rep_id,
            "seed": seed,
            "hostname": socket.gethostname(),
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "output_dir": self.run_dir,
        }

        # Git commit hash (best-effort)
        try:
            result = _sp.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                manifest["git_commit"] = result.stdout.strip()
        except Exception:
            manifest["git_commit"] = "unknown"

        # pip freeze (best-effort)
        try:
            result = _sp.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                manifest["pip_freeze"] = result.stdout.strip().split("\n")
        except Exception:
            manifest["pip_freeze"] = []

        if cli_args:
            manifest["cli_args"] = cli_args
        if extra:
            manifest.update(extra)

        path = os.path.join(self.run_dir, "run_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=_json_serializer)
        return path

    def save_pareto_front(
        self,
        name: str,
        pareto_data: ParetoFrontData,
    ) -> str:
        """
        Save Pareto front data.
        
        Parameters
        ----------
        name : str
            Name for the file (e.g., "vae_space")
        pareto_data : ParetoFrontData
            Pareto front data to save
            
        Returns
        -------
        str
            Path to saved file
        """
        path = os.path.join(self.results_dir, f"{name}_pareto.npz")
        
        # Prepare data dict
        data = {
            "F": pareto_data.F,
            "X": pareto_data.X,
            "objectives": np.array(pareto_data.objectives, dtype=object),
            "rep_names": np.array(list(pareto_data.representatives.keys()), dtype=object),
            "rep_indices": np.array(list(pareto_data.representatives.values())),
        }
        
        # Add selected indices
        for rep_name, indices in pareto_data.selected_indices.items():
            data[f"idx_{rep_name}"] = indices
        
        np.savez_compressed(path, **data)
        return path

    def save_coreset(
        self,
        name: str,
        indices: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a coreset (selected indices).
        
        Parameters
        ----------
        name : str
            Name for the coreset
        indices : np.ndarray
            Selected indices
        metadata : Optional[Dict[str, Any]]
            Additional metadata
            
        Returns
        -------
        str
            Path to saved file
        """
        path = os.path.join(self.coresets_dir, f"{name}.npz")
        
        data = {"indices": indices}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        np.savez_compressed(path, **data)
        return path

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        name: str = "metrics",
    ) -> Tuple[str, str]:
        """
        Save metrics as JSON and CSV.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            Metrics dictionary
        name : str
            Base name for files
            
        Returns
        -------
        Tuple[str, str]
            Paths to (JSON file, CSV file)
        """
        json_path = os.path.join(self.results_dir, f"{name}.json")
        csv_path = os.path.join(self.results_dir, f"{name}.csv")
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=_json_serializer)
        
        # Save flattened CSV
        flat_metrics = _flatten_dict(metrics)
        with open(csv_path, 'w') as f:
            f.write("metric,value\n")
            for key, value in flat_metrics.items():
                f.write(f"{key},{value}\n")
        
        return json_path, csv_path

    def save_config(self, config: Any) -> str:
        """
        Save experiment configuration.
        
        Parameters
        ----------
        config : Any
            Configuration object (dataclass or dict)
            
        Returns
        -------
        str
            Path to saved file
        """
        path = os.path.join(self.run_dir, "config.json")
        
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = dict(config)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=_json_serializer)
        
        return path

    def save_rows(
        self,
        rows: List[Dict[str, Any]],
        name: str = "results",
    ) -> str:
        """
        Save list of result rows as CSV.
        
        Parameters
        ----------
        rows : List[Dict[str, Any]]
            List of row dictionaries
        name : str
            File name
            
        Returns
        -------
        str
            Path to saved file
        """
        if not rows:
            return ""
        
        path = os.path.join(self.results_dir, f"{name}.csv")
        
        # Get all keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        keys = sorted(all_keys)
        
        with open(path, 'w') as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                values = [str(row.get(k, "")) for k in keys]
                f.write(",".join(values) + "\n")
        
        return path

    def save_wall_clock(
        self,
        timing: Dict[str, Any],
    ) -> str:
        """Save wall-clock timing breakdown as JSON.

        This is always written (not gated by ``CORESET_DEBUG``) so that
        every experiment run has a machine-readable timing record.

        Parameters
        ----------
        timing : dict
            Timing breakdown.  Expected keys include at minimum
            ``wall_clock_total_s``; additional phase-level keys
            (e.g. ``wall_clock_solver_s``, ``wall_clock_eval_s``) are
            encouraged.

        Returns
        -------
        str
            Path to the saved ``wall_clock.json`` file.
        """
        path = os.path.join(self.results_dir, "wall_clock.json")
        with open(path, "w") as f:
            json.dump(timing, f, indent=2, default=_json_serializer)
        return path

    def get_plot_path(self, name: str, ext: str = "png") -> str:
        """Get path for a plot file."""
        return os.path.join(self.plots_dir, f"{name}.{ext}")

    # ------------------------------------------------------------------
    # Phase 7 (R11): Proxy stability & objective–metric alignment CSVs
    # ------------------------------------------------------------------

    def get_proxy_stability_csv_path(self) -> str:
        """Return the canonical path for ``proxy_stability.csv`` (Table IV).

        Returns
        -------
        str
            ``<results_dir>/proxy_stability.csv``
        """
        return os.path.join(self.results_dir, "proxy_stability.csv")

    def get_objective_metric_alignment_csv_path(self) -> str:
        """Return the canonical path for ``objective_metric_alignment.csv`` (Fig 4).

        Returns
        -------
        str
            ``<results_dir>/objective_metric_alignment.csv``
        """
        return os.path.join(self.results_dir, "objective_metric_alignment.csv")

    def get_alignment_heatmap_csv_path(self) -> str:
        """Return the canonical path for the heatmap pivot-table CSV (Fig 4).

        Returns
        -------
        str
            ``<results_dir>/objective_metric_alignment_heatmap.csv``
        """
        return os.path.join(self.results_dir, "objective_metric_alignment_heatmap.csv")

    # ------------------------------------------------------------------
    # G10: Structured effort-sweep output helpers
    # ------------------------------------------------------------------

    def save_effort_grid_csv(
        self,
        grid: List[Tuple[int, int]],
        *,
        k: int = 0,
        objectives: Tuple[str, ...] = (),
        constraint_regime: str = "",
        space: str = "",
    ) -> str:
        """Save the effort-sweep parameter grid as a structured CSV.

        Produces ``effort_grid_config.csv`` with one row per ``(P, T)``
        effort level.  Downstream artifact generators (e.g.
        ``ManuscriptArtifacts.fig_effort_quality``) can read this file
        to annotate plots with the exact grid used.

        Parameters
        ----------
        grid : list of (int, int)
            ``(pop_size, n_gen)`` pairs.
        k : int
            Coreset cardinality.
        objectives : tuple of str
            Objective function names.
        constraint_regime : str
            Human-readable constraint label.
        space : str
            Representation space name.

        Returns
        -------
        str
            Path to the saved CSV.
        """
        path = os.path.join(self.results_dir, "effort_grid_config.csv")
        with open(path, "w") as f:
            f.write(
                "level,pop_size,n_gen,effort_P_x_T,k,objectives,constraint_regime,space\n"
            )
            obj_str = "+".join(objectives) if objectives else ""
            for i, (P, T) in enumerate(grid):
                f.write(
                    f"{i},{P},{T},{P * T},{k},{obj_str},{constraint_regime},{space}\n"
                )
        return path

    def save_effort_sweep_summary(
        self,
        rows: List[Dict[str, Any]],
    ) -> str:
        """Save a dedicated effort-sweep summary CSV.

        This produces ``effort_sweep_summary.csv`` with the *knee*
        representative from each effort level, including wall-clock
        time, front size, and key quality metrics.  The file is a
        filtered, structured view of the full ``effort_sweep_results.csv``
        for convenient consumption by tables and plots.

        Parameters
        ----------
        rows : list of dict
            Rows from the effort sweep (may contain multiple rep_names).

        Returns
        -------
        str
            Path to the saved CSV.
        """
        # Filter to knee representative only (most commonly used in tables)
        knee_rows = [r for r in rows if r.get("rep_name") == "knee"]
        if not knee_rows:
            knee_rows = rows  # fall back to all if no knee

        if not knee_rows:
            return ""

        path = os.path.join(self.results_dir, "effort_sweep_summary.csv")

        # Columns in presentation order
        priority_cols = [
            "pop_size", "n_gen", "effort_P_x_T", "wall_clock_s",
            "front_size", "geo_kl", "geo_l1", "geo_kl_muni", "geo_l1_muni",
            "geo_kl_pop", "geo_l1_pop",
        ]
        all_keys = set()
        for r in knee_rows:
            all_keys.update(r.keys())
        # priority first, then rest alphabetically
        ordered = [c for c in priority_cols if c in all_keys]
        ordered += sorted(all_keys - set(ordered))

        with open(path, "w") as f:
            f.write(",".join(ordered) + "\n")
            for r in knee_rows:
                vals = [str(r.get(c, "")) for c in ordered]
                f.write(",".join(vals) + "\n")

        return path


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_pareto_front(path: str) -> ParetoFrontData:
    """
    Load Pareto front data from npz file.
    
    Parameters
    ----------
    path : str
        Path to npz file
        
    Returns
    -------
    ParetoFrontData
    """
    data = np.load(path, allow_pickle=True)
    
    objectives = tuple(data["objectives"])
    rep_names = list(data["rep_names"])
    rep_indices = data["rep_indices"]
    representatives = dict(zip(rep_names, rep_indices))
    
    # Load selected indices
    selected_indices = {}
    for key in data.files:
        if key.startswith("idx_"):
            name = key[4:]  # Remove "idx_" prefix
            selected_indices[name] = data[key]
    
    return ParetoFrontData(
        F=data["F"],
        X=data["X"],
        objectives=objectives,
        representatives=representatives,
        selected_indices=selected_indices,
    )
