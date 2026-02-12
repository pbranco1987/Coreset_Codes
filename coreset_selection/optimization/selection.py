"""
Pareto front analysis and representative selection.

Contains:
- select_knee for finding knee point via maximum curvature
- select_pareto_representatives for selecting diverse representatives
- feasible_filter for filtering to feasible solutions
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def select_knee(F: np.ndarray, normalize: bool = True, eps_norm: float = 1e-12) -> int:
    """
    Select balanced (knee-like) solution from Pareto front.
    
    Per manuscript Section 6.2:
    - For each objective j, compute min-max normalized values:
      f̃_j(s) = (f_j(s) - f_j^min) / (f_j^max - f_j^min + ε_norm)
    - Select s_knee ∈ argmin_{s ∈ P*} ||f̃(s)||_2
    
    This selects the feasible non-dominated point closest to the
    utopia point (which is the origin after normalization).
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_points, n_objectives)
    normalize : bool
        Whether to normalize objectives (should always be True per manuscript)
    eps_norm : float
        Small constant to prevent division by zero (ε_norm > 0)
        
    Returns
    -------
    int
        Index of the knee (balanced) point
    """
    F = np.asarray(F)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    
    n_points, n_obj = F.shape
    
    if n_points <= 1:
        return 0
        
    if normalize:
        # Min-max normalization per manuscript
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        denom = F_max - F_min + eps_norm  # ε_norm prevents division by zero
        F_norm = (F - F_min) / denom
    else:
        F_norm = F.copy()
    
    # Under this normalization, the utopia point is the origin (0, 0, ..., 0)
    # because each normalized objective attains its minimum value 0 on P*
    # The knee solution is closest to this utopia point in L2 distance
    distances_to_utopia = np.linalg.norm(F_norm, axis=1)
    
    return int(np.argmin(distances_to_utopia))


def select_pareto_representatives(
    F: np.ndarray,
    objectives: Tuple[str, ...],
    *,
    add_pairwise_knees: bool = True,
) -> Dict[str, int]:
    """
    Select representative solutions from a Pareto front.
    
    Selects:
    - Per-objective minimizers (e.g., "best_skl")
    - Overall knee point (e.g., "knee")
    - Pairwise knee points if requested (e.g., "knee_skl_mmd")
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_points, n_objectives)
    objectives : Tuple[str, ...]
        Names of objectives (e.g., ("skl", "mmd", "sinkhorn"))
    add_pairwise_knees : bool
        Whether to include pairwise 2D knee points
        
    Returns
    -------
    Dict[str, int]
        Mapping from representative name to index in F
    """
    F = np.asarray(F)
    n_points, n_obj = F.shape
    
    if n_obj != len(objectives):
        raise ValueError(f"F has {n_obj} objectives but {len(objectives)} names given")
    
    sel_map: Dict[str, int] = {}
    
    # Per-objective minimizers
    for j, obj_name in enumerate(objectives):
        sel_map[f"best_{obj_name}"] = int(np.argmin(F[:, j]))
    
    # Overall knee point
    sel_map["knee"] = select_knee(F, normalize=True)
    
    # Pairwise knees
    if add_pairwise_knees and n_obj > 2:
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                F_pair = F[:, [i, j]]
                knee_idx = select_knee(F_pair, normalize=True)
                key = f"knee_{objectives[i]}_{objectives[j]}"
                sel_map[key] = knee_idx
    
    return sel_map


def feasible_filter(
    problem,
    Xcand: np.ndarray,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter candidate solutions to keep only feasible ones.
    
    Parameters
    ----------
    problem : Problem
        Pymoo problem with constraint evaluation
    Xcand : np.ndarray
        Candidate solutions, shape (n_candidates, n_var)
    tol : float
        Tolerance for constraint violation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (feasible_mask, feasible_solutions)
    """
    Xcand = np.asarray(Xcand)
    
    if not hasattr(problem, 'n_ieq_constr') or problem.n_ieq_constr == 0:
        # No constraints, all feasible
        return np.ones(len(Xcand), dtype=bool), Xcand
    
    # Evaluate constraints
    out = {}
    problem._evaluate(Xcand, out)
    
    G = out.get("G", None)
    if G is None:
        return np.ones(len(Xcand), dtype=bool), Xcand
    
    # Feasible if all constraints <= 0 (within tolerance)
    feasible_mask = np.all(G <= tol, axis=1)
    
    return feasible_mask, Xcand[feasible_mask]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for a Pareto front.
    
    Points at the boundary get infinite distance. Interior points
    get distance based on objective-wise neighbors.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_points, n_objectives)
        
    Returns
    -------
    np.ndarray
        Crowding distances, shape (n_points,)
    """
    F = np.asarray(F)
    n_points, n_obj = F.shape
    
    if n_points <= 2:
        return np.full(n_points, np.inf)
    
    distances = np.zeros(n_points)
    
    for j in range(n_obj):
        # Sort by this objective
        sorted_idx = np.argsort(F[:, j])
        
        # Boundary points get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        # Normalize range
        f_min = F[sorted_idx[0], j]
        f_max = F[sorted_idx[-1], j]
        denom = f_max - f_min
        if denom < 1e-12:
            continue
        
        # Interior points
        for i in range(1, n_points - 1):
            curr = sorted_idx[i]
            prev = sorted_idx[i - 1]
            next_ = sorted_idx[i + 1]
            distances[curr] += (F[next_, j] - F[prev, j]) / denom
    
    return distances


def hypervolume_contribution(
    F: np.ndarray,
    ref_point: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute hypervolume contribution of each point.
    
    The contribution is the hypervolume lost if that point is removed.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_points, n_objectives)
    ref_point : Optional[np.ndarray]
        Reference point for hypervolume. If None, uses nadir + 10%
        
    Returns
    -------
    np.ndarray
        Hypervolume contributions, shape (n_points,)
    """
    F = np.asarray(F)
    n_points, n_obj = F.shape
    
    if ref_point is None:
        # Use nadir point + 10% margin
        nadir = F.max(axis=0)
        margin = 0.1 * (nadir - F.min(axis=0))
        ref_point = nadir + margin
    
    contributions = np.zeros(n_points)
    
    # Simple 2D case
    if n_obj == 2:
        # Sort by first objective
        sorted_idx = np.argsort(F[:, 0])
        F_sorted = F[sorted_idx]
        
        for i, orig_idx in enumerate(sorted_idx):
            # Compute rectangle from this point to next point's y (or ref)
            x = F_sorted[i, 0]
            y = F_sorted[i, 1]
            
            if i == n_points - 1:
                x_next = ref_point[0]
            else:
                x_next = F_sorted[i + 1, 0]
            
            if i == 0:
                y_bound = ref_point[1]
            else:
                y_bound = F_sorted[i - 1, 1]
            
            # Contribution is the exclusive rectangle
            dx = x_next - x
            dy = y_bound - y
            contributions[orig_idx] = max(0, dx * dy)
    else:
        # For higher dimensions, would need proper HV computation
        # Fall back to crowding distance as proxy
        contributions = crowding_distance(F)
    
    return contributions
