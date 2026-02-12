"""
Plotting functions for manuscript figures.

Contains:
- Pareto front visualization
- Metric vs. k curves
- Geographic distribution plots
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.plotting import (
    set_manuscript_style,
    get_method_style,
    get_objective_color,
    objective_label,
    method_label,
    figure_size,
    add_panel_label,
    PALETTE_METHODS,
)


def plot_pareto_front_2d(
    F: np.ndarray,
    objectives: Tuple[str, str],
    representatives: Optional[Dict[str, int]] = None,
    ax=None,
    title: Optional[str] = None,
    highlight_knee: bool = True,
) -> Any:
    """
    Plot a 2D Pareto front.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_solutions, 2)
    objectives : Tuple[str, str]
        Names of the two objectives
    representatives : Optional[Dict[str, int]]
        Mapping from representative names to indices
    ax : Optional[matplotlib.axes.Axes]
        Axes to plot on. If None, creates new figure.
    title : Optional[str]
        Plot title
    highlight_knee : bool
        Whether to highlight the knee point
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size("single"))
    
    # Sort by first objective for line plot
    sort_idx = np.argsort(F[:, 0])
    F_sorted = F[sort_idx]
    
    # Plot Pareto front
    ax.plot(F_sorted[:, 0], F_sorted[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax.scatter(F[:, 0], F[:, 1], c='gray', s=20, alpha=0.5, label='Pareto front')
    
    # Highlight representatives
    if representatives:
        for rep_name, rep_idx in representatives.items():
            if rep_idx >= len(F):
                continue
            color, marker = get_method_style(f"pareto_{rep_name}")
            ax.scatter(
                F[rep_idx, 0], F[rep_idx, 1],
                c=color, marker=marker, s=100, 
                edgecolors='black', linewidths=0.5,
                label=method_label(f"pareto_{rep_name}"), zorder=10
            )
    
    ax.set_xlabel(objective_label(objectives[0]))
    ax.set_ylabel(objective_label(objectives[1]))
    
    if title:
        ax.set_title(title)
    
    ax.legend(loc='upper right', fontsize=8)
    
    return ax


def plot_pareto_front_3d(
    F: np.ndarray,
    objectives: Tuple[str, str, str],
    representatives: Optional[Dict[str, int]] = None,
    ax=None,
    title: Optional[str] = None,
    elev: float = 20,
    azim: float = 45,
) -> Any:
    """
    Plot a 3D Pareto front.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values, shape (n_solutions, 3)
    objectives : Tuple[str, str, str]
        Names of the three objectives
    representatives : Optional[Dict[str, int]]
        Mapping from representative names to indices
    ax : Optional
        3D axes to plot on
    title : Optional[str]
        Plot title
    elev : float
        Elevation angle for 3D view
    azim : float
        Azimuth angle for 3D view
        
    Returns
    -------
    Axes3D
        The 3D axes object
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=figure_size("single", aspect=1.0))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto front
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='gray', s=10, alpha=0.3)
    
    # Highlight representatives
    if representatives:
        for rep_name, rep_idx in representatives.items():
            if rep_idx >= len(F):
                continue
            color, marker = get_method_style(f"pareto_{rep_name}")
            ax.scatter(
                [F[rep_idx, 0]], [F[rep_idx, 1]], [F[rep_idx, 2]],
                c=color, marker=marker, s=100,
                edgecolors='black', linewidths=0.5,
                label=method_label(f"pareto_{rep_name}")
            )
    
    ax.set_xlabel(objective_label(objectives[0]))
    ax.set_ylabel(objective_label(objectives[1]))
    ax.set_zlabel(objective_label(objectives[2]))
    
    ax.view_init(elev=elev, azim=azim)
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_metric_vs_k(
    results_df,
    metric: str,
    methods: Optional[List[str]] = None,
    ax=None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_std: bool = True,
) -> Any:
    """
    Plot a metric vs. k for multiple methods.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Results dataframe with columns: method, k, {metric}
    metric : str
        Metric column name
    methods : Optional[List[str]]
        Methods to include. If None, uses all.
    ax : Optional[matplotlib.axes.Axes]
        Axes to plot on
    title : Optional[str]
        Plot title
    ylabel : Optional[str]
        Y-axis label
    show_std : bool
        Whether to show standard deviation bands
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size("single"))
    
    if methods is None:
        methods = results_df['method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        
        # Group by k
        grouped = method_data.groupby('k')[metric]
        k_values = grouped.mean().index.values
        means = grouped.mean().values
        stds = grouped.std().values
        
        color, marker = get_method_style(method)
        
        ax.plot(k_values, means, color=color, marker=marker, 
                label=method_label(method), markersize=5)
        
        if show_std and len(stds) > 0 and not np.all(np.isnan(stds)):
            ax.fill_between(k_values, means - stds, means + stds, 
                           color=color, alpha=0.2)
    
    ax.set_xlabel('Coreset size $k$')
    ax.set_ylabel(ylabel or metric)
    
    if title:
        ax.set_title(title)
    
    ax.legend(loc='best', fontsize=8)
    
    return ax


def plot_geographic_distribution(
    counts: np.ndarray,
    group_names: List[str],
    target_counts: Optional[np.ndarray] = None,
    ax=None,
    title: Optional[str] = None,
) -> Any:
    """
    Plot geographic distribution as a bar chart.
    
    Parameters
    ----------
    counts : np.ndarray
        Counts per geographic group
    group_names : List[str]
        Names of geographic groups
    target_counts : Optional[np.ndarray]
        Target quota counts (for comparison)
    ax : Optional[matplotlib.axes.Axes]
        Axes to plot on
    title : Optional[str]
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size("double", aspect=0.4))
    
    n_groups = len(counts)
    x = np.arange(n_groups)
    width = 0.35
    
    ax.bar(x - width/2, counts, width, label='Selected', color='steelblue')
    
    if target_counts is not None:
        ax.bar(x + width/2, target_counts, width, label='Target', 
               color='orange', alpha=0.7)
    
    ax.set_xlabel('State')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=8)
    
    if target_counts is not None:
        ax.legend()
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    return ax


def save_pareto_front_plots(
    F: np.ndarray,
    objectives: Tuple[str, ...],
    representatives: Dict[str, int],
    out_prefix: str,
    title: Optional[str] = None,
) -> List[str]:
    """
    Save Pareto front plots for all 2D projections.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values
    objectives : Tuple[str, ...]
        Objective names
    representatives : Dict[str, int]
        Representative indices
    out_prefix : str
        Output file prefix (without extension)
    title : Optional[str]
        Base title
        
    Returns
    -------
    List[str]
        Paths to saved files
    """
    import matplotlib.pyplot as plt
    
    set_manuscript_style()
    
    n_obj = len(objectives)
    saved_paths = []
    
    # 2D projections
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            fig, ax = plt.subplots(figsize=figure_size("single"))
            
            F_2d = F[:, [i, j]]
            obj_pair = (objectives[i], objectives[j])
            
            plot_pareto_front_2d(
                F_2d, obj_pair, representatives,
                ax=ax,
                title=f"{title} ({objectives[i]} vs {objectives[j]})" if title else None
            )
            
            path_png = f"{out_prefix}_{objectives[i]}_{objectives[j]}.png"
            path_pdf = f"{out_prefix}_{objectives[i]}_{objectives[j]}.pdf"
            
            fig.savefig(path_png, dpi=300, bbox_inches='tight')
            fig.savefig(path_pdf, bbox_inches='tight')
            plt.close(fig)
            
            saved_paths.extend([path_png, path_pdf])
    
    # 3D plot if 3 objectives
    if n_obj == 3:
        fig = plt.figure(figsize=figure_size("single", aspect=1.0))
        ax = fig.add_subplot(111, projection='3d')
        
        plot_pareto_front_3d(F, objectives[:3], representatives, ax=ax, title=title)
        
        path_png = f"{out_prefix}_3d.png"
        path_pdf = f"{out_prefix}_3d.pdf"
        
        fig.savefig(path_png, dpi=300, bbox_inches='tight')
        fig.savefig(path_pdf, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.extend([path_png, path_pdf])
    
    return saved_paths


def plot_convergence(
    history: List[np.ndarray],
    metric: str = "hypervolume",
    ax=None,
) -> Any:
    """
    Plot optimization convergence over generations.
    
    Parameters
    ----------
    history : List[np.ndarray]
        List of metric values per generation
    metric : str
        Metric name for labeling
    ax : Optional[matplotlib.axes.Axes]
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size("single"))
    
    generations = np.arange(len(history))
    
    ax.plot(generations, history, 'b-', linewidth=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Convergence')
    
    return ax
