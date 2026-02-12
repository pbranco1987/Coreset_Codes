r"""Weighted proportionality constraints (manuscript Sections IV-B and V-D).

We model proportionality constraints through:
  - a group label g_i (here: state)
  - nonnegative weights w_i (e.g., 1 for municipality-share; population for
    population-share)

For a subset S, define group totals W_g(S)=Σ_{i∈S, g_i=g} w_i and W(S)=Σ_{i∈S} w_i.
We use Laplace smoothing (pseudo-count α>0) to avoid infinite KL:
  \hatπ_g(S) = (W_g(S)+α)/(W(S)+αG)

The proportionality metric is the forward KL:
  D(S)=KL(π || \hatπ(S)), where π is the full-data weighted distribution.

This module provides:
  - ProportionalityConstraint: computes D(S)
  - ProportionalityConstraintSet: aggregates multiple constraints
  - swap-based repair heuristics used inside NSGA-II (Algorithm 2 style)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geo.info import GeoInfo


def _safe_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p||q) for probability vectors with p>0; assumes q>0."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-30))))


@dataclass(frozen=True)
class ProportionalityConstraint:
    """One KL proportionality constraint D(S) ≤ τ."""

    name: str
    weights: np.ndarray  # shape (N,)
    target_pi: np.ndarray  # shape (G,)
    alpha: float = 1.0
    tau: float = 0.02

    def value(self, mask: np.ndarray, geo: GeoInfo) -> float:
        r"""Compute D(S)=KL(π || \hatπ(S))."""
        mask = np.asarray(mask, dtype=bool)
        w = self.weights
        if mask.sum() == 0:
            return float("inf")

        # Per-group totals
        Wg = np.zeros(geo.G, dtype=np.float64)
        for g in range(geo.G):
            idx_g = geo.group_to_indices[g]
            if idx_g.size == 0:
                continue
            if mask[idx_g].any():
                Wg[g] = float(w[idx_g[mask[idx_g]]].sum())

        W = float(Wg.sum())
        # Smoothed subset distribution (Laplace)
        q = (Wg + float(self.alpha)) / (W + float(self.alpha) * geo.G)
        return _safe_kl(self.target_pi, q)

    def violation(self, mask: np.ndarray, geo: GeoInfo) -> float:
        """Positive part of constraint violation."""
        v = self.value(mask, geo)
        return float(max(v - float(self.tau), 0.0))


def _build_target_pi(geo: GeoInfo, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    Wg = np.zeros(geo.G, dtype=np.float64)
    for g in range(geo.G):
        idx_g = geo.group_to_indices[g]
        if idx_g.size == 0:
            continue
        Wg[g] = float(weights[idx_g].sum())
    W = float(Wg.sum())
    if W <= 0:
        raise ValueError("Weights must have positive total mass.")
    pi = Wg / W
    # Avoid exact zeros to keep numeric stability; leave true zeros as zeros
    return pi


def build_population_share_constraint(
    *,
    geo: GeoInfo,
    population: np.ndarray,
    alpha: float,
    tau: float,
) -> ProportionalityConstraint:
    r"""Build a population-share proportionality constraint.

    Implements the population-weighted KL constraint from manuscript
    Section IV-B: weights w_i = pop_i, target π_g = Σ_{i:g_i=g} pop_i / Σ_i pop_i.
    The constraint is D^(pop)(S) = KL(π || π̂^(α)(S)) ≤ τ.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group structure.
    population : np.ndarray
        Per-municipality population weights, shape (N,).
    alpha : float
        Dirichlet smoothing parameter (α_geo, default 1.0).
    tau : float
        KL tolerance threshold (τ_population).

    Returns
    -------
    ProportionalityConstraint
    """
    pop = np.asarray(population, dtype=np.float64)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)
    pi = _build_target_pi(geo, pop)
    return ProportionalityConstraint(
        name="population_share",
        weights=pop,
        target_pi=pi,
        alpha=float(alpha),
        tau=float(tau),
    )


def build_municipality_share_constraint(
    *,
    geo: GeoInfo,
    alpha: float,
    tau: float,
) -> ProportionalityConstraint:
    r"""Build a municipality-share (count-based) proportionality constraint.

    Implements the count-weighted KL constraint from manuscript Section IV-B:
    weights w_i ≡ 1, target π_g = n_g / N (municipality share).
    The constraint is D^(muni)(S) = KL(π || π̂^(α)(S)) ≤ τ.

    Parameters
    ----------
    geo : GeoInfo
        Geographic group structure.
    alpha : float
        Dirichlet smoothing parameter (α_geo, default 1.0).
    tau : float
        KL tolerance threshold (τ_municipality).

    Returns
    -------
    ProportionalityConstraint
    """
    w = np.ones(geo.N, dtype=np.float64)
    pi = _build_target_pi(geo, w)
    return ProportionalityConstraint(
        name="municipality_share",
        weights=w,
        target_pi=pi,
        alpha=float(alpha),
        tau=float(tau),
    )


class ProportionalityConstraintSet:
    """A set of proportionality constraints with swap-based repair."""

    def __init__(
        self,
        *,
        geo: GeoInfo,
        constraints: Sequence[ProportionalityConstraint],
        min_one_per_group: bool = True,
        preserve_group_counts: bool = False,
        max_iters: int = 200,
    ):
        self.geo = geo
        self.constraints = list(constraints)
        self.min_one_per_group = bool(min_one_per_group)
        self.preserve_group_counts = bool(preserve_group_counts)
        self.max_iters = int(max_iters)

        # Per-group count bounds
        self.lower = np.ones(geo.G, dtype=int) if self.min_one_per_group else np.zeros(geo.G, dtype=int)
        self.upper = np.asarray(geo.group_sizes, dtype=int)

    def values(self, mask: np.ndarray) -> Dict[str, float]:
        r"""Compute D^(h)(S) for each active constraint h (manuscript Eq. 4)."""
        return {c.name: c.value(mask, self.geo) for c in self.constraints}

    def total_violation(self, mask: np.ndarray) -> float:
        r"""Total constraint violation: V(S) = Σ_h max{D^(h)(S) − τ_h, 0} (Sec V-D)."""
        return float(sum(c.violation(mask, self.geo) for c in self.constraints))

    def is_feasible(self, mask: np.ndarray) -> bool:
        """Check if S satisfies all KL constraints and lower-bound requirements."""
        if self.total_violation(mask) > 0:
            return False
        # Count lower bounds
        if self.min_one_per_group:
            for g in range(self.geo.G):
                if mask[self.geo.group_to_indices[g]].sum() < self.lower[g]:
                    return False
        return True

    def repair(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Heuristic repair to reduce KL violations while preserving |S|."""
        mask = np.asarray(mask, dtype=bool).copy()
        if not self.constraints:
            return mask

        for _ in range(self.max_iters):
            # Pick the most violated constraint
            vals = [c.value(mask, self.geo) for c in self.constraints]
            viols = [max(v - c.tau, 0.0) for v, c in zip(vals, self.constraints)]
            v_max = float(max(viols))
            if v_max <= 0:
                break
            h = int(np.argmax(viols))
            c = self.constraints[h]

            if self.preserve_group_counts:
                mask2 = self._repair_within_groups(mask, c)
            else:
                mask2 = self._repair_across_groups(mask, c, rng)

            if self.total_violation(mask2) < self.total_violation(mask):
                mask = mask2
            else:
                break

        return mask

    def _repair_across_groups(
        self,
        mask: np.ndarray,
        c: ProportionalityConstraint,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Swap one item between two groups (Algorithm 2 style)."""
        geo = self.geo
        w = c.weights

        # Compute smoothed subset π-hat
        Wg = np.zeros(geo.G, dtype=np.float64)
        counts = np.zeros(geo.G, dtype=int)
        for g in range(geo.G):
            idx_g = geo.group_to_indices[g]
            sel = idx_g[mask[idx_g]]
            counts[g] = int(sel.size)
            if sel.size:
                Wg[g] = float(w[sel].sum())
        W = float(Wg.sum())
        q = (Wg + c.alpha) / (W + c.alpha * geo.G)
        s = c.target_pi / (q + 1e-30)

        # donor: over-represented => small s_g
        donor_candidates = [g for g in range(geo.G) if counts[g] > self.lower[g]]
        recv_candidates = [g for g in range(geo.G) if counts[g] < self.upper[g]]
        if not donor_candidates or not recv_candidates:
            return mask

        g_minus = int(min(donor_candidates, key=lambda g: s[g]))
        g_plus = int(max(recv_candidates, key=lambda g: s[g]))

        idx_minus = geo.group_to_indices[g_minus]
        idx_plus = geo.group_to_indices[g_plus]
        sel_minus = idx_minus[mask[idx_minus]]
        unsel_plus = idx_plus[~mask[idx_plus]]
        if sel_minus.size == 0 or unsel_plus.size == 0:
            return mask

        i_minus = int(rng.choice(sel_minus))
        i_plus = int(rng.choice(unsel_plus))
        mask2 = mask.copy()
        mask2[i_minus] = False
        mask2[i_plus] = True
        return mask2

    def _repair_within_groups(self, mask: np.ndarray, c: ProportionalityConstraint) -> np.ndarray:
        """Adjust weighted totals while preserving per-group counts (joint mode)."""
        geo = self.geo
        w = c.weights

        Wg = np.zeros(geo.G, dtype=np.float64)
        counts = np.zeros(geo.G, dtype=int)
        for g in range(geo.G):
            idx_g = geo.group_to_indices[g]
            sel = idx_g[mask[idx_g]]
            counts[g] = int(sel.size)
            if sel.size:
                Wg[g] = float(w[sel].sum())
        W = float(Wg.sum())
        q = (Wg + c.alpha) / (W + c.alpha * geo.G)
        s = c.target_pi / (q + 1e-30)

        # Choose an over-weighted and under-weighted group with swappable items
        candidates = list(range(geo.G))
        if not candidates:
            return mask

        g_over = int(np.argmin(s))
        g_under = int(np.argmax(s))

        best_mask = mask
        best_v = self.total_violation(mask)

        for g, mode in [(g_over, "reduce"), (g_under, "increase")]:
            idx_g = geo.group_to_indices[g]
            sel = idx_g[mask[idx_g]]
            unsel = idx_g[~mask[idx_g]]
            if sel.size == 0 or unsel.size == 0:
                continue
            if mode == "reduce":
                # swap out highest-weight selected, swap in lowest-weight unselected
                i_out = int(sel[np.argmax(w[sel])])
                i_in = int(unsel[np.argmin(w[unsel])])
            else:
                i_out = int(sel[np.argmin(w[sel])])
                i_in = int(unsel[np.argmax(w[unsel])])
            m2 = mask.copy()
            m2[i_out] = False
            m2[i_in] = True
            v2 = self.total_violation(m2)
            if v2 < best_v:
                best_v = v2
                best_mask = m2

        return best_mask
