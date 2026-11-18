"""Break-even arrival rates for opening lane c+1 (spec 4.6).

Lane c+1 pays off when  C_w * lam * Delta * [Wq(c; lam) - Wq(c+1; lam)] >= C_s.
The break-even rate lam* is where the two sides are equal. The function is
scanned over (0, c*mu) on a grid and every sign change is refined with brentq.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .queueing import mmc_metrics, mm1_split_metrics


def _wq(design: str, lam: float, mu: float, c: int, cs2: float) -> float:
    if design == "pooled":
        return mmc_metrics(lam, mu, c, cs2=cs2).Wq
    return mm1_split_metrics(lam, mu, c, cs2=cs2).Wq


def net_benefit(lam: float, c: int, mu: float, Cs: float, Cw: float, delta: float,
                design: str = "pooled", cs2: float = 1.0) -> float:
    """Waiting cost removed by lane c+1 minus its staffing cost (INR per block)."""
    return Cw * lam * delta * (_wq(design, lam, mu, c, cs2)
                               - _wq(design, lam, mu, c + 1, cs2)) - Cs


def breakeven_lambda(c: int, mu: float, Cs: float, Cw: float, delta: float,
                     design: str = "pooled", cs2: float = 1.0,
                     n_grid: int = 2000) -> list[float]:
    """All lam* in (0, c*mu) where opening lane c+1 exactly pays for itself."""
    hi = c * mu
    grid = np.linspace(hi * 1e-6, hi * (1 - 1e-9), n_grid)
    f = lambda l: net_benefit(l, c, mu, Cs, Cw, delta, design, cs2)
    vals = np.array([f(l) for l in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0:
            roots.append(float(grid[i]))
        elif a * b < 0:
            roots.append(float(brentq(f, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-12)))
    return roots


def breakeven_table(c_max: int, mu: float, Cs: float, Cw: float, delta: float,
                    profile=None, design: str = "pooled", cs2: float = 1.0) -> pd.DataFrame:
    """lam* for every step c -> c+1, with the blocks whose rate is above it."""
    rows = []
    for c in range(1, c_max):
        roots = breakeven_lambda(c, mu, Cs, Cw, delta, design, cs2)
        lam_star = roots[0] if roots else float("nan")
        row = {"from_c": c, "to_c": c + 1, "lambda_star": lam_star,
               "all_crossings": ", ".join(f"{r:.4f}" for r in roots) or "none",
               "stability_limit": c * mu}
        if profile is not None and roots:
            row["blocks_above"] = int(sum(1 for l in profile if lam_star <= l < c * mu))
        rows.append(row)
    return pd.DataFrame(rows)


def net_benefit_curve(c: int, mu: float, Cs: float, Cw: float, delta: float,
                      design: str = "pooled", cs2: float = 1.0, n: int = 300) -> pd.DataFrame:
    lams = np.linspace(c * mu * 0.005, c * mu * 0.995, n)
    return pd.DataFrame({"lam": lams,
                         "net_benefit": [net_benefit(l, c, mu, Cs, Cw, delta, design, cs2)
                                         for l in lams]})
