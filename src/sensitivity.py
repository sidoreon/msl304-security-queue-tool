"""One-at-a-time sensitivity, two-way grid and tornado chart data (spec 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .staffing import Params, optimize_staffing, baseline_plan, saving


def _run(profile, P: Params, design: str) -> dict:
    opt = optimize_staffing(profile, P, design)
    base = baseline_plan(profile, P, design)
    tc = opt["total_cost"]
    return {"total_cost": tc if np.isfinite(tc) else np.nan,
            "peak_lanes": max(opt["lanes"]),
            "saving_pct": 100 * saving(opt, base),
            "infeasible_blocks": opt["n_infeasible"]}


def _apply(name: str, value: float, profile, P: Params):
    prof = np.asarray(profile, float)
    if name == "arrival_scale":
        return prof * value, P
    if name == "mu":
        return prof, P.copy(mu=value)
    if name == "T":
        return prof, P.copy(T=value)
    if name == "alpha":
        return prof, P.copy(alpha=value)
    if name == "cost_ratio":           # multiplier on the base C_w / C_s ratio
        return prof, P.copy(Cw=P.Cw * value)
    if name == "p":
        return prof, P.copy(p=value)
    if name == "cs2":
        return prof, P.copy(cs2=value)
    if name == "c_max":
        return prof, P.copy(c_max=int(value))
    raise KeyError(name)


def default_ranges(P: Params) -> dict:
    return {
        "arrival_scale": list(np.round(np.linspace(0.6, 1.4, 9), 3)),
        "mu": list(np.round(P.mu * np.linspace(0.8, 1.2, 9), 4)),
        "T": [P.T * f for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)],
        "alpha": [0.01, 0.025, 0.05, 0.10, 0.15, 0.20],
        "cost_ratio": [0.25, 0.5, 1.0, 2.0, 4.0],
        "p": [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
        "cs2": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        "c_max": list(range(max(1, P.c_max - 3), P.c_max + 4)),
    }


LABELS = {
    "arrival_scale": "Arrival scale (x base profile)",
    "mu": "Service rate mu (pax/min)",
    "T": "Target wait T (min)",
    "alpha": "Violation probability alpha",
    "cost_ratio": "Cost ratio Cw/Cs (x base)",
    "p": "Priority share p",
    "cs2": "Service variability cs2",
    "c_max": "Lane limit c_max",
}


def one_way(profile, P: Params, design: str, name: str, values=None) -> pd.DataFrame:
    values = default_ranges(P)[name] if values is None else values
    rows = []
    for v in values:
        prof, Q = _apply(name, v, profile, P)
        r = _run(prof, Q, design)
        r.update(parameter=name, value=v)
        rows.append(r)
    return pd.DataFrame(rows)[["parameter", "value", "total_cost", "peak_lanes",
                               "saving_pct", "infeasible_blocks"]]


def all_one_way(profile, P: Params, design: str) -> pd.DataFrame:
    return pd.concat([one_way(profile, P, design, n) for n in default_ranges(P)],
                     ignore_index=True)


def two_way_grid(profile, P: Params, design: str,
                 scales=(0.6, 0.8, 1.0, 1.2, 1.4),
                 cw_values=None) -> pd.DataFrame:
    cw_values = cw_values or [P.Cw * f for f in (0.25, 0.5, 1, 2, 4)]
    rows = []
    for s in scales:
        for cw in cw_values:
            r = _run(np.asarray(profile, float) * s, P.copy(Cw=cw), design)
            rows.append({"arrival_scale": s, "Cw": cw, "total_cost": r["total_cost"],
                         "peak_lanes": r["peak_lanes"]})
    return pd.DataFrame(rows)


def tornado(profile, P: Params, design: str) -> pd.DataFrame:
    lows_highs = {
        "arrival_scale": (0.8, 1.2),
        "mu": (P.mu * 0.8, P.mu * 1.2),
        "T": (P.T * 0.5, P.T * 1.5),
        "alpha": (max(P.alpha * 0.5, 1e-3), min(P.alpha * 2, 0.99)),
        "cost_ratio": (0.5, 2.0),
        "p": (max(P.p - 0.1, 0.0), min(P.p + 0.1, 1.0)),
        "cs2": (max(P.cs2 * 0.5, 0.0), P.cs2 * 2 if P.cs2 > 0 else 1.0),
        "c_max": (max(P.c_max - 2, 1), P.c_max + 2),
    }
    base = _run(profile, P, design)["total_cost"]
    rows = []
    for n, (lo, hi) in lows_highs.items():
        pl, Ql = _apply(n, lo, profile, P)
        ph, Qh = _apply(n, hi, profile, P)
        cl = _run(pl, Ql, design)["total_cost"]
        ch = _run(ph, Qh, design)["total_cost"]
        rows.append({"parameter": n, "label": LABELS[n], "low_value": lo, "high_value": hi,
                     "cost_low": cl, "cost_high": ch, "base_cost": base,
                     "swing": abs(ch - cl) if np.isfinite(cl) and np.isfinite(ch) else np.nan})
    return pd.DataFrame(rows).sort_values("swing", ascending=False, na_position="first")
