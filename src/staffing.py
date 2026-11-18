"""Staffing over the day at the lowest total cost (spec 4.5).

cost_t(c) = C_s * c + C_w * lambda_t * Delta * Wq(c)
minimize   sum_t cost_t(c_t) + C_sw * sum_t max(0, c_t - c_{t-1})
subject to the wait target, rho < 1 and 1 <= c_t <= c_max in every block.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .queueing import INF, mmc_metrics, mm1_split_metrics
from .priority import fast_lane_metrics, cobham_metrics

DESIGNS = ["pooled", "separate", "fast_lane", "inlane_priority"]
DESIGN_LABELS = {
    "pooled": "Pooled queue (M/M/c)",
    "separate": "Separate queues (c x M/M/1)",
    "fast_lane": "Priority P1: dedicated fast lane",
    "inlane_priority": "Priority P2: priority inside each lane",
}


@dataclass
class Params:
    delta: float = 30.0          # block length, minutes
    mu: float = 2.0              # service rate of one lane, passengers/min
    cs2: float = 1.0             # squared CV of service time
    c_max: int = 10
    Cs: float = 400.0            # INR per lane per block
    Cw: float = 5.0              # INR per passenger-minute of waiting
    T: float = 10.0              # target wait, minutes
    alpha: float = 0.10          # allowed P(wait > T)
    metric: str = "mean"         # "mean" -> Wq <= T ; "tail" -> P(W > T) <= alpha
    p: float = 0.10              # priority share
    mu_p: float | None = None    # priority service rate (None -> mu)
    Csw: float = 0.0             # cost to open one extra lane between blocks
    c0: int = 1                  # lanes open before the first block
    max_step: int | None = None  # optional |c_t - c_{t-1}| <= max_step

    @property
    def mu_priority(self) -> float:
        return self.mu if self.mu_p is None else self.mu_p

    def copy(self, **kw) -> "Params":
        return replace(self, **kw)


def min_lanes(design: str) -> int:
    return 2 if design == "fast_lane" else 1


def evaluate_block(design: str, lam: float, c: int, P: Params) -> dict:
    """All metrics and the cost of running `design` with c lanes at arrival rate lam."""
    out = {"c": c, "lam": lam}
    if design == "pooled":
        m = mmc_metrics(lam, P.mu, c, cs2=P.cs2)
    elif design == "separate":
        m = mm1_split_metrics(lam, P.mu, c, cs2=P.cs2)
    if design in ("pooled", "separate"):
        out.update(rho=m.rho, Wq=m.Wq, Lq=m.Lq, p_wait=m.p_wait, stable=m.stable,
                   tail_T=m.tail(P.T) if m.stable else 1.0,
                   Wq_priority=np.nan, Wq_regular=np.nan)
        if P.metric == "mean":
            meets = m.stable and m.Wq <= P.T + 1e-12
        else:
            meets = m.stable and out["tail_T"] <= P.alpha + 1e-12
    else:
        if design == "fast_lane" and c < 2:
            r = {"rho": lam / (c * P.mu), "Wq": INF, "stable": False,
                 "Wq_priority": INF, "Wq_regular": INF}
        elif design == "fast_lane":
            r = fast_lane_metrics(lam, P.p, P.mu_priority, P.mu, c, P.cs2)
        else:
            r = cobham_metrics(lam, P.p, P.mu_priority, P.mu, c, P.cs2)
        out.update(rho=r["rho"], Wq=r["Wq"], Lq=lam * r["Wq"] if r["stable"] else INF,
                   p_wait=np.nan, stable=r["stable"], tail_T=np.nan,
                   Wq_priority=r["Wq_priority"], Wq_regular=r["Wq_regular"])
        # Priority designs: mean metric analytically, for both classes.
        meets = r["stable"] and max(r["Wq_priority"], r["Wq_regular"]) <= P.T + 1e-12
    out["meets_target"] = bool(meets)
    out["staff_cost"] = P.Cs * c
    out["wait_cost"] = P.Cw * lam * P.delta * out["Wq"] if out["stable"] else INF
    if lam <= 0:
        out["wait_cost"] = 0.0
    out["cost"] = out["staff_cost"] + out["wait_cost"]
    return out


def _cost_table(profile: Sequence[float], P: Params, design: str):
    K, C = len(profile), P.c_max
    evals = [[None] * (C + 1) for _ in range(K)]
    cost = np.full((K, C + 1), INF)
    allowed = np.zeros((K, C + 1), dtype=bool)
    infeasible = np.zeros(K, dtype=bool)
    for t, lam in enumerate(profile):
        for c in range(1, C + 1):
            e = evaluate_block(design, float(lam), c, P)
            evals[t][c] = e
            cost[t, c] = e["cost"]
            allowed[t, c] = e["meets_target"] and c >= min_lanes(design)
        if not allowed[t].any():
            infeasible[t] = True
            allowed[t, C] = True
            if not np.isfinite(cost[t, C]):     # unstable: surrogate (staff only)
                cost[t, C] = evals[t][C]["staff_cost"]
    return evals, cost, allowed, infeasible


def _switch(c_new: int, c_old: int, P: Params) -> float:
    return P.Csw * max(0, c_new - c_old)


def _build_result(profile, P: Params, design: str, plan, evals, infeasible, method,
                  notes=None) -> dict:
    rows, prev = [], P.c0
    for t, (lam, c) in enumerate(zip(profile, plan)):
        e = evals[t][c]
        sw = _switch(c, prev, P)
        prev = c
        rows.append({
            "block": t + 1, "lam": float(lam), "c": int(c), "rho": e["rho"],
            "Wq": e["Wq"], "P(wait>T)": e["tail_T"],
            "Wq_priority": e["Wq_priority"], "Wq_regular": e["Wq_regular"],
            "staff_cost": e["staff_cost"], "wait_cost": e["wait_cost"],
            "switch_cost": sw, "cost": e["staff_cost"] + e["wait_cost"] + sw,
            "infeasible": bool(infeasible[t]),
            "unstable": not e["stable"],
            "pax_wait_min": float(lam) * P.delta * e["Wq"] if e["stable"] else INF,
        })
    df = pd.DataFrame(rows)
    notes = list(notes or [])
    if design in ("fast_lane", "inlane_priority") and P.metric == "tail":
        notes.append("Priority designs use the mean-wait target analytically; "
                     "check the tail in the Simulation tab.")
    return {
        "design": design, "method": method, "plan": df,
        "lanes": [int(c) for c in plan],
        "total_cost": float(df["cost"].sum()),
        "staff_cost": float(df["staff_cost"].sum()),
        "wait_cost": float(df["wait_cost"].sum()),
        "switch_cost": float(df["switch_cost"].sum()),
        "pax_wait_min": float(df["pax_wait_min"].sum()),
        "n_infeasible": int(df["infeasible"].sum()),
        "n_unstable": int(df["unstable"].sum()),
        "notes": notes,
    }


def _dp(cost, allowed, P: Params, step):
    K, C = cost.shape[0], P.c_max
    V = np.full((K, C + 1), INF)
    arg = np.zeros((K, C + 1), dtype=int)
    for c in range(1, C + 1):
        if allowed[0, c] and (step is None or abs(c - P.c0) <= step):
            V[0, c] = cost[0, c] + _switch(c, P.c0, P)
    for t in range(1, K):
        for c in range(1, C + 1):
            if not allowed[t, c]:
                continue
            best, bi = INF, 0
            for cp in range(1, C + 1):
                if not np.isfinite(V[t - 1, cp]):
                    continue
                if step is not None and abs(c - cp) > step:
                    continue
                v = V[t - 1, cp] + _switch(c, cp, P)
                if v < best - 1e-12:
                    best, bi = v, cp
            if np.isfinite(best):
                V[t, c] = cost[t, c] + best
                arg[t, c] = bi
    if not np.isfinite(V[-1]).any():
        return None
    c = int(np.argmin(V[-1]))
    plan = [c]
    for t in range(K - 1, 0, -1):
        c = int(arg[t, c])
        plan.append(c)
    return plan[::-1]


def optimize_staffing(profile: Sequence[float], params: Params, design: str = "pooled",
                      method: str = "auto") -> dict:
    """Lanes per block, costs and flags.

    method "auto": exact enumeration when C_sw = 0 and no step limit, otherwise DP.
    """
    P = params
    evals, cost, allowed, infeasible = _cost_table(profile, P, design)
    notes = []
    if method == "auto":
        method = "enumeration" if (P.Csw == 0 and P.max_step is None) else "dp"
    if method == "enumeration":
        plan = []
        for t in range(len(profile)):
            cand = [c for c in range(1, P.c_max + 1) if allowed[t, c]]
            plan.append(min(cand, key=lambda c: (cost[t, c], c)))
    else:
        plan = _dp(cost, allowed, P, P.max_step)
        if plan is None:
            notes.append("The lane-change limit made the problem infeasible; "
                         "it was relaxed for this run.")
            plan = _dp(cost, allowed, P, None)
    return _build_result(profile, P, design, plan, evals, infeasible, method, notes)


def optimize_staffing_mip(profile: Sequence[float], params: Params,
                          design: str = "pooled") -> dict:
    """Same output as optimize_staffing, solved as an integer program with PuLP.

    Like the DP, an infeasible lane-change limit is relaxed and noted.
    """
    try:
        return _mip(profile, params, design)
    except RuntimeError:
        if params.max_step is None:
            raise
        r = _mip(profile, params.copy(max_step=None), design)
        r["notes"].append("The lane-change limit made the problem infeasible; "
                          "it was relaxed for this run.")
        return r


def _mip(profile: Sequence[float], params: Params, design: str) -> dict:
    import pulp

    P = params
    evals, cost, allowed, infeasible = _cost_table(profile, P, design)
    K, C = len(profile), P.c_max
    prob = pulp.LpProblem("staffing", pulp.LpMinimize)
    if hasattr(prob, "add_variable"):            # PuLP >= 4
        newvar = prob.add_variable
    else:                                         # PuLP 2.x / 3.x
        newvar = pulp.LpVariable
    x = {(t, c): newvar(f"x_{t}_{c}", lowBound=0, upBound=1, cat="Binary")
         for t in range(K) for c in range(1, C + 1)}
    u = {t: newvar(f"u_{t}", lowBound=0) for t in range(K)}
    ct = {t: pulp.lpSum(c * x[t, c] for c in range(1, C + 1)) for t in range(K)}
    big = 0.0
    for t in range(K):
        prob += pulp.lpSum(x[t, c] for c in range(1, C + 1)) == 1
        for c in range(1, C + 1):
            if not allowed[t, c]:
                prob += x[t, c] == 0
        prev = P.c0 if t == 0 else ct[t - 1]
        prob += u[t] >= ct[t] - prev
        if P.max_step is not None:
            prob += ct[t] - prev <= P.max_step
            prob += prev - ct[t] <= P.max_step
    obj = []
    for t in range(K):
        for c in range(1, C + 1):
            if allowed[t, c]:
                obj.append(float(cost[t, c]) * x[t, c])
    prob += pulp.lpSum(obj) + P.Csw * pulp.lpSum(u.values())
    if hasattr(pulp, "PULP_CBC_CMD"):
        solver = pulp.PULP_CBC_CMD(msg=0)
    else:                                         # PuLP >= 4 ships without CBC
        solver = pulp.HiGHS(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MIP status {pulp.LpStatus[status]}")
    plan = []
    for t in range(K):
        plan.append(max(range(1, C + 1), key=lambda c: x[t, c].value() or 0))
    return _build_result(profile, P, design, plan, evals, infeasible, "mip")


def baseline_lanes(profile: Sequence[float], P: Params, design: str = "pooled") -> int:
    """Smallest c that meets the target in the peak block (c_max if none does)."""
    peak = float(np.max(profile))
    for c in range(min_lanes(design), P.c_max + 1):
        if evaluate_block(design, peak, c, P)["meets_target"]:
            return c
    return P.c_max


def baseline_plan(profile: Sequence[float], P: Params, design: str = "pooled",
                  c_fixed: int | None = None) -> dict:
    """Fixed staffing: the same number of lanes in every block."""
    c = baseline_lanes(profile, P, design) if c_fixed is None else int(c_fixed)
    evals = [[None] * (P.c_max + 1) for _ in profile]
    infeasible = np.zeros(len(profile), dtype=bool)
    for t, lam in enumerate(profile):
        evals[t][c] = evaluate_block(design, float(lam), c, P)
        infeasible[t] = not evals[t][c]["meets_target"]
    return _build_result(profile, P, design, [c] * len(profile), evals, infeasible,
                         "baseline")


def saving(opt: dict, base: dict) -> float:
    b, o = base["total_cost"], opt["total_cost"]
    if not (np.isfinite(b) and np.isfinite(o)) or b == 0:
        return float("nan")
    return (b - o) / b


def compare_designs(profile: Sequence[float], P: Params, c_fixed: int | None = None):
    """Optimize every design and recommend the cheapest one that meets the target."""
    rows, results = [], {}
    for d in DESIGNS:
        opt = optimize_staffing(profile, P, d)
        base = baseline_plan(profile, P, d, c_fixed)
        results[d] = (opt, base)
        rows.append({
            "design": d, "label": DESIGN_LABELS[d],
            "total_cost": opt["total_cost"], "baseline_cost": base["total_cost"],
            "saving_pct": 100 * saving(opt, base),
            "peak_lanes": max(opt["lanes"]), "lane_blocks": sum(opt["lanes"]),
            "pax_wait_min": opt["pax_wait_min"],
            "infeasible_blocks": opt["n_infeasible"],
        })
    df = pd.DataFrame(rows)
    order = df.sort_values(["infeasible_blocks", "total_cost"])
    return df, str(order.iloc[0]["design"]), results


def cost_curve(lam: float, P: Params, design: str = "pooled") -> pd.DataFrame:
    """Staffing, waiting and total cost against c for one block (trade-off curve)."""
    rows = []
    for c in range(min_lanes(design), P.c_max + 1):
        e = evaluate_block(design, lam, c, P)
        rows.append({"c": c, "staff_cost": e["staff_cost"], "wait_cost": e["wait_cost"],
                     "total_cost": e["cost"], "Wq": e["Wq"], "meets_target": e["meets_target"]})
    return pd.DataFrame(rows)
