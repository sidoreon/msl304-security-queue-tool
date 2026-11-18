"""Simulate a short stretch of time and record who is where, for the queue animations.

Each design is a set of "groups". A group is one or more lanes with one shared line:

    pooled           one group with all c lanes
    separate         c groups of one lane; passengers pick a lane at random
    fast_lane        one lane for priority passengers + one group of c-1 lanes for the rest
    inlane_priority  c groups of one lane; priority passengers go to the front of their line

All designs get the same passengers, so differences come only from the layout.
"""
import heapq

import numpy as np


def make_passengers(lam, mu, minutes, p=0.1, mu_p=None, seed=0):
    rng = np.random.default_rng(seed)
    mu_p = mu if mu_p is None else mu_p
    n = int(lam * minutes * 1.4) + 30
    arrive = np.cumsum(rng.exponential(1 / lam, n))
    arrive = arrive[arrive < minutes]
    m = len(arrive)
    priority = rng.random(m) < p
    screen = np.where(priority, rng.exponential(1 / mu_p, m), rng.exponential(1 / mu, m))
    pick = rng.integers(0, 10 ** 6, m)
    return {"arrive": arrive, "screen": screen, "priority": priority, "pick": pick}


def groups_for(design, c):
    """List of groups: (lane numbers, uses priority order, who may join)."""
    if design == "pooled":
        return [(list(range(c)), False, "all")]
    if design == "separate":
        return [([k], False, "all") for k in range(c)]
    if design == "fast_lane":
        return [([0], False, "priority"), (list(range(1, c)), False, "regular")]
    return [([k], True, "all") for k in range(c)]


def _serve(idx, arrive, screen, priority, lanes, use_priority):
    """First come first served (or priority first) for one group. Returns start time and lane."""
    start, lane = {}, {}
    free = [(0.0, l) for l in lanes]
    heapq.heapify(free)
    waiting = []
    i = 0
    while i < len(idx) or waiting:
        f = free[0][0]
        if i < len(idx) and (arrive[idx[i]] <= f or not waiting):
            j = idx[i]
            key = (0 if (use_priority and priority[j]) else 1, arrive[j])
            heapq.heappush(waiting, (key, j))
            i += 1
            continue
        _, j = heapq.heappop(waiting)
        f, l = heapq.heappop(free)
        start[j] = max(f, arrive[j])
        lane[j] = l
        heapq.heappush(free, (start[j] + screen[j], l))
    return start, lane


def run(design, pax, c):
    """Start time, lane and group of every passenger."""
    a, s, pr, pick = pax["arrive"], pax["screen"], pax["priority"], pax["pick"]
    groups = groups_for(design, c)
    n = len(a)
    group_of = np.empty(n, dtype=int)
    for j in range(n):
        if design in ("separate", "inlane_priority"):
            group_of[j] = pick[j] % c
        elif design == "fast_lane":
            group_of[j] = 0 if pr[j] else 1
        else:
            group_of[j] = 0
    start, lane = np.empty(n), np.empty(n, dtype=int)
    for g, (lanes, use_pr, _) in enumerate(groups):
        idx = np.where(group_of == g)[0]
        st, ln = _serve(idx, a, s, pr, lanes, use_pr)
        for j in idx:
            start[j], lane[j] = st[j], ln[j]
    return start, lane, group_of, groups


def snapshots(design, pax, c, times):
    """For each time: the line in front of each group (priority flags, front first) and busy lanes."""
    a, s, pr = pax["arrive"], pax["screen"], pax["priority"]
    start, lane, group_of, groups = run(design, pax, c)
    end = start + s
    out = []
    for t in times:
        lines = []
        for g, (_, use_pr, _) in enumerate(groups):
            w = np.where((group_of == g) & (a <= t) & (start > t))[0]
            order = sorted(w, key=lambda j: ((0 if (use_pr and pr[j]) else 1), a[j]))
            lines.append([bool(pr[j]) for j in order])
        serving = np.where((start <= t) & (end > t))[0]
        busy = {int(lane[j]): bool(pr[j]) for j in serving}
        done = (start <= t)
        out.append({"t": float(t), "lines": lines, "busy": busy,
                    "avg_wait": float(np.mean((start - a)[done])) if done.any() else 0.0})
    return out, groups


def average_waits(design, pax, c, warmup=0.0):
    """Average wait overall, for priority and for regular passengers."""
    a, pr = pax["arrive"], pax["priority"]
    start, *_ = run(design, pax, c)
    keep = a >= warmup
    w = (start - a)[keep]
    p = pr[keep]
    return {"all": float(w.mean()),
            "priority": float(w[p].mean()) if p.any() else float("nan"),
            "regular": float(w[~p].mean()) if (~p).any() else float("nan")}
