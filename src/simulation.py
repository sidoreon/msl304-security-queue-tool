"""Discrete-event simulation of the checkpoint with SimPy (spec 5).

Mode A  validation: simulate a design at one block's (lam, c), discard a warm-up,
        replicate, and report means with 95% confidence intervals.
Mode B  stress tests: bursty arrivals (groups or a two-state rate), a lane that
        closes for part of every block, and join-shortest-queue in the separate layout.

Service times are exponential when cs2 = 1, gamma with shape 1/cs2 otherwise
(deterministic when cs2 = 0). Every run stores its seed so it can be repeated.

A second engine ("fast") uses the Kiefer-Wolfowitz / Lindley recursions for the
FIFO designs (pooled, separate). It produces the same waits as SimPy for the same
random numbers and is used to cross-check SimPy and to speed up long runs.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import simpy
from scipy import stats

FIFO_DESIGNS = ("pooled", "separate")


@dataclass
class StressOptions:
    group_mean: float = 3.0       # mean group size for "groups"
    high_factor: float = 0.6      # rate = lam*(1 +/- high_factor) for "two_state"
    state_mean: float = 15.0      # mean minutes spent in each state
    closure_fraction: float = 0.25  # share of each block one lane is closed
    block_length: float = 30.0


def _service_sampler(rng: np.random.Generator, cs2: float):
    def draw(mu: float, n: int) -> np.ndarray:
        if cs2 <= 1e-12:
            return np.full(n, 1.0 / mu)
        if abs(cs2 - 1.0) < 1e-12:
            return rng.exponential(1.0 / mu, n)
        k = 1.0 / cs2
        return rng.gamma(k, 1.0 / (k * mu), n)
    return draw


def _arrival_times(rng, lam: float, n: int, stress: str | None, so: StressOptions):
    """Arrival epochs of n passengers (sorted)."""
    if stress == "groups":
        g = max(so.group_mean, 1.0)
        times, t = [], 0.0
        while len(times) < n:
            t += rng.exponential(g / lam)
            size = int(rng.geometric(1.0 / g))
            times.extend([t] * size)
        return np.array(times[:n])
    if stress == "two_state":
        hi, lo = lam * (1 + so.high_factor), lam * max(1 - so.high_factor, 1e-9)
        times, t = [], 0.0
        state_hi = bool(rng.integers(2))
        switch = rng.exponential(so.state_mean)
        while len(times) < n:
            rate = hi if state_hi else lo
            nxt = t + rng.exponential(1.0 / rate)
            if nxt > switch:              # memoryless: restart at the switch time
                t = switch
                state_hi = not state_hi
                switch = t + rng.exponential(so.state_mean)
                continue
            t = nxt
            times.append(t)
        return np.array(times)
    return np.cumsum(rng.exponential(1.0 / lam, n))


def _prepare(rng, design, lam, mu, c, cs2, p, mu_p, n, stress, so, weights):
    arr = _arrival_times(rng, lam, n, stress, so)
    is_pri = (rng.random(n) < p) if design in ("fast_lane", "inlane_priority") else np.zeros(n, bool)
    draw = _service_sampler(rng, cs2)
    svc = draw(mu, n)
    if is_pri.any():
        svc_p = draw(mu_p, n)
        svc = np.where(is_pri, svc_p, svc)
    w = np.full(c, 1.0 / c) if weights is None else np.asarray(weights, float) / np.sum(weights)
    route = rng.choice(c, size=n, p=w)
    return arr, is_pri, svc, route


def _run_simpy(design, c, arr, is_pri, svc, route, stress, so):
    env = simpy.Environment()
    n = len(arr)
    waits = np.full(n, np.nan)
    closure = stress == "lane_closure"

    def make(cap):
        return simpy.PriorityResource(env, capacity=cap)

    if design == "pooled":
        pools = [make(c)]
    elif design == "separate":
        pools = [make(1) for _ in range(c)]
    elif design == "fast_lane":
        pools = [make(1), make(c - 1)]     # 0: priority lane, 1: regular pool
    else:
        pools = [make(1) for _ in range(c)]

    def pick(i):
        if design == "pooled":
            return pools[0]
        if design == "fast_lane":
            return pools[0] if is_pri[i] else pools[1]
        if design == "separate" and stress == "jsq":
            loads = [len(r.queue) + r.count for r in pools]
            m = min(loads)
            best = [k for k, v in enumerate(loads) if v == m]
            return pools[best[int(route[i]) % len(best)]]
        return pools[int(route[i])]

    all_done = env.event()
    finished = [0]

    def passenger(i):
        res = pick(i)
        prio = 0 if (design == "inlane_priority" and is_pri[i]) else 1
        with res.request(priority=prio) as req:
            yield req
            waits[i] = env.now - arr[i]
            yield env.timeout(svc[i])
        finished[0] += 1
        if finished[0] == n:
            all_done.succeed()

    def source():
        for i in range(n):
            dt = arr[i] - env.now
            if dt > 0:
                yield env.timeout(dt)
            env.process(passenger(i))

    def blocker():
        target = pools[1] if design == "fast_lane" else pools[0]
        while True:
            yield env.timeout((1 - so.closure_fraction) * so.block_length)
            with target.request(priority=-1) as req:
                yield req
                yield env.timeout(so.closure_fraction * so.block_length)

    env.process(source())
    if closure:
        env.process(blocker())
    env.run(until=all_done)
    return waits


def _run_fast(design, c, arr, svc, route):
    """Exact FIFO recursions (same results as SimPy for pooled / separate)."""
    n = len(arr)
    waits = np.empty(n)
    if design == "pooled":
        free = [0.0] * c
        heapq.heapify(free)
        for i in range(n):
            f = heapq.heappop(free)
            start = f if f > arr[i] else arr[i]
            waits[i] = start - arr[i]
            heapq.heappush(free, start + svc[i])
    else:
        free = np.zeros(c)
        for i in range(n):
            k = route[i]
            start = free[k] if free[k] > arr[i] else arr[i]
            waits[i] = start - arr[i]
            free[k] = start + svc[i]
    return waits


def simulate_replication(design: str, lam: float, mu: float, c: int, cs2: float = 1.0,
                         T: float = 10.0, p: float = 0.1, mu_p: float | None = None,
                         n_warm: int = 500, n_rec: int = 5000, seed=None,
                         weights=None, stress: str | None = None,
                         stress_opts: StressOptions | None = None,
                         engine: str = "simpy") -> dict:
    so = stress_opts or StressOptions()
    mu_p = mu if mu_p is None else mu_p
    rng = np.random.default_rng(seed)
    n = n_warm + n_rec
    arr, is_pri, svc, route = _prepare(rng, design, lam, mu, c, cs2, p, mu_p, n,
                                       stress, so, weights)
    use_fast = engine == "fast" and design in FIFO_DESIGNS and stress in (None, "groups", "two_state")
    waits = _run_fast(design, c, arr, svc, route) if use_fast else \
        _run_simpy(design, c, arr, is_pri, svc, route, stress, so)
    w = waits[n_warm:]
    pri = is_pri[n_warm:]
    span = arr[-1] - arr[n_warm]
    util = svc[n_warm:].sum() / (c * span) if span > 0 else float("nan")
    return {
        "mean_wait": float(np.mean(w)),
        "tail": float(np.mean(w > T)),
        "util": float(util),
        "mean_wait_priority": float(np.mean(w[pri])) if pri.any() else float("nan"),
        "mean_wait_regular": float(np.mean(w[~pri])) if (~pri).any() else float("nan"),
        "waits": w,
    }


def _ci(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    h = float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / math.sqrt(len(x)))
    return m, m - h, m + h


def simulate_block(design: str, lam: float, mu: float, c: int, cs2: float = 1.0,
                   T: float = 10.0, p: float = 0.1, mu_p: float | None = None,
                   reps: int = 100, n_warm: int = 500, n_rec: int = 5000,
                   seed: int = 2026, weights=None, stress: str | None = None,
                   stress_opts: StressOptions | None = None, engine: str = "simpy",
                   keep_waits: int = 5) -> dict:
    """Replicate a block and return means with 95% confidence intervals."""
    if design == "fast_lane" and c < 2:
        raise ValueError("fast-lane design needs c >= 2")
    seeds = np.random.SeedSequence(seed).spawn(reps)
    runs = [simulate_replication(design, lam, mu, c, cs2, T, p, mu_p, n_warm, n_rec,
                                 s, weights, stress, stress_opts, engine) for s in seeds]
    out = {"design": design, "lam": lam, "mu": mu, "c": c, "reps": reps, "seed": seed,
           "stress": stress or "none", "engine": engine}
    for key in ("mean_wait", "tail", "util", "mean_wait_priority", "mean_wait_regular"):
        m, lo, hi = _ci([r[key] for r in runs])
        out[key], out[key + "_lo"], out[key + "_hi"] = m, lo, hi
    out["waits_sample"] = np.concatenate([r["waits"] for r in runs[:keep_waits]])
    return out


def passes(analytic: float, sim_mean: float, lo: float, hi: float,
           rel: float = 0.05, abs_tol: float = 0.0) -> bool:
    """Spec pass criterion: inside the CI, or within 5% of the simulated mean.

    abs_tol lets very small probabilities (e.g. P(wait > T) ~ 0.001) pass when the
    gap is negligible in absolute terms.
    """
    if not (np.isfinite(analytic) and np.isfinite(sim_mean)):
        return False
    if lo - 1e-12 <= analytic <= hi + 1e-12:
        return True
    if abs(analytic - sim_mean) <= rel * abs(sim_mean):
        return True
    return abs(analytic - sim_mean) <= abs_tol
