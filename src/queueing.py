"""Pooled M/M/c, separate c x M/M/1 queues and the pooling benefit (spec 4.1-4.3).

All rates are per minute; all times are in minutes.

When the squared coefficient of variation of service time ``cs2`` differs from 1
the separate layout becomes c independent M/G/1 queues and is solved exactly with
the Pollaczek-Khinchine formula. The pooled M/G/c queue has no closed form, so the
Allen-Cunneen approximation Wq ~ Wq(M/M/c) * (1 + cs2) / 2 is used. With the
default cs2 = 1 both reduce to the exact M/M/c and M/M/1 results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

INF = float("inf")


@dataclass
class QueueMetrics:
    """Steady-state metrics of one design for one arrival rate."""

    lam: float
    mu: float
    c: int
    rho: float
    p_wait: float
    Lq: float
    Wq: float
    tail: Callable[[float], float] = field(repr=False)
    stable: bool = True
    per_lane: list | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {"lam": self.lam, "mu": self.mu, "c": self.c, "rho": self.rho,
                "p_wait": self.p_wait, "Lq": self.Lq, "Wq": self.Wq,
                "stable": self.stable}


def _unstable(lam: float, mu: float, c: int, rho: float) -> QueueMetrics:
    return QueueMetrics(lam, mu, c, rho, 1.0, INF, INF, lambda t: 1.0, stable=False)


def erlang_c(a: float, c: int) -> float:
    """Probability that an arrival waits in M/M/c with offered load a = lam/mu.

    The sum is built iteratively (term_n = term_{n-1} * a / n) so factorials
    never overflow for large c.
    """
    rho = a / c
    if rho >= 1:
        return 1.0
    term = 1.0          # a^0 / 0!
    s = 1.0
    for n in range(1, c):
        term *= a / n
        s += term
    term_c = term * a / c  # a^c / c!
    last = term_c / (1.0 - rho)
    p0 = 1.0 / (s + last)
    return last * p0


def mmc_metrics(lam: float, mu: float, c: int, cs2: float = 1.0) -> QueueMetrics:
    """Pooled queue: one FIFO line feeding c identical lanes.

    Returns rho, p_wait, Lq, Wq and tail(t) = P(wait > t).
    """
    c = int(c)
    if c < 1 or mu <= 0:
        raise ValueError("need c >= 1 and mu > 0")
    rho = lam / (c * mu)
    if lam <= 0:
        return QueueMetrics(0.0, mu, c, 0.0, 0.0, 0.0, 0.0, lambda t: 0.0)
    if rho >= 1:
        return _unstable(lam, mu, c, rho)
    a = lam / mu
    pw = erlang_c(a, c)
    factor = (1.0 + cs2) / 2.0
    Lq = pw * rho / (1.0 - rho) * factor
    Wq = Lq / lam
    if abs(cs2 - 1.0) < 1e-12:
        decay = c * mu - lam
    else:  # mean-matched exponential tail (approximation)
        decay = pw / Wq if Wq > 0 else INF

    def tail(t: float, pw=pw, decay=decay) -> float:
        if t < 0:
            return 1.0
        return pw * math.exp(-decay * t)

    return QueueMetrics(lam, mu, c, rho, pw, Lq, Wq, tail)


def _mg1_lane(lam_i: float, mu: float, cs2: float):
    rho_i = lam_i / mu
    if lam_i <= 0:
        return rho_i, 0.0, (lambda t: 0.0)
    if rho_i >= 1:
        return rho_i, INF, (lambda t: 1.0)
    # P-K: Wq = lam E[S^2] / (2(1-rho)), E[S^2] = (1+cs2)/mu^2
    wq = lam_i * (1.0 + cs2) / mu ** 2 / (2.0 * (1.0 - rho_i))
    if abs(cs2 - 1.0) < 1e-12:
        decay = mu - lam_i
    else:
        decay = rho_i / wq
    return rho_i, wq, (lambda t, r=rho_i, d=decay: 1.0 if t < 0 else r * math.exp(-d * t))


def mm1_split_metrics(lam: float, mu: float, c: int,
                      weights: Sequence[float] | None = None,
                      cs2: float = 1.0) -> QueueMetrics:
    """Separate layout: c independent single-lane queues, random routing by weights."""
    c = int(c)
    if weights is None:
        weights = [1.0 / c] * c
    if len(weights) != c:
        raise ValueError("weights must have length c")
    tot = float(sum(weights))
    w = [x / tot for x in weights]
    lanes = []
    for wi in w:
        rho_i, wq_i, tail_i = _mg1_lane(wi * lam, mu, cs2)
        lanes.append({"weight": wi, "lam": wi * lam, "rho": rho_i, "Wq": wq_i, "tail": tail_i})
    rho = lam / (c * mu)
    if lam <= 0:
        return QueueMetrics(0.0, mu, c, 0.0, 0.0, 0.0, 0.0, lambda t: 0.0, per_lane=lanes)
    if any(l["rho"] >= 1 for l in lanes if l["weight"] > 0):
        m = _unstable(lam, mu, c, rho)
        m.per_lane = lanes
        return m
    Wq = sum(l["weight"] * l["Wq"] for l in lanes)
    p_wait = sum(l["weight"] * l["rho"] for l in lanes)

    def tail(t: float) -> float:
        return sum(l["weight"] * l["tail"](t) for l in lanes)

    return QueueMetrics(lam, mu, c, rho, p_wait, lam * Wq, Wq, tail, per_lane=lanes)


def pooling_benefit(lam: float, mu: float, c: int, cs2: float = 1.0) -> dict:
    """Minutes and percent of mean wait saved by pooling, for the same c and mu."""
    sep = mm1_split_metrics(lam, mu, c, cs2=cs2)
    poo = mmc_metrics(lam, mu, c, cs2=cs2)
    if not (sep.stable and poo.stable):
        return {"Wq_separate": sep.Wq, "Wq_pooled": poo.Wq,
                "minutes_saved": float("nan"), "percent_saved": float("nan")}
    saved = sep.Wq - poo.Wq
    pct = 100.0 * saved / sep.Wq if sep.Wq > 0 else 0.0
    return {"Wq_separate": sep.Wq, "Wq_pooled": poo.Wq,
            "minutes_saved": saved, "percent_saved": pct}


def meets_target(m: QueueMetrics, metric: str, T: float, alpha: float) -> bool:
    if not m.stable:
        return False
    if metric == "mean":
        return m.Wq <= T + 1e-12
    return m.tail(T) <= alpha + 1e-12


def lanes_needed(design: str, lam: float, mu: float, c_max: int, metric: str,
                 T: float, alpha: float, cs2: float = 1.0) -> int | None:
    """Smallest c (<= c_max) whose design meets the wait target, else None."""
    f = mmc_metrics if design == "pooled" else (
        lambda l, m, c, cs2=1.0: mm1_split_metrics(l, m, c, cs2=cs2))
    for c in range(1, c_max + 1):
        if meets_target(f(lam, mu, c, cs2=cs2), metric, T, alpha):
            return c
    return None


def lanes_saved(lam: float, mu: float, c_max: int, metric: str, T: float,
                alpha: float, cs2: float = 1.0):
    s = lanes_needed("separate", lam, mu, c_max, metric, T, alpha, cs2)
    p = lanes_needed("pooled", lam, mu, c_max, metric, T, alpha, cs2)
    if s is None or p is None:
        return None
    return s - p
