"""Priority (fast-lane) designs analysed with M/G/1 formulas (spec 4.4).

Class k has mean service time 1/mu_k and second moment E[S_k^2] = (1 + cs2)/mu_k^2.

P1  dedicated fast lane: one M/G/1 lane for priority passengers, the other
    c-1 lanes serve regular passengers as a pooled M/M/(c-1) queue.
P2  non-preemptive priority inside each of c separate lanes (Cobham's formula).

Tail probabilities are not available in closed form; they come from simulation.
"""
from __future__ import annotations

from .queueing import INF, mmc_metrics


def second_moment(mu: float, cs2: float) -> float:
    return (1.0 + cs2) / mu ** 2


def mg1_wait(lam: float, ES: float, ES2: float) -> float:
    """Pollaczek-Khinchine mean wait in queue of an M/G/1 queue."""
    rho = lam * ES
    if lam <= 0:
        return 0.0
    if rho >= 1:
        return INF
    return lam * ES2 / (2.0 * (1.0 - rho))


def _mixture(p: float, mu_p: float, mu_r: float, cs2: float):
    ES = p / mu_p + (1 - p) / mu_r
    ES2 = p * second_moment(mu_p, cs2) + (1 - p) * second_moment(mu_r, cs2)
    return ES, ES2


def no_priority_reference(lam: float, p: float, mu_p: float, mu_r: float, c: int,
                          cs2: float = 1.0) -> dict:
    """Mean wait when the same passengers are served without any priority.

    pooled: one FIFO line feeding c lanes (M/M/c at the mixture service rate),
    separate: c FIFO M/G/1 lanes, each with 1/c of all passengers.
    """
    ES, ES2 = _mixture(p, mu_p, mu_r, cs2)
    mu_mix = 1.0 / ES
    cs2_mix = ES2 / ES ** 2 - 1.0
    pooled = mmc_metrics(lam, mu_mix, c, cs2=cs2_mix).Wq
    separate = mg1_wait(lam / c, ES, ES2)
    return {"pooled": pooled, "separate": separate}


def fast_lane_metrics(lam: float, p: float, mu_p: float, mu_r: float, c: int,
                      cs2: float = 1.0) -> dict:
    """Design P1: one dedicated priority lane plus a pooled M/M/(c-1) regular queue."""
    c = int(c)
    if c < 2:
        raise ValueError("the fast-lane design needs c >= 2")
    lam_p, lam_r = p * lam, (1 - p) * lam
    rho_p = lam_p / mu_p
    Wq_p = mg1_wait(lam_p, 1.0 / mu_p, second_moment(mu_p, cs2))
    reg = mmc_metrics(lam_r, mu_r, c - 1, cs2=cs2)
    Wq_r = reg.Wq
    stable = rho_p < 1 and reg.stable
    Wq = p * Wq_p + (1 - p) * Wq_r if stable else INF
    ref = no_priority_reference(lam, p, mu_p, mu_r, c, cs2)
    return {
        "design": "fast_lane", "c": c, "lam_p": lam_p, "lam_r": lam_r,
        "rho_priority": rho_p, "rho_regular": reg.rho,
        "rho": (lam_p / mu_p + lam_r / mu_r) / c,
        "Wq_priority": Wq_p, "Wq_regular": Wq_r, "Wq": Wq, "stable": stable,
        "Wq_ref": ref["pooled"],
        "regular_change": Wq_r - ref["pooled"] if stable else INF,
    }


def cobham_metrics(lam: float, p: float, mu_p: float, mu_r: float, c: int,
                   cs2: float = 1.0) -> dict:
    """Design P2: non-preemptive priority inside each of c separate lanes."""
    c = int(c)
    l1, l2 = p * lam / c, (1 - p) * lam / c
    rho1, rho2 = l1 / mu_p, l2 / mu_r
    W0 = (l1 * second_moment(mu_p, cs2) + l2 * second_moment(mu_r, cs2)) / 2.0
    stable = rho1 + rho2 < 1
    if stable:
        W1 = W0 / (1 - rho1)
        W2 = W0 / ((1 - rho1) * (1 - rho1 - rho2))
        Wq = p * W1 + (1 - p) * W2
    else:
        W1 = W0 / (1 - rho1) if rho1 < 1 else INF
        W2 = Wq = INF
    ref = no_priority_reference(lam, p, mu_p, mu_r, c, cs2)
    return {
        "design": "inlane_priority", "c": c, "lam_p": p * lam, "lam_r": (1 - p) * lam,
        "rho_priority": rho1, "rho_regular": rho2, "rho": rho1 + rho2, "W0": W0,
        "Wq_priority": W1, "Wq_regular": W2, "Wq": Wq, "stable": stable,
        "Wq_ref": ref["separate"],
        "regular_change": W2 - ref["separate"] if stable else INF,
    }
