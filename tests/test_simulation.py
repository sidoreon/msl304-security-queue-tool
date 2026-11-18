"""Simulation against formulas, 5 parameter sets (spec section 10 and 5).

The default spec run uses 100 replications; set SIM_REPS to change it.
"""
import os

import numpy as np
import pytest

from src.queueing import mmc_metrics, mm1_split_metrics
from src.priority import fast_lane_metrics, cobham_metrics
from src.simulation import simulate_block, simulate_replication, passes

REPS = int(os.environ.get("SIM_REPS", "100"))

CASES = [
    # design, lam, mu, c, T
    ("pooled", 1.0, 1.0, 2, 1.0),
    ("pooled", 11.7, 2.0, 7, 1.0),
    ("separate", 6.0, 2.0, 4, 2.0),
    ("fast_lane", 8.0, 2.0, 5, 1.0),
    ("inlane_priority", 5.0, 2.0, 4, 2.0),
]


def analytic(design, lam, mu, c, T, p=0.1):
    if design == "pooled":
        m = mmc_metrics(lam, mu, c)
        return {"mean_wait": m.Wq, "tail": m.tail(T)}
    if design == "separate":
        m = mm1_split_metrics(lam, mu, c)
        return {"mean_wait": m.Wq, "tail": m.tail(T)}
    r = fast_lane_metrics(lam, p, mu, mu, c) if design == "fast_lane" else cobham_metrics(lam, p, mu, mu, c)
    return {"mean_wait": r["Wq"], "mean_wait_priority": r["Wq_priority"],
            "mean_wait_regular": r["Wq_regular"]}


@pytest.mark.parametrize("design,lam,mu,c,T", CASES)
def test_simulation_matches_formulas(design, lam, mu, c, T):
    sim = simulate_block(design, lam, mu, c, T=T, reps=REPS, seed=7)
    a = analytic(design, lam, mu, c, T)
    for key, val in a.items():
        assert passes(val, sim[key], sim[key + "_lo"], sim[key + "_hi"], abs_tol=0.005), \
            (key, val, sim[key], sim[key + "_lo"], sim[key + "_hi"])
    assert sim["util"] == pytest.approx(lam / (c * mu), rel=0.05)


@pytest.mark.parametrize("design", ["pooled", "separate"])
def test_fast_engine_equals_simpy(design):
    a = simulate_replication(design, 7.0, 2.0, 4, seed=3)
    b = simulate_replication(design, 7.0, 2.0, 4, seed=3, engine="fast")
    np.testing.assert_allclose(a["waits"], b["waits"], atol=1e-9)


def test_seed_reproducible():
    a = simulate_block("pooled", 5.0, 2.0, 3, reps=3, seed=11)
    b = simulate_block("pooled", 5.0, 2.0, 3, reps=3, seed=11)
    assert a["mean_wait"] == b["mean_wait"]


@pytest.mark.parametrize("stress", ["groups", "two_state", "lane_closure", "jsq"])
def test_stress_tests_run(stress):
    design = "separate" if stress == "jsq" else "pooled"
    r = simulate_block(design, 6.0, 2.0, 4, reps=3, stress=stress, seed=5)
    assert np.isfinite(r["mean_wait"]) and r["mean_wait"] >= 0


def test_jsq_beats_random_routing():
    rnd = simulate_block("separate", 6.0, 2.0, 4, reps=10, seed=1)
    jsq = simulate_block("separate", 6.0, 2.0, 4, reps=10, seed=1, stress="jsq")
    pooled = mmc_metrics(6.0, 2.0, 4).Wq
    assert pooled < jsq["mean_wait"] < rnd["mean_wait"]
