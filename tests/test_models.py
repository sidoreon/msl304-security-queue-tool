"""Analytical tests from spec section 10."""
import math

import numpy as np
import pytest

from src.queueing import mmc_metrics, mm1_split_metrics, pooling_benefit, erlang_c
from src.priority import mg1_wait, fast_lane_metrics, cobham_metrics, second_moment


def test_mm1_known_values():
    m = mmc_metrics(0.5, 1.0, 1)
    assert m.rho == pytest.approx(0.5)
    assert m.Wq == pytest.approx(1.0)
    assert m.Lq == pytest.approx(0.5)


def test_mm2_known_values():
    m = mmc_metrics(1.0, 1.0, 2)
    assert m.p_wait == pytest.approx(1 / 3)
    assert m.Lq == pytest.approx(1 / 3)
    assert m.Wq == pytest.approx(1 / 3)


def test_pooling_hand_check():
    b = pooling_benefit(1.0, 1.0, 2)
    assert b["Wq_separate"] == pytest.approx(1.0)
    assert b["Wq_pooled"] == pytest.approx(1 / 3)
    assert b["minutes_saved"] == pytest.approx(2 / 3)
    assert b["percent_saved"] == pytest.approx(66.6667, abs=1e-3)


@pytest.mark.parametrize("lam,mu,c", [(0.5, 1, 1), (1, 1, 2), (7, 2, 5), (11.7, 2, 7), (3, 0.8, 6)])
def test_littles_law_pooled_separate(lam, mu, c):
    for m in (mmc_metrics(lam, mu, c), mm1_split_metrics(lam, mu, c)):
        assert m.Lq == pytest.approx(lam * m.Wq)


@pytest.mark.parametrize("design", ["fast", "cobham"])
def test_littles_law_priority(design):
    lam, p, mu = 6.0, 0.2, 2.0
    r = fast_lane_metrics(lam, p, mu, mu, 4) if design == "fast" else cobham_metrics(lam, p, mu, mu, 4)
    Lq_classes = r["lam_p"] * r["Wq_priority"] + r["lam_r"] * r["Wq_regular"]
    assert Lq_classes == pytest.approx(lam * r["Wq"])


@pytest.mark.parametrize("lam,mu", [(0.3, 1), (1.5, 2), (0.9, 1)])
def test_c1_pooled_equals_separate(lam, mu):
    a, b = mmc_metrics(lam, mu, 1), mm1_split_metrics(lam, mu, 1)
    assert a.Wq == pytest.approx(b.Wq)
    assert a.Lq == pytest.approx(b.Lq)
    for t in (0, 1, 5):
        assert a.tail(t) == pytest.approx(b.tail(t))


@pytest.mark.parametrize("lam,mu", [(0.5, 1), (1.2, 2), (1.9, 2)])
def test_pk_equals_mm1_when_cs2_is_1(lam, mu):
    assert mg1_wait(lam, 1 / mu, second_moment(mu, 1.0)) == pytest.approx(mmc_metrics(lam, mu, 1).Wq)


@pytest.mark.parametrize("p", [0.05, 0.1, 0.3, 0.6])
@pytest.mark.parametrize("cs2", [0.5, 1.0, 2.0])
def test_priority_conservation(p, cs2):
    lam, mu, c = 6.0, 2.0, 4
    r = cobham_metrics(lam, p, mu, mu, c, cs2)
    weighted = p * r["Wq_priority"] + (1 - p) * r["Wq_regular"]
    single = mg1_wait(lam / c, 1 / mu, second_moment(mu, cs2))
    assert weighted == pytest.approx(single)


@pytest.mark.parametrize("lam,mu", [(5.0, 2.0), (11.7, 2.0), (0.8, 1.0)])
def test_wq_strictly_decreasing_in_c(lam, mu):
    c0 = math.floor(lam / mu) + 1
    for f in (mmc_metrics, mm1_split_metrics):
        w = [f(lam, mu, c).Wq for c in range(c0, c0 + 10)]
        assert all(a > b for a, b in zip(w, w[1:]))


def test_tail_formula_matches_definition():
    lam, mu, c = 9.0, 2.0, 6
    m = mmc_metrics(lam, mu, c)
    assert m.tail(0) == pytest.approx(m.p_wait)
    assert m.tail(2.0) == pytest.approx(m.p_wait * math.exp(-(c * mu - lam) * 2.0))
    # integral of the tail equals the mean wait
    ts = np.linspace(0, 20, 200001)
    trap = getattr(np, "trapezoid", None) or np.trapz
    assert trap([m.tail(t) for t in ts], ts) == pytest.approx(m.Wq, rel=1e-4)


def test_erlang_c_large_c_no_overflow():
    pw = erlang_c(190.0, 200)
    assert 0 < pw < 1


def test_unstable_reports_inf():
    assert not mmc_metrics(4.0, 2.0, 2).stable
    assert math.isinf(mm1_split_metrics(4.0, 2.0, 2).Wq)


def test_fast_lane_needs_two_lanes():
    with pytest.raises(ValueError):
        fast_lane_metrics(1.0, 0.1, 2.0, 2.0, 1)
