"""Standard textbook values for M/M/1 and M/M/c, and a server-count cost trade-off.

Rates here are per hour; the formulas do not depend on the time unit.
"""
import pytest

from src.queueing import mmc_metrics


def test_mm1_reference_values():
    # lambda = 12/hr, mu = 15/hr: Lq = 3.2, L = 4, Wq = 16 min, W = 20 min
    m = mmc_metrics(12, 15, 1)
    assert m.rho == pytest.approx(0.8)
    assert m.Lq == pytest.approx(3.2)
    assert m.Lq + 12 / 15 == pytest.approx(4.0)
    assert m.Wq * 60 == pytest.approx(16.0)
    assert (m.Wq + 1 / 15) * 60 == pytest.approx(20.0)


def test_mm1_wait_target():
    # Wq = 7.14 min with mu = 6/hr (target 5 min missed); 4 min with mu = 7.5/hr
    assert mmc_metrics(2.5, 6, 1).Wq * 60 == pytest.approx(7.14, abs=0.01)
    assert mmc_metrics(2.5, 7.5, 1).Wq * 60 == pytest.approx(4.0)


def test_mm2_reference_values():
    # lambda = 10/hr, mu = 6/hr, c = 2: P0 = 1/11, Lq = 3.79, L = 5.45, Wq = 22.7 min
    m = mmc_metrics(10, 6, 2)
    a = 10 / 6
    p0 = m.p_wait / (a ** 2 / (2 * (1 - a / 2)))
    assert p0 == pytest.approx(1 / 11)
    assert m.Lq == pytest.approx(3.79, abs=0.01)
    assert m.Lq + a == pytest.approx(5.45, abs=0.01)
    assert m.Wq * 60 == pytest.approx(22.7, abs=0.1)


def test_server_count_cost_tradeoff():
    # lambda = 45/hr, mu = 50/hr per server, waiting $15/hr, server $10/hr.
    # Total cost/hr = 45 * W * 15 + 10 * c: about 145.0, 36.9, 43.9 -> two servers is cheapest.
    totals = {}
    for c in (1, 2, 3):
        m = mmc_metrics(45, 50, c)
        W = m.Wq + 1 / 50
        totals[c] = 45 * W * 15 + 10 * c
    assert totals[1] == pytest.approx(145.0)
    assert totals[2] == pytest.approx(36.9, abs=0.05)
    assert totals[3] == pytest.approx(43.9, abs=0.1)
    assert min(totals, key=totals.get) == 2
    assert mmc_metrics(45, 50, 2).Lq == pytest.approx(0.2285, abs=1e-4)
