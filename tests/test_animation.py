"""The animation simulator should agree with the formulas on long runs."""
import pytest

from src import animation as anim
from src.queueing import mmc_metrics, mm1_split_metrics
from src.priority import fast_lane_metrics, cobham_metrics


@pytest.fixture(scope="module")
def pax():
    return anim.make_passengers(9.0, 2.0, 40000, p=0.1, seed=3)


@pytest.mark.parametrize("design,formula", [
    ("pooled", lambda: mmc_metrics(9.0, 2.0, 6).Wq),
    ("separate", lambda: mm1_split_metrics(9.0, 2.0, 6).Wq),
    ("fast_lane", lambda: fast_lane_metrics(9.0, 0.1, 2.0, 2.0, 6)["Wq"]),
    ("inlane_priority", lambda: cobham_metrics(9.0, 0.1, 2.0, 2.0, 6)["Wq"]),
])
def test_long_run_matches_formula(pax, design, formula):
    sim = anim.average_waits(design, pax, 6, warmup=500)["all"]
    assert sim == pytest.approx(formula(), rel=0.1)


def test_snapshots_shape(pax):
    snaps, groups = anim.snapshots("separate", pax, 4, [10, 20])
    assert len(snaps) == 2 and len(groups) == 4
    assert all(len(s["lines"]) == 4 for s in snaps)
