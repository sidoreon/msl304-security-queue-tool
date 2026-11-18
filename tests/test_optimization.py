"""Staffing optimization and break-even tests from spec section 10."""
import numpy as np
import pytest

from src.staffing import (Params, optimize_staffing, optimize_staffing_mip,
                          baseline_plan, compare_designs, evaluate_block)
from src.breakeven import breakeven_lambda, net_benefit
from src.sample_data import sample_profile
from src.validation import validate_inputs

PROFILE = sample_profile().lam.values


@pytest.mark.parametrize("design", ["pooled", "separate", "fast_lane", "inlane_priority"])
@pytest.mark.parametrize("metric", ["mean", "tail"])
def test_enumeration_equals_dp_when_no_switching(design, metric):
    P = Params(metric=metric, T=3.0)
    e = optimize_staffing(PROFILE, P, design, method="enumeration")
    d = optimize_staffing(PROFILE, P, design, method="dp")
    assert e["lanes"] == d["lanes"]
    assert e["total_cost"] == pytest.approx(d["total_cost"])


@pytest.mark.parametrize("seed", range(20))
def test_dp_equals_mip_random_profiles(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(6, 16))
    prof = rng.uniform(0.2, 15.0, K)
    P = Params(Csw=float(rng.uniform(0, 600)), metric=str(rng.choice(["mean", "tail"])),
               T=float(rng.uniform(1, 10)), c0=int(rng.integers(1, 4)),
               max_step=None if seed % 3 else 3)
    design = ["pooled", "separate"][seed % 2]
    dp = optimize_staffing(prof, P, design, method="dp")
    mip = optimize_staffing_mip(prof, P, design)
    assert dp["total_cost"] == pytest.approx(mip["total_cost"], rel=1e-7)


def test_plan_meets_target_and_beats_baseline():
    P = Params()
    opt = optimize_staffing(PROFILE, P, "pooled")
    base = baseline_plan(PROFILE, P, "pooled")
    assert opt["n_infeasible"] == 0
    assert (opt["plan"]["Wq"] <= P.T).all()
    assert (opt["plan"]["rho"] < 1).all()
    assert opt["total_cost"] <= base["total_cost"]


def test_infeasible_block_flagged_not_hidden():
    P = Params(c_max=3)
    opt = optimize_staffing([1.0, 5.9, 7.0], P, "pooled")
    assert opt["lanes"][1] == 3 and opt["lanes"][2] == 3
    assert opt["plan"]["infeasible"].tolist()[2]
    assert opt["plan"]["unstable"].tolist()[2]


def test_waiting_cost_includes_block_length():
    P = Params(delta=30)
    e = evaluate_block("pooled", 6.0, 4, P)
    assert e["wait_cost"] == pytest.approx(P.Cw * 6.0 * 30 * e["Wq"])


def test_compare_designs_recommends_pooled_by_default():
    df, rec, _ = compare_designs(PROFILE, Params())
    assert rec == "pooled"


@pytest.mark.parametrize("c", range(1, 9))
def test_breakeven_root_is_zero(c):
    for root in breakeven_lambda(c, 2.0, 400.0, 5.0, 30.0):
        assert abs(net_benefit(root, c, 2.0, 400.0, 5.0, 30.0)) < 1e-6


def test_validation_messages():
    P = Params(c_max=2)
    errors, warnings = validate_inputs([1, -1], P)
    assert errors
    errors, warnings = validate_inputs([1, 5], P)
    assert not errors and warnings
