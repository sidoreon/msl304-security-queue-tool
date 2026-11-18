"""Input checks (spec 7): errors block a run, warnings are shown to the user."""
from __future__ import annotations

import math


def validate_inputs(profile, P, design: str | None = None) -> tuple[list[str], list[str]]:
    errors, warnings = [], []

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    lams = [num(x) for x in profile]
    if len(lams) == 0:
        errors.append("The arrival profile is empty.")
    for i, l in enumerate(lams, start=1):
        if math.isnan(l):
            errors.append(f"Block {i}: arrival rate is not a number.")
        elif l < 0:
            errors.append(f"Block {i}: arrival rate cannot be negative.")
    if not P.delta > 0:
        errors.append("Block length must be > 0.")
    if not P.mu > 0:
        errors.append("Service rate mu must be > 0.")
    if P.mu_p is not None and not P.mu_p > 0:
        errors.append("Priority service rate must be > 0.")
    if P.cs2 < 0:
        errors.append("Service variability cs2 must be >= 0.")
    if int(P.c_max) != P.c_max or P.c_max < 1:
        errors.append("Lane limit c_max must be an integer >= 1.")
    if P.Cs < 0 or P.Cw < 0 or P.Csw < 0:
        errors.append("Costs must be >= 0.")
    if not P.T > 0:
        errors.append("Target wait T must be > 0.")
    if not (0 < P.alpha < 1):
        errors.append("Violation probability alpha must be between 0 and 1.")
    if not (0 <= P.p <= 1):
        errors.append("Priority share p must be between 0 and 1.")
    if int(P.c0) != P.c0 or P.c0 < 1:
        errors.append("Initial open lanes c0 must be an integer >= 1.")
    if P.max_step is not None and P.max_step < 1:
        errors.append("Lane-change limit must be >= 1.")
    if not errors:
        cap = P.c_max * P.mu
        bad = [i for i, l in enumerate(lams, start=1) if l >= cap]
        if bad:
            warnings.append(f"Blocks {bad} have lambda >= c_max x mu = {cap:g}: "
                            "no number of lanes can make them stable.")
        if design in ("fast_lane", "inlane_priority") and P.c_max < 2:
            warnings.append("A priority design was selected with c = 1; "
                            "the fast-lane design needs at least 2 lanes.")
    return errors, warnings
