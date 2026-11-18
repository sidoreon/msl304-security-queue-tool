"""Assumed arrival profile with a morning and an evening peak (spec 7).

lambda(h) = 2 + 10 exp(-(h-8)^2 / (2*1.2^2)) + 8 exp(-(h-18)^2 / (2*1.5^2))   [passengers/min]

This shape is ASSUMED for illustration; it is not measured data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def arrival_rate(h):
    h = np.asarray(h, float)
    return (2 + 10 * np.exp(-(h - 8) ** 2 / (2 * 1.2 ** 2))
            + 8 * np.exp(-(h - 18) ** 2 / (2 * 1.5 ** 2)))


def sample_profile(start_hour: float = 5.0, end_hour: float = 23.0,
                   block_minutes: float = 30.0, scale: float = 1.0) -> pd.DataFrame:
    """One row per block; the rate is evaluated at the block midpoint."""
    n = int(round((end_hour - start_hour) * 60 / block_minutes))
    starts = start_hour + np.arange(n) * block_minutes / 60
    mids = starts + block_minutes / 120
    lam = np.round(scale * arrival_rate(mids), 3)

    def hhmm(h):
        return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"

    return pd.DataFrame({
        "block": np.arange(1, n + 1),
        "start": [hhmm(h) for h in starts],
        "end": [hhmm(h + block_minutes / 60) for h in starts],
        "lam": lam,
    })
