"""AC-W-07: OU recovery on 1,000 simulated paths of 1,000 daily steps, half-life 10 days."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statarb.models.ou import fit_ou


def test_half_life_recovery_and_ar1_vs_mle() -> None:
    rng = np.random.default_rng(7)
    true = 10.0
    b = 2.0 ** (-1.0 / true)
    n, paths = 1000, 1000
    shocks = rng.normal(0.0, 0.01, (paths, n))
    x = np.empty((paths, n))
    x[:, 0] = 0.3 + shocks[:, 0] / math.sqrt(1 - b * b)
    for t in range(1, n):
        x[:, t] = 0.3 + b * (x[:, t - 1] - 0.3) + shocks[:, t]
    ar1, gaps = [], []
    for path in x:
        fast = fit_ou(path, method="ar1")
        exact = fit_ou(path, method="mle")
        assert fast is not None and exact is not None
        ar1.append(fast.half_life_days)
        gaps.append(abs(exact.half_life_days - fast.half_life_days) / fast.half_life_days)
    relative = np.array(ar1) / true - 1.0
    assert float(np.mean(np.abs(relative) <= 0.35)) >= 0.90
    assert float(np.median(gaps)) <= 0.02
    # Both readings of "median relative error", reported, neither claimed as the AC's pass.
    assert -0.06 <= float(np.median(relative)) <= -0.02  # signed: about -3.9%
    assert 0.10 <= float(np.median(np.abs(relative))) <= 0.14  # absolute: about 12.2%


def _cramer_rao_median_abs(half_life: float, n: int) -> float:
    """Median |relative error| of the half-life implied by the AR(1) slope's Cramer-Rao bound."""
    b = 2.0 ** (-1.0 / half_life)
    sd_b = math.sqrt((1.0 - b * b) / n)
    slope = math.log(2.0) / (b * math.log(b) ** 2)  # d half-life / d b
    return 0.6745 * slope * sd_b / half_life


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC-W-07, owner ruling 2026-11-03: accepted as a finding, no pass claimed. The median "
        "absolute half-life error is 12.2%; the Cramer-Rao floor is near 12% at 1,000 steps, "
        "below the 10% target for any estimator."
    ),
)
def test_median_absolute_error_target() -> None:
    floor = _cramer_rao_median_abs(10.0, 1000)
    assert 0.11 <= floor <= 0.13, floor
    assert floor <= 0.10
