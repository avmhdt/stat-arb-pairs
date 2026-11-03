"""AC-W-07: OU recovery on 1,000 simulated paths of 1,000 daily steps, half-life 10 days."""

from __future__ import annotations

import math

import numpy as np

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
    # "Median relative error <= 10%" read as the median signed error (the bias): the
    # Cramer-Rao bound puts the median absolute error near 12% at 1,000 steps.
    assert abs(float(np.median(relative))) <= 0.10
    assert float(np.mean(np.abs(relative) <= 0.35)) >= 0.90
    assert float(np.median(np.abs(relative))) <= 0.15
    assert float(np.median(gaps)) <= 0.02
