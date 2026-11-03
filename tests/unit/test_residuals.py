from __future__ import annotations

import math

import numpy as np
import pytest

from statarb.models.residuals import (
    eigenportfolios,
    etf_residuals,
    half_life_days,
    ou_params,
    pca_residuals,
    regression_residuals,
    s_score,
    s_scores,
)


def test_eigenportfolios_recover_a_strong_factor() -> None:
    rng = np.random.default_rng(0)
    factor = rng.normal(0, 0.01, 300)
    returns = factor[:, None] * rng.uniform(0.5, 1.5, 40) + rng.normal(0, 0.003, (300, 40))
    weights, eig = eigenportfolios(returns, 3)
    assert weights.shape == (40, 3) and eig[0] > 10 * eig[1]
    assert abs(np.corrcoef(returns @ weights[:, 0], factor)[0, 1]) > 0.99
    by_share, _ = eigenportfolios(returns, None, variance_explained=0.5)
    assert by_share.shape[1] == 1


def test_regression_and_pca_residuals() -> None:
    rng = np.random.default_rng(1)
    f = rng.normal(0, 0.01, (100, 2))
    noise = rng.normal(0, 0.001, (100, 5))
    stocks = f @ np.array([[1.0, 0.5, 0.2, 1.2, 0.8], [0.3, -0.2, 1.0, 0.0, 0.5]]) + noise
    resid, betas = regression_residuals(stocks, f)
    assert np.allclose(betas, [[1.0, 0.5, 0.2, 1.2, 0.8], [0.3, -0.2, 1.0, 0.0, 0.5]], atol=0.05)
    assert np.allclose(resid.sum(axis=0), 0.0, atol=1e-12)
    assert pca_residuals(np.vstack([stocks] * 3), 252, 2, 60).shape == (60, 5)
    etf = etf_residuals(stocks, np.repeat(f[:, :1], 5, axis=1))
    assert etf.shape == (60, 5)


def test_ou_params_and_s_scores() -> None:
    rng = np.random.default_rng(2)
    b = math.exp(-25 / 252)
    x = np.zeros((2000, 3))
    for t in range(1, 2000):
        x[t] = b * x[t - 1] + rng.normal(0, 0.01, 3)
    kappa, _, sigma_eq = ou_params(x)
    assert np.allclose(kappa, 25, rtol=0.35)
    assert np.allclose(sigma_eq, 0.01 / math.sqrt(1 - b * b), rtol=0.1)
    walk = np.cumsum(np.ones((50, 1)), axis=0)
    assert np.isnan(ou_params(walk)[0][0])
    resid = np.diff(np.vstack([np.zeros((1, 3)), x[-61:]]), axis=0)[-60:]
    s, k = s_scores(resid, kappa_min=8.4)
    assert s.shape == (3,) and np.isfinite(k).all()
    strict, _ = s_scores(resid, kappa_min=1e9)
    assert np.isnan(strict).all()
    assert s_score(np.cumsum(resid, axis=0), 8.4).shape == (3,)
    assert half_life_days(25.0) == pytest.approx(math.log(2) / 25 * 252)
    assert half_life_days(0.0) == math.inf
