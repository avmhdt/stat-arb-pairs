"""AC-W-06: Kalman filter against a slow reference, recursive OLS, a beta step and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from statarb.models.kalman import formation_start, kalman_filter, kalman_hedge, resume


def slow_reference(y, x, delta, obs_var, theta0, p0):  # type: ignore[no-untyped-def]
    """Textbook matrix form, one step at a time."""
    w = delta / (1 - delta) * np.eye(2)
    theta, p = theta0.astype(float).copy(), p0.astype(float).copy()
    out = []
    for yt, xt in zip(y, x, strict=True):
        r = p + w
        f = np.array([1.0, xt])
        e = yt - f @ theta
        q = f @ r @ f + obs_var
        k = r @ f / q
        theta = theta + k * e
        p = r - np.outer(k, f) @ r
        out.append((theta[0], theta[1], e, q))
    return np.array(out)


def test_matches_slow_reference() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 500)) + 3.5
    y = 0.2 + 1.3 * x + rng.normal(0, 0.01, 500)
    theta0, p0, v = formation_start(y[:250], x[:250])
    fast = kalman_hedge(y, x, 1e-4, v, theta0, p0)
    ref = slow_reference(y, x, 1e-4, v, theta0, p0)
    for column, series in enumerate((fast.alpha, fast.beta, fast.innovation, fast.innovation_var)):
        assert np.allclose(series, ref[:, column], rtol=1e-10, atol=1e-12)
    assert np.allclose(fast.z, ref[:, 2] / np.sqrt(ref[:, 3]), rtol=1e-10)


def test_small_delta_converges_to_recursive_ols() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.02, 300)) + 4.0
    y = -0.5 + 0.8 * x + rng.normal(0, 0.01, 300)
    result = kalman_hedge(y, x, 1e-14, 1e-4, np.zeros(2), 1e8 * np.eye(2))
    for t in (20, 100, 299):
        design = np.column_stack([np.ones(t + 1), x[: t + 1]])
        coef, *_ = np.linalg.lstsq(design, y[: t + 1], rcond=None)
        fitted = result.alpha[t] + result.beta[t] * x[: t + 1]
        assert np.allclose(fitted, design @ coef, atol=1e-8)
        # alpha and beta are nearly collinear when log x barely moves, so loosely.
        assert result.beta[t] == pytest.approx(coef[1], abs=1e-3)


def test_tracks_a_beta_step() -> None:
    rng = np.random.default_rng(2)
    n = 400
    x = np.cumsum(rng.normal(0, 0.02, n)) + 3.0
    beta = np.where(np.arange(n) < 200, 1.0, 1.5)
    y = 0.1 + beta * x + rng.normal(0, 0.005, n)
    theta0, p0, v = formation_start(y[:200], x[:200])
    result = kalman_hedge(y, x, 1e-3, v, theta0, p0)
    assert abs(result.beta[260] - 1.5) < 0.1


def test_correctly_specified_innovations_are_standard_normal() -> None:
    rng = np.random.default_rng(3)
    n, delta, v = 10_000, 1e-4, 1e-4
    w = delta / (1 - delta)
    theta = np.array([0.2, 1.1]) + np.cumsum(rng.normal(0, np.sqrt(w), (n, 2)), axis=0)
    x = 3.0 + np.cumsum(rng.normal(0, 0.01, n))
    y = theta[:, 0] + theta[:, 1] * x + rng.normal(0, np.sqrt(v), n)
    result = kalman_hedge(y, x, delta, v, np.array([0.2, 1.1]), np.eye(2) * 1e-6)
    z = result.z[100:]
    assert abs(float(z.mean())) < 0.05
    assert 0.9 <= float(z.var()) <= 1.1


def test_nan_skips_update_and_resume_continues() -> None:
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(0, 0.01, 200)) + 3.0
    y = 1.2 * x + rng.normal(0, 0.01, 200)
    theta0, p0, v = formation_start(y[:100], x[:100])
    whole, _ = kalman_filter(y, x, 1e-4, v, theta0, p0)
    _, state = kalman_filter(y[:120], x[:120], 1e-4, v, theta0, p0)
    rest, _ = resume(state, y[120:], x[120:])
    assert np.allclose(rest.z, whole.z[120:], rtol=1e-12)
    gappy = y.copy()
    gappy[50] = np.nan
    result, _ = kalman_filter(gappy, x, 1e-4, v, theta0, p0)
    assert np.isnan(result.z[50]) and result.beta[50] == result.beta[49]


@pytest.mark.parametrize(("delta", "obs_var"), [(0.0, 1.0), (1.0, 1.0), (0.1, 0.0)])
def test_rejects_bad_settings(delta: float, obs_var: float) -> None:
    with pytest.raises(ValueError):
        kalman_filter(np.ones(3), np.ones(3), delta, obs_var, np.zeros(2), np.eye(2))


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        kalman_filter(np.ones(3), np.ones(4), 0.1, 1.0, np.zeros(2), np.eye(2))
