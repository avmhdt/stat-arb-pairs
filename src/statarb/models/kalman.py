"""Kalman-filter hedge ratio (FR-12). Filtering only, never smoothing (spec section 16).

State ``theta = [alpha, beta]`` follows a random walk with covariance
``W = delta / (1 - delta) I``; the observation is ``log y = alpha + beta log x + e`` with
``var(e) = V``. Each step predicts with the previous state, so the innovation ``e_t`` and
its variance ``Q_t`` use nothing dated after ``t``:

    R = P + W,  F = [1, log x_t],  e = log y_t - F theta,  Q = F R F' + V,
    K = R F' / Q,  theta <- theta + K e,  P <- R - K F R.

``z_t = e_t / sqrt(Q_t)`` is the trading signal. A NaN observation skips the update and
carries the state forward with ``P`` grown by ``W``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class KalmanResult:
    """Filtered series, one value per observation."""

    alpha: F64
    beta: F64
    innovation: F64
    innovation_var: F64
    z: F64


@dataclass(frozen=True, slots=True)
class KalmanState:
    """The filter's state after the last observation: what crosses ``T`` (spec Figure 1)."""

    theta: tuple[float, float]
    cov: tuple[float, float, float, float]  # row-major 2 x 2
    delta: float
    obs_var: float


@njit(cache=True, fastmath=False)
def _filter(
    log_y: F64, log_x: F64, delta: float, obs_var: float, theta0: F64, p0: F64
) -> tuple[F64, F64, F64, F64, F64, F64]:
    n = log_y.size
    w = delta / (1.0 - delta)
    a0, b0 = theta0[0], theta0[1]
    p00, p01, p10, p11 = p0[0, 0], p0[0, 1], p0[1, 0], p0[1, 1]
    alpha = np.empty(n)
    beta = np.empty(n)
    innov = np.empty(n)
    qvar = np.empty(n)
    for t in range(n):
        r00, r01, r10, r11 = p00 + w, p01, p10, p11 + w
        x = log_x[t]
        y = log_y[t]
        if math.isnan(x) or math.isnan(y):
            p00, p01, p10, p11 = r00, r01, r10, r11
            alpha[t], beta[t] = a0, b0
            innov[t] = np.nan
            qvar[t] = np.nan
            continue
        e = y - (a0 + b0 * x)
        rf0 = r00 + r01 * x  # R F'
        rf1 = r10 + r11 * x
        q = rf0 + x * rf1 + obs_var
        k0, k1 = rf0 / q, rf1 / q
        a0 += k0 * e
        b0 += k1 * e
        fr0 = r00 + x * r10  # F R
        fr1 = r01 + x * r11
        p00, p01 = r00 - k0 * fr0, r01 - k0 * fr1
        p10, p11 = r10 - k1 * fr0, r11 - k1 * fr1
        alpha[t], beta[t] = a0, b0
        innov[t] = e
        qvar[t] = q
    final = np.array([a0, b0])
    cov = np.array([[p00, p01], [p10, p11]])
    return alpha, beta, innov, qvar, final, cov


def kalman_filter(
    log_y: F64, log_x: F64, delta: float, obs_var: float, theta0: F64, P0: F64
) -> tuple[KalmanResult, KalmanState]:
    """Run the filter and return the series and the final state."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if obs_var <= 0.0:
        raise ValueError("obs_var must be positive")
    y = np.ascontiguousarray(log_y, dtype=np.float64)
    x = np.ascontiguousarray(log_x, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("log_y and log_x differ in length")
    alpha, beta, innov, qvar, final, cov = _filter(
        y,
        x,
        float(delta),
        float(obs_var),
        np.ascontiguousarray(theta0, dtype=np.float64),
        np.ascontiguousarray(P0, dtype=np.float64),
    )
    result = KalmanResult(alpha, beta, innov, qvar, innov / np.sqrt(qvar))
    state = KalmanState(
        (float(final[0]), float(final[1])),
        (float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 0]), float(cov[1, 1])),
        float(delta),
        float(obs_var),
    )
    return result, state


def kalman_hedge(
    log_y: F64, log_x: F64, delta: float, obs_var: float, theta0: F64, P0: F64
) -> KalmanResult:
    """The spec's interface: filtered ``alpha``, ``beta``, innovations and ``z``."""
    return kalman_filter(log_y, log_x, delta, obs_var, theta0, P0)[0]


def resume(state: KalmanState, log_y: F64, log_x: F64) -> tuple[KalmanResult, KalmanState]:
    """Continue filtering from a frozen state, for the trading window."""
    theta = np.array(state.theta)
    cov = np.array(state.cov).reshape(2, 2)
    return kalman_filter(log_y, log_x, state.delta, state.obs_var, theta, cov)


def formation_start(log_y: F64, log_x: F64) -> tuple[F64, F64, float]:
    """OLS initial state, its covariance and the residual variance from formation data.

    ``P0`` is the OLS coefficient covariance ``s^2 (X'X)^-1``, computed through QR.
    """
    design = np.column_stack([np.ones_like(log_x), log_x])
    q, r = np.linalg.qr(design)
    theta = np.linalg.solve(r, q.T @ log_y)
    resid = log_y - design @ theta
    obs_var = float(resid @ resid) / (log_y.size - 2)
    rinv = np.linalg.inv(r)
    return theta, obs_var * (rinv @ rinv.T), obs_var
