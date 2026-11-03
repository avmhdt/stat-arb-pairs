"""Ornstein-Uhlenbeck estimation (FR-13).

``dX = kappa (mu - X) dt + sigma dW`` sampled every ``dt`` years is the AR(1)
``X[t+1] = a + b X[t] + e`` with ``b = exp(-kappa dt)``, ``a = mu (1 - b)`` and
``var(e) = sigma^2 (1 - b^2) / (2 kappa)``. The stationary deviation is
``sigma_eq = sd(e) / sqrt(1 - b^2)`` and the half-life ``ln 2 / kappa`` years, which is
``ln 2 / -ln b`` sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OUParams:
    """Fitted OU parameters: ``kappa`` and ``sigma`` per year, ``half_life_days`` in sessions."""

    kappa: float
    mu: float
    sigma: float
    sigma_eq: float
    half_life_days: float


@dataclass(frozen=True, slots=True)
class OUFailure:
    """Why no OU fit exists: the AR(1) slope is outside (0, 1), or the data are unusable."""

    reason: str


def _from_ar1(a: float, b: float, resid_sd: float, dt: float) -> OUParams:
    kappa = -math.log(b) / dt
    sigma_eq = resid_sd / math.sqrt(1.0 - b * b)
    return OUParams(
        kappa=kappa,
        mu=a / (1.0 - b),
        sigma=sigma_eq * math.sqrt(2.0 * kappa),
        sigma_eq=sigma_eq,
        half_life_days=math.log(2.0) / -math.log(b),
    )


def ar1_fit(x: F64) -> tuple[float, float, float] | OUFailure:
    """OLS of ``x[t+1]`` on ``[1, x[t]]`` by least squares: ``(a, b, residual sd)``.

    The residual deviation uses ``n - 2`` degrees of freedom.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        return OUFailure("need at least 4 finite observations")
    if float(np.ptp(values)) <= 1e-12 * (1.0 + float(np.abs(values).max())):
        return OUFailure("constant series")
    design = np.column_stack([np.ones(values.size - 1), values[:-1]])
    coef, *_ = np.linalg.lstsq(design, values[1:], rcond=None)
    resid = values[1:] - design @ coef
    dof = resid.size - 2
    return float(coef[0]), float(coef[1]), float(math.sqrt(float(resid @ resid) / dof))


def _neg_loglik(theta: F64, x: F64) -> float:
    """Exact Gaussian likelihood of the AR(1), the first point from the stationary law."""
    b = math.tanh(float(theta[0]))
    mu = float(theta[1])
    sigma_eq = math.exp(float(theta[2]))
    if not 0.0 < b < 1.0:
        return 1e300
    step_var = sigma_eq * sigma_eq * (1.0 - b * b)
    resid = x[1:] - mu - b * (x[:-1] - mu)
    n = resid.size
    first = float((x[0] - mu) ** 2) / (sigma_eq * sigma_eq)
    return 0.5 * (
        (n + 1) * math.log(2.0 * math.pi)
        + 2.0 * math.log(sigma_eq)
        + first
        + n * math.log(step_var)
        + float(resid @ resid) / step_var
    )


def fit_ou(x: F64, dt: float = 1 / 252, method: Literal["ar1", "mle"] = "ar1") -> OUParams | None:
    """Fit an OU process to ``x``; ``None`` when ``b`` is not in ``(0, 1)``.

    Parameters
    ----------
    x
        The series, for example a spread, one value per session.
    dt
        Years per step.
    method
        ``ar1``: the OLS mapping above. ``mle``: the exact Gaussian likelihood including
        the first observation from the stationary law, started at the AR(1) estimate.
    """
    result = fit_ou_detail(x, dt, method)
    return result if isinstance(result, OUParams) else None


def fit_ou_detail(
    x: F64, dt: float = 1 / 252, method: Literal["ar1", "mle"] = "ar1"
) -> OUParams | OUFailure:
    """Like ``fit_ou``, but says why a fit failed."""
    fitted = ar1_fit(x)
    if isinstance(fitted, OUFailure):
        return fitted
    a, b, resid_sd = fitted
    if not 0.0 < b < 1.0:
        return OUFailure(f"AR(1) slope {b:.6g} is not in (0, 1)")
    if resid_sd <= 0.0:
        return OUFailure("zero residual variance")
    start = _from_ar1(a, b, resid_sd, dt)
    if method == "ar1":
        return start
    values = np.asarray(x, dtype=np.float64)
    theta0 = np.array([math.atanh(b), start.mu, math.log(start.sigma_eq)])
    found = minimize(
        _neg_loglik,
        theta0,
        args=(values,),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
    )
    b_hat = math.tanh(float(found.x[0]))
    if not 0.0 < b_hat < 1.0:
        return OUFailure(f"MLE slope {b_hat:.6g} is not in (0, 1)")
    mu_hat = float(found.x[1])
    sigma_eq = math.exp(float(found.x[2]))
    step_sd = sigma_eq * math.sqrt(1.0 - b_hat * b_hat)
    return _from_ar1(mu_hat * (1.0 - b_hat), b_hat, step_sd, dt)
