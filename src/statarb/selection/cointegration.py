"""Engle-Granger and Johansen tests (FR-09, FR-10).

``engle_granger`` reproduces ``statsmodels.tsa.stattools.coint(y, x, trend="c")`` with an
explicit ``maxlag``: OLS of ``y`` on ``[x, 1]``, then an ADF regression without
deterministic terms on the residuals, the lag chosen by AIC (or BIC) over a common sample
with ties going to the shorter lag, rerun at the chosen lag on its full sample. The
p-value is MacKinnon's (2010) approximation for two variables with a constant. The batch
version runs the same code in Numba over many pairs.

All least squares go through QR or ``lstsq``, never the normal equations (NFR-04).
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.stats import norm
from statsmodels.tsa import adfvalues

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

SQRTEPS = math.sqrt(np.finfo(np.float64).eps)
_PENALTY = {"aic": 0, "bic": 1}

_N_VARS = 2
_TAU_MAX = float(adfvalues._tau_maxs["c"][_N_VARS - 1])
_TAU_MIN = float(adfvalues._tau_mins["c"][_N_VARS - 1])
_TAU_STAR = float(adfvalues._tau_stars["c"][_N_VARS - 1])
_TAU_SMALL = np.asarray(adfvalues._tau_smallps["c"][_N_VARS - 1], dtype=np.float64)
_TAU_LARGE = np.asarray(adfvalues._tau_largeps["c"][_N_VARS - 1], dtype=np.float64)


def default_maxlag(nobs: int) -> int:
    """``floor(12 (n / 100)^(1/4))``, capped at ``n // 2 - 1`` as ``adfuller`` requires."""
    root: float = (nobs / 100.0) ** 0.25
    return min(math.floor(12.0 * root), nobs // 2 - 1)


def mackinnon_pvalue(stat: F64) -> F64:
    """Vectorised ``statsmodels.tsa.adfvalues.mackinnonp(stat, "c", N=2)``."""
    values = np.asarray(stat, dtype=np.float64)
    small = np.polyval(_TAU_SMALL[::-1], values)
    large = np.polyval(_TAU_LARGE[::-1], values)
    with np.errstate(invalid="ignore"):
        p = np.where(values <= _TAU_STAR, norm.cdf(small), norm.cdf(large))
        p = np.where(values > _TAU_MAX, 1.0, p)
        p = np.where(values < _TAU_MIN, 0.0, p)
    return np.where(np.isnan(values), np.nan, p)


@njit(cache=True, fastmath=False)
def _adf_design(u: F64, lags: int) -> tuple[F64, F64]:
    """``adfuller``'s design for regression "n": lagged level, then ``lags`` lagged diffs."""
    n = u.size
    nobs = n - 1 - lags
    design = np.empty((nobs, lags + 1))
    target = np.empty(nobs)
    for r in range(nobs):
        t = lags + r
        design[r, 0] = u[t]
        target[r] = u[t + 1] - u[t]
        for j in range(1, lags + 1):
            design[r, j] = u[t + 1 - j] - u[t - j]
    return design, target


@njit(cache=True, fastmath=False)
def _adf_stat(u: F64, maxlag: int, autolag: int) -> tuple[float, int]:
    """ADF t-statistic without deterministic terms and the lag used.

    ``autolag``: 0 = AIC, 1 = BIC, -1 = use ``maxlag`` as given.
    """
    lag = maxlag
    if autolag >= 0:
        design, target = _adf_design(u, maxlag)
        nobs = target.size
        q, r = np.linalg.qr(design)
        z = np.ascontiguousarray(q.T) @ target
        fitted = np.ascontiguousarray(q) @ z
        ssr = 0.0
        for i in range(nobs):
            ssr += (target[i] - fitted[i]) ** 2
        best = np.inf
        for k in range(maxlag + 1, 0, -1):
            if k <= maxlag:
                ssr += z[k] ** 2
            penalty = 2.0 * k if autolag == 0 else math.log(nobs) * k
            crit = nobs * math.log(ssr / nobs) + penalty
            if crit <= best:
                best = crit
                lag = k - 1
    design, target = _adf_design(u, lag)
    q, r = np.linalg.qr(design)
    coef = np.linalg.solve(r, np.ascontiguousarray(q.T) @ target)
    resid = target - design @ coef
    dof = target.size - (lag + 1)
    scale = (resid @ resid) / dof
    rinv = np.linalg.inv(r)
    var0 = 0.0
    for j in range(lag + 1):
        var0 += rinv[0, j] ** 2
    return coef[0] / math.sqrt(scale * var0), lag


@njit(cache=True, fastmath=False)
def _eg_stat(y: F64, x: F64, maxlag: int, autolag: int) -> tuple[float, int]:
    """First-stage OLS of ``y`` on ``[x, 1]``, then the ADF statistic of its residuals."""
    n = y.size
    design = np.empty((n, 2))
    design[:, 0] = x
    design[:, 1] = 1.0
    coef, _, _, _ = np.linalg.lstsq(design, y)
    resid = y - design @ coef
    mean = y.mean()
    tss = 0.0
    ssr = 0.0
    for i in range(n):
        tss += (y[i] - mean) ** 2
        ssr += resid[i] ** 2
    if tss <= 0.0 or 1.0 - ssr / tss >= 1.0 - 100.0 * 1.4901161193847656e-08:
        return -np.inf, 0
    return _adf_stat(resid, maxlag, autolag)


@njit(cache=True, fastmath=False)
def _eg_batch(logp: F64, pairs: I64, maxlag: int, autolag: int) -> tuple[F64, I64]:
    count = pairs.shape[0]
    stats = np.empty(count)
    lags = np.empty(count, dtype=np.int64)
    for i in range(count):
        y = np.ascontiguousarray(logp[:, pairs[i, 0]])
        x = np.ascontiguousarray(logp[:, pairs[i, 1]])
        stats[i], lags[i] = _eg_stat(y, x, maxlag, autolag)
    return stats, lags


def _autolag_code(autolag: Literal["aic", "bic"] | None) -> int:
    return -1 if autolag is None else _PENALTY[autolag]


def engle_granger(
    y: F64,
    x: F64,
    autolag: Literal["aic", "bic"] | None = "aic",
    maxlag: int | None = None,
) -> tuple[float, float, int]:
    """Engle-Granger test of ``y`` on ``x`` with a constant: ``(stat, pvalue, lag)``.

    Parameters
    ----------
    y, x
        Log prices, equal length, finite.
    autolag
        Lag selection criterion, or None to use ``maxlag`` lags.
    maxlag
        Largest ADF lag; ``default_maxlag(len(y))`` when omitted.
    """
    yy = np.ascontiguousarray(y, dtype=np.float64)
    xx = np.ascontiguousarray(x, dtype=np.float64)
    if yy.shape != xx.shape or yy.ndim != 1:
        raise ValueError("y and x must be 1-d and the same length")
    lags = default_maxlag(yy.size) if maxlag is None else maxlag
    stat, lag = _eg_stat(yy, xx, lags, _autolag_code(autolag))
    return float(stat), float(mackinnon_pvalue(np.array([stat]))[0]), int(lag)


def engle_granger_batch(
    logp: F64,
    pairs: I64,
    autolag: Literal["aic", "bic"] | None = "aic",
    maxlag: int | None = None,
) -> F64:
    """Engle-Granger for many pairs of columns of ``logp``: an ``(n_pairs, 2)`` array.

    Column 0 is the statistic and column 1 the p-value, regressing column ``pairs[i, 0]``
    on column ``pairs[i, 1]``. Swap the pair columns for the other ordering.
    """
    panel = np.asfortranarray(logp, dtype=np.float64)
    index = np.ascontiguousarray(pairs, dtype=np.int64).reshape(-1, 2)
    lags = default_maxlag(panel.shape[0]) if maxlag is None else maxlag
    stats, _ = _eg_batch(panel, index, lags, _autolag_code(autolag))
    return np.column_stack([stats, mackinnon_pvalue(stats)])


def johansen_lag(Y: F64, max_k_ar_diff: int = 5) -> int:
    """VAR lag in differences chosen by AIC, at most ``max_k_ar_diff`` (FR-10)."""
    from statsmodels.tsa.vector_ar.vecm import select_order

    return int(
        select_order(np.asarray(Y, dtype=np.float64), maxlags=max_k_ar_diff, deterministic="co").aic
    )


def johansen_trace(Y: F64, det_order: int = 0, k_ar_diff: int = 1) -> tuple[float, float]:
    """Johansen trace statistic for ``r = 0`` and its 95% critical value.

    statsmodels gives critical values only, so Johansen confirms EG discoveries and never
    feeds the FDR procedure.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    with warnings.catch_warnings():
        # statsmodels drops round-off imaginary parts of the eigenvalues and says so.
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        result = coint_johansen(np.asarray(Y, dtype=np.float64), det_order, k_ar_diff)
    return float(result.lr1[0]), float(result.cvt[0, 1])
