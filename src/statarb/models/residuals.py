"""Avellaneda-Lee (2010) residuals and s-scores (FR-18, FR-19).

PCA factors: standardise each stock's returns over ``corr_window`` days, take the top
eigenvectors ``v_j`` of their correlation matrix (from an SVD of the standardised panel)
and form eigenportfolios with weights ``Q_ji = v_ji / sigma_i``; factor returns are
``F_jt = sum_i Q_ji R_it``. Each stock's last ``resid_window`` returns are regressed on
``[1, F]`` (one least-squares solve for all stocks) and the residuals summed into
``X``. An AR(1) on ``X`` gives ``kappa = -ln(b) 252``, ``m = a / (1 - b)`` and
``sigma_eq = sd(zeta) / sqrt(1 - b^2)``; the s-score is ``(X_T - m_centred) / sigma_eq``
with the means centred across stocks, as A-L do. ``X_T`` is zero by construction because
the regression has an intercept.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def eigenportfolios(
    returns: F64, n_factors: int | None = 15, variance_explained: float | None = None
) -> tuple[F64, F64]:
    """Eigenportfolio weights ``(n_stocks, k)`` and the eigenvalues used.

    ``returns`` is ``(days, stocks)`` with no NaN. With ``variance_explained`` the number of
    factors is the smallest whose eigenvalues reach that share of the trace.
    """
    values = np.asarray(returns, dtype=np.float64)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    standard = (values - mean) / sd
    standard = np.nan_to_num(standard)
    _, singular, vt = np.linalg.svd(standard, full_matrices=False)
    eig = singular**2 / (values.shape[0] - 1)
    if variance_explained is not None:
        share = np.cumsum(eig) / eig.sum()
        k = int(np.searchsorted(share, variance_explained) + 1)
    else:
        k = min(int(n_factors or 1), eig.size)
    vectors = vt[:k].T
    # A sign convention, so that the same factor keeps its sign from day to day.
    vectors = vectors * np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    weights = vectors / np.nan_to_num(sd, nan=np.inf)[:, None]
    return weights, eig[:k]


def regression_residuals(stock_returns: F64, factors: F64) -> tuple[F64, F64]:
    """Residuals ``(days, stocks)`` and betas ``(factors, stocks)`` of OLS on ``[1, factors]``."""
    design = np.column_stack([np.ones(factors.shape[0]), factors])
    coef, *_ = np.linalg.lstsq(design, stock_returns, rcond=None)
    return stock_returns - design @ coef, coef[1:]


def pca_residuals(
    returns: F64, corr_window: int = 252, n_factors: int = 15, resid_window: int = 60
) -> F64:
    """Residuals of the last ``resid_window`` days on PCA factors estimated over ``corr_window``."""
    window = returns[-corr_window:]
    weights, _ = eigenportfolios(window, n_factors)
    recent = returns[-resid_window:]
    return regression_residuals(recent, recent @ weights)[0]


def etf_residuals(stock_returns: F64, etf_returns: F64, resid_window: int = 60) -> F64:
    """Residuals of each stock on its own ETF: column ``i`` of both inputs is one stock's pair."""
    stocks = np.asarray(stock_returns, dtype=np.float64)[-resid_window:]
    etfs = np.asarray(etf_returns, dtype=np.float64)[-resid_window:]
    resid = np.empty_like(stocks)
    for i in range(stocks.shape[1]):
        design = np.column_stack([np.ones(resid_window), etfs[:, i]])
        coef, *_ = np.linalg.lstsq(design, stocks[:, i], rcond=None)
        resid[:, i] = stocks[:, i] - design @ coef
    return resid


def ou_params(cum_residual: F64) -> tuple[F64, F64, F64]:
    """``kappa`` (per year), ``m`` and ``sigma_eq`` per column of a cumulative residual panel.

    NaN where the AR(1) slope is outside ``(0, 1)``.
    """
    x = np.asarray(cum_residual, dtype=np.float64)
    lagged, ahead = x[:-1], x[1:]
    n = lagged.shape[0]
    mean_l, mean_a = lagged.mean(axis=0), ahead.mean(axis=0)
    dl, da = lagged - mean_l, ahead - mean_a
    var_l = (dl * dl).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (dl * da).sum(axis=0) / var_l
        a = mean_a - b * mean_l
        zeta = ahead - (a + b * lagged)
        var_zeta = (zeta * zeta).sum(axis=0) / (n - 2)
        good = (b > 0) & (b < 1)
        kappa = np.where(good, -np.log(np.where(good, b, 0.5)) * 252.0, np.nan)
        m = np.where(good, a / (1.0 - b), np.nan)
        sigma_eq = np.where(good, np.sqrt(var_zeta / (1.0 - b * b)), np.nan)
    return kappa, m, sigma_eq


def s_scores(residuals: F64, kappa_min: float = 252 / 30, center: bool = True) -> tuple[F64, F64]:
    """A-L s-scores and kappas from a residual panel ``(days, stocks)``.

    The s-score is NaN where the fit fails or ``kappa <= kappa_min`` (A-L reject the model
    there). Means are centred across the stocks with a valid fit.
    """
    cum = np.cumsum(residuals, axis=0)
    kappa, m, sigma_eq = ou_params(cum)
    valid = np.isfinite(kappa) & (kappa > kappa_min) & (sigma_eq > 0)
    if center and valid.any():
        m = m - np.nanmean(np.where(valid, m, np.nan))
    s = np.where(valid, (cum[-1] - m) / sigma_eq, np.nan)
    return s, kappa


def s_score(cum_residual: F64, kappa_min: float = 252 / 30) -> F64:
    """Spec interface: s-scores from cumulative residuals, uncentred, NaN where rejected."""
    kappa, m, sigma_eq = ou_params(cum_residual)
    valid = np.isfinite(kappa) & (kappa > kappa_min) & (sigma_eq > 0)
    last = np.asarray(cum_residual, dtype=np.float64)[-1]
    return np.where(valid, (last - m) / sigma_eq, np.nan)


def half_life_days(kappa: float) -> float:
    """Sessions for a deviation to halve at ``kappa`` per year."""
    return math.log(2.0) / kappa * 252.0 if kappa > 0 else math.inf
