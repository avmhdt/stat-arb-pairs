"""AC-W-05: Engle-Granger and Johansen against statsmodels, plus size and power by simulation."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statarb.selection.cointegration import (
    default_maxlag,
    engle_granger,
    engle_granger_batch,
    johansen_trace,
    mackinnon_pvalue,
)

N = 252
SEED = 20261103


def _ou(rng: np.random.Generator, half_life: float, n: int, sd: float = 0.01) -> np.ndarray:
    b = 2.0 ** (-1.0 / half_life)
    shocks = rng.normal(0.0, sd, n)
    spread = np.empty(n)
    spread[0] = shocks[0] / math.sqrt(1 - b * b)
    for t in range(1, n):
        spread[t] = b * spread[t - 1] + shocks[t]
    return spread


def _planted(
    rng: np.random.Generator, half_life: float, n: int = N
) -> tuple[np.ndarray, np.ndarray]:
    x = np.cumsum(rng.normal(0.0, 0.015, n))
    return 0.1 + 1.2 * x + _ou(rng, half_life, n), x


def test_matches_statsmodels_on_500_pairs() -> None:
    rng = np.random.default_rng(SEED)
    maxlag = default_maxlag(N)
    assert maxlag == 15
    worst_stat = worst_p = 0.0
    lags = set()
    for i in range(500):
        if i % 2:
            y, x = _planted(rng, float(rng.uniform(2, 40)))
        else:
            x = np.cumsum(rng.normal(0, 0.01, N))
            y = np.cumsum(rng.normal(0, 0.01, N))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected, p_expected, _ = coint(y, x, trend="c", maxlag=maxlag, autolag="aic")
        stat, p, lag = engle_granger(y, x, "aic", maxlag)
        lags.add(lag)
        worst_stat = max(worst_stat, abs(stat - expected) / abs(expected))
        worst_p = max(worst_p, abs(p - p_expected))
    assert worst_stat < 1e-8
    assert worst_p < 1e-8
    assert len(lags) > 3


@pytest.mark.parametrize("autolag", ["bic", None])
def test_other_lag_rules_match(autolag: str | None) -> None:
    rng = np.random.default_rng(1)
    for _ in range(30):
        y, x = _planted(rng, 8.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected, _, _ = coint(y, x, trend="c", maxlag=6, autolag=autolag)
        assert engle_granger(y, x, autolag, 6)[0] == pytest.approx(expected, rel=1e-8)  # type: ignore[arg-type]


def test_batch_equals_single() -> None:
    rng = np.random.default_rng(2)
    logp = np.cumsum(rng.normal(0, 0.01, (N, 12)), axis=0)
    pairs = np.array([[0, 1], [3, 2], [11, 5]])
    batch = engle_granger_batch(logp, pairs)
    for row, (a, b) in zip(batch, pairs, strict=True):
        stat, p, _ = engle_granger(logp[:, a], logp[:, b])
        assert row[0] == stat and row[1] == p


def test_collinear_and_bad_input() -> None:
    x = np.cumsum(np.random.default_rng(3).normal(0, 0.01, N))
    stat, p, _ = engle_granger(2.0 * x + 1.0, x)
    assert stat == -math.inf and p == 0.0
    with pytest.raises(ValueError, match="same length"):
        engle_granger(x, x[:-1])


def test_mackinnon_matches_statsmodels() -> None:
    grid = np.linspace(-25, 3, 4001)
    expected = np.array([mackinnonp(value, regression="c", N=2) for value in grid])
    assert np.array_equal(mackinnon_pvalue(grid), expected)
    assert np.isnan(mackinnon_pvalue(np.array([np.nan]))[0])


def _null_pairs(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logp = np.cumsum(rng.normal(0, 0.01, (N, 2 * count)), axis=0)
    return logp, np.arange(2 * count).reshape(count, 2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC-W-05 asks for size 5% +/- 1.5 pp on 2,000 independent random-walk pairs. One "
        "ordering with AIC lags and MacKinnon's asymptotic p-values rejects 7.3% here and "
        "6.55% on 20,000 pairs, as statsmodels' coint does. Owner ruling 2026-11-03: "
        "accepted as a documented miss."
    ),
)
def test_size_on_independent_random_walks() -> None:
    logp, pairs = _null_pairs(2000, SEED + 1)
    rate = float((engle_granger_batch(logp, pairs)[:, 1] < 0.05).mean())
    assert 0.06 <= rate <= 0.085, rate  # documents the measured level
    assert 0.035 <= rate <= 0.065, rate


def test_order_invariant_rule_is_conservative() -> None:
    """Selection keeps the larger p-value of the two orderings; its size is below 5%."""
    logp, pairs = _null_pairs(2000, SEED + 1)
    larger = np.maximum(
        engle_granger_batch(logp, pairs)[:, 1], engle_granger_batch(logp, pairs[:, ::-1])[:, 1]
    )
    assert float((larger < 0.05).mean()) <= 0.05


def _power(half_life: float, count: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    columns = []
    for _ in range(count):
        y, x = _planted(rng, half_life)
        columns += [y, x]
    logp = np.column_stack(columns)
    pairs = np.arange(2 * count).reshape(count, 2)
    return float((engle_granger_batch(logp, pairs)[:, 1] < 0.05).mean())


def test_power_at_short_half_life() -> None:
    assert _power(5.0, 1000, SEED + 2) >= 0.90


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC-W-05 asks for power >= 0.90 at a 10-day half-life with n = 252. Engle-Granger "
        "measures about 0.5 here and in statsmodels' coint: the expected ADF statistic is "
        "about -3 against a critical value near -3.34. Owner ruling 2026-11-03: accepted as "
        "a documented miss; threshold unchanged."
    ),
)
def test_power_at_ten_day_half_life() -> None:
    power = _power(10.0, 1000, SEED + 3)
    assert 0.35 <= power <= 0.65, power  # documents the measured level
    assert power >= 0.90


def test_johansen_wrapper_reproduces_statsmodels() -> None:
    rng = np.random.default_rng(4)
    y, x = _planted(rng, 6.0)
    data = np.column_stack([y, x])
    for k in (0, 1, 3):
        result = coint_johansen(data, 0, k)
        assert johansen_trace(data, 0, k) == (float(result.lr1[0]), float(result.cvt[0, 1]))
