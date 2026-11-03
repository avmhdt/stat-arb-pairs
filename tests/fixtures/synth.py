"""Synthetic market with known answers (spec section 11). Deterministic for a given seed.

60 stocks over 6 years of weekday sessions: 10 planted cointegrated pairs with known
hedge ratio and half-life, a common market factor, 3 delistings (one performance-related
with an imputed return, one merger with a real return, one unknown), one ticker change and
one thinly traded name. Three sector ETFs and a market ETF follow the factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from statarb.data.market import MarketData

N_STOCKS = 60
N_PAIRS = 10
SIC2_GROUPS = (28, 35, 36, 49, 60, 73)
HALF_LIVES = (5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0)
BETAS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 0.7, 1.5)
ETFS = {"SPY": None, "XLB": 28, "XLI": 35, "XLK": 73}


@dataclass(frozen=True)
class PlantedPair:
    """``log y = alpha + beta * log x + s`` with ``s`` an OU process of known half-life."""

    leg_y: str
    leg_x: str
    beta: float
    half_life_days: float
    group: str


@dataclass(frozen=True)
class SynthMarket:
    """The generated panel and what was planted in it."""

    data: MarketData
    planted: tuple[PlantedPair, ...]
    delisted: tuple[str, ...]
    renamed: str
    thin: str


def sessions(start: date = date(2015, 1, 1), years: int = 6) -> list[date]:
    """Weekdays from ``start`` to the end of the last year."""
    end = date(start.year + years - 1, 12, 31)
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def security_id(i: int) -> str:
    """``syn:007`` style ids."""
    return f"syn:{i:03d}"


def make_market(seed: int = 7, years: int = 6) -> SynthMarket:
    """Generate the synthetic market."""
    rng = np.random.default_rng(seed)
    days = sessions(years=years)
    n = len(days)
    factor = rng.normal(0.0003, 0.009, n)
    log_prices = np.empty((N_STOCKS, n))
    sic = np.empty(N_STOCKS, dtype=np.int64)
    planted: list[PlantedPair] = []
    for i in range(N_STOCKS):
        loading = rng.uniform(0.6, 1.4)
        noise = rng.normal(0.0, 0.014, n)
        log_prices[i] = math.log(rng.uniform(15.0, 120.0)) + np.cumsum(loading * factor + noise)
        sic[i] = SIC2_GROUPS[i % len(SIC2_GROUPS)] * 100 + int(rng.integers(0, 99))
    for k in range(N_PAIRS):
        x, y = 2 * k, 2 * k + 1
        b = math.exp(-math.log(2.0) / HALF_LIVES[k])
        sigma = 0.025 * math.sqrt(1.0 - b * b)
        spread = np.empty(n)
        spread[0] = 0.0
        shocks = rng.normal(0.0, sigma, n)
        for t in range(1, n):
            spread[t] = b * spread[t - 1] + shocks[t]
        alpha = rng.uniform(-0.3, 0.3)
        log_prices[y] = alpha + BETAS[k] * log_prices[x] + spread
        sic[y] = sic[x] // 100 * 100 + int(rng.integers(0, 99))
        planted.append(
            PlantedPair(
                security_id(y), security_id(x), BETAS[k], HALF_LIVES[k], f"{sic[x] // 100:02d}"
            )
        )
    shares = rng.uniform(5e7, 2e9, N_STOCKS)
    volume = rng.uniform(2e5, 5e6, (N_STOCKS, n))
    thin = N_STOCKS - 2
    volume[thin, rng.random(n) < 0.03] = 0.0
    delist_at = {40: int(n * 0.45), 41: int(n * 0.62), 42: int(n * 0.80)}
    delist_ret = {40: (-0.30, True), 41: (0.05, False), 42: (0.0, True)}
    rows: list[pl.DataFrame] = []
    master_rows = []
    for i in range(N_STOCKS):
        last = delist_at.get(i, n - 1)
        close = np.exp(log_prices[i, : last + 1])
        ret = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
        frame = pl.DataFrame(
            {
                "date": days[: last + 1],
                "security_id": security_id(i),
                "open": close * (1 + rng.normal(0, 0.002, last + 1)),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "close_is_mid": False,
                "volume": volume[i, : last + 1],
                "ret_total": ret,
                "tr_index": np.cumprod(1.0 + ret),
                "shares_out": shares[i],
                "mktcap_usd": close * shares[i],
            }
        )
        delisting = delist_ret.get(i)
        ret_delist = [None] * last + [delisting[0] if delisting else None]
        imputed = [None] * last + [delisting[1] if delisting else None]
        frame = frame.with_columns(
            pl.Series("ret_delist", ret_delist, dtype=pl.Float64),
            pl.Series("ret_delist_imputed", imputed, dtype=pl.Boolean),
            pl.lit(None, pl.Float64).alias("bid"),
            pl.lit(None, pl.Float64).alias("ask"),
            pl.lit("synthetic").alias("source"),
        )
        rows.append(frame)
        master_rows.append(
            {
                "security_id": security_id(i),
                "ticker": f"S{i:03d}",
                "name": f"Synthetic {i}",
                "security_type": "common",
                "exchange": "NYSE" if i % 2 else "NASDAQ",
                "first_date": days[0],
                "last_date": days[last] if i in delist_at else None,
                "delist_code": None,
                "source": "synthetic",
            }
        )
    master = pl.DataFrame(
        master_rows, schema_overrides={"last_date": pl.Date, "delist_code": pl.Int32}
    )
    renamed = N_STOCKS - 1
    change = days[n // 2]
    history = master.select(
        "security_id",
        "ticker",
        "name",
        "security_type",
        "exchange",
        "source",
        pl.lit(days[0]).alias("valid_from"),
        pl.col("last_date").alias("valid_to"),
    )
    old = history.filter(pl.col("security_id") == security_id(renamed)).with_columns(
        pl.lit("OLDT").alias("ticker"), pl.lit(change - timedelta(days=1)).alias("valid_to")
    )
    new = history.filter(pl.col("security_id") == security_id(renamed)).with_columns(
        pl.lit(change).alias("valid_from")
    )
    history = pl.concat(
        [history.filter(pl.col("security_id") != security_id(renamed)), old, new], how="vertical"
    ).sort("security_id", "valid_from")
    industry = pl.DataFrame(
        {
            "security_id": [security_id(i) for i in range(N_STOCKS)],
            "sic": sic.astype(np.int32),
            "valid_from": [days[0]] * N_STOCKS,
            "valid_to": [None] * N_STOCKS,
            "source": "synthetic",
            "is_point_in_time": True,
        },
        schema_overrides={"valid_to": pl.Date},
    )
    etf_rows = []
    for name, group in ETFS.items():
        ret = factor + rng.normal(0.0, 0.003 if group is None else 0.006, n)
        etf_rows.append(
            pl.DataFrame(
                {
                    "date": days,
                    "etf": name,
                    "ret_total": ret,
                    "close": 100.0 * np.cumprod(1.0 + ret),
                    "volume": 1e7,
                    "inception_date": days[0],
                }
            )
        )
    borrow = pl.DataFrame(
        {
            "date": [days[0]] * 3,
            "security_id": [None] * 3,
            "bucket": ["top500", "top1000", "rest"],
            "fee_rate_annual": [0.0025, 0.01, 0.05],
            "source": "assumed",
        },
        schema_overrides={"security_id": pl.String},
    )
    data = MarketData(
        prices=pl.concat(rows, how="diagonal").sort("security_id", "date"),
        master=master,
        ticker_history=history,
        industry=industry,
        etf_returns=pl.concat(etf_rows).sort("etf", "date"),
        borrow=borrow,
        data_profile="prototype",
        manifest_hash="synthetic",
    )
    return SynthMarket(
        data=data,
        planted=tuple(planted),
        delisted=tuple(security_id(i) for i in delist_at),
        renamed=security_id(renamed),
        thin=security_id(thin),
    )
