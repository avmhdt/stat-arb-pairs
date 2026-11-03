"""Synthetic factor market with planted OU residuals (AC-W-14). Deterministic for a seed.

Returns follow three factors; 30% of the names add an OU residual with kappa = 25 a year
(half-life about 7 sessions), the rest add independent noise, so their cumulative
residuals are random walks. SPY tracks the first factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from fixtures.synth import sessions
from statarb.data.market import MarketData

KAPPA = 25.0


@dataclass(frozen=True)
class AlMarket:
    """The panel and the true kappa of every name (0 for the random-walk residuals)."""

    data: MarketData
    true_kappa: dict[str, float]


def make_al_market(
    seed: int = 3, n_stocks: int = 100, years: int = 3, planted_share: float = 0.3
) -> AlMarket:
    """Generate the market."""
    rng = np.random.default_rng(seed)
    days = sessions(years=years)
    n = len(days)
    factors = rng.normal(0.0, [0.010, 0.006, 0.004], (n, 3))
    loadings = np.column_stack(
        [rng.uniform(0.6, 1.4, n_stocks), rng.normal(0, 1, n_stocks), rng.normal(0, 1, n_stocks)]
    )
    planted = rng.permutation(n_stocks)[: int(planted_share * n_stocks)]
    b = math.exp(-KAPPA / 252.0)
    sd = 0.012
    idio = rng.normal(0.0, sd, (n, n_stocks))
    for i in planted:
        x = np.empty(n)
        x[0] = 0.0
        shocks = rng.normal(0.0, sd * math.sqrt(1 - b * b) * 2.5, n)
        for t in range(1, n):
            x[t] = b * x[t - 1] + shocks[t]
        idio[:, i] = np.diff(np.concatenate([[0.0], x]))
    returns = factors @ loadings.T + idio
    rows = []
    for i in range(n_stocks):
        sid = f"syn:{i:03d}"
        close = 50.0 * np.cumprod(1.0 + returns[:, i])
        rows.append(
            pl.DataFrame(
                {
                    "date": days,
                    "security_id": sid,
                    "open": close,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "close_is_mid": False,
                    "volume": 1e6,
                    "ret_total": returns[:, i],
                    "ret_delist": None,
                    "tr_index": np.cumprod(1.0 + returns[:, i]),
                    "shares_out": 1e8,
                    "mktcap_usd": close * 1e8,
                },
                schema_overrides={"ret_delist": pl.Float64},
            )
        )
    ids = [f"syn:{i:03d}" for i in range(n_stocks)]
    master = pl.DataFrame(
        {
            "security_id": ids,
            "ticker": ids,
            "name": ids,
            "security_type": "common",
            "exchange": "NYSE",
            "first_date": days[0],
            "last_date": None,
            "delist_code": None,
            "source": "synthetic",
        },
        schema_overrides={"last_date": pl.Date, "delist_code": pl.Int32},
    )
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
    industry = pl.DataFrame(
        {
            "security_id": ids,
            "sic": [3571] * n_stocks,
            "valid_from": [days[0]] * n_stocks,
            "valid_to": [None] * n_stocks,
            "source": "synthetic",
            "is_point_in_time": True,
        },
        schema_overrides={"valid_to": pl.Date, "sic": pl.Int32},
    )
    spy = factors[:, 0]
    etf = pl.concat(
        [
            pl.DataFrame(
                {
                    "date": days,
                    "etf": name,
                    "ret_total": spy + rng.normal(0, 0.001, n),
                    "close": 100.0,
                    "volume": 1e8,
                    "inception_date": days[0],
                }
            )
            for name in ("SPY", "XLK")
        ]
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
        prices=pl.concat(rows).sort("security_id", "date"),
        master=master,
        ticker_history=history,
        industry=industry,
        etf_returns=etf.sort("etf", "date"),
        borrow=borrow,
        data_profile="prototype",
        manifest_hash="synthetic-al",
    )
    truth = {sid: (KAPPA if i in set(planted.tolist()) else 0.0) for i, sid in enumerate(ids)}
    return AlMarket(data, truth)
