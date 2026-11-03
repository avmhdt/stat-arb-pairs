"""Leakage harness (FR-03): perturb everything dated on or after T and demand identical output.

A selector takes a ``FormationView`` and returns a frozen result: anything comparable with
``==`` whose floats are finite. The harness runs it on the clean data and on each perturbed
copy, window by window, and fails on the first difference.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from datetime import timedelta

import numpy as np
import polars as pl

from statarb.config import UniverseConfig
from statarb.data.market import MarketData
from statarb.windows import FormationView, Window

Selector = Callable[[FormationView], Hashable]
PERTURBATIONS = ("noise", "permutation", "nan", "future_industry")
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "ret_total", "tr_index", "mktcap_usd")


def _future_industry(data: MarketData, window: Window) -> MarketData:
    """Every security changes SIC on ``T`` and ETF returns change from ``T`` on."""
    t = window.formation_end
    closed = data.industry.with_columns(
        pl.when(pl.col("valid_to").is_null())
        .then(pl.lit(t - timedelta(days=1)))
        .otherwise(pl.col("valid_to"))
        .alias("valid_to")
    )
    moved = data.industry.with_columns(
        ((pl.col("sic") + 4100) % 9000).alias("sic"), pl.lit(t).alias("valid_from")
    )
    etf = data.etf_returns.with_columns(
        pl.when(pl.col("date") >= t)
        .then(pl.col("ret_total") * 3 + 0.01)
        .otherwise(pl.col("ret_total"))
        .alias("ret_total")
    )
    return MarketData(
        prices=data.prices,
        master=data.master,
        ticker_history=data.ticker_history,
        industry=pl.concat([closed, moved]),
        etf_returns=etf,
        borrow=data.borrow,
        data_profile=data.data_profile,
        manifest_hash=data.manifest_hash,
    )


def perturb(data: MarketData, window: Window, how: str, seed: int) -> MarketData:
    """A copy of ``data`` with every row dated on or after ``T`` changed by ``how``."""
    if how == "future_industry":
        return _future_industry(data, window)
    rng = np.random.default_rng(seed)
    t = window.formation_end
    prices = data.prices.with_row_index("_row")
    future = prices.filter(pl.col("date") >= t)
    fields = [name for name in PRICE_FIELDS if name in prices.columns]
    if how == "noise":
        factor = pl.Series(np.exp(rng.normal(0.0, 0.2, future.height)))
        changed = future.with_columns(pl.col(name) * factor for name in fields)
    elif how == "permutation":
        order = rng.permutation(future.height)
        changed = future.with_columns(
            pl.Series(name, future[name].to_numpy()[order]) for name in fields
        )
    elif how == "nan":
        changed = future.with_columns(pl.lit(float("nan")).alias(name) for name in fields)
    else:
        raise ValueError(f"unknown perturbation {how!r}")
    rows = pl.concat([prices.filter(pl.col("date") < t), changed]).sort("_row").drop("_row")
    return data.replace_prices(rows)


def check_no_lookahead(
    selector: Selector,
    data: MarketData,
    windows: Sequence[Window],
    universe: UniverseConfig,
    perturbations: Sequence[str] = PERTURBATIONS,
    seed: int = 0,
) -> int:
    """Raise ``AssertionError`` if any perturbation changes any window's selection.

    Returns the number of comparisons made.
    """
    compared = 0
    for number, window in enumerate(windows):
        clean = selector(FormationView(data, window, universe))
        for offset, how in enumerate(perturbations):
            noisy = perturb(data, window, how, seed + 100 * number + offset)
            result = selector(FormationView(noisy, window, universe))
            assert result == clean, f"{how} perturbation changed window {window.window_id}"
            compared += 1
    return compared
