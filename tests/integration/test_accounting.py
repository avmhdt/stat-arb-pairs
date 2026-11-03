"""AC-W-08 accounting: a hand-computed round trip, delisting and stale closes, exact netting."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from statarb.config import RunConfig
from statarb.costs import LegCostInputs
from statarb.data.market import MarketData
from statarb.selection.pairs import PairParams, PairSet
from statarb.trading.engine import FrozenPair, FrozenWindow, combine, run_window
from statarb.windows import TradingView, Window

D0 = date(2024, 3, 4)  # a Monday


def _days(n: int) -> list[date]:
    days, day = [], D0
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _market(
    paths: dict[str, list[float | None]], delist: dict[str, tuple[int, float]] | None = None
) -> tuple[MarketData, Window, list[date]]:
    n = len(next(iter(paths.values())))
    days = _days(n)
    rows = []
    for sid, path in paths.items():
        previous = None
        for i, (day, level) in enumerate(zip(days, path, strict=True)):
            if level is None:
                continue
            ret = 0.0 if previous is None else level / previous - 1.0
            d = (delist or {}).get(sid)
            rows.append(
                {
                    "date": day,
                    "security_id": sid,
                    "close": level,
                    "tr_index": level,
                    "ret_total": ret,
                    "ret_delist": d[1] if d and d[0] == i else None,
                }
            )
            previous = level
    prices = pl.DataFrame(rows, schema_overrides={"ret_delist": pl.Float64})
    empty = pl.DataFrame(
        {"security_id": [], "valid_from": [], "valid_to": []},
        schema={"security_id": pl.String, "valid_from": pl.Date, "valid_to": pl.Date},
    )
    data = MarketData(
        prices=prices,
        master=pl.DataFrame(
            {"security_id": list(paths), "last_date": [None] * len(paths)},
            schema_overrides={"last_date": pl.Date},
        ),
        ticker_history=empty,
        industry=empty,
        etf_returns=pl.DataFrame(
            {"date": [], "etf": [], "ret_total": []},
            schema={"date": pl.Date, "etf": pl.String, "ret_total": pl.Float64},
        ),
        borrow=pl.DataFrame({"date": [D0]}),
    )
    window = Window("w", D0 - timedelta(days=365), D0, D0, days[-1] + timedelta(days=1))
    return data, window, days


def _frozen(window: Window, fee: float = 0.05) -> FrozenWindow:
    params = PairParams("A", "B", 1.0, 0.0, 0.0, 0.01, None, None, None, None, 1)
    leg = LegCostInputs(half_spread=0.0, adv_usd=1e9, sigma_daily=0.01, fee_rate_annual=fee)
    pairset = PairSet(window, (params,), pl.DataFrame())
    return FrozenWindow(
        pairset, (FrozenPair(params, None, 1.0, None),), {"A": leg, "B": leg}, True, 1.0
    )


def _cfg(lag: int = 1) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {
                "name": "acct",
                "strategy_family": "pairs_distance",
                "data_profile": "prototype",
            },
            "data": {"provider": "free"},
            "selection": {"method": "distance", "top_k": 1},
            "windows": {"execution_lag_days": lag},
            "rules": {"stop_z": None, "time_stop_half_lives": None},
            "sizing": {"gross_per_pair": 2.0},
            "costs": {
                "commission_bps": 1.0,
                "half_spread": {"model": "none"},
                "impact": {"model": "none"},
            },
        }
    )


def test_hand_computed_round_trip() -> None:
    data, window, days = _market({"A": [1.0, 1.03, 1.03, 1.0, 1.0, 1.0], "B": [1.0] * 6})
    result = run_window(_frozen(window), TradingView(data, window), _cfg(lag=1))
    daily = result.daily
    # z: 0, 3, 3, 0, ... Signal at d1 (short A, long B, 1 unit each), executed at d2's close.
    # Exit signal at d3, executed at d4's close.
    commission = 1e-4
    r_a = 1.0 / 1.03 - 1.0
    fee = 0.05 / 360.0
    pnl = [0.0, 0.0, 0.0, -1.0 * r_a, 0.0, 0.0]
    trading = [0.0, 0.0, 2 * commission, 0.0, commission * (1.0 * (1 + r_a) + 1.0), 0.0]
    borrow = [0.0, 0.0, 0.0, fee * 1.0, fee * (1 + r_a), 0.0]
    capital = [1.0]
    for t in range(5):
        capital.append(capital[-1] + pnl[t] - trading[t] - borrow[t])
    for t in range(6):
        assert daily["ret_gross"][t] == pytest.approx(pnl[t] / capital[t], abs=1e-12)
        assert daily["cost_trading"][t] == pytest.approx(trading[t] / capital[t], abs=1e-12)
        assert daily["cost_borrow"][t] == pytest.approx(borrow[t] / capital[t], abs=1e-12)
    assert (daily["ret"] == daily["ret_gross"] - daily["cost_trading"] - daily["cost_borrow"]).all()
    assert daily["n_open"].to_list() == [0, 0, 1, 1, 0, 0]
    assert daily["employed"].to_list() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    trades = result.trades.sort("date_exec", "security_id")
    assert trades["reason"].to_list() == ["entry", "entry", "exit", "exit"]
    assert trades["date_signal"].to_list() == [days[1], days[1], days[3], days[3]]
    assert trades["date_exec"].to_list() == [days[2], days[2], days[4], days[4]]
    assert trades["side"].to_list() == [-1, 1, 1, -1]


def test_no_wait_executes_on_the_signal_day() -> None:
    data, window, days = _market({"A": [1.0, 1.03, 1.03, 1.0, 1.0, 1.0], "B": [1.0] * 6})
    trades = run_window(_frozen(window), TradingView(data, window), _cfg(lag=0)).trades
    assert sorted(set(trades["date_exec"].to_list())) == [days[1], days[3]]


def test_delisting_closes_the_pair_with_its_return() -> None:
    data, window, days = _market(
        {"A": [1.0, 1.03, 1.03, 1.02, 1.02, 1.02], "B": [1.0, 1.0, 1.0, 1.0, None, None]},
        {"B": (3, -0.30)},
    )
    result = run_window(_frozen(window, fee=0.0), TradingView(data, window), _cfg(lag=1))
    daily, trades = result.daily, result.trades
    r_a = 1.02 / 1.03 - 1.0
    r_b = (1.0 + 0.0) * (1.0 - 0.30) - 1.0
    capital = 1.0 - 2 * 1e-4  # the entry commission at d2
    assert daily["ret_gross"][3] == pytest.approx((-1.0 * r_a + 1.0 * r_b) / capital, abs=1e-12)
    closes = trades.filter(pl.col("reason") != "entry")
    assert closes["reason"].unique().to_list() == ["delist"]
    assert closes["date_exec"].unique().to_list() == [days[3]]
    assert daily["n_open"][4] == 0


def test_stale_prices_close_after_five_sessions() -> None:
    a = [1.0, 1.03] + [1.03] * 10
    b: list[float | None] = [1.0, 1.0, 1.0] + [None] * 9
    data, window, days = _market({"A": a, "B": b})
    trades = run_window(_frozen(window, fee=0.0), TradingView(data, window), _cfg(lag=1)).trades
    closes = trades.filter(pl.col("reason") != "entry")
    assert closes["reason"].unique().to_list() == ["stale"]
    assert closes["date_exec"].unique().to_list() == [days[8]]


def test_combine_averages_active_windows() -> None:
    first = pl.DataFrame(
        {
            "date": [D0, D0 + timedelta(days=1)],
            "window_id": "a",
            "ret_gross": [0.02, 0.01],
            "cost_trading": [0.001, 0.0],
            "cost_borrow": [0.0, 0.0],
            "ret": [0.019, 0.01],
            "pnl_net": [0.019, 0.01],
            "employed": [1.0, 0.0],
            "gross_exposure": [2.0, 0.0],
            "n_open": [1, 0],
            "turnover": [0.1, 0.0],
            "cost_impact": [0.0, 0.0],
        }
    )
    second = first.with_columns(
        pl.lit("b").alias("window_id"),
        pl.col("ret_gross") * 0,
        pl.col("pnl_net") * 0,
        pl.col("employed") * 0,
    ).head(1)
    combined = combine([first, second])
    assert combined["windows_active"].to_list() == [2, 1]
    assert combined["ret_gross"].to_list() == pytest.approx([0.01, 0.01])
    assert combined["ret"].to_list() == pytest.approx([0.01 - 0.001, 0.01])
    assert combined["ret_employed"].to_list() == pytest.approx([0.019, 0.0])
    assert combine([]).is_empty()
