from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from fixtures.synth import make_market, sessions
from statarb.windows import FormationView, LookAheadError, TradingView, Window, rolling_windows


def test_window_boundaries_on_month_starts() -> None:
    days = sessions(date(2015, 1, 1), years=3)
    windows = rolling_windows(days, 12, 6, 1)
    first = windows[0]
    assert first == Window(
        window_id="2016-01-01",
        formation_start=date(2015, 1, 1),
        formation_end=date(2016, 1, 1),
        trading_start=date(2016, 1, 1),
        trading_end=date(2016, 7, 1),
    )
    last = windows[-1]
    assert last.trading_start == date(2017, 7, 3)
    assert last.trading_end == date(2017, 12, 30)  # day after the last session
    assert len(windows) == 19


def test_leap_year_and_holiday_gaps() -> None:
    days = [day for day in sessions(date(2019, 1, 1), years=3) if day != date(2020, 3, 2)]
    windows = {window.window_id: window for window in rolling_windows(days, 12, 3, 1)}
    march = windows["2020-03-03"]
    assert march.formation_start == date(2019, 3, 1)
    assert march.trading_end == date(2020, 6, 1)
    assert "2020-02-03" in windows


def test_step_and_lengths() -> None:
    days = sessions(date(2015, 1, 1), years=4)
    assert len(rolling_windows(days, 24, 6, 6)) == 4
    assert rolling_windows(days, 48, 1, 1) == []


@pytest.mark.parametrize("args", [(0, 6, 1), (12, 0, 1), (12, 6, 0)])
def test_rejects_bad_lengths(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        rolling_windows(sessions(), *args)


def test_rejects_unsorted_or_gappy_sessions() -> None:
    days = sessions(date(2015, 1, 1), years=2)
    with pytest.raises(ValueError, match="increasing"):
        rolling_windows([days[1], days[0], *days[2:]], 12, 6, 1)
    gappy = [day for day in days if day.month != 5]
    with pytest.raises(ValueError, match="no sessions"):
        rolling_windows(gappy, 12, 6, 1)


@pytest.fixture(scope="module")
def market_window():  # type: ignore[no-untyped-def]
    market = make_market(seed=3, years=3)
    return market, rolling_windows(market.data.sessions, 12, 6, 1)[4]


def test_formation_view_guards(market_window) -> None:  # type: ignore[no-untyped-def]
    market, window = market_window
    view = FormationView(market.data, window)
    prices = view.prices(["close"])
    assert prices.columns == ["date", "security_id", "close"]
    assert prices["date"].max() < window.formation_end
    assert prices["date"].min() >= window.formation_start
    assert view.last_session() == window.formation_end - timedelta(days=3)
    assert view.etf_returns()["date"].max() < window.formation_end
    assert view.query("prices", date(2000, 1, 1), window.formation_end).height > 0
    with pytest.raises(LookAheadError):
        view.query("prices", window.formation_start, window.formation_end + timedelta(days=1))
    with pytest.raises(KeyError):
        view.query("secrets", window.formation_start, window.formation_end)
    with pytest.raises(ValueError, match="universe config"):
        view.universe()
    assert view.data_profile == "prototype"


def test_formation_view_needs_sessions(market_window) -> None:  # type: ignore[no-untyped-def]
    market, _ = market_window
    early = Window("x", date(2000, 1, 1), date(2000, 2, 1), date(2000, 2, 1), date(2000, 3, 1))
    with pytest.raises(ValueError, match="no formation sessions"):
        FormationView(market.data, early).last_session()


def test_as_of_listing_and_industry(market_window) -> None:  # type: ignore[no-untyped-def]
    market, _ = market_window
    windows = rolling_windows(market.data.sessions, 12, 6, 1)
    before, after = windows[0], windows[-1]
    renamed = pl.col("security_id") == market.renamed
    assert FormationView(market.data, before).master().filter(renamed)["ticker"].to_list() == [
        "OLDT"
    ]
    assert FormationView(market.data, after).master().filter(renamed)["ticker"].to_list() == [
        "S059"
    ]
    assert FormationView(market.data, after).industry().height == 60


def test_trading_view(market_window) -> None:  # type: ignore[no-untyped-def]
    market, window = market_window
    view = TradingView(market.data, window)
    prices = view.prices()
    assert prices["date"].min() == window.trading_start
    assert prices["date"].max() < window.trading_end
    assert view.sessions()[0] == window.trading_start
    assert view.etf_returns()["date"].min() == window.trading_start
    assert view.borrow().height == 3
    assert view.prices(["date", "close"]).columns == ["date", "security_id", "close"]
