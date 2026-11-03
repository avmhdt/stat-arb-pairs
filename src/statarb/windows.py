"""Rolling formation and trading windows, and the date guards around them (FR-01, FR-02).

Selection and fitting code only ever receives a ``FormationView``, which cannot return a
row dated on or after ``T``; trading code receives a ``TradingView`` limited to
``[T, T + H)``. ``tests/integration/test_no_lookahead.py`` checks that nothing gets round
them (FR-03).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from statarb.config import UniverseConfig
    from statarb.data.market import MarketData


@dataclass(frozen=True, slots=True)
class Window:
    """One formation/trading split. Both end dates are exclusive."""

    window_id: str
    formation_start: date
    formation_end: date  # exclusive; equals trading_start (T)
    trading_start: date
    trading_end: date  # exclusive


class LookAheadError(RuntimeError):
    """Code in the formation window asked for data dated on or after ``T``."""


def _month(day: date) -> int:
    return day.year * 12 + day.month - 1


def rolling_windows(
    sessions: list[date],
    formation_months: int = 12,
    trading_months: int = 6,
    step_months: int = 1,
) -> list[Window]:
    """Windows on a trading calendar, with ``T`` the first session of a month.

    Formation covers the ``formation_months`` calendar months before ``T`` and trading the
    ``trading_months`` months from ``T``, so the two are disjoint and adjacent. ``T`` advances
    ``step_months`` at a time, giving overlapping trading windows when the step is shorter
    than the trading length (GGR 2006). The sessions are taken to start on a month's first
    session and to end on a month's last one; only complete windows are returned.

    Parameters
    ----------
    sessions
        Trading dates, strictly increasing, with no calendar month missing.
    formation_months, trading_months, step_months
        Window lengths and step, in calendar months; all positive.

    Returns
    -------
    list[Window]
        In order of ``T``. ``window_id`` is ``T`` in ISO format.
    """
    if min(formation_months, trading_months, step_months) < 1:
        raise ValueError("window lengths and step must be positive")
    if any(later <= earlier for earlier, later in itertools.pairwise(sessions)):
        raise ValueError("sessions must be strictly increasing")
    firsts: list[date] = []
    for day in sessions:
        if not firsts or _month(day) != _month(firsts[-1]):
            if firsts and _month(day) != _month(firsts[-1]) + 1:
                raise ValueError(f"no sessions in the month before {day}")
            firsts.append(day)
    after_last = sessions[-1] + timedelta(days=1) if sessions else None
    windows: list[Window] = []
    for i in range(formation_months, len(firsts) - trading_months + 1, step_months):
        end_index = i + trading_months
        trading_end = firsts[end_index] if end_index < len(firsts) else after_last
        assert trading_end is not None
        windows.append(
            Window(
                window_id=firsts[i].isoformat(),
                formation_start=firsts[i - formation_months],
                formation_end=firsts[i],
                trading_start=firsts[i],
                trading_end=trading_end,
            )
        )
    return windows


def _between(frame: pl.DataFrame, start: date | None, end: date) -> pl.DataFrame:
    keep = pl.col("date") < end
    if start is not None:
        keep = keep & (pl.col("date") >= start)
    return frame.filter(keep)


def _select(frame: pl.DataFrame, columns: list[str] | None) -> pl.DataFrame:
    if columns is None:
        return frame
    wanted = [
        "date",
        "security_id",
        *(name for name in columns if name not in ("date", "security_id")),
    ]
    return frame.select(wanted)


class FormationView:
    """Read-only view of processed data restricted to dates < ``window.formation_end``.

    Parameters
    ----------
    data
        The full processed panel. The view keeps it private and filters every read.
    window
        The window being formed.
    universe_cfg
        Eligibility rules for ``universe()``; required only by callers that use it.
    """

    __slots__ = ("_data", "_universe", "_universe_cfg", "window")

    def __init__(
        self,
        data: MarketData,
        window: Window,
        universe_cfg: UniverseConfig | None = None,
    ) -> None:
        self._data = data
        self.window = window
        self._universe_cfg = universe_cfg
        self._universe: pl.DataFrame | None = None

    @property
    def data_profile(self) -> str:
        """``headline`` or ``prototype``, carried into every result."""
        return self._data.data_profile

    def sessions(self) -> list[date]:
        """Formation sessions, ``[formation_start, T)``."""
        return [
            day
            for day in self._data.sessions
            if self.window.formation_start <= day < self.window.formation_end
        ]

    def prices(self, columns: list[str] | None = None) -> pl.DataFrame:
        """Daily rows in ``[formation_start, T)``."""
        frame = _between(self._data.prices, self.window.formation_start, self.window.formation_end)
        return _select(frame, columns)

    def universe(self) -> pl.DataFrame:
        """Eligible securities at ``T``, from data dated at most ``T - 1`` (DQ-02)."""
        if self._universe is None:
            if self._universe_cfg is None:
                raise ValueError("this view was built without a universe config")
            from statarb.universe import eligible_universe

            self._universe = eligible_universe(self, self._universe_cfg)
        return self._universe

    def last_session(self) -> date:
        """``T - 1``, the last formation session."""
        sessions = self.sessions()
        if not sessions:
            raise ValueError(f"no formation sessions before {self.window.formation_end}")
        return sessions[-1]

    def master(self) -> pl.DataFrame:
        """Listing attributes in force on ``T - 1``: ticker, type, exchange (DQ-09)."""
        return _as_of(self._data.ticker_history, self.last_session())

    def industry(self) -> pl.DataFrame:
        """SIC codes in force on ``T - 1`` (DQ-06)."""
        return _as_of(self._data.industry, self.last_session())

    def etf_returns(self) -> pl.DataFrame:
        """ETF returns dated before ``T``, with inception dates (DQ-07)."""
        return _between(self._data.etf_returns, None, self.window.formation_end)

    def delisted_ids(self) -> set[str]:
        """Securities whose eventual delisting the survivors-only flag needs.

        This is future information by design: it serves only the deliberate
        survivorship-biased comparison (FR-04), never a default run.
        """
        master = self._data.master
        return set(master.filter(pl.col("last_date").is_not_null())["security_id"].to_list())

    def query(self, table: str, start: date, end: date) -> pl.DataFrame:
        """Rows of ``table`` dated in ``[start, end)``; raises ``LookAheadError`` if ``end > T``."""
        if end > self.window.formation_end:
            raise LookAheadError(f"query to {end} past T = {self.window.formation_end}")
        return _between(self._data.table(table), start, end)


class TradingView:
    """Read-only view restricted to ``[trading_start, trading_end)``."""

    __slots__ = ("_data", "window")

    def __init__(self, data: MarketData, window: Window) -> None:
        self._data = data
        self.window = window

    def sessions(self) -> list[date]:
        """Trading sessions, ``[T, trading_end)``."""
        return [
            day
            for day in self._data.sessions
            if self.window.trading_start <= day < self.window.trading_end
        ]

    def prices(self, columns: list[str] | None = None) -> pl.DataFrame:
        """Daily rows in ``[T, trading_end)``."""
        frame = _between(self._data.prices, self.window.trading_start, self.window.trading_end)
        return _select(frame, columns)

    def etf_returns(self) -> pl.DataFrame:
        """ETF returns in ``[T, trading_end)``."""
        return _between(self._data.etf_returns, self.window.trading_start, self.window.trading_end)

    def borrow(self) -> pl.DataFrame:
        """Borrow schedule rows effective before ``trading_end`` (DQ-12)."""
        return self._data.borrow.filter(pl.col("date") < self.window.trading_end)


def _as_of(history: pl.DataFrame, day: date) -> pl.DataFrame:
    return history.filter(
        pl.col("valid_from") <= day,
        pl.col("valid_to").is_null() | (pl.col("valid_to") >= day),
    ).drop("valid_from", "valid_to")
