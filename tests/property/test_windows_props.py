"""AC-W-02: windows are disjoint, adjacent, and tile the sample when the step equals H."""

from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from statarb.windows import rolling_windows


def _sessions(start: date, months: int, holes: list[int]) -> list[date]:
    end_year = start.year + (start.month - 1 + months) // 12
    end_month = (start.month - 1 + months) % 12 + 1
    end = date(end_year, end_month, 1)
    days = []
    day = start
    while day < end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    dropped = {days[i % len(days)] for i in holes}
    firsts = {min(d for d in days if (d.year, d.month) == (x.year, x.month)) for x in days}
    return [d for d in days if d not in dropped or d in firsts]


@settings(max_examples=60, deadline=None)
@given(
    year=st.integers(1998, 2024),
    month=st.integers(1, 12),
    months=st.integers(8, 60),
    formation=st.integers(1, 24),
    trading=st.integers(1, 12),
    step=st.integers(1, 6),
    holes=st.lists(st.integers(0, 2000), max_size=20),
)
def test_windows_disjoint_and_adjacent(
    year: int, month: int, months: int, formation: int, trading: int, step: int, holes: list[int]
) -> None:
    days = _sessions(date(year, month, 1), months, holes)
    for window in rolling_windows(days, formation, trading, step):
        formation_days = [d for d in days if window.formation_start <= d < window.formation_end]
        trading_days = [d for d in days if window.trading_start <= d < window.trading_end]
        assert formation_days and trading_days
        assert not set(formation_days) & set(trading_days)
        assert window.formation_end == window.trading_start
        following = days.index(formation_days[-1]) + 1
        assert days[following] == trading_days[0]
        assert window.window_id == window.trading_start.isoformat()


@settings(max_examples=40, deadline=None)
@given(months=st.integers(14, 60), formation=st.integers(1, 12), trading=st.integers(1, 6))
def test_step_equal_to_trading_tiles_without_gaps(
    months: int, formation: int, trading: int
) -> None:
    days = _sessions(date(2010, 1, 1), months, [])
    windows = rolling_windows(days, formation, trading, trading)
    covered = [d for w in windows for d in days if w.trading_start <= d < w.trading_end]
    assert len(covered) == len(set(covered))
    if windows:
        start = days.index(windows[0].trading_start)
        stop = days.index(covered[-1]) + 1
        assert covered == days[start:stop]
