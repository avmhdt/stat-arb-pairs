"""AC-W-09 on local headline data: every formation year's universe holds names that later delist.

Needs ``statarb build`` on the Sharadar profile; never runs in CI (marker ``data``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from statarb.config import UniverseConfig
from statarb.data.market import load_market_data
from statarb.data.schemas import read_build_info
from statarb.windows import FormationView, rolling_windows

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def processed() -> Path:
    if read_build_info(PROCESSED).get("data_profile") != "headline":
        pytest.skip("no headline build under data/processed")
    return PROCESSED


def test_each_formation_year_has_later_delisted_names(processed: Path) -> None:
    data = load_market_data(
        processed,
        date(1998, 1, 2),
        date(2025, 12, 31),
        ["close", "volume", "close_is_mid", "mktcap_usd"],
        ("common",),
    )
    delisted = data.master.filter(pl.col("last_date").is_not_null()).select(
        "security_id", "last_date"
    )
    counts = {}
    for window in rolling_windows(data.sessions, 12, 6, 12):
        universe = FormationView(data, window, UniverseConfig()).universe()
        later = universe.join(delisted, on="security_id").filter(
            pl.col("last_date") >= window.formation_end
        )
        counts[window.window_id] = later.height
    assert len(counts) >= 25
    assert all(count > 0 for count in counts.values()), counts
