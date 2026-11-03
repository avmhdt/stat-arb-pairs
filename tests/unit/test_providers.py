from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from statarb.data.providers import edgar, sharadar, stooq
from statarb.settings import SettingsError

FAKE_KEY = "fake-key-for-tests"


def test_sharadar_key_goes_in_header_not_url() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="ticker,date,close\nXLE,2026-10-30,90.1\n")

    result = sharadar.check_access("SFP", FAKE_KEY, transport=httpx.MockTransport(handler))
    assert result.rows == 1
    assert result.columns == ("ticker", "date", "close")
    assert (result.table, result.ticker) == ("funds", "XLE")
    request = seen[0]
    assert request.headers["x-api-key"] == FAKE_KEY
    assert FAKE_KEY not in str(request.url)
    assert request.url.path == "/v1.0/data/funds"
    assert request.url.params["limit"] == "1"


def test_sharadar_check_tickers_are_outside_free_tier() -> None:
    free_tier = {"AAPL", "MSFT", "SPY"}
    assert not {ticker for _, ticker in sharadar.KEY_CHECKS.values()} & free_tier


def test_sharadar_refused_key_raises() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(403, text="Exceeds free tier"))
    with pytest.raises(httpx.HTTPStatusError):
        sharadar.check_access("SEP", FAKE_KEY, transport=transport)


def test_sharadar_needs_a_key() -> None:
    with pytest.raises(ValueError, match="SHARADAR_API_KEY"):
        sharadar.sharadar_client("")


def test_edgar_submissions_url_and_user_agent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"name": "Apple Inc.", "sic": "3571"})

    with edgar.EdgarClient('"Test User test@example.com"', httpx.MockTransport(handler)) as client:
        payload = client.submissions(320193)
    assert payload["sic"] == "3571"
    assert str(seen[0].url) == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert seen[0].headers["User-Agent"] == "Test User test@example.com"


def test_edgar_rejects_missing_user_agent() -> None:
    with pytest.raises(SettingsError):
        edgar.EdgarClient("")


def test_edgar_stays_under_the_sec_limit() -> None:
    assert edgar.MAX_REQUESTS_PER_SECOND <= 10


def test_stooq_index(stooq_dir: Path) -> None:
    index = stooq.build_index(stooq_dir)
    assert index["ticker"].to_list() == ["AAPL.US", "ABC.US", "SPY.US"]
    spy = index.row(2, named=True)
    assert spy == {
        "ticker": "SPY.US",
        "symbol": "SPY",
        "exchange": "NYSE",
        "security_type": "etf",
        "path": "d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt",
    }
    assert index.row(1, named=True)["exchange"] == "NYSEMKT"
    assert index.row(0, named=True)["security_type"] == "stock"
    assert stooq.duplicate_tickers(index) == []


def test_stooq_duplicates(stooq_dir: Path) -> None:
    us = stooq_dir / stooq.US_DAILY
    copy = us / "nasdaq etfs" / "spy.us.txt"
    copy.parent.mkdir(parents=True)
    copy.write_text((us / "nyse etfs" / "2" / "spy.us.txt").read_text())
    assert stooq.duplicate_tickers(stooq.build_index(stooq_dir)) == ["SPY.US"]


def test_stooq_missing_tree(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stooq.build_index(tmp_path)


def test_stooq_read_daily(stooq_dir: Path) -> None:
    frame = stooq.read_daily(stooq_dir / stooq.US_DAILY / "nyse etfs" / "2" / "spy.us.txt")
    assert frame.columns == ["date", "open", "high", "low", "close", "volume"]
    assert frame["date"].to_list() == [date(2026, 10, 29), date(2026, 10, 30)]
    assert frame["volume"].to_list() == [1000.5, 77666350.0]
    assert frame["close"][1] == 755.72


def test_stooq_rejects_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "x.us.txt"
    path.write_text("ticker,date\nX,1\n")
    with pytest.raises(stooq.StooqFormatError, match="header"):
        stooq.read_daily(path)


def test_stooq_rejects_intraday(tmp_path: Path) -> None:
    path = tmp_path / "x.us.txt"
    path.write_text(
        "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
        "X.US,5,20261030,153000,1,1,1,1,1,0\n"
    )
    with pytest.raises(stooq.StooqFormatError, match="daily"):
        stooq.read_daily(path)
