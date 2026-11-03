"""Shared fixtures: an isolated environment and a fake Stooq tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from statarb.settings import Settings

ENV_VARS = (
    "SEC_USER_AGENT",
    "STATARB_EQUITY_PROVIDER",
    "WRDS_USERNAME",
    "SHARADAR_API_KEY",
    "STOOQ_DIR",
    "TIINGO_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_DATA_FEED",
    "STATARB_BORROW_SOURCE",
    "DATA_ROOT",
    "STATARB_DATA_DIR",
    "STATARB_RESULTS_DIR",
    "STATARB_LOG_LEVEL",
)

STOOQ_HEADER = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sees the developer's ``.env`` or environment."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def write_stooq_file(path: Path, ticker: str, rows: list[str]) -> None:
    """Write a file in Stooq's daily layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{ticker},D,{row},0\n" for row in rows)
    path.write_text(STOOQ_HEADER + body, encoding="utf-8")


@pytest.fixture
def stooq_dir(tmp_path: Path) -> Path:
    """A three-file snapshot, with SPY in a numbered subfolder as in the real one."""
    root = tmp_path / "stooq"
    us = root / "d_us_txt" / "data" / "daily" / "us"
    write_stooq_file(
        us / "nyse etfs" / "2" / "spy.us.txt",
        "SPY.US",
        [
            "20261029,000000,760.0,762.0,755.0,758.0,1000.5",
            "20261030,000000,761.38,761.46,750.36,755.72,77666350",
        ],
    )
    write_stooq_file(
        us / "nasdaq stocks" / "1" / "aapl.us.txt",
        "AAPL.US",
        ["20261030,000000,1.0,2.0,0.5,1.5,100"],
    )
    write_stooq_file(
        us / "nysemkt stocks" / "abc.us.txt",
        "ABC.US",
        ["20261030,000000,1.0,2.0,0.5,1.5,100"],
    )
    (us / "nysemkt stocks" / "readme.md").write_text("not data", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path: Path, stooq_dir: Path) -> Settings:
    """Settings with a valid User-Agent, no Sharadar key and a data root under tmp_path."""
    return Settings(
        _env_file=None,
        sec_user_agent="Test User test@example.com",
        stooq_dir=stooq_dir,
        statarb_data_dir=str(tmp_path / "data"),
        statarb_results_dir=tmp_path / "results",
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries happen at once in tests."""
    monkeypatch.setattr("statarb.data.http.BACKOFF_SECONDS", 0.0)
