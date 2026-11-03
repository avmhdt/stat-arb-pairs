"""``statarb smoke``: settings, data access and one SEC call (M1, AC-W-15).

Checks one Sharadar SEP ticker and SFP access when ``SHARADAR_API_KEY`` is set, else the
local Stooq snapshot, plus one SEC submissions call. Nothing is written outside
``<data root>/raw/_smoke/``. Secrets are never printed: only whether a key is set.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from statarb.data.providers import edgar, sharadar, stooq
from statarb.settings import Settings, SettingsError, is_on_c_drive, validate_user_agent

Status = Literal["ok", "warn", "fail"]

CHECK_CIK = 320193  # Apple Inc.
STOOQ_CHECK_TICKER = "SPY.US"
MIN_FREE_GB = 50.0  # P-10 reserve


@dataclass(frozen=True, slots=True)
class Check:
    """One line of the smoke report."""

    name: str
    status: Status
    detail: str


def smoke_dir(settings: Settings) -> Path:
    """``<data root>/raw/_smoke``, the only place smoke writes to."""
    return settings.data_dir / "raw" / "_smoke"


def check_user_agent(settings: Settings) -> Check:
    """SEC_USER_AGENT holds a name and a contact email."""
    try:
        validate_user_agent(settings.sec_user_agent)
    except SettingsError as error:
        return Check("sec_user_agent", "fail", str(error))
    return Check("sec_user_agent", "ok", "set, with a contact email")


def check_data_dir(settings: Settings, on_c: Callable[[Path], bool] = is_on_c_drive) -> list[Check]:
    """Data root resolves off C:, is writable, and has the P-10 reserve free."""
    root = settings.data_dir.expanduser().resolve()
    if on_c(root):
        return [Check("data_dir", "fail", f"{root} is on C:, where no project data may live")]
    target = smoke_dir(settings)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return [Check("data_dir", "fail", f"{root} not writable: {error}")]
    free_gb = shutil.disk_usage(target).free / 1e9
    return [
        Check("data_dir", "ok", str(root)),
        Check(
            "disk_free",
            "ok" if free_gb >= MIN_FREE_GB else "warn",
            f"{free_gb:,.0f} GB free, {MIN_FREE_GB:.0f} GB reserved (P-10)",
        ),
    ]


def check_sharadar(settings: Settings, transport: httpx.BaseTransport | None = None) -> list[Check]:
    """One-row SEP, SFP and TICKERS queries on tickers outside the free tier."""
    api_key = settings.sharadar_api_key.get_secret_value().strip()
    checks: list[Check] = []
    for dataset in sharadar.KEY_CHECKS:
        name = f"sharadar_{dataset.lower()}"
        try:
            result = sharadar.check_access(dataset, api_key, transport=transport)
        except httpx.HTTPStatusError as error:
            checks.append(Check(name, "fail", f"HTTP {error.response.status_code}"))
            continue
        except httpx.HTTPError as error:
            checks.append(Check(name, "fail", type(error).__name__))
            continue
        status: Status = "ok" if result.rows >= 1 else "fail"
        checks.append(
            Check(name, status, f"table {result.table}, {result.ticker}: {result.rows} row(s)")
        )
    return checks


def check_stooq(settings: Settings) -> list[Check]:
    """Index the local snapshot once, write the index, and read one ETF file."""
    if settings.stooq_dir is None:
        return [Check("stooq", "fail", "STOOQ_DIR not set and SHARADAR_API_KEY absent")]
    try:
        index = stooq.build_index(settings.stooq_dir)
    except FileNotFoundError as error:
        return [Check("stooq", "fail", str(error))]
    target = smoke_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    index.write_parquet(target / "stooq_index.parquet", compression="zstd")
    duplicates = stooq.duplicate_tickers(index)
    checks = [
        Check(
            "stooq_index",
            "ok" if index.height else "fail",
            f"{index.height:,} files, {len(duplicates)} duplicate tickers",
        )
    ]
    match = index.filter(index["ticker"] == STOOQ_CHECK_TICKER)
    if match.is_empty():
        checks.append(Check("stooq_read", "fail", f"{STOOQ_CHECK_TICKER} not in the index"))
        return checks
    try:
        frame = stooq.read_daily(settings.stooq_dir / match["path"][0])
    except stooq.StooqFormatError as error:
        checks.append(Check("stooq_read", "fail", str(error)))
        return checks
    last = frame.select(pl.col("date").max()).item()
    checks.append(
        Check(
            "stooq_read",
            "ok" if frame.height else "fail",
            f"{STOOQ_CHECK_TICKER}: {frame.height:,} rows, last {last}, "
            f"snapshot {stooq.SNAPSHOT_DATE}",
        )
    )
    return checks


def check_sec(settings: Settings, transport: httpx.BaseTransport | None = None) -> Check:
    """One submissions call; the JSON is kept in the smoke folder."""
    try:
        with edgar.EdgarClient(settings.sec_user_agent, transport=transport) as client:
            payload = client.submissions(CHECK_CIK)
    except SettingsError as error:
        return Check("sec_submissions", "fail", str(error))
    except httpx.HTTPStatusError as error:
        return Check("sec_submissions", "fail", f"HTTP {error.response.status_code}")
    except httpx.HTTPError as error:
        return Check("sec_submissions", "fail", type(error).__name__)
    target = smoke_dir(settings)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"sec_submissions_CIK{CHECK_CIK:010d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    sic = payload.get("sic", "")
    status: Status = "ok" if sic else "fail"
    return Check(
        "sec_submissions", status, f"CIK {CHECK_CIK:010d}: {payload.get('name')}, SIC {sic}"
    )


def run_smoke(
    settings: Settings,
    transport: httpx.BaseTransport | None = None,
    on_c: Callable[[Path], bool] = is_on_c_drive,
) -> list[Check]:
    """Run every smoke check in order and return the report."""
    checks = [check_user_agent(settings), *check_data_dir(settings, on_c)]
    if checks[1].status == "fail":
        return checks
    if settings.has_sharadar_key:
        checks.append(Check("sharadar_api_key", "ok", "set"))
        checks.extend(check_sharadar(settings, transport))
    else:
        checks.append(
            Check("sharadar_api_key", "warn", "not set, needed before M2 (P-06); using Stooq")
        )
        checks.extend(check_stooq(settings))
    checks.append(check_sec(settings, transport))
    return checks
