"""Sharadar's direct API (D-01). The key travels in the ``x-api-key`` header, never the URL.

SEP, TICKERS and ACTIONS go through eqpit (``eqpit_store``); this module holds the
access check and the SFP bulk table, which eqpit does not read.
"""

from __future__ import annotations

import csv
import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

from statarb.data.http import DEFAULT_TIMEOUT, get_with_retries

SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"

# The API answers a few tickers (AAPL and MSFT in SEP, SPY in SFP) without a key, and
# TICKERS needs no key at all. A key check must therefore ask for tickers outside the
# free tier, or a missing key would pass (equity-factor-pipeline, decisions M1).
KEY_CHECKS: dict[str, tuple[str, str]] = {
    "SEP": ("stocks", "GME"),
    "SFP": ("funds", "XLE"),
    "TICKERS": ("tickers", "GME"),
}


@dataclass(frozen=True, slots=True)
class AccessResult:
    """Outcome of a one-row query on one table."""

    dataset: str
    table: str
    ticker: str
    rows: int
    columns: tuple[str, ...]


def sharadar_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return a client for the API with the key in its headers."""
    if not api_key:
        raise ValueError("SHARADAR_API_KEY is not set")
    return httpx.Client(
        base_url=SHARADAR_BASE_URL,
        headers={"x-api-key": api_key},
        transport=transport,
        timeout=DEFAULT_TIMEOUT,
    )


def check_access(
    dataset: str,
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> AccessResult:
    """Fetch one row of ``dataset`` (``SEP``, ``SFP`` or ``TICKERS``) for its check ticker.

    Raises ``httpx.HTTPStatusError`` on a refused key (HTTP 403 ``Exceeds free tier``).
    """
    table, ticker = KEY_CHECKS[dataset]
    with sharadar_client(api_key, transport) as client:
        response = get_with_retries(
            client, f"/data/{table}", params={"ticker": ticker, "limit": 1}, attempts=3
        )
    reader = csv.reader(io.StringIO(response.text))
    header = next(reader, [])
    rows = sum(1 for row in reader if row)
    return AccessResult(dataset, table, ticker, rows, tuple(header))


SFP_DATASET = "SFP"
SFP_TABLE = "funds"
_SFP_TEXT = ("ticker",)
_SFP_DATES = ("date", "lastupdated")


def sfp_to_parquet(archive: Path, target: Path) -> int:
    """Convert the SFP bulk zip to Parquet with fixed types; returns the row count."""
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.namelist()[0]
        workdir = target.parent / ".extract-SFP"
        extracted = Path(bundle.extract(member, workdir))
    header = extracted.open(encoding="utf-8").readline().strip().split(",")
    overrides: dict[str, pl.DataType | type[pl.DataType]] = {
        name: pl.Date if name in _SFP_DATES else pl.String if name in _SFP_TEXT else pl.Float64
        for name in header
    }
    pl.scan_csv(extracted, schema_overrides=overrides).sink_parquet(target, compression="zstd")
    shutil.rmtree(workdir)
    return int(pl.scan_parquet(target).select(pl.len()).collect().item())


def pull_funds(raw_root: Path, api_key: str, refresh: bool = False) -> tuple[Path, bool]:
    """Download SFP (ETF and fund prices) as an eqpit-style snapshot at 10 MB/s.

    Returns the snapshot folder and whether anything was downloaded. An existing snapshot is
    kept unless ``refresh``.
    """
    from eqpit.providers.sharadar import SharadarClient
    from eqpit.snapshot import latest_snapshot, list_snapshots, open_snapshot

    existing = list_snapshots(raw_root, SFP_DATASET)
    if existing and not refresh:
        return existing[-1], False
    counts: dict[str, int] = {}
    with SharadarClient(
        api_key, max_bytes_per_second=10_000_000
    ) as client:  # pragma: no cover - network
        details = client.status(SFP_TABLE)
        with open_snapshot(
            raw_root,
            "sharadar",
            SFP_DATASET,
            False,
            source=f"{SHARADAR_BASE_URL}/data/{SFP_TABLE}?years=full",
            row_counts=counts,
            metadata={"vendor_modified": str(details.get("modified", ""))},
        ) as staging:
            archive = staging / f"{SFP_TABLE}.csv.zip"
            client.download(SFP_TABLE, archive)
            counts[f"{SFP_DATASET}.parquet"] = sfp_to_parquet(
                archive, staging / f"{SFP_DATASET}.parquet"
            )
    return latest_snapshot(raw_root, SFP_DATASET), True


def etf_returns_from_sfp(sfp: pl.LazyFrame, etfs: list[str]) -> pl.DataFrame:
    """Daily total returns of ``etfs`` from SFP rows.

    ``closeadj`` carries splits and distributions, so its ratio is the total return.
    ``close`` and ``volume`` are split-adjusted; they are restated to the unadjusted basis
    with ``closeunadj / close``, as for SEP.
    """
    basis = pl.col("closeunadj") / pl.col("close")
    rows = (
        sfp.filter(pl.col("ticker").is_in(etfs), pl.col("closeadj") > 0)
        .select(
            "date",
            pl.col("ticker").alias("etf"),
            pl.col("closeadj"),
            pl.col("closeunadj").alias("close_unadj"),
            (pl.col("volume") / basis).alias("volume_unadj"),
        )
        .collect()
        .sort("etf", "date")
    )
    return rows.select(
        "date",
        "etf",
        (pl.col("closeadj") / pl.col("closeadj").shift(1).over("etf") - 1.0).alias("ret_total"),
        pl.col("close_unadj").alias("close"),
        pl.col("volume_unadj").alias("volume"),
        pl.col("date").min().over("etf").alias("inception_date"),
    )
