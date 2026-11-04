"""IBKR short-stock availability snapshots (FR-34, DQ-12): forward-only, never back-filled.

Interactive Brokers publishes per-country pipe-delimited files by anonymous FTP
(``ftp2.interactivebrokers.com``, user ``shortstock``). Each run stores today's US file as a
dated snapshot under the data root; raw files are not redistributed.
"""

from __future__ import annotations

import ftplib
import io
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

HOST = "ftp2.interactivebrokers.com"
USER = "shortstock"
US_FILE = "usa.txt"
COLUMNS = ("SYM", "CUR", "NAME", "CON", "ISIN", "REBATERATE", "FEERATE", "AVAILABLE")


def parse(text: str) -> pl.DataFrame:
    """``symbol, currency, rebate_rate_annual, fee_rate_annual, available_shares`` (decimals)."""
    header: list[str] | None = None
    rows = []
    for line in text.splitlines():
        if line.startswith("#SYM"):
            header = [cell for cell in line.lstrip("#").split("|") if cell]
            continue
        if line.startswith("#") or not line.strip() or header is None:
            continue
        cells = line.split("|")[: len(header)]
        rows.append(dict(zip(header, cells, strict=False)))
    frame = pl.DataFrame(rows, schema=dict.fromkeys(header or COLUMNS, pl.String))

    def number(name: str) -> pl.Expr:
        return pl.col(name).str.strip_chars().str.replace(r"^>", "").cast(pl.Float64, strict=False)

    return frame.select(
        pl.col("SYM").alias("symbol"),
        pl.col("CUR").alias("currency"),
        (number("REBATERATE") / 100.0).alias("rebate_rate_annual"),
        (number("FEERATE") / 100.0).alias("fee_rate_annual"),
        number("AVAILABLE").alias("available_shares"),
    )


def fetch(raw_root: Path) -> tuple[Path, bool]:  # pragma: no cover - network
    """Download today's US file once per UTC day: the snapshot folder and whether it is new."""
    day = datetime.now(UTC).strftime("%Y%m%d")
    target = raw_root / f"snapshot={day}"
    if (target / US_FILE).exists():
        return target, False
    buffer = io.BytesIO()
    with ftplib.FTP(HOST, timeout=60) as ftp:
        ftp.login(USER, "")
        ftp.retrbinary(f"RETR {US_FILE}", buffer.write)
    target.mkdir(parents=True, exist_ok=True)
    (target / US_FILE).write_bytes(buffer.getvalue())
    parse(buffer.getvalue().decode("utf-8", errors="replace")).write_parquet(target / "usa.parquet")
    return target, True


def compare(snapshot: pl.DataFrame, latest: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """IBKR fees by this project's size bucket against the assumed schedule (FR-34).

    ``latest`` has ``ticker, rank`` (market-cap rank of listed common stocks on the last
    date); ``schedule`` the assumed ``bucket, fee_rate_annual``.
    """
    bucket = (
        pl.when(pl.col("rank") <= 500)
        .then(pl.lit("top500"))
        .when(pl.col("rank") <= 1000)
        .then(pl.lit("top1000"))
        .otherwise(pl.lit("rest"))
    )
    joined = latest.join(
        snapshot.filter(pl.col("currency") == "USD"), left_on="ticker", right_on="symbol"
    ).with_columns(bucket.alias("bucket"))
    table = joined.group_by("bucket").agg(
        pl.len().alias("names_matched"),
        pl.col("fee_rate_annual").median().alias("ibkr_fee_median"),
        pl.col("fee_rate_annual").quantile(0.75).alias("ibkr_fee_p75"),
        pl.col("fee_rate_annual").quantile(0.90).alias("ibkr_fee_p90"),
        pl.col("fee_rate_annual").mean().alias("ibkr_fee_mean"),
    )
    return (
        table.join(
            schedule.select("bucket", pl.col("fee_rate_annual").alias("assumed_fee")), on="bucket"
        )
        .with_columns((pl.col("ibkr_fee_mean") / pl.col("assumed_fee")).alias("mean_over_assumed"))
        .sort("bucket")
    )
