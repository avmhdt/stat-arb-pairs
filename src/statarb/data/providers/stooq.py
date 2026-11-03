"""The owner's local Stooq bulk daily snapshot (P-12), the labelled prototype fallback.

Files are located by walking the tree once into a ticker -> path index, never by guessing
paths: some sit in numbered subfolders such as ``nyse etfs/2/spy.us.txt``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

from statarb.data.manifest import publish

SNAPSHOT_DATE = date(2026, 10, 30)
US_DAILY = Path("d_us_txt") / "data" / "daily" / "us"
HEADER = ("<TICKER>", "<PER>", "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>")
INDEX_SCHEMA = {
    "ticker": pl.String,
    "symbol": pl.String,
    "exchange": pl.String,
    "security_type": pl.String,
    "path": pl.String,
}


class StooqFormatError(ValueError):
    """A Stooq file does not have the documented layout."""


def build_index(stooq_dir: Path) -> pl.DataFrame:
    """Walk the US daily tree once and return one row per file.

    Parameters
    ----------
    stooq_dir
        ``STOOQ_DIR``, the folder holding ``d_us_txt``.

    Returns
    -------
    polars.DataFrame
        Columns ``ticker`` (``SPY.US``), ``symbol`` (``SPY``), ``exchange`` (``NYSE``,
        ``NASDAQ``, ``NYSEMKT``), ``security_type`` (``etf`` or ``stock``) and ``path``
        relative to ``stooq_dir``, sorted by ticker.
    """
    root = stooq_dir / US_DAILY
    if not root.is_dir():
        raise FileNotFoundError(f"no US daily tree under {stooq_dir}")
    rows: list[dict[str, str]] = []
    for folder, _dirs, files in os.walk(root):
        relative = Path(folder).relative_to(root)
        if not relative.parts:
            continue
        exchange, _, kind = relative.parts[0].partition(" ")
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            ticker = name[: -len(".txt")].upper()
            rows.append(
                {
                    "ticker": ticker,
                    "symbol": ticker.removesuffix(".US"),
                    "exchange": exchange.upper(),
                    "security_type": "etf" if kind.startswith("etf") else "stock",
                    "path": (US_DAILY / relative / name).as_posix(),
                }
            )
    return pl.DataFrame(rows, schema=INDEX_SCHEMA).sort("ticker")


def duplicate_tickers(index: pl.DataFrame) -> list[str]:
    """Tickers that appear under more than one path."""
    counts = index.group_by("ticker").len()
    return sorted(counts.filter(pl.col("len") > 1)["ticker"].to_list())


def read_daily(path: Path) -> pl.DataFrame:
    """Read one Stooq daily file into ``date, open, high, low, close, volume``.

    The header is checked exactly; ``<PER>`` must be ``D`` and ``<TIME>`` ``000000``.
    Volumes may be fractional, so they are read as floats.
    """
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    if tuple(first.split(",")[: len(HEADER)]) != HEADER:
        raise StooqFormatError(f"{path.name}: unexpected header {first!r}")
    frame = pl.read_csv(
        path,
        schema_overrides={
            "<DATE>": pl.String,
            "<TIME>": pl.String,
            "<VOL>": pl.Float64,
            "<OPEN>": pl.Float64,
            "<HIGH>": pl.Float64,
            "<LOW>": pl.Float64,
            "<CLOSE>": pl.Float64,
        },
    )
    if frame.height and (
        frame["<PER>"].unique().to_list() != ["D"]
        or frame["<TIME>"].unique().to_list() != ["000000"]
    ):
        raise StooqFormatError(f"{path.name}: not a daily file")
    return frame.select(
        pl.col("<DATE>").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        pl.col("<OPEN>").alias("open"),
        pl.col("<HIGH>").alias("high"),
        pl.col("<LOW>").alias("low"),
        pl.col("<CLOSE>").alias("close"),
        pl.col("<VOL>").alias("volume"),
    )


def snapshot_root(raw_root: Path) -> Path:
    """``raw/stooq/snapshot=<snapshot date>``."""
    return raw_root / f"snapshot={SNAPSHOT_DATE:%Y%m%d}"


def ingest_snapshot(
    stooq_dir: Path, raw_root: Path, tickers: set[str] | None = None
) -> tuple[Path, bool]:
    """Convert the US daily tree once into ``index.parquet`` and ``us_daily.parquet``.

    The source on E: is read once; later runs use the Parquet copy (P-12). Returns the
    snapshot folder and whether anything was written. ``tickers`` (``SPY.US`` style) limits
    the copy, for tests and quick prototypes.
    """
    target = snapshot_root(raw_root)
    daily = target / "us_daily.parquet"
    if daily.exists():
        return target, False
    index = build_index(stooq_dir)
    if tickers is not None:
        index = index.filter(pl.col("ticker").is_in(sorted(tickers)))
    frames = []
    for ticker, path in index.select("ticker", "path").iter_rows():
        frame = read_daily(stooq_dir / path)
        if frame.height:
            frames.append(frame.with_columns(pl.lit(ticker).alias("ticker")))
    staging = target.with_name(f".staging-{target.name}")
    staging.mkdir(parents=True, exist_ok=True)
    index.write_parquet(staging / "index.parquet", compression="zstd")
    pl.concat(frames, how="vertical").sort("ticker", "date").write_parquet(
        staging / "us_daily.parquet", compression="zstd"
    )
    publish(staging, target)
    return target, True
