"""The processed panel that views slice: prices, listings, industries, ETFs, borrow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from statarb.data import schemas

TABLES = ("prices", "etf_returns", "borrow")


@dataclass(frozen=True)
class MarketData:
    """Everything a run reads, held in memory and sorted by ``security_id, date``.

    Code never reads these frames directly: it gets a ``FormationView`` or a
    ``TradingView`` (FR-02).
    """

    prices: pl.DataFrame
    master: pl.DataFrame
    ticker_history: pl.DataFrame
    industry: pl.DataFrame
    etf_returns: pl.DataFrame
    borrow: pl.DataFrame
    data_profile: str = "headline"
    manifest_hash: str = ""
    sessions: list[date] = field(init=False)

    def __post_init__(self) -> None:
        dates = self.prices["date"].unique().sort().to_list()
        object.__setattr__(self, "sessions", dates)

    def table(self, name: str) -> pl.DataFrame:
        """A dated table by name, for ``FormationView.query``."""
        if name not in TABLES:
            raise KeyError(f"unknown table {name!r}; expected one of {TABLES}")
        frame: pl.DataFrame = getattr(self, name)
        return frame

    def replace_prices(self, prices: pl.DataFrame) -> MarketData:
        """A copy with different prices, used by the leakage test's perturbations."""
        return MarketData(
            prices=prices.sort("security_id", "date"),
            master=self.master,
            ticker_history=self.ticker_history,
            industry=self.industry,
            etf_returns=self.etf_returns,
            borrow=self.borrow,
            data_profile=self.data_profile,
            manifest_hash=self.manifest_hash,
        )


def load_market_data(
    processed: Path,
    start: date,
    end: date,
    price_columns: list[str] | None = None,
    security_types: tuple[str, ...] | None = None,
    ever_min_mktcap_usd: float | None = None,
) -> MarketData:
    """Read processed tables for ``[start, end]``.

    Parameters
    ----------
    processed
        ``<data root>/processed``.
    start, end
        Inclusive date range for prices and ETF returns.
    price_columns
        Price columns to keep besides ``date`` and ``security_id``; all when omitted.
    security_types
        Keep prices only for these master types, which keeps memory down.
    ever_min_mktcap_usd
        Keep only securities whose market cap reaches this level on some day in the range.
        It changes no universe: a name below the floor on every day fails it at every T.
    """
    info = schemas.read_build_info(processed)
    master = pl.read_parquet(processed / "security_master" / "*.parquet")
    scan = pl.scan_parquet(processed / "prices_daily" / "**" / "*.parquet", hive_partitioning=False)
    scan = scan.filter(pl.col("date").is_between(start, end))
    if security_types is not None:
        wanted = master.filter(pl.col("security_type").is_in(list(security_types)))["security_id"]
        scan = scan.filter(pl.col("security_id").is_in(wanted.to_list()))
    if ever_min_mktcap_usd is not None:
        big = (
            scan.group_by("security_id")
            .agg(pl.col("mktcap_usd").max())
            .filter(pl.col("mktcap_usd") >= ever_min_mktcap_usd)
            .select("security_id")
            .collect()
        )
        scan = scan.join(big.lazy(), on="security_id", how="semi")
    if price_columns is not None:
        scan = scan.select("date", "security_id", *price_columns)
    prices = scan.collect().sort("security_id", "date").rechunk()
    etf = pl.read_parquet(processed / "etf_returns" / "*.parquet").filter(
        pl.col("date").is_between(start, end)
    )
    return MarketData(
        prices=prices,
        master=master,
        ticker_history=pl.read_parquet(processed / "ticker_history" / "*.parquet"),
        industry=pl.read_parquet(processed / "industry_history" / "*.parquet"),
        etf_returns=etf.sort("etf", "date"),
        borrow=pl.read_parquet(processed / "borrow_rates" / "*.parquet"),
        data_profile=str(info.get("data_profile", "prototype")),
        manifest_hash=str(info.get("manifest_hash", "")),
    )
