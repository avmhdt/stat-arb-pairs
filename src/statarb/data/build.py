"""Raw snapshots to processed tables (``statarb build``).

Two profiles:

- ``headline`` (Sharadar through eqpit): survivorship-free, point-in-time SIC history,
  share counts from filings, delisting returns imputed per DQ-03.
- ``prototype`` (the local Stooq snapshot plus EDGAR current SIC): survivorship-biased,
  adjusted prices only, no share counts, SIC not point in time (DQ-06).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

from statarb.data import manifest, schemas
from statarb.data.providers import sharadar as sharadar_provider
from statarb.industry import SicEtfMap

log = structlog.get_logger(__name__)

ALL_TABLES = ("prices", "industry", "etf", "borrow")
FLAG_MIDPOINT = 1


def _shr(column: str) -> pl.Expr:
    """eqpit's ``SHR:<permaticker>`` as this repo's ``shr:<permaticker>`` (DQ-09)."""
    return pl.col(column).str.replace("^SHR:", "shr:")


def _security_type(share_type: pl.Expr) -> pl.Expr:
    """CRSP share codes 10 and 11 hold US commons and REITs; everything else is ``other``."""
    return (
        pl.when(share_type.is_in(["common", "reit"]))
        .then(pl.lit("common"))
        .otherwise(pl.lit("other"))
    )


def tr_index(ret: pl.Expr, by: str = "security_id") -> pl.Expr:
    """Cumulative total-return index, 1.0 on each security's first date (DQ-04)."""
    return (1.0 + ret.fill_null(0.0)).cum_prod().over(by)


def attach_delistings(prices: pl.DataFrame, delistings: pl.DataFrame) -> pl.DataFrame:
    """Put each delisting return on the security's last price row on or before its date.

    ``delistings`` has ``security_id, delist_date, dlret, imputed``. A delisting with no
    price row on or before its date is dropped and counted by DQ-03.
    """
    last_rows = (
        prices.select("security_id", "date")
        .join(delistings.select("security_id", "delist_date"), on="security_id", how="inner")
        .filter(pl.col("date") <= pl.col("delist_date"))
        .group_by("security_id")
        .agg(pl.col("date").max())
        .join(delistings.select("security_id", "dlret", "imputed"), on="security_id")
    )
    return prices.join(last_rows, on=["security_id", "date"], how="left").rename(
        {"dlret": "ret_delist", "imputed": "ret_delist_imputed"}
    )


def sharadar_tables(store: Any) -> dict[str, pl.DataFrame]:
    """Map eqpit's securities, prices and delistings onto the processed schemas."""
    windows = store.securities.with_columns(
        _shr("security_id").alias("security_id"),
        _security_type(pl.col("share_type")).alias("security_type"),
    )
    prices = store.prices.with_columns(_shr("security_id").alias("security_id")).sort(
        "security_id", "date"
    )
    delistings = store.delistings.select(
        _shr("security_id").alias("security_id"),
        "delist_date",
        "dlret",
        (pl.col("imputation_rule") != "none").alias("imputed"),
    )
    span = prices.group_by("security_id").agg(
        pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_price")
    )
    latest = (
        windows.sort("security_id", "valid_from")
        .group_by("security_id", maintain_order=True)
        .last()
    )
    delisted = delistings["security_id"].unique()
    master = latest.join(span, on="security_id", how="inner").select(
        "security_id",
        "ticker",
        "name",
        "security_type",
        "exchange",
        "first_date",
        pl.when(pl.col("security_id").is_in(delisted.to_list()))
        .then(pl.col("last_price"))
        .alias("last_date"),
        pl.lit(None, pl.Int32).alias("delist_code"),
        pl.lit("sharadar").alias("source"),
    )
    history = windows.select(
        "security_id",
        "ticker",
        "name",
        "security_type",
        "exchange",
        "valid_from",
        "valid_to",
        pl.lit("sharadar").alias("source"),
    )
    industry = windows.filter(pl.col("sic").is_not_null()).select(
        "security_id",
        pl.col("sic").cast(pl.Int32),
        "valid_from",
        "valid_to",
        pl.lit("sharadar_tickers").alias("source"),
        pl.lit(True).alias("is_point_in_time"),
    )
    daily = attach_delistings(prices, delistings).select(
        "date",
        "security_id",
        "open",
        "high",
        "low",
        "close",
        ((pl.col("price_flag") & FLAG_MIDPOINT) > 0).alias("close_is_mid"),
        "volume",
        pl.col("ret").alias("ret_total"),
        "ret_delist",
        "ret_delist_imputed",
        tr_index(pl.col("ret")).alias("tr_index"),
        "shares_out",
        pl.col("mcap").alias("mktcap_usd"),
        pl.lit("sharadar").alias("source"),
    )
    return {
        "security_master": master,
        "ticker_history": history,
        "industry_history": industry,
        "prices_daily": daily,
    }


def stooq_tables(
    daily: pl.DataFrame, index: pl.DataFrame, sic: pl.DataFrame | None
) -> dict[str, pl.DataFrame]:
    """Prototype tables from the Stooq snapshot and optional EDGAR current SIC.

    ``sic`` has ``symbol, sic``. Stooq prices are already adjusted, so ``close`` is not the
    unadjusted price the schema asks for; there are no share counts, so ``mktcap_usd`` is
    null and prototype universes rank by dollar volume.
    """
    stocks = daily.join(index.select("ticker", "symbol", "exchange", "security_type"), on="ticker")
    spans = stocks.group_by("ticker").agg(
        pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_seen")
    )
    ids = spans.with_columns(
        pl.format(
            "free:{}:{}", pl.col("ticker").str.replace(r"\.US$", ""), pl.col("first_date")
        ).alias("security_id")
    )
    snapshot_end = daily["date"].max()
    rows = stocks.join(ids.select("ticker", "security_id"), on="ticker").sort("security_id", "date")
    master = (
        index.join(ids, on="ticker")
        .select(
            "security_id",
            pl.col("symbol").alias("ticker"),
            pl.col("symbol").alias("name"),
            pl.when(pl.col("security_type") == "etf")
            .then(pl.lit("etf"))
            .otherwise(pl.lit("common"))
            .alias("security_type"),
            "exchange",
            "first_date",
            pl.when(pl.col("last_seen") < snapshot_end)
            .then(pl.col("last_seen"))
            .alias("last_date"),
            pl.lit(None, pl.Int32).alias("delist_code"),
            pl.lit("stooq").alias("source"),
        )
        .sort("security_id")
    )
    history = master.select(
        "security_id",
        "ticker",
        "name",
        "security_type",
        "exchange",
        pl.col("first_date").alias("valid_from"),
        pl.col("last_date").alias("valid_to"),
        "source",
    )
    ret = pl.col("close") / pl.col("close").shift(1).over("security_id") - 1.0
    prices = rows.select(
        "date",
        "security_id",
        "open",
        "high",
        "low",
        "close",
        pl.lit(False).alias("close_is_mid"),
        "volume",
        ret.alias("ret_total"),
        tr_index(ret).alias("tr_index"),
        pl.lit("stooq").alias("source"),
    )
    if sic is None:
        industry = pl.DataFrame(schema=dict(schemas.INDUSTRY_HISTORY))
    else:
        industry = master.join(sic, left_on="ticker", right_on="symbol").select(
            "security_id",
            pl.col("sic").cast(pl.Int32),
            pl.col("first_date").alias("valid_from"),
            pl.lit(None, pl.Date).alias("valid_to"),
            pl.lit("edgar_current").alias("source"),
            pl.lit(False).alias("is_point_in_time"),
        )
    return {
        "security_master": master,
        "ticker_history": history,
        "industry_history": industry.drop_nulls("sic"),
        "prices_daily": prices,
    }


def stooq_etf_returns(daily: pl.DataFrame, etfs: Sequence[str]) -> pl.DataFrame:
    """ETF total returns from the Stooq snapshot, whose prices are already adjusted."""
    wanted = [f"{name}.US" for name in etfs]
    rows = daily.filter(pl.col("ticker").is_in(wanted)).sort("ticker", "date")
    return rows.select(
        "date",
        pl.col("ticker").str.replace(r"\.US$", "").alias("etf"),
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_total"),
        "close",
        "volume",
        pl.col("date").min().over("ticker").alias("inception_date"),
    )


def borrow_rates(schedule: Path) -> pl.DataFrame:
    """The assumed borrow schedule as ``borrow_rates`` rows, one per size bucket (D-07)."""
    raw = yaml.safe_load(schedule.read_text(encoding="utf-8"))
    effective = raw["effective_from"]
    start = effective if isinstance(effective, date) else date.fromisoformat(str(effective))
    return pl.DataFrame(
        {
            "date": [start] * len(raw["buckets"]),
            "security_id": [None] * len(raw["buckets"]),
            "bucket": [str(item["bucket"]) for item in raw["buckets"]],
            "fee_rate_annual": [float(item["fee_rate_annual"]) for item in raw["buckets"]],
            "source": str(raw.get("source", "assumed")),
        },
        schema_overrides={"security_id": pl.String},
    )


def write_tables(processed: Path, tables: dict[str, pl.DataFrame]) -> dict[str, int]:
    """Validate and write every table; returns row counts."""
    counts = {}
    for name, frame in tables.items():
        counts[name] = schemas.write_table(frame, processed, name)
        log.info("processed_table", table=name, rows=counts[name])
    return counts


def build_info(
    data_dir: Path, profile: str, provider: str, counts: dict[str, int], sources: Sequence[str]
) -> dict[str, object]:
    """Provenance written next to the processed tables (NFR-12)."""
    return {
        "data_profile": profile,
        "provider": provider,
        "manifest_hash": manifest.manifest_hash(data_dir, sources),
        "row_counts": counts,
    }


def etf_list(sic_map: SicEtfMap) -> list[str]:
    """ETFs the map can name, the market proxy included."""
    return sic_map.etfs


def read_sfp(raw_root: Path) -> pl.LazyFrame:
    """The latest SFP snapshot as a lazy frame."""
    from eqpit.snapshot import latest_snapshot

    return pl.scan_parquet(latest_snapshot(raw_root, sharadar_provider.SFP_DATASET) / "SFP.parquet")
