"""Polars schemas for every processed table (spec section 7b), validated on write.

Validity intervals (``valid_from``, ``valid_to``) are inclusive; a null ``valid_to`` is open.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

Schema = Mapping[str, pl.DataType | type[pl.DataType]]

SECURITY_MASTER: Schema = {
    "security_id": pl.String,
    "ticker": pl.String,
    "name": pl.String,
    "security_type": pl.String,
    "exchange": pl.String,
    "first_date": pl.Date,
    "last_date": pl.Date,
    "delist_code": pl.Int32,
    "source": pl.String,
}

TICKER_HISTORY: Schema = {
    "security_id": pl.String,
    "ticker": pl.String,
    "name": pl.String,
    "security_type": pl.String,
    "exchange": pl.String,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
    "source": pl.String,
}

PRICES_DAILY: Schema = {
    "date": pl.Date,
    "security_id": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "close_is_mid": pl.Boolean,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "volume": pl.Float64,
    "ret_total": pl.Float64,
    "ret_delist": pl.Float64,
    "ret_delist_imputed": pl.Boolean,
    "tr_index": pl.Float64,
    "shares_out": pl.Float64,
    "mktcap_usd": pl.Float64,
    "source": pl.String,
}

INDUSTRY_HISTORY: Schema = {
    "security_id": pl.String,
    "sic": pl.Int32,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
    "source": pl.String,
    "is_point_in_time": pl.Boolean,
}

# close and volume are extra columns beyond the spec's four: the A-L ETF hedge leg needs
# them for spread and impact costs (M5).
ETF_RETURNS: Schema = {
    "date": pl.Date,
    "etf": pl.String,
    "ret_total": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "inception_date": pl.Date,
}

BORROW_RATES: Schema = {
    "date": pl.Date,
    "security_id": pl.String,
    "bucket": pl.String,
    "fee_rate_annual": pl.Float64,
    "rebate_rate_annual": pl.Float64,
    "available_shares": pl.Float64,
    "source": pl.String,
}

TABLES: Mapping[str, Schema] = {
    "security_master": SECURITY_MASTER,
    "ticker_history": TICKER_HISTORY,
    "prices_daily": PRICES_DAILY,
    "industry_history": INDUSTRY_HISTORY,
    "etf_returns": ETF_RETURNS,
    "borrow_rates": BORROW_RATES,
}

KEYS: Mapping[str, tuple[str, ...]] = {"borrow_rates": ("date",)}
"""Key columns where they are not the first two: schedule rows have no ``security_id``."""

BUILD_INFO = "build_info.json"


class SchemaError(ValueError):
    """A frame does not match its table schema."""


def conform(
    frame: pl.DataFrame, schema: Schema, keys: tuple[str, ...] | None = None
) -> pl.DataFrame:
    """Select the schema's columns in order, cast strictly; missing columns become null.

    Key columns, the first two unless ``keys`` says otherwise, must be present and non-null.
    """
    keys = keys if keys is not None else tuple(schema)[:2]
    missing = [name for name in keys if name not in frame.columns]
    if missing:
        raise SchemaError(f"missing key columns {missing}")
    extra = sorted(set(frame.columns) - set(schema))
    if extra:
        raise SchemaError(f"columns outside the schema: {extra}")
    shaped = frame.select(
        pl.col(name).cast(dtype, strict=True)
        if name in frame.columns
        else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
    )
    nulls = [name for name in keys if shaped[name].null_count()]
    if nulls:
        raise SchemaError(f"null keys in {nulls}")
    return shaped


def write_table(frame: pl.DataFrame, processed: Path, table: str) -> int:
    """Validate and write ``table``; daily tables are partitioned by year (section 7d)."""
    shaped = conform(frame, TABLES[table], KEYS.get(table))
    target = processed / table
    if target.exists():
        for old in sorted(target.rglob("*.parquet")):
            old.unlink()
    target.mkdir(parents=True, exist_ok=True)
    if table == "prices_daily":
        years = shaped.with_columns(pl.col("date").dt.year().alias("_year"))
        for year in sorted(years["_year"].unique().to_list()):
            part = years.filter(pl.col("_year") == year).drop("_year").sort("security_id", "date")
            folder = target / f"year={year}"
            folder.mkdir(parents=True, exist_ok=True)
            part.write_parquet(folder / "part-0.parquet", compression="zstd", statistics=True)
    else:
        shaped.write_parquet(target / "part-0.parquet", compression="zstd", statistics=True)
    return shaped.height


def write_build_info(processed: Path, info: Mapping[str, Any]) -> None:
    """Record provenance for the processed tables (NFR-12)."""
    processed.mkdir(parents=True, exist_ok=True)
    (processed / BUILD_INFO).write_text(
        json.dumps(dict(info), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_build_info(processed: Path) -> dict[str, Any]:
    """Provenance of the processed tables, or an empty mapping before the first build."""
    path = processed / BUILD_INFO
    if not path.exists():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
