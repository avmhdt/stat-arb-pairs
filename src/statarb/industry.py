"""SIC to sector-ETF map (``configs/sic_to_etf.yaml``) and industry group keys (FR-07)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

Grouping = Literal["none", "sic2", "sic3", "etf_sector"]


@dataclass(frozen=True, slots=True)
class SicRange:
    """SIC codes ``first`` to ``last`` inclusive map to ``etf``."""

    first: int
    last: int
    etf: str


@dataclass(frozen=True, slots=True)
class SicEtfMap:
    """Ordered SIC ranges (the first match wins) and per-ETF fallbacks before inception."""

    ranges: tuple[SicRange, ...]
    fallback: dict[str, str]
    market: str

    @property
    def etfs(self) -> list[str]:
        """Every ETF the map can name, market proxy included, sorted."""
        names = {item.etf for item in self.ranges} | set(self.fallback.values()) | {self.market}
        return sorted(names | set(self.fallback))

    def etf_for(self, sic: int | None) -> str | None:
        """The sector ETF for one SIC code, or None if no range covers it."""
        if sic is None:
            return None
        for item in self.ranges:
            if item.first <= sic <= item.last:
                return item.etf
        return None

    def usable(self, etf: str, inception: dict[str, date], cutoff: date) -> str | None:
        """``etf`` if it started trading by ``cutoff`` (DQ-07), else its fallback chain's first."""
        seen: set[str] = set()
        current: str | None = etf
        while current is not None and current not in seen:
            seen.add(current)
            started = inception.get(current)
            if started is not None and started <= cutoff:
                return current
            current = self.fallback.get(current)
        return None


def load_sic_etf_map(path: Path) -> SicEtfMap:
    """Read the YAML map: ``ranges: [[first, last, etf], ...]``, ``fallback``, ``market``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ranges = tuple(SicRange(int(a), int(b), str(etf)) for a, b, etf in raw["ranges"])
    for item in ranges:
        if item.first > item.last:
            raise ValueError(f"empty SIC range {item}")
    return SicEtfMap(
        ranges, {str(k): str(v) for k, v in raw.get("fallback", {}).items()}, str(raw["market"])
    )


def group_keys(
    industry: pl.DataFrame, grouping: Grouping, sic_map: SicEtfMap | None = None
) -> pl.DataFrame:
    """``security_id, group`` for one grouping scheme; securities without a SIC drop out.

    ``none`` puts everything in group ``all``; ``sic2`` and ``sic3`` use the first two or
    three SIC digits; ``etf_sector`` uses the sector ETF from ``sic_map``.
    """
    if grouping == "none":
        return industry.select("security_id", pl.lit("all").alias("group")).unique(
            maintain_order=True
        )
    coded = industry.filter(pl.col("sic").is_not_null())
    if grouping == "sic2":
        key = (pl.col("sic") // 100).cast(pl.String).str.zfill(2)
    elif grouping == "sic3":
        key = (pl.col("sic") // 10).cast(pl.String).str.zfill(3)
    else:
        if sic_map is None:
            raise ValueError("etf_sector grouping needs the SIC to ETF map")
        etfs = [sic_map.etf_for(int(sic)) for sic in coded["sic"].to_list()]
        coded = coded.with_columns(pl.Series("_etf", etfs, dtype=pl.String))
        key = pl.col("_etf")
    return coded.select("security_id", key.alias("group")).drop_nulls("group").sort("security_id")
