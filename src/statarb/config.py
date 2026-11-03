"""Validated YAML run configs (spec section 10).

A config may name a parent with ``extends``; the child's keys are merged over the parent's.
Sections are added as the milestones that use them land, and unknown keys are an error, so
a typo never passes silently.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

StrategyFamily = Literal[
    "pairs_distance", "pairs_coint", "al_pca", "al_etf", "intraday_pairs", "crypto_pairs"
]
DataProfile = Literal["headline", "prototype"]
SecurityType = Literal["common", "etf", "other"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSection(_Section):
    """Run identity and provenance label."""

    name: str
    strategy_family: StrategyFamily
    data_profile: DataProfile = "headline"
    seed: int = 20261030


class DataConfig(_Section):
    """Provider and sample period (D-01, D-03)."""

    provider: Literal["sharadar", "crsp", "free"] = "sharadar"
    start: date = date(1998, 1, 2)
    end: date = date(2025, 12, 31)
    calendar: str = "XNYS"


class UniverseConfig(_Section):
    """Eligibility at each formation date (FR-04, DQ-02, DQ-05, D-06)."""

    security_types: tuple[SecurityType, ...] = ("common",)
    exchanges: tuple[str, ...] = ("NYSE", "AMEX", "NASDAQ")
    rank_by: Literal["mktcap_usd", "dollar_volume"] = "mktcap_usd"
    top_n: int | None = Field(default=1000, ge=1)
    min_price_usd: float | None = 5.0
    min_mktcap_usd: float | None = None
    min_traded_frac: float = Field(default=0.95, ge=0.0, le=1.0)
    require_full_history: bool = False
    survivors_only: bool = False


class WindowsConfig(_Section):
    """Formation and trading lengths in months (D-04, D-13)."""

    formation_months: int = Field(default=12, ge=1)
    trading_months: int = Field(default=6, ge=1)
    step_months: int = Field(default=1, ge=1)
    execution_lag_days: int = Field(default=1, ge=0)


class RunConfig(_Section):
    """A fully resolved run config."""

    run: RunSection
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    windows: WindowsConfig = WindowsConfig()

    @model_validator(mode="after")
    def _free_data_is_prototype(self) -> RunConfig:
        if self.data.provider == "free" and self.run.data_profile != "prototype":
            raise ValueError("data.provider = free forces run.data_profile = prototype")
        return self


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` merged into a copy of ``base``; mappings merge, everything else replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"circular extends at {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a mapping")
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    return deep_merge(_read(path.parent / str(parent), (*seen, resolved)), loaded)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.sub=value`` strings; values are parsed as YAML scalars or lists."""
    result = raw
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} is not key=value")
        nested: Any = yaml.safe_load(text)
        for part in reversed(key.split(".")):
            nested = {part: nested}
        result = deep_merge(result, nested)
    return result


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    """Read ``path``, resolve ``extends`` and ``--set`` overrides, and validate."""
    return RunConfig.model_validate(apply_overrides(_read(path), overrides or []))
