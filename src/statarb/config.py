"""Validated YAML run configs (spec section 10).

A config may name a parent with ``extends``; the child's keys are merged over the parent's.
Sections are added as the milestones that use them land, and unknown keys are an error, so
a typo never passes silently.
"""

from __future__ import annotations

import itertools
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


class EgConfig(_Section):
    """Engle-Granger settings (FR-09). Only a constant enters the cointegrating regression."""

    trend: Literal["c"] = "c"
    autolag: Literal["aic", "bic"] | None = "aic"
    maxlag: int | None = Field(default=None, ge=0)


class JohansenConfig(_Section):
    """Johansen confirmation of FDR discoveries (FR-10); statsmodels tabulates 95% here."""

    enabled: bool = True
    det_order: int = 0
    max_k_ar_diff: int = Field(default=5, ge=0)
    level: float = Field(default=0.95, ge=0.95, le=0.95)


class FdrConfig(_Section):
    """FDR procedure and level (D-05)."""

    primary: Literal["bh", "by"] = "bh"
    also_report: tuple[Literal["bh", "by"], ...] = ("by",)
    q: float = Field(default=0.05, gt=0.0, lt=1.0)


class HalfLifeConfig(_Section):
    """Half-life filter in sessions (FR-11)."""

    min: float = 1.0
    max: float = 63.0


class SelectionConfig(_Section):
    """Pair selection (FR-08 to FR-11, D-05, D-06)."""

    method: Literal["distance", "engle_granger"] = "engle_granger"
    grouping: Literal["none", "sic2", "sic3", "etf_sector"] = "sic2"
    max_pairs_per_stock: int | None = Field(default=1, ge=1)
    top_k: int = Field(default=20, ge=1)
    rank_from: int = Field(default=1, ge=1)
    store_top: int = Field(default=200, ge=1)
    eg: EgConfig = EgConfig()
    johansen_confirm: JohansenConfig = JohansenConfig()
    fdr: FdrConfig = FdrConfig()
    half_life_days: HalfLifeConfig = HalfLifeConfig()


class KalmanConfig(_Section):
    """Kalman hedge settings (FR-12)."""

    delta: float = Field(default=1.0e-4, gt=0.0, lt=1.0)
    obs_var: Literal["formation_ols"] = "formation_ols"


class ModelConfig(_Section):
    """Hedge model for cointegration pairs; distance pairs always use normalised prices."""

    hedge: Literal["static", "kalman"] = "kalman"
    kalman: KalmanConfig = KalmanConfig()


class RulesConfig(_Section):
    """Trading rules (FR-14). ``None`` disables a stop."""

    entry_z: float = Field(default=2.0, gt=0.0)
    exit_mode: Literal["cross_zero", "band"] = "cross_zero"
    exit_z: float = Field(default=0.5, ge=0.0)
    stop_z: float | None = 4.0
    time_stop_half_lives: float | None = 3.0
    break_policy: Literal["hold", "stop_only", "close_on_break", "close_and_blacklist"] = (
        "stop_only"
    )
    break_window_days: int = 60
    break_pvalue: float = 0.20


class SizingConfig(_Section):
    """Sizing (FR-15). One pair slot is ``1 / top_k`` of committed capital."""

    scheme: Literal["equal_gross", "inverse_half_life", "vol_target"] = "equal_gross"
    gross_per_pair: float = Field(default=1.0, gt=0.0)


class HalfSpreadConfig(_Section):
    """Half-spread model (D-08)."""

    model: Literal["quotes_then_corwin_schultz", "flat", "none"] = "quotes_then_corwin_schultz"
    floor_bps: float = Field(default=2.0, ge=0.0)
    flat_bps: float = Field(default=0.0, ge=0.0)
    window_days: int = Field(default=20, ge=2)


class BorrowConfig(_Section):
    """Short-borrow fees (D-07)."""

    source: Literal["assumed"] = "assumed"
    schedule: str = "configs/borrow_schedule.yaml"
    multiplier: float = Field(default=1.0, ge=0.0)


class ImpactConfig(_Section):
    """Square-root impact (D-08)."""

    model: Literal["sqrt", "none"] = "sqrt"
    y_coeff: float = Field(default=1.0, ge=0.0)
    adv_window_days: int = Field(default=20, ge=1)
    max_participation: float = Field(default=0.10, gt=0.0)


class CostsConfig(_Section):
    """Costs (FR-16)."""

    commission_bps: float = Field(default=1.0, ge=0.0)
    half_spread: HalfSpreadConfig = HalfSpreadConfig()
    borrow: BorrowConfig = BorrowConfig()
    impact: ImpactConfig = ImpactConfig()
    aum_usd: float = Field(default=10_000_000.0, gt=0.0)


class AlConfig(_Section):
    """Avellaneda-Lee residual strategies (FR-18, FR-19)."""

    factors: Literal["pca", "etf"] = "pca"
    corr_window_days: int = Field(default=252, ge=20)
    n_factors: int = Field(default=15, ge=1)
    variance_explained: float | None = Field(default=None, gt=0.0, lt=1.0)
    resid_window_days: int = Field(default=60, ge=10)
    kappa_min: float = Field(default=252 / 30, ge=0.0)
    s_open: float = Field(default=1.25, gt=0.0)
    s_close_short: float = 0.75
    s_close_long: float = 0.50
    universe_min_mktcap_usd: float = 1.0e9
    rebalance: Literal["daily"] = "daily"
    center_means: bool = True
    gross_per_side: float = Field(default=2.0, gt=0.0)
    min_positions_per_side: int = Field(default=100, ge=1)
    hedge_etf: str = "SPY"


class OutputConfig(_Section):
    """Run outputs."""

    dir: str = "results/runs"
    save_trades: bool = True


class RunConfig(_Section):
    """A fully resolved run config."""

    run: RunSection
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    windows: WindowsConfig = WindowsConfig()
    selection: SelectionConfig = SelectionConfig()
    model: ModelConfig = ModelConfig()
    rules: RulesConfig = RulesConfig()
    sizing: SizingConfig = SizingConfig()
    costs: CostsConfig = CostsConfig()
    output: OutputConfig = OutputConfig()
    al: AlConfig = AlConfig()

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


def grid_members(spec: dict[str, Any]) -> list[list[str]]:
    """``--set`` lists for a sweep grid: ``members`` crossed with the product of the rest."""
    members = spec.get("members") or [{}]
    keys = sorted(key for key in spec if key != "members")
    out: list[list[str]] = []
    for member in members:
        for values in itertools.product(*(spec[key] for key in keys)):
            chosen = {**member, **dict(zip(keys, values, strict=True))}
            out.append([f"{key}={_scalar(value)}" for key, value in chosen.items()])
    return out


def _scalar(value: Any) -> str:
    """A value as YAML text on one line."""
    text: str = yaml.safe_dump(value, default_flow_style=True).strip()
    return text.removesuffix("...").strip()
