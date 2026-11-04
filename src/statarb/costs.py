"""Transaction and borrow costs (FR-16, D-07, D-08). Every component is reported separately.

Per side, as a fraction of traded notional:

- commission: ``commission_bps / 1e4``;
- half-spread: closing quotes where they exist, else Corwin-Schultz (2012) from daily
  highs and lows, floored at ``floor_bps``;
- square-root impact: ``Y * sigma_daily * sqrt(q / ADV)`` with ``q`` the traded USD and
  ``ADV`` the average daily dollar volume.

Borrow accrues on short notional at ``fee_rate_annual / 360`` per calendar day (ACT/360),
so a short held over a weekend pays three days on Monday.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

F64 = npt.NDArray[np.float64]

BPS = 1e-4
_CS_DENOM = 3.0 - 2.0 * math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class LegCostInputs:
    """Cost inputs of one security, frozen at ``T`` from formation data."""

    half_spread: float  # fraction of price
    adv_usd: float
    sigma_daily: float
    fee_rate_annual: float


@dataclass(frozen=True, slots=True)
class CostParams:
    """The resolved cost settings of a run."""

    commission_bps: float = 1.0
    half_spread_floor_bps: float = 2.0
    half_spread_model: str = "quotes_then_corwin_schultz"
    flat_half_spread_bps: float = 0.0
    impact_y: float = 1.0
    impact_on: bool = True
    max_participation: float = 0.10
    borrow_multiplier: float = 1.0
    aum_usd: float = 10_000_000.0


def corwin_schultz(high: F64, low: F64) -> F64:
    """Corwin-Schultz spread estimate (full spread, fraction of price) for each day pair.

    Element ``i`` uses days ``i`` and ``i + 1``; negative estimates are set to 0, as in the
    paper's preferred version.
    """
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(h / lo) ** 2
        beta = logs[:-1] + logs[1:]
        gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
        alpha = (math.sqrt(2.0) - 1.0) * np.sqrt(beta) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(np.isfinite(spread), np.maximum(spread, 0.0), np.nan)


def half_spread(
    high: F64,
    low: F64,
    bid: F64 | None,
    ask: F64 | None,
    model: str,
    floor_bps: float,
    flat_bps: float = 0.0,
) -> float:
    """Half-spread as a fraction of price from a trailing window of daily rows."""
    if model == "none":
        return 0.0
    if model == "flat":
        return flat_bps * BPS
    estimate = math.nan
    if bid is not None and ask is not None:
        b, a = np.asarray(bid, dtype=np.float64), np.asarray(ask, dtype=np.float64)
        ok = np.isfinite(b) & np.isfinite(a) & (a > b) & (b > 0)
        if ok.any():
            estimate = float(np.mean((a[ok] - b[ok]) / (a[ok] + b[ok])))
    if math.isnan(estimate) and len(high) >= 2:
        spreads = corwin_schultz(high, low)
        if np.isfinite(spreads).any():
            estimate = float(np.nanmean(spreads)) / 2.0
    if math.isnan(estimate):
        estimate = 0.0
    return max(estimate, floor_bps * BPS)


def trade_cost(
    notional: float, leg: LegCostInputs, params: CostParams
) -> tuple[float, float, float]:
    """``(commission, spread, impact)`` for one side, in units of ``notional``'s currency.

    ``notional`` is in units of committed capital; impact converts it to USD with
    ``aum_usd`` to compare against ADV.
    """
    size = abs(notional)
    commission = size * params.commission_bps * BPS
    spread = size * leg.half_spread
    impact = 0.0
    if params.impact_on and params.impact_y > 0 and leg.adv_usd > 0 and size > 0:
        participation = size * params.aum_usd / leg.adv_usd
        impact = size * params.impact_y * leg.sigma_daily * math.sqrt(participation)
    return commission, spread, impact


def participation_scale(
    notionals: tuple[float, ...], legs: tuple[LegCostInputs, ...], params: CostParams
) -> float:
    """Factor (at most 1) that keeps every leg's trade within ``max_participation`` of ADV.

    Applies only with the impact model on: the cap belongs to the impact block (FR-24), so
    gross-of-cost replications trade unscaled. A leg without ADV gets no position.
    """
    if not params.impact_on:
        return 1.0
    worst = max(participation(n, leg, params) for n, leg in zip(notionals, legs, strict=True))
    if math.isinf(worst):
        return 0.0
    return 1.0 if worst <= params.max_participation else params.max_participation / worst


def participation(notional: float, leg: LegCostInputs, params: CostParams) -> float:
    """Traded USD over ADV, for the 10% cap diagnostics (FR-24)."""
    if leg.adv_usd <= 0:
        return math.inf
    return abs(notional) * params.aum_usd / leg.adv_usd


def borrow_cost(
    short_notional: float,
    fee_rate_annual: float,
    previous: date,
    current: date,
    multiplier: float = 1.0,
) -> float:
    """ACT/360 borrow fee on ``|short_notional|`` from ``previous`` to ``current``."""
    days = (current - previous).days
    return abs(short_notional) * fee_rate_annual * multiplier * days / 360.0


def trade_costs(
    trades: pl.DataFrame, spreads: pl.DataFrame, adv: pl.DataFrame, cfg: CostParams
) -> pl.DataFrame:
    """Cost columns for a trades table (spec interface).

    ``trades`` has ``security_id, notional_usd``; ``spreads`` has ``security_id,
    half_spread``; ``adv`` has ``security_id, adv_usd, sigma_daily``.
    """
    rows = trades.join(spreads, on="security_id", how="left").join(
        adv, on="security_id", how="left"
    )
    size = pl.col("notional_usd").abs()
    impact = (
        size * cfg.impact_y * pl.col("sigma_daily") * (size / pl.col("adv_usd")).sqrt()
        if cfg.impact_on
        else pl.lit(0.0)
    )
    return rows.with_columns(
        (size * cfg.commission_bps * BPS).alias("cost_commission"),
        (size * pl.col("half_spread")).alias("cost_spread"),
        impact.fill_null(0.0).alias("cost_impact"),
    )


def borrow_accrual(positions: pl.DataFrame, borrow: pl.DataFrame) -> pl.DataFrame:
    """Daily borrow fees (spec interface).

    ``positions`` has ``security_id, date, prev_date, notional_usd`` (signed); ``borrow``
    has ``security_id, fee_rate_annual``. Only shorts pay.
    """
    rows = positions.join(borrow, on="security_id", how="left")
    days = (pl.col("date") - pl.col("prev_date")).dt.total_days()
    short = pl.when(pl.col("notional_usd") < 0).then(-pl.col("notional_usd")).otherwise(0.0)
    return rows.with_columns(
        (short * pl.col("fee_rate_annual") * days / 360.0).alias("cost_borrow")
    )
