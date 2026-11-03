"""Return metrics (FR-28, basic set) and Newey-West means."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def summary_metrics(ret: pl.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Annualised mean, volatility, Sharpe, Sortino, maximum drawdown and hit rate.

    Sharpe follows the trial-log contract: ``mean / std(ddof=1) * sqrt(q)``. Sortino uses
    the root mean square of negative returns. Drawdown is on the compounded series.
    """
    values = ret.drop_nulls().cast(pl.Float64).to_numpy()
    n = values.size
    if n < 2:
        return {"n_obs": float(n)}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    root = math.sqrt(periods_per_year)
    return {
        "n_obs": float(n),
        "mean_annual": mean * periods_per_year,
        "vol_annual": std * root,
        "sharpe_annual": mean / std * root if std > 0 else 0.0,
        "sortino_annual": mean / downside * root if downside > 0 else 0.0,
        "max_drawdown": float((wealth / peak - 1.0).min()),
        "hit_rate": float((values > 0).mean()),
        "total_return": float(wealth[-1] - 1.0),
    }


def newey_west_mean(values: np.ndarray, lags: int = 6) -> tuple[float, float, float]:
    """Mean, Newey-West standard error and t-statistic (GGR use six lags)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return math.nan, math.nan, math.nan
    mean = float(x.mean())
    e = x - mean
    gamma0 = float(e @ e) / n
    total = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    se = math.sqrt(max(total, 0.0) / n)
    return mean, se, mean / se if se > 0 else math.nan


def monthly(daily: pl.DataFrame, column: str = "ret") -> pl.DataFrame:
    """Compounded calendar-month returns of a daily column."""
    return (
        daily.sort("date")
        .group_by(pl.col("date").dt.truncate("1mo").alias("month"), maintain_order=True)
        .agg(((1.0 + pl.col(column)).product() - 1.0).alias(column))
        .sort("month")
    )


def trade_stats(trades: pl.DataFrame, daily: pl.DataFrame, aum_usd: float) -> dict[str, float]:
    """Round trips, average holding days, annual turnover and cost totals (units of capital)."""
    if trades.is_empty():
        return {"round_trips": 0.0, "avg_holding_days": 0.0, "turnover_annual": 0.0}
    entries = (
        trades.filter(pl.col("reason") == "entry")
        .group_by("window_id", "pair_id", "date_exec")
        .len()
    )
    closes = trades.filter(pl.col("reason") != "entry")
    spans = (
        trades.sort("date_exec")
        .group_by("window_id", "pair_id", maintain_order=True)
        .agg(pl.col("date_exec"), pl.col("reason"))
    )
    holds: list[int] = []
    for _, _, dates, reasons in spans.iter_rows():
        opened = None
        for day, why in zip(dates, reasons, strict=True):
            if why == "entry" and opened is None:
                opened = day
            elif why != "entry" and opened is not None:
                holds.append((day - opened).days)
                opened = None
    years = max(daily.height / 252.0, 1e-9)
    traded = float(trades["notional_usd"].abs().sum()) / aum_usd
    active = (
        daily.select(pl.col("windows_active").mean()).item()
        if "windows_active" in daily.columns
        else 1.0
    )
    windows_active = float(active or 1.0)
    return {
        "round_trips": float(entries.height),
        "closes": float(closes["pair_id"].n_unique()),
        "avg_holding_days": float(np.mean(holds)) if holds else 0.0,
        "turnover_annual": traded / max(windows_active, 1.0) / years,
        "cost_commission_usd": float(trades["cost_commission"].sum()),
        "cost_spread_usd": float(trades["cost_spread"].sum()),
        "cost_impact_usd": float(trades["cost_impact"].sum()),
        "share_trades_over_participation_cap": float(
            trades.select((pl.col("participation") > 0.10).mean()).item() or 0.0
        ),
    }
