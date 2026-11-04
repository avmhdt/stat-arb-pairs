"""Decay, break-even cost, capacity, survivorship, break policies and sizing (FR-21 to FR-26,
FR-31, AC-W-09). Everything here reads finished runs under ``results/runs`` and returns
aggregate tables.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml

from statarb.evaluation.bootstrap import sharpe, stationary_indices
from statarb.evaluation.metrics import monthly, newey_west_mean

F64 = npt.NDArray[np.float64]

SUB_PERIODS = {
    "1962-1988": (date(1962, 1, 1), date(1988, 12, 31)),
    "1989-2002": (date(1989, 1, 1), date(2002, 12, 31)),
    "2003-2009": (date(2003, 1, 1), date(2009, 12, 31)),
    "2010-2017": (date(2010, 1, 1), date(2017, 12, 31)),
    "2018-2025": (date(2018, 1, 1), date(2025, 12, 31)),
}
"""FR-21's sub-periods, truncated to the data (the first two hold only 1999-2002 here)."""
AUM_GRID = (1e6, 1e7, 3e7, 1e8, 3e8, 1e9)
HALF_LIFE_BUCKETS = ((1.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 63.0))


@dataclass(frozen=True)
class Run:
    """A finished run: its saved params, daily returns, trades and round trips."""

    run_id: str
    params: dict[str, object]
    returns: pl.DataFrame
    trades: pl.DataFrame
    roundtrips: pl.DataFrame
    label: str

    @property
    def family(self) -> str:
        return str(self.params["run"]["strategy_family"])  # type: ignore[index]

    @property
    def aum(self) -> float:
        return float(self.params["costs"]["aum_usd"])  # type: ignore[index]


def read_run(runs: Path, run_id: str, label: str | None = None) -> Run:
    """Load one complete run folder; one without ``metrics.json`` (written last) is refused."""
    folder = runs / run_id
    if not (folder / "metrics.json").exists():
        raise FileNotFoundError(f"run {run_id} is incomplete or missing: no metrics.json")
    params = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    trades = (
        pl.read_parquet(folder / "trades.parquet")
        if (folder / "trades.parquet").exists()
        else pl.DataFrame()
    )
    trips = (
        pl.read_parquet(folder / "roundtrips.parquet")
        if (folder / "roundtrips.parquet").exists()
        else pl.DataFrame()
    )
    returns = pl.read_parquet(folder / "returns.parquet").sort("date")
    return Run(
        run_id,
        params,
        returns,
        trades,
        trips,
        label or f"{params['run']['strategy_family']}:{run_id[:8]}",
    )


def _total(series: pl.Series) -> float:
    return float(series.cast(pl.Float64).to_numpy().sum())


def _mean(series: pl.Series) -> float:
    values = series.cast(pl.Float64).to_numpy()
    return float(values.mean()) if values.size else math.nan


def _between(frame: pl.DataFrame, start: date, end: date, column: str = "date") -> pl.DataFrame:
    return frame.filter(pl.col(column).is_between(start, end))


def daily_turnover(run: Run) -> pl.DataFrame:
    """``date, turnover, cost_impact`` per unit of capital: the returns' own columns when the
    run recorded them, else rebuilt from its trades (runs made before M6)."""
    if {"turnover", "cost_impact"} <= set(run.returns.columns):
        return run.returns.select("date", "turnover", "cost_impact")
    active = run.returns.select("date", pl.col("windows_active").cast(pl.Float64))
    from_trades = run.trades.group_by(pl.col("date_exec").alias("date")).agg(
        (pl.col("notional_usd").abs().sum() / run.aum).alias("_turn"),
        (pl.col("cost_impact").sum() / run.aum).alias("_impact"),
    )
    return active.join(from_trades, on="date", how="left").select(
        "date",
        (pl.col("_turn").fill_null(0.0) / pl.col("windows_active")).alias("turnover"),
        (pl.col("_impact").fill_null(0.0) / pl.col("windows_active")).alias("cost_impact"),
    )


def decay(run: Run, column: str = "ret") -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """Sub-period table, rolling 36-month mean and Sharpe, and a time-trend regression.

    The trend is monthly return on months elapsed (scaled per decade), with a Newey-West
    standard error (six lags); the post-2002 mean has a Newey-West 95% interval.
    """
    months = monthly(run.returns.select("date", column), column).rename({column: "ret"})
    rows = []
    for name, (start, end) in SUB_PERIODS.items():
        part = _between(months, start, end, "month")
        if part.height < 6:
            continue
        mean, se, t = newey_west_mean(part["ret"].to_numpy())
        values = part["ret"].to_numpy()
        rows.append(
            {
                "run": run.label,
                "period": name,
                "first": part["month"][0],
                "last": part["month"][-1],
                "months": part.height,
                "mean_monthly": mean,
                "t_stat": t,
                "sharpe_annual": float(values.mean() / values.std(ddof=1) * math.sqrt(12))
                if values.std(ddof=1) > 0
                else 0.0,
            }
        )
    roll = months.with_columns(
        pl.col("ret").rolling_mean(36).alias("mean_36m"),
        (pl.col("ret").rolling_mean(36) / pl.col("ret").rolling_std(36) * math.sqrt(12)).alias(
            "sharpe_36m"
        ),
        pl.lit(run.label).alias("run"),
    )
    y = months["ret"].to_numpy()
    x = np.arange(y.size) / 120.0
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    trend_t = _newey_west_slope_t(design, resid, coef)
    late = _between(months, date(2003, 1, 1), date(2100, 1, 1), "month")["ret"].to_numpy()
    mean, se, _ = newey_west_mean(late)
    summary = {
        "trend_per_decade": float(coef[1]),
        "trend_t_newey_west": trend_t,
        "post_2002_mean_monthly": mean,
        "post_2002_ci_low": mean - 1.96 * se,
        "post_2002_ci_high": mean + 1.96 * se,
    }
    return pl.DataFrame(rows), roll, summary


def _newey_west_slope_t(design: F64, resid: F64, coef: F64, lags: int = 6) -> float:
    """t-statistic of the slope with a Newey-West (Bartlett, ``lags``) covariance."""
    xe = design * resid[:, None]
    s = xe.T @ xe
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = xe[lag:].T @ xe[:-lag]
        s += weight * (gamma + gamma.T)
    bread = np.linalg.inv(design.T @ design)
    cov = bread @ s @ bread
    return float(coef[1] / math.sqrt(cov[1, 1])) if cov[1, 1] > 0 else math.nan


def break_even(
    run: Run,
    multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
    n_boot: int = 2000,
    block: float = 20.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Per-side cost in basis points at which the mean net return is zero (FR-23).

    ``mean(gross - k * borrow) / mean(turnover)``, turnover being one-way traded notional per
    unit of capital, so the answer is per side. The interval resamples days with a
    stationary bootstrap (mean block 20). ``modelled_bps`` is the run's own trading cost per
    side, for comparison.
    """
    flows = run.returns.select("date", "ret_gross", "cost_borrow", "cost_trading").join(
        daily_turnover(run), on="date"
    )
    rows = []
    rng = np.random.default_rng(seed)
    periods = {"all": (date(1900, 1, 1), date(2100, 1, 1)), **SUB_PERIODS}
    for name, (start, end) in periods.items():
        part = _between(flows, start, end)
        if part.height < 60 or float(part["turnover"].sum()) <= 0:
            continue
        gross, borrow, turn = (part[c].to_numpy() for c in ("ret_gross", "cost_borrow", "turnover"))
        modelled = _total(part["cost_trading"]) / _total(part["turnover"]) * 1e4
        draws = [stationary_indices(part.height, block, rng) for _ in range(n_boot)]
        for k in multipliers:
            edge = gross - k * borrow
            point = float(edge.mean() / turn.mean() * 1e4)
            boot = np.array([edge[i].mean() / max(turn[i].mean(), 1e-18) * 1e4 for i in draws])
            rows.append(
                {
                    "run": run.label,
                    "period": name,
                    "borrow_multiplier": k,
                    "break_even_bps": point,
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "modelled_bps": modelled,
                    "turnover_annual": float(turn.mean() * 252),
                }
            )
    return pl.DataFrame(rows)


def capacity(
    run: Run, years: int = 5, y_coeffs: tuple[float, ...] = (0.5, 1.0, 2.0), cap: float = 0.10
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Net Sharpe and return against AUM for the last ``years`` of the run (FR-24).

    Square-root impact scales each trade's impact cost by ``sqrt(AUM / AUM_run)`` and by
    ``Y / Y_run``; commissions and spreads are proportional and unchanged. Participation
    scales with AUM, giving the share of trades above the 10% ADV cap.

    Two bases: ``all_costs`` (the run's net return, impact rescaled) and ``impact_only``
    (gross return less borrow and the rescaled impact, no commission or spread), which
    shows how far impact alone erodes the edge when the full cost model leaves none.
    """
    last = run.returns["date"].max()
    assert isinstance(last, date)
    start = date(last.year - years + 1, 1, 1)
    flows = _between(
        run.returns.select("date", "ret", "ret_gross", "cost_borrow").join(
            daily_turnover(run), on="date"
        ),
        start,
        last,
    )
    base_aum = run.aum
    base_y = float(run.params["costs"]["impact"]["y_coeff"])  # type: ignore[index]
    trades = _between(run.trades, start, last, "date_exec") if run.trades.height else run.trades
    impact = flows["cost_impact"].to_numpy()
    bases = {
        "all_costs": flows["ret"].to_numpy() + impact,
        "impact_only": flows["ret_gross"].to_numpy() - flows["cost_borrow"].to_numpy(),
    }
    rows = []
    fine = np.geomspace(1e5, 1e10, 400)
    summary = []
    for basis, zero_impact in bases.items():
        for y in y_coeffs:

            def net(aum: float, y: float = y, base: F64 = zero_impact) -> F64:
                scale = math.sqrt(aum / base_aum) * (y / base_y if base_y > 0 else 0.0)
                out: F64 = base - impact * scale
                return out

            for aum in AUM_GRID:
                series = net(aum)
                share = (
                    _mean((trades["participation"] * aum / base_aum > cap).cast(pl.Float64))
                    if trades.height
                    else math.nan
                )
                rows.append(
                    {
                        "run": run.label,
                        "basis": basis,
                        "y_coeff": y,
                        "aum_usd": aum,
                        "net_sharpe": sharpe(series),
                        "net_return_annual": float(series.mean() * 252),
                        "share_trades_over_cap": share,
                    }
                )
            base_sharpe = sharpe(zero_impact)
            sharpes = np.array([sharpe(net(a)) for a in fine])
            means = np.array([net(a).mean() for a in fine])
            half = (
                fine[np.argmax(sharpes <= base_sharpe / 2)]
                if base_sharpe > 0 and (sharpes <= base_sharpe / 2).any()
                else math.nan
            )
            # No AUM qualifies when the strategy already loses money at zero impact.
            zero = (
                fine[np.argmax(means <= 0)]
                if zero_impact.mean() > 0 and (means <= 0).any()
                else math.nan
            )
            summary.append(
                {
                    "run": run.label,
                    "basis": basis,
                    "y_coeff": y,
                    "zero_impact_sharpe": base_sharpe,
                    "zero_impact_return_annual": float(zero_impact.mean() * 252),
                    "aum_half_sharpe_usd": float(half),
                    "aum_zero_return_usd": float(zero),
                    "first": start,
                    "last": last,
                }
            )
    return pl.DataFrame(rows), pl.DataFrame(summary)


def survivorship(full: Run, survivors: Run) -> dict[str, object]:
    """Full point-in-time universe against survivors only, on common days (AC-W-09)."""
    joined = full.returns.select("date", "ret").join(
        survivors.returns.select("date", pl.col("ret").alias("ret_surv")), on="date"
    )
    a, b = joined["ret"].to_numpy(), joined["ret_surv"].to_numpy()
    return {
        "strategy": full.family,
        "first": joined["date"][0],
        "last": joined["date"][-1],
        "full_return_annual": float(a.mean() * 252),
        "survivors_return_annual": float(b.mean() * 252),
        "difference_annual": float((b - a).mean() * 252),
        "full_sharpe": sharpe(a),
        "survivors_sharpe": sharpe(b),
        "full_run": full.run_id,
        "survivors_run": survivors.run_id,
    }


def cvar(values: F64, level: float = 0.05) -> float:
    """Mean of the worst ``level`` share of values."""
    if values.size == 0:
        return math.nan
    cut = np.quantile(values, level)
    return float(values[values <= cut].mean())


def policies(runs: list[Run]) -> pl.DataFrame:
    """Break policies on identical windows (FR-26, AC-H-08)."""
    rows = []
    for run in runs:
        trips = run.roundtrips
        ret = run.returns["ret"].to_numpy()
        trip_ret = trips["ret_on_slot"].to_numpy() if trips.height else np.zeros(0)
        rows.append(
            {
                "policy": run.params["rules"]["break_policy"],  # type: ignore[index]
                "return_annual": float(ret.mean() * 252),
                "sharpe": sharpe(ret),
                "trades": trips.height,
                "mean_trade_return": float(trip_ret.mean()) if trip_ret.size else math.nan,
                "cvar5_trade_return": cvar(trip_ret),
                "mean_holding_days": _mean(trips["holding_days"]) if trips.height else math.nan,
                "break_exits": int((trips["exit_reason"] == "break").sum()) if trips.height else 0,
                "run_id": run.run_id,
            }
        )
    return pl.DataFrame(rows)


def sizing(runs: list[Run]) -> pl.DataFrame:
    """Sizing schemes by half-life bucket: P&L per capital-day, Sharpe, drawdown (FR-31)."""
    rows = []
    for run in runs:
        ret = run.returns["ret"].to_numpy()
        wealth = np.cumprod(1.0 + ret)
        drawdown = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
        trips = run.roundtrips.with_columns(
            (pl.col("slot_capital") * pl.col("holding_days").clip(lower_bound=1)).alias(
                "capital_days"
            )
        )
        scheme = run.params["sizing"]["scheme"]  # type: ignore[index]
        for low, high in (*HALF_LIFE_BUCKETS, (0.0, math.inf)):
            part = (
                trips.filter(pl.col("half_life_days").is_between(low, high, closed="left"))
                if math.isfinite(high)
                else trips
            )
            rows.append(
                {
                    "family": run.family,
                    "scheme": scheme,
                    "half_life_bucket": f"{low:g}-{high:g}" if math.isfinite(high) else "all",
                    "trades": part.height,
                    "pnl_per_capital_day_bps": float(
                        _total(part["pnl"]) / _total(part["capital_days"]) * 1e4
                    )
                    if part.height
                    else math.nan,
                    "mean_trade_return": _mean(part["ret_on_slot"]) if part.height else math.nan,
                    "run_sharpe": sharpe(ret),
                    "run_max_drawdown": drawdown,
                    "run_id": run.run_id,
                }
            )
    return pl.DataFrame(rows)


def metrics_of(runs: Path, run_id: str) -> dict[str, float]:
    """A run's saved metrics."""
    loaded: dict[str, float] = json.loads(
        (runs / run_id / "metrics.json").read_text(encoding="utf-8")
    )
    return loaded
