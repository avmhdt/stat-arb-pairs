"""Window-level trading and overlapping-portfolio accounting (FR-12, FR-14 to FR-17).

``freeze`` runs on a ``FormationView``: selection, the Kalman filter through formation, and
each leg's cost inputs. Its ``FrozenWindow`` is the only thing ``run_window`` receives from
before ``T``; ``run_window`` itself reads only a ``TradingView``.

Accounting, in units of the window's committed capital (1 at ``T``):

- legs are bought and held between trades, so their values drift with prices;
- a signal at close ``t`` executes at close ``t + lag`` (D-13); delisting and stale-price
  closes execute at once, and every open pair closes at the window's last close;
- a delisting return is earned on the delisting day, then the pair is closed (DQ-03);
- a missing price carries forward for up to 5 sessions, then the pair closes (DQ-11);
- daily return = P&L / capital at the previous close, and ``ret = ret_gross -
  cost_trading - cost_borrow`` holds exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from statarb.costs import (
    BPS,
    CostParams,
    LegCostInputs,
    borrow_cost,
    half_spread,
    participation,
    trade_cost,
)
from statarb.data.build import borrow_rates
from statarb.models.kalman import KalmanState, formation_start, kalman_filter, resume
from statarb.selection.pairs import PairParams, PairSet
from statarb.selection.select import select_window
from statarb.trading import rules
from statarb.trading.sizing import leg_weights, pair_scales

if TYPE_CHECKING:
    from pathlib import Path

    from statarb.config import RunConfig
    from statarb.industry import SicEtfMap
    from statarb.windows import FormationView, TradingView

F64 = npt.NDArray[np.float64]

STALE_LIMIT = 5
COST_COLUMNS = ("tr_index", "high", "low", "close", "volume", "ret_total", "mktcap_usd")
TRADE_COLUMNS = {
    "run_id": pl.String,
    "window_id": pl.String,
    "pair_id": pl.String,
    "date_signal": pl.Date,
    "date_exec": pl.Date,
    "security_id": pl.String,
    "side": pl.Int8,
    "notional_usd": pl.Float64,
    "price": pl.Float64,
    "reason": pl.String,
    "z_signal": pl.Float64,
    "cost_commission": pl.Float64,
    "cost_spread": pl.Float64,
    "cost_impact": pl.Float64,
    "participation": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class FrozenPair:
    """One traded pair and everything fixed for it at ``T``."""

    params: PairParams
    kalman: KalmanState | None
    scale: float
    max_holding: int | None


@dataclass(frozen=True)
class FrozenWindow:
    """The only object that crosses ``T``."""

    pairset: PairSet
    pairs: tuple[FrozenPair, ...]
    legs: dict[str, LegCostInputs]
    distance: bool
    slot: float


@dataclass(frozen=True)
class WindowResult:
    """Daily totals and trades of one window."""

    window_id: str
    daily: pl.DataFrame
    trades: pl.DataFrame


def cost_params(cfg: RunConfig) -> CostParams:
    """Resolve the run's cost settings."""
    costs = cfg.costs
    return CostParams(
        commission_bps=costs.commission_bps,
        half_spread_floor_bps=costs.half_spread.floor_bps,
        half_spread_model=costs.half_spread.model,
        flat_half_spread_bps=costs.half_spread.flat_bps,
        impact_y=costs.impact.y_coeff,
        impact_on=costs.impact.model == "sqrt",
        max_participation=costs.impact.max_participation,
        borrow_multiplier=costs.borrow.multiplier,
        aum_usd=costs.aum_usd,
    )


def _fee_for_rank(rank: int | None, schedule: pl.DataFrame) -> float:
    fees = dict(schedule.select("bucket", "fee_rate_annual").iter_rows())
    if rank is not None and rank <= 500:
        return float(fees["top500"])
    if rank is not None and rank <= 1000:
        return float(fees["top1000"])
    return float(fees["rest"])


def leg_inputs(
    view: FormationView, ids: Sequence[str], cfg: RunConfig, schedule: pl.DataFrame
) -> dict[str, LegCostInputs]:
    """Cost inputs per security from the last formation sessions, frozen at ``T``."""
    window = max(cfg.costs.half_spread.window_days, cfg.costs.impact.adv_window_days)
    sessions = view.sessions()[-(window + 1) :]
    rows = view.query("prices", sessions[0], view.window.formation_end).filter(
        pl.col("security_id").is_in(list(ids))
    )
    universe = view.universe()
    by_cap = universe.sort(["mktcap_usd", "security_id"], descending=[True, False], nulls_last=True)
    rank = {sid: i + 1 for i, sid in enumerate(by_cap["security_id"].to_list())}
    out: dict[str, LegCostInputs] = {}
    for sid in ids:
        leg = rows.filter(pl.col("security_id") == sid).sort("date")
        tail_spread = leg.tail(cfg.costs.half_spread.window_days + 1)
        bid = leg["bid"].to_numpy() if "bid" in leg.columns else None
        ask = leg["ask"].to_numpy() if "ask" in leg.columns else None
        spread = half_spread(
            tail_spread["high"].to_numpy(),
            tail_spread["low"].to_numpy(),
            bid,
            ask,
            cfg.costs.half_spread.model,
            cfg.costs.half_spread.floor_bps,
            cfg.costs.half_spread.flat_bps,
        )
        tail = leg.tail(cfg.costs.impact.adv_window_days)
        dollar = (tail["close"] * tail["volume"]).drop_nulls()
        rets = tail["ret_total"].drop_nulls().drop_nans()
        out[sid] = LegCostInputs(
            half_spread=spread,
            adv_usd=float(dollar.mean()) if dollar.len() else 0.0,  # type: ignore[arg-type]
            sigma_daily=float(rets.std()) if rets.len() > 1 else 0.0,  # type: ignore[arg-type]
            fee_rate_annual=_fee_for_rank(rank.get(sid), schedule),
        )
    return out


def freeze(
    view: FormationView,
    cfg: RunConfig,
    schedule: pl.DataFrame,
    sic_map: SicEtfMap | None = None,
    run_id: str | None = None,
) -> FrozenWindow:
    """Select pairs and fix every parameter the trading window will use."""
    pairset = select_window(view, cfg.selection, sic_map, run_id)
    distance = cfg.selection.method == "distance"
    scales = pair_scales(
        [p.ou_half_life_days for p in pairset.pairs],
        [p.spread_sigma for p in pairset.pairs],
        cfg.sizing.scheme,
    )
    legs = sorted({leg for p in pairset.pairs for leg in (p.leg_y, p.leg_x)})
    log_tr: dict[str, F64] = {}
    if not distance and cfg.model.hedge == "kalman" and legs:
        from statarb.selection.panel import formation_panel

        panel = np.log(formation_panel(view, legs))
        log_tr = {sid: panel[:, i] for i, sid in enumerate(legs)}
    frozen = []
    for params, scale in zip(pairset.pairs, scales, strict=True):
        state = None
        if log_tr:
            y, x = log_tr[params.leg_y], log_tr[params.leg_x]
            theta0, p0, obs_var = formation_start(y, x)
            _, state = kalman_filter(y, x, cfg.model.kalman.delta, obs_var, theta0, p0)
        hold = (
            None
            if distance
            else rules.max_holding_days(params.ou_half_life_days, cfg.rules.time_stop_half_lives)
        )
        frozen.append(FrozenPair(params, state, scale, hold))
    return FrozenWindow(
        pairset=pairset,
        pairs=tuple(frozen),
        legs=leg_inputs(view, legs, cfg, schedule) if legs else {},
        distance=distance,
        slot=1.0 / cfg.selection.top_k,
    )


def _leg_panel(
    view: TradingView, ids: list[str], sessions: list[date]
) -> dict[str, dict[str, F64]]:
    rows = view.prices(["tr_index", "ret_total", "ret_delist", "close"]).filter(
        pl.col("security_id").is_in(ids)
    )
    frame = pl.DataFrame({"date": sessions})
    out: dict[str, dict[str, F64]] = {}
    for sid in ids:
        leg = frame.join(rows.filter(pl.col("security_id") == sid), on="date", how="left").sort(
            "date"
        )
        out[sid] = {
            name: leg[name].cast(pl.Float64).fill_nan(None).to_numpy().astype(np.float64)
            for name in ("tr_index", "ret_total", "ret_delist", "close")
        }
    return out


def _signal(
    pair: FrozenPair, y: dict[str, F64], x: dict[str, F64], distance: bool
) -> tuple[F64, F64]:
    """``z`` and the hedge ratio on each day, from trading-window prices only."""
    p = pair.params
    if distance:
        first_y = y["tr_index"][np.isfinite(y["tr_index"])]
        first_x = x["tr_index"][np.isfinite(x["tr_index"])]
        if not first_y.size or not first_x.size:
            nan = np.full(y["tr_index"].size, np.nan)
            return nan, np.ones_like(nan)
        spread = y["tr_index"] / first_y[0] - x["tr_index"] / first_x[0]
        return spread / p.spread_sigma, np.ones_like(spread)
    log_y, log_x = np.log(y["tr_index"]), np.log(x["tr_index"])
    if pair.kalman is not None:
        result, _ = resume(pair.kalman, log_y, log_x)
        return result.z, result.beta
    spread = log_y - p.alpha0 - p.beta0 * log_x
    return (spread - p.spread_mu) / p.spread_sigma, np.full(spread.size, p.beta0)


def _end_of_life(y: dict[str, F64], x: dict[str, F64]) -> tuple[int | None, str]:
    """Index of the forced close: a delisting, or the sixth stale session in a row."""
    n = y["tr_index"].size
    for leg in (y, x):
        hits = np.nonzero(np.isfinite(leg["ret_delist"]))[0]
        if hits.size:
            first = int(hits[0])
            return first, "delist"
    for t in range(n):
        for leg in (y, x):
            seen = np.isfinite(leg["tr_index"][: t + 1])
            if seen.any():
                last = int(np.nonzero(seen)[0][-1])
                if t - last > STALE_LIMIT:
                    return t, "stale"
    return None, ""


def _leg_return(leg: dict[str, F64], t: int) -> float:
    r = float(leg["ret_total"][t])
    if not math.isfinite(r):
        return 0.0
    d = float(leg["ret_delist"][t])
    return (1.0 + r) * (1.0 + d) - 1.0 if math.isfinite(d) else r


def _last_price(leg: dict[str, F64], t: int) -> float:
    seen = leg["close"][: t + 1]
    finite = seen[np.isfinite(seen)]
    return float(finite[-1]) if finite.size else math.nan


def run_window(
    frozen: FrozenWindow, view: TradingView, cfg: RunConfig, run_id: str = ""
) -> WindowResult:
    """Trade one window's frozen pairs through ``[T, T + H)``."""
    sessions = view.sessions()
    n = len(sessions)
    params = cost_params(cfg)
    lag = cfg.windows.execution_lag_days
    ids = sorted(frozen.legs)
    panel = _leg_panel(view, ids, sessions) if ids else {}
    pnl = np.zeros(n)
    trading = np.zeros(n)
    borrow = np.zeros(n)
    employed = np.zeros(n)
    exposure = np.zeros(n)
    open_count = np.zeros(n, dtype=np.int32)
    trades: list[dict[str, object]] = []
    for pair in frozen.pairs:
        p = pair.params
        pair_id = f"{p.leg_y}|{p.leg_x}"
        y, x = panel[p.leg_y], panel[p.leg_x]
        z, beta = _signal(pair, y, x, frozen.distance)
        dead, dead_reason = _end_of_life(y, x)
        horizon = n if dead is None else dead + 1
        signal, reason = rules.state_machine(
            z[:horizon],
            cfg.rules.entry_z,
            cfg.rules.exit_z,
            cfg.rules.stop_z,
            cfg.rules.exit_mode,
            pair.max_holding,
            close_at_end=True,
        )
        target = np.zeros(n, dtype=np.int8)
        target[lag:horizon] = signal[: horizon - lag] if horizon > lag else 0
        target[horizon - 1 :] = 0
        gross = frozen.slot * cfg.sizing.gross_per_pair * pair.scale
        value_y = value_x = 0.0
        position = 0
        for t in range(n):
            if t > 0 and position != 0:
                r_y, r_x = _leg_return(y, t), _leg_return(x, t)
                pnl[t] += value_y * r_y + value_x * r_x
                for value, leg_id in ((value_y, p.leg_y), (value_x, p.leg_x)):
                    if value < 0:
                        borrow[t] += borrow_cost(
                            value,
                            frozen.legs[leg_id].fee_rate_annual,
                            sessions[t - 1],
                            sessions[t],
                            params.borrow_multiplier,
                        )
                employed[t] += frozen.slot * pair.scale
                value_y *= 1.0 + r_y
                value_x *= 1.0 + r_x
            wanted = int(target[t])
            if wanted != position:
                signal_day = t - lag if t - lag >= 0 else t
                if wanted == 0:
                    why = (
                        "window_end"
                        if t == n - 1 and dead is None
                        else rules.REASONS.get(
                            int(reason[signal_day]) if signal_day < horizon else 0, "exit"
                        )
                    )
                    if dead is not None and t == dead:
                        why = dead_reason
                    legs = ((p.leg_y, -value_y), (p.leg_x, -value_x))
                    value_y = value_x = 0.0
                else:
                    w_y, w_x = leg_weights(wanted, float(beta[signal_day]), gross, frozen.distance)
                    legs = ((p.leg_y, w_y), (p.leg_x, w_x))
                    value_y, value_x = w_y, w_x
                    why = "entry"
                for leg_id, traded in legs:
                    if traded == 0.0:
                        continue
                    leg = frozen.legs[leg_id]
                    commission, spread, impact = trade_cost(traded, leg, params)
                    trading[t] += commission + spread + impact
                    trades.append(
                        {
                            "run_id": run_id,
                            "window_id": frozen.pairset.window.window_id,
                            "pair_id": pair_id,
                            "date_signal": sessions[signal_day],
                            "date_exec": sessions[t],
                            "security_id": leg_id,
                            "side": 1 if traded > 0 else -1,
                            "notional_usd": traded * params.aum_usd,
                            "price": _last_price(panel[leg_id], t),
                            "reason": why,
                            "z_signal": float(z[signal_day]) if signal_day < z.size else math.nan,
                            "cost_commission": commission * params.aum_usd,
                            "cost_spread": spread * params.aum_usd,
                            "cost_impact": impact * params.aum_usd,
                            "participation": participation(traded, leg, params),
                        }
                    )
                position = wanted
            if position != 0:
                exposure[t] += abs(value_y) + abs(value_x)
                open_count[t] += 1
    capital = np.empty(n)
    level = 1.0
    for t in range(n):
        capital[t] = level
        level += pnl[t] - trading[t] - borrow[t]
    ret_gross = pnl / capital
    cost_trading = trading / capital
    cost_borrow = borrow / capital
    daily = pl.DataFrame(
        {
            "date": sessions,
            "window_id": frozen.pairset.window.window_id,
            "ret_gross": ret_gross,
            "cost_trading": cost_trading,
            "cost_borrow": cost_borrow,
            "ret": ret_gross - cost_trading - cost_borrow,
            "pnl_net": pnl - trading - borrow,
            "employed": employed,
            "gross_exposure": exposure / capital,
            "n_open": open_count,
        }
    )
    trade_frame = (
        pl.DataFrame(trades, schema=TRADE_COLUMNS) if trades else pl.DataFrame(schema=TRADE_COLUMNS)
    )
    return WindowResult(frozen.pairset.window.window_id, daily, trade_frame)


def combine(dailies: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Overlapping portfolios (GGR): each day, the average of the active windows (FR-17).

    ``ret_employed`` pools the active windows: net P&L over the capital of pairs open that
    day, 0 when none is open.
    """
    if not dailies:
        return pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64})
    rows = pl.concat(dailies)
    combined = (
        rows.group_by("date", maintain_order=False)
        .agg(
            pl.col("ret_gross").sort().mean(),
            pl.col("cost_trading").sort().mean(),
            pl.col("cost_borrow").sort().mean(),
            pl.col("pnl_net").sort().sum().alias("_pnl"),
            pl.col("employed").sort().sum().alias("_employed"),
            pl.col("gross_exposure").sort().mean(),
            pl.col("n_open").sum(),
            pl.len().alias("windows_active"),
        )
        .sort("date")
    )
    return combined.select(
        "date",
        (pl.col("ret_gross") - pl.col("cost_trading") - pl.col("cost_borrow")).alias("ret"),
        "ret_gross",
        pl.when(pl.col("_employed") > 0)
        .then(pl.col("_pnl") / pl.col("_employed"))
        .otherwise(0.0)
        .alias("ret_employed"),
        "cost_trading",
        "cost_borrow",
        "gross_exposure",
        pl.col("n_open").cast(pl.Int32),
        pl.col("windows_active").cast(pl.Int32),
    )


def borrow_schedule(path: Path) -> pl.DataFrame:
    """The assumed schedule's bucket fees (D-07)."""
    return borrow_rates(path)


__all__ = ["BPS", "FrozenPair", "FrozenWindow", "WindowResult", "combine", "freeze", "run_window"]
