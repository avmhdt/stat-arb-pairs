"""Avellaneda-Lee residual strategies: freeze at T, then re-estimate daily (FR-18, FR-19).

``freeze_al`` runs on a ``FormationView``: the universe (market cap at least
``al.universe_min_mktcap_usd`` on ``T - 1``), the last ``corr_window`` days of returns,
each stock's hedge instrument (SPY for PCA, its sector ETF for the ETF variant, with the
DQ-07 inception rule) and the s-scores on ``T - 1``. ``run_al_window`` then walks the
trading window: at each close it appends that day's returns, re-estimates factors,
residuals and s-scores from the trailing windows, and applies A-L's rules:

- open long when ``s < -s_open``, short when ``s > s_open``;
- close a long when ``s > -s_close_long``, a short when ``s < s_close_short``;
- close, and do not open, when the fit fails or ``kappa <= kappa_min``.

A new position gets ``gross_per_side / max(m, min_positions_per_side)`` of current equity
(A-L's ``Q = E Lambda``), with ``m`` the average number of positions open on a side over the last 20
sessions (seeded with the count of ``|s| > s_open`` on ``T - 1``): A-L's ``Lambda = 2 / 100``
for "2+2" leverage with about 100 names a side, adjusted only for new positions. It is
hedged with ``-beta`` dollars of its hedge instrument; size and beta are fixed at entry and
marked daily. Hedges are netted into one book per instrument, traded once a day.
Signals execute with the run's lag; delisted or stale names close at once; all positions
close at the window's last close. Accounting matches the pairs engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from statarb.costs import LegCostInputs, borrow_cost, participation, trade_cost
from statarb.models.residuals import eigenportfolios, regression_residuals, s_scores
from statarb.selection.pairs import PairSet
from statarb.trading.engine import TRADE_COLUMNS, WindowResult, cost_params, leg_inputs

if TYPE_CHECKING:
    from statarb.config import RunConfig
    from statarb.industry import SicEtfMap
    from statarb.windows import FormationView, TradingView

F64 = npt.NDArray[np.float64]
STALE_LIMIT = 5
SIZE_LOOKBACK = 20
ETF_FEE = 0.0025
MIN_HISTORY_SESSIONS = 252
AL_TESTED: dict[str, pl.DataType | type[pl.DataType]] = {
    "run_id": pl.String,
    "window_id": pl.String,
    "formation_start": pl.Date,
    "formation_end": pl.Date,
    "security_id": pl.String,
    "hedge": pl.String,
    "s_score": pl.Float64,
    "ou_kappa": pl.Float64,
    "selected": pl.Boolean,
}
"""One row per universe stock and window: its hedge and its s-score on ``T - 1``."""


@dataclass(frozen=True)
class FrozenAl:
    """Everything the trading window of an A-L strategy receives from before ``T``."""

    pairset: PairSet  # ``tested`` follows AL_TESTED, one row per universe stock
    ids: tuple[str, ...]
    history: F64  # (corr_window, stocks) returns up to T - 1
    hedges: tuple[str, ...]  # hedge instrument per stock
    hedge_history: dict[str, F64]
    legs: dict[str, LegCostInputs]
    s_initial: F64
    kappa_initial: F64


def _wide(frame: pl.DataFrame, sessions: list[date], ids: list[str], column: str) -> F64:
    base = pl.DataFrame({"date": sessions})
    if frame.is_empty() or not ids:
        return np.full((len(sessions), len(ids)), np.nan)
    wide = frame.pivot(on="security_id", index="date", values=column)
    joined = base.join(wide, on="date", how="left").sort("date")
    for name in ids:
        if name not in joined.columns:
            joined = joined.with_columns(pl.lit(None, pl.Float64).alias(name))
    return joined.select(pl.col(ids).cast(pl.Float64).fill_nan(None)).to_numpy().astype(np.float64)


def _etf_panel(etf: pl.DataFrame, sessions: list[date], names: list[str]) -> dict[str, F64]:
    base = pl.DataFrame({"date": sessions})
    out = {}
    for name in names:
        rows = etf.filter(pl.col("etf") == name).select("date", "ret_total")
        joined = base.join(rows, on="date", how="left").sort("date")
        out[name] = joined["ret_total"].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy()
    return out


def _signals(history: F64, hedge: F64, cfg: RunConfig) -> tuple[F64, F64, F64]:
    """s-scores, kappas and hedge betas from the trailing windows of ``history``."""
    al = cfg.al
    recent = history[-al.resid_window_days :]
    hedge_recent = hedge[-al.resid_window_days :]
    if al.factors == "pca":
        weights, _ = eigenportfolios(
            history[-al.corr_window_days :], al.n_factors, al.variance_explained
        )
        residuals, _ = regression_residuals(recent, recent @ weights)
    else:
        mean_h = hedge_recent.mean(axis=0)
        mean_r = recent.mean(axis=0)
        dh = hedge_recent - mean_h
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_etf = (dh * (recent - mean_r)).sum(axis=0) / (dh * dh).sum(axis=0)
        beta_etf = np.nan_to_num(beta_etf)
        residuals = recent - mean_r - dh * beta_etf
    s, kappa = s_scores(residuals, al.kappa_min, al.center_means)
    mean_h = hedge_recent.mean(axis=0)
    dh = hedge_recent - mean_h
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (dh * (recent - recent.mean(axis=0))).sum(axis=0) / (dh * dh).sum(axis=0)
    return s, kappa, np.nan_to_num(beta)


def _hedge_matrix(hedges: tuple[str, ...], panel: dict[str, F64]) -> F64:
    return np.column_stack([panel[name] for name in hedges]) if hedges else np.empty((0, 0))


def freeze_al(
    view: FormationView,
    cfg: RunConfig,
    schedule: pl.DataFrame,
    sic_map: SicEtfMap | None = None,
    run_id: str | None = None,
) -> FrozenAl:
    """Universe, formation history, hedges and ``T - 1`` signals of one window."""
    al = cfg.al
    universe = view.universe().filter(pl.col("mktcap_usd") >= al.universe_min_mktcap_usd)
    sessions = view.sessions()[-al.corr_window_days :]
    candidates = sorted(universe["security_id"].to_list())
    rows = view.prices(["ret_total"]).filter(
        pl.col("security_id").is_in(candidates) & pl.col("date").is_in(sessions)
    )
    returns = _wide(rows, sessions, candidates, "ret_total")
    complete = np.mean(np.isfinite(returns), axis=0) >= 0.95
    ids = [name for name, ok in zip(candidates, complete, strict=True) if ok]
    history = np.nan_to_num(returns[:, complete])
    etf = view.etf_returns()
    counts = dict(etf.group_by("etf").len().iter_rows())
    inception = {name: date.min for name, n in counts.items() if n >= MIN_HISTORY_SESSIONS}
    cutoff = view.last_session()
    if al.factors == "etf" and sic_map is not None:
        sic = dict(view.industry().select("security_id", "sic").iter_rows())
        hedges = tuple(
            sic_map.usable(sic_map.etf_for(sic.get(sid)) or al.hedge_etf, inception, cutoff)
            or al.hedge_etf
            for sid in ids
        )
    else:
        hedges = tuple(al.hedge_etf for _ in ids)
    names = sorted(set(hedges))
    hedge_history = _etf_panel(etf, sessions, names)
    if len(sessions) < al.corr_window_days or not ids:
        s0 = kappa0 = np.full(len(ids), np.nan)
    else:
        s0, kappa0, _ = _signals(history, _hedge_matrix(hedges, hedge_history), cfg)
    legs = leg_inputs(view, ids, cfg, schedule) if ids else {}
    tail = etf.filter(pl.col("date") >= sessions[-20]) if sessions else etf
    for name in names:
        part = tail.filter(pl.col("etf") == name)
        dollar = (part["close"] * part["volume"]).drop_nulls()
        rets = part["ret_total"].drop_nulls().drop_nans()
        legs[name] = LegCostInputs(
            half_spread=cfg.costs.half_spread.floor_bps * 1e-4
            if cfg.costs.half_spread.model != "none"
            else 0.0,
            adv_usd=float(dollar.mean()) if dollar.len() else 0.0,  # type: ignore[arg-type]
            sigma_daily=float(rets.std()) if rets.len() > 1 else 0.0,  # type: ignore[arg-type]
            fee_rate_annual=ETF_FEE,
        )
    tested = pl.DataFrame(
        {
            "run_id": run_id or "",
            "window_id": view.window.window_id,
            "formation_start": view.window.formation_start,
            "formation_end": view.window.formation_end,
            "security_id": ids,
            "hedge": list(hedges),
            "s_score": s0,
            "ou_kappa": kappa0,
            "selected": np.isfinite(s0),
        },
        schema=AL_TESTED,
    )
    return FrozenAl(
        PairSet(view.window, (), tested),
        tuple(ids),
        history,
        hedges,
        hedge_history,
        legs,
        s0,
        kappa0,
    )


def _states(previous: npt.NDArray[np.int8], s: F64, cfg: RunConfig) -> npt.NDArray[np.int8]:
    al = cfg.al
    valid = np.isfinite(s)
    state = previous.copy()
    long_close = (previous == 1) & (~valid | (s > -al.s_close_long))
    short_close = (previous == -1) & (~valid | (s < al.s_close_short))
    state[long_close | short_close] = 0
    flat = previous == 0
    state[flat & valid & (s < -al.s_open)] = 1
    state[flat & valid & (s > al.s_open)] = -1
    return state


def run_al_window(
    frozen: FrozenAl, view: TradingView, cfg: RunConfig, run_id: str = ""
) -> WindowResult:
    """Trade one window with daily re-estimation from trailing data only."""
    sessions = view.sessions()
    n, k = len(sessions), len(frozen.ids)
    params = cost_params(cfg)
    lag = cfg.windows.execution_lag_days
    sizes = np.zeros(k)
    s0 = np.nan_to_num(frozen.s_initial)
    seed = max(int((s0 > cfg.al.s_open).sum()), int((s0 < -cfg.al.s_open).sum()))
    side_counts: list[float] = [float(seed)] * SIZE_LOOKBACK
    ids = list(frozen.ids)
    prices = view.prices(["ret_total", "ret_delist", "close"]).filter(
        pl.col("security_id").is_in(ids)
    )
    rets = _wide(prices, sessions, ids, "ret_total")
    delist = _wide(prices, sessions, ids, "ret_delist")
    closes = _wide(prices, sessions, ids, "close")
    hedge_panel = _etf_panel(view.etf_returns(), sessions, sorted(set(frozen.hedges)))
    hedge_returns = _hedge_matrix(frozen.hedges, hedge_panel)
    history = frozen.history.copy()
    hedge_hist = _hedge_matrix(frozen.hedges, frozen.hedge_history)
    seen = np.isfinite(rets)
    dead = np.zeros(k, dtype=bool)
    stale = np.zeros(k, dtype=np.int32)
    desired = np.zeros((n, k), dtype=np.int8)
    state = np.zeros(k, dtype=np.int8)
    s_hist = np.full((n, k), np.nan)
    beta_hist = np.zeros((n, k))
    for t in range(n):
        day = np.nan_to_num(rets[t])
        history = np.vstack([history[1:], day])
        hedge_hist = np.vstack([hedge_hist[1:], hedge_returns[t]]) if k else hedge_hist
        stale = np.where(seen[t], 0, stale + 1)
        dead |= np.isfinite(delist[t]) | (stale > STALE_LIMIT)
        s, _, beta = (
            _signals(history, hedge_hist, cfg) if k else (np.zeros(0), np.zeros(0), np.zeros(0))
        )
        s = np.where(dead, np.nan, s)
        s_hist[t], beta_hist[t] = s, beta
        state = _states(state, s, cfg)
        state[dead] = 0
        desired[t] = state
    effective = np.zeros((n, k), dtype=np.int8)
    if n > lag:
        effective[lag:] = desired[: n - lag]
    effective[-1:] = 0
    first_dead = np.full(k, n, dtype=np.int64)
    for i in range(k):
        hits = np.nonzero(np.isfinite(delist[:, i]))[0]
        if hits.size:
            first_dead[i] = hits[0]
        else:
            run = 0
            for t in range(n):
                run = 0 if seen[t, i] else run + 1
                if run > STALE_LIMIT:
                    first_dead[i] = t
                    break
    for i in range(k):
        if first_dead[i] < n:
            effective[first_dead[i] :, i] = 0
    pnl = np.zeros(n)
    trading = np.zeros(n)
    borrow = np.zeros(n)
    employed = np.zeros(n)
    exposure = np.zeros(n)
    capital = np.empty(n)
    count = np.zeros(n, dtype=np.int32)
    trades: list[dict[str, object]] = []
    held_beta = np.zeros(k)
    position = np.zeros(k, dtype=np.int8)
    instruments = sorted(set(frozen.hedges))
    column_of = {name: j for j, name in enumerate(instruments)}
    hedge_column = np.array([column_of[name] for name in frozen.hedges], dtype=np.int64)
    instrument_returns = (
        np.column_stack([hedge_panel[name] for name in instruments])
        if instruments
        else np.zeros((n, 0))
    )
    book = np.zeros(len(instruments))  # net hedge notional per instrument
    level = 1.0
    for t in range(n):
        capital[t] = level
        if t > 0:
            open_now = position != 0
            r = np.nan_to_num(rets[t])
            d = np.where(np.isfinite(delist[t]), delist[t], 0.0)
            total = (1.0 + r) * (1.0 + d) - 1.0
            stock_values = position * sizes
            pnl[t] = float(np.sum(stock_values * total) + np.sum(book * instrument_returns[t]))
            for i in np.nonzero(open_now & (stock_values < 0))[0]:
                borrow[t] += borrow_cost(
                    stock_values[i],
                    frozen.legs[ids[i]].fee_rate_annual,
                    sessions[t - 1],
                    sessions[t],
                    params.borrow_multiplier,
                )
            for short in np.nonzero(book < 0)[0]:
                borrow[t] += borrow_cost(
                    book[short], ETF_FEE, sessions[t - 1], sessions[t], params.borrow_multiplier
                )
            employed[t] = float(sizes[open_now].sum())
        equity = level + pnl[t] - borrow[t]
        changes = np.nonzero(effective[t] != position)[0]
        expected = max(
            float(np.mean(side_counts[-SIZE_LOOKBACK:])), float(cfg.al.min_positions_per_side)
        )
        for i in changes:
            new = int(effective[t, i])
            old = int(position[i])
            signal_day = t - lag if t - lag >= 0 else t
            if new == 0:
                reason = (
                    "delist"
                    if first_dead[i] == t and np.isfinite(delist[t, i])
                    else "stale"
                    if first_dead[i] == t
                    else "window_end"
                    if t == n - 1
                    else "exit"
                )
                traded = float(-old * sizes[i])
                held_beta[i] = 0.0
                sizes[i] = 0.0
            else:
                reason = "entry"
                held_beta[i] = float(beta_hist[signal_day, i])
                sizes[i] = equity * cfg.al.gross_per_side / expected
                traded = float(new * sizes[i])
            position[i] = new
            leg = frozen.legs[ids[i]]
            commission, spread, impact = trade_cost(traded, leg, params)
            trading[t] += commission + spread + impact
            trades.append(
                _trade_row(
                    run_id,
                    frozen,
                    ids[i],
                    ids[i],
                    sessions[signal_day],
                    sessions[t],
                    traded,
                    float(closes[t, i]),
                    reason,
                    float(s_hist[signal_day, i]),
                    (commission, spread, impact),
                    participation(traded, leg, params),
                    params.aum_usd,
                )
            )
        target = np.zeros(len(instruments))
        if k:
            np.add.at(target, hedge_column, -position * sizes * held_beta)
        for j, name in enumerate(instruments):
            hedge_trade = float(target[j] - book[j])
            if abs(hedge_trade) <= 1e-15:
                continue
            leg = frozen.legs[name]
            commission, spread, impact = trade_cost(hedge_trade, leg, params)
            trading[t] += commission + spread + impact
            trades.append(
                _trade_row(
                    run_id,
                    frozen,
                    name,
                    name,
                    sessions[t],
                    sessions[t],
                    hedge_trade,
                    math.nan,
                    "hedge",
                    math.nan,
                    (commission, spread, impact),
                    participation(hedge_trade, leg, params),
                    params.aum_usd,
                )
            )
        book = target
        count[t] = int((position != 0).sum())
        side_counts.append(((position == 1).sum() + (position == -1).sum()) / 2.0)
        exposure[t] = float(np.sum(sizes) + np.abs(book).sum())
        level += pnl[t] - trading[t] - borrow[t]
    ret_gross, cost_trading, cost_borrow = pnl / capital, trading / capital, borrow / capital
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
            "n_open": count,
        }
    )
    frame = (
        pl.DataFrame(trades, schema=TRADE_COLUMNS) if trades else pl.DataFrame(schema=TRADE_COLUMNS)
    )
    return WindowResult(frozen.pairset.window.window_id, daily, frame)


def _trade_row(
    run_id: str,
    frozen: FrozenAl,
    pair_id: str,
    security: str,
    signal_day: date,
    exec_day: date,
    traded: float,
    price: float,
    reason: str,
    z: float,
    costs: tuple[float, float, float],
    share: float,
    aum: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "window_id": frozen.pairset.window.window_id,
        "pair_id": pair_id,
        "date_signal": signal_day,
        "date_exec": exec_day,
        "security_id": security,
        "side": 1 if traded > 0 else -1,
        "notional_usd": traded * aum,
        "price": price,
        "reason": reason,
        "z_signal": z,
        "cost_commission": costs[0] * aum,
        "cost_spread": costs[1] * aum,
        "cost_impact": costs[2] * aum,
        "participation": share,
    }
