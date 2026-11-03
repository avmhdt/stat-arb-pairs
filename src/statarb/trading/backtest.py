"""``run_backtest``: every window selected, frozen and traded, then logged (FR-17, FR-27).

The trial-log row is written here, inside the one function every run goes through, so no
evaluation can skip it; a run that fails is logged too, with ``n_obs = 0`` (contract v1.0).
Windows run in up to 8 worker processes and come back in window order, so results do not
depend on the worker count (NFR-02, NFR-03).
"""

from __future__ import annotations

import json
import multiprocessing
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

from statarb.config import RunConfig
from statarb.data.market import MarketData, load_market_data
from statarb.evaluation.metrics import summary_metrics, trade_stats
from statarb.industry import SicEtfMap, load_sic_etf_map
from statarb.tracking import log_failed, log_trial
from statarb.trading.engine import borrow_schedule, combine, freeze, run_window
from statarb.trading.residual import freeze_al, run_al_window
from statarb.windows import FormationView, TradingView, Window, rolling_windows

log = structlog.get_logger(__name__)

PRICE_COLUMNS = [
    "close",
    "high",
    "low",
    "volume",
    "close_is_mid",
    "mktcap_usd",
    "tr_index",
    "ret_total",
    "ret_delist",
]
MAX_WORKERS = 8
SIC_MAP = Path("configs") / "sic_to_etf.yaml"

_STATE: dict[str, Any] = {}


@dataclass
class RunResult:
    """Everything a run produced. ``returns`` follows the ``daily_returns`` schema."""

    run_id: str
    cfg: RunConfig
    returns: pl.DataFrame
    trades: pl.DataFrame
    pairs: pl.DataFrame
    metrics: dict[str, float]
    data_profile: str
    manifest_hash: str
    windows: int
    path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_for(cfg: RunConfig, processed: Path) -> MarketData:
    """The processed panel a run needs."""
    return load_market_data(
        processed, cfg.data.start, cfg.data.end, PRICE_COLUMNS, tuple(cfg.universe.security_types)
    )


def _context(cfg: RunConfig, data: MarketData) -> None:
    _STATE["cfg"] = cfg
    _STATE["data"] = data
    _STATE["sic"] = load_sic_etf_map(SIC_MAP) if SIC_MAP.exists() else None
    _STATE["borrow"] = borrow_schedule(Path(cfg.costs.borrow.schedule))


def _init(processed: str, cfg_json: str) -> None:
    cfg = RunConfig.model_validate_json(cfg_json)
    _context(cfg, load_for(cfg, Path(processed)))


def _window(window: Window, run_id: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cfg: RunConfig = _STATE["cfg"]
    data: MarketData = _STATE["data"]
    sic: SicEtfMap | None = _STATE["sic"]
    view = FormationView(data, window, cfg.universe)
    if cfg.run.strategy_family in ("al_pca", "al_etf"):
        frozen_al = freeze_al(view, cfg, _STATE["borrow"], sic, run_id)
        result = run_al_window(frozen_al, TradingView(data, window), cfg, run_id)
        return frozen_al.pairset.tested, result.daily, result.trades
    frozen = freeze(view, cfg, _STATE["borrow"], sic, run_id)
    result = run_window(frozen, TradingView(data, window), cfg, run_id)
    return frozen.pairset.tested, result.daily, result.trades


def windows_for(cfg: RunConfig, data: MarketData) -> list[Window]:
    """The run's windows on the data's calendar."""
    return rolling_windows(
        data.sessions,
        cfg.windows.formation_months,
        cfg.windows.trading_months,
        cfg.windows.step_months,
    )


def params_for(cfg: RunConfig, data: MarketData) -> dict[str, Any]:
    """The trial's ``params_json``: the full resolved config plus data provenance."""
    params: dict[str, Any] = json.loads(cfg.model_dump_json())
    params["data_profile"] = data.data_profile
    params["manifest_hash"] = data.manifest_hash
    return params


def _simulate(
    cfg: RunConfig, data: MarketData, processed: Path | None, workers: int, run_id: str
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int]:
    windows = windows_for(cfg, data)
    if not windows:
        raise ValueError("the sample is too short for one formation and trading window")
    if workers <= 1 or processed is None or len(windows) == 1:
        _context(cfg, data)
        parts = [_window(window, run_id) for window in windows]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, MAX_WORKERS),
            # Fork after Polars or Numba threads have started can deadlock on Linux.
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=(str(processed), cfg.model_dump_json()),
        ) as pool:
            parts = list(pool.map(_window, windows, [run_id] * len(windows)))
    pairs = pl.concat([part[0] for part in parts], how="vertical")
    returns = combine([part[1] for part in parts])
    trade_parts = [part[2] for part in parts if part[2].height]
    trades = pl.concat(trade_parts, how="vertical") if trade_parts else parts[0][2]
    return pairs, returns, trades, len(windows)


def run_backtest(
    cfg: RunConfig,
    workers: int = 1,
    *,
    data: MarketData | None = None,
    processed: Path | None = None,
    results_dir: Path = Path("results"),
    note: str = "",
) -> RunResult:
    """Run every window, write ``results/runs/<run_id>/`` and append one trial-log row."""
    panel = data if data is not None else load_for(cfg, processed or Path("data") / "processed")
    log_path = results_dir / "trial_log.csv"
    params = params_for(cfg, panel)
    family = cfg.run.strategy_family
    provisional = str(uuid.uuid4())
    try:
        pairs, returns, trades, count = _simulate(
            cfg, panel, processed if data is None else None, workers, provisional
        )
        if returns.height < 2:
            raise ValueError("the run produced fewer than two daily returns")
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        log_failed(log_path, family, params, detail + "\n" + traceback.format_exc(limit=3))
        raise
    run_id = log_trial(
        log_path,
        family,
        params,
        returns["date"].to_list(),
        returns["ret"].to_list(),
        note=note,
        extra={
            name: returns[name].to_list() for name in returns.columns if name not in ("date", "ret")
        },
    )
    pairs = pairs.with_columns(pl.lit(run_id).alias("run_id"))
    trades = trades.with_columns(pl.lit(run_id).alias("run_id"))
    metrics = summary_metrics(returns["ret"]) | {
        f"gross_{k}": v
        for k, v in summary_metrics(returns["ret_gross"]).items()
        if k in ("mean_annual", "sharpe_annual")
    }
    metrics |= trade_stats(trades, returns, cfg.costs.aum_usd)
    metrics |= {
        "cost_trading_annual": float(returns["cost_trading"].mean()) * 252,  # type: ignore[arg-type]
        "cost_borrow_annual": float(returns["cost_borrow"].mean()) * 252,  # type: ignore[arg-type]
        "pairs_traded_per_window": float(pairs["selected"].sum()) / count,
        "windows": float(count),
    }
    target = results_dir / "runs" / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.yaml").write_text(yaml.safe_dump(params, sort_keys=True), encoding="utf-8")
    pairs.write_parquet(target / "pairs.parquet", compression="zstd")
    if cfg.output.save_trades:
        trades.write_parquet(target / "trades.parquet", compression="zstd")
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info(
        "run_done", run_id=run_id, family=family, windows=count, sharpe=metrics.get("sharpe_annual")
    )
    return RunResult(
        run_id,
        cfg,
        returns,
        trades,
        pairs,
        metrics,
        panel.data_profile,
        panel.manifest_hash,
        count,
        target,
    )


def append_trial(result: RunResult, path: str = "results/trial_log.csv") -> str:
    """Spec interface: the row was appended inside ``run_backtest``; returns its ``run_id``.

    Logging happens in ``run_backtest`` so that no run can skip it; calling this again does
    not log twice.
    """
    del path
    return result.run_id
