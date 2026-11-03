"""``statarb select``: selection over every window of a config, nothing traded.

Windows are independent, so they may run in worker processes; each worker loads the
processed panel once. Results come back in window order, so the output does not depend
on the worker count (NFR-03).
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from statarb.config import RunConfig
from statarb.data.market import MarketData, load_market_data
from statarb.industry import SicEtfMap, load_sic_etf_map
from statarb.selection.select import select_window
from statarb.tracking import git_sha
from statarb.windows import FormationView, Window, rolling_windows

PRICE_COLUMNS = ["close", "volume", "close_is_mid", "mktcap_usd", "tr_index"]
MAX_WORKERS = 8
SIC_MAP = Path("configs") / "sic_to_etf.yaml"
SELECTION_INFO = "selection.json"

_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class SelectionRun:
    """Where a selection run wrote its rows, and what it covered."""

    run_id: str
    path: Path
    windows: int
    rows: int


def load_for(cfg: RunConfig, processed: Path) -> MarketData:
    """The processed panel a config needs."""
    return load_market_data(
        processed, cfg.data.start, cfg.data.end, PRICE_COLUMNS, tuple(cfg.universe.security_types)
    )


def _sic_map() -> SicEtfMap | None:
    return load_sic_etf_map(SIC_MAP) if SIC_MAP.exists() else None


def _init(processed: str, cfg_json: str) -> None:
    cfg = RunConfig.model_validate_json(cfg_json)
    _STATE["cfg"] = cfg
    _STATE["data"] = load_for(cfg, Path(processed))
    _STATE["sic"] = _sic_map()


def _one(window: Window, run_id: str) -> pl.DataFrame:
    cfg = _STATE["cfg"]
    data = _STATE["data"]
    assert isinstance(cfg, RunConfig) and isinstance(data, MarketData)
    sic = _STATE["sic"]
    view = FormationView(data, window, cfg.universe)
    return select_window(
        view, cfg.selection, sic if isinstance(sic, SicEtfMap) else None, run_id
    ).tested


def run_selection(
    cfg: RunConfig,
    processed: Path,
    out_root: Path,
    window_ids: list[str] | None = None,
    workers: int = 1,
    data: MarketData | None = None,
) -> SelectionRun:
    """Select every window (or the given ones) and write ``pairs.parquet``.

    The output folder is ``<out_root>/select-<run_id>/``, git-ignored because the rows name
    securities. ``selection.json`` records the config and data provenance.
    """
    run_id = str(uuid.uuid4())
    panel = data if data is not None else load_for(cfg, processed)
    windows = rolling_windows(
        panel.sessions,
        cfg.windows.formation_months,
        cfg.windows.trading_months,
        cfg.windows.step_months,
    )
    if window_ids is not None:
        wanted = set(window_ids)
        windows = [w for w in windows if w.window_id in wanted]
        missing = sorted(wanted - {w.window_id for w in windows})
        if missing:
            raise ValueError(f"no such windows: {missing}")
    frames: list[pl.DataFrame]
    if workers <= 1 or len(windows) <= 1:
        _STATE.update(cfg=cfg, data=panel, sic=_sic_map())
        frames = [_one(window, run_id) for window in windows]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, MAX_WORKERS),
            initializer=_init,
            initargs=(str(processed), cfg.model_dump_json()),
        ) as pool:
            frames = list(pool.map(_one, windows, [run_id] * len(windows)))
    rows = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    target = out_root / f"select-{run_id}"
    target.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(target / "pairs.parquet", compression="zstd")
    info = {
        "run_id": run_id,
        "name": cfg.run.name,
        "strategy_family": cfg.run.strategy_family,
        "method": cfg.selection.method,
        "data_profile": panel.data_profile,
        "manifest_hash": panel.manifest_hash,
        "windows": len(windows),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(out_root / "trial_log.csv"),
        "config": json.loads(cfg.model_dump_json()),
    }
    (target / SELECTION_INFO).write_text(
        json.dumps(info, indent=2, sort_keys=True), encoding="utf-8"
    )
    return SelectionRun(run_id, target, len(windows), rows.height)
