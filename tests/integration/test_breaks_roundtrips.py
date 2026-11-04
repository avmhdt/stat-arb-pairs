"""FR-26 break policies and round-trip accounting on the synthetic market."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fixtures.synth import SynthMarket, make_market
from statarb.config import RunConfig
from statarb.trading.backtest import run_backtest
from statarb.trading.engine import break_flags

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def market() -> SynthMarket:
    return make_market(seed=11, years=3)


@pytest.fixture(autouse=True)
def configs(tmp_path: Path) -> None:
    shutil.copytree(REPO / "configs", tmp_path / "configs")


def _cfg(policy: str = "stop_only", **rules: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"name": "b", "strategy_family": "pairs_coint", "data_profile": "prototype"},
            "data": {"provider": "free", "start": "2015-01-01", "end": "2017-12-31"},
            "universe": {"top_n": None, "min_price_usd": None},
            "selection": {"top_k": 5},
            "model": {"hedge": "static"},
            "windows": {"step_months": 3},
            "rules": {"break_policy": policy, **rules},
        }
    )


def test_break_flags_detect_a_decoupling() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 200))
    spread = np.zeros(200)
    for t in range(1, 200):
        spread[t] = 0.5 * spread[t - 1] + rng.normal(0, 0.005)
    y = x + spread
    y[120:] += np.cumsum(rng.normal(0.004, 0.02, 80))  # the relation breaks at 120
    tail = (y[:60], x[:60])
    legs_y = {"tr_index": np.exp(y[60:])}
    legs_x = {"tr_index": np.exp(x[60:])}
    flags = break_flags(tail, legs_y, legs_x, 60, 0.20)
    assert not flags[:40].any()
    assert flags[-20:].mean() > 0.5
    gappy = {"tr_index": np.where(np.arange(140) == 10, np.nan, np.exp(y[60:]))}
    assert not break_flags(tail, gappy, legs_x, 60, 0.20)[10]


def test_roundtrips_add_up_to_net_pnl(market: SynthMarket, tmp_path: Path) -> None:
    result = run_backtest(_cfg(), data=market.data, results_dir=tmp_path)
    trips = pl.read_parquet(result.path / "roundtrips.parquet")  # type: ignore[operator]
    assert trips.height > 0
    entries = (
        result.trades.filter(pl.col("reason") == "entry")
        .select("window_id", "pair_id", "date_exec")
        .unique()
        .height
    )
    assert trips.height == entries
    assert (trips["holding_days"] >= 0).all()
    assert {"turnover", "cost_impact"} <= set(result.returns.columns)


@pytest.mark.parametrize("policy", ["hold", "close_on_break", "close_and_blacklist"])
def test_policies_run_and_label_exits(market: SynthMarket, tmp_path: Path, policy: str) -> None:
    result = run_backtest(_cfg(policy, break_pvalue=0.05), data=market.data, results_dir=tmp_path)
    reasons = set(result.trades["reason"].to_list())
    if policy == "hold":
        assert "stop" not in reasons and "break" not in reasons
    else:
        assert "break" in reasons


def test_entries_respect_the_participation_cap(market: SynthMarket, tmp_path: Path) -> None:
    cfg = _cfg()
    big = cfg.model_copy(update={"costs": cfg.costs.model_copy(update={"aum_usd": 1e12})})
    assert big.costs.impact.model == "sqrt"
    result = run_backtest(big, data=market.data, results_dir=tmp_path)
    entries = result.trades.filter(pl.col("reason") == "entry")
    assert entries.height > 0
    cap = big.costs.impact.max_participation
    assert (entries["participation"] <= cap * (1 + 1e-9)).all()
    assert entries["participation"].max() == pytest.approx(cap)
