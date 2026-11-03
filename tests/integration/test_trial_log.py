"""AC-W-10: every run appends exactly one compatible trial-log row; failures count too."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from fixtures.synth import SynthMarket, make_market
from statarb import cli
from statarb.config import RunConfig
from statarb.settings import Settings
from statarb.tracking import LOG_COLUMNS, TRIAL_LOG_COLUMNS, read_trials
from statarb.trading.backtest import append_trial, run_backtest
from unit.test_data_layer import write_processed

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def repo_configs(tmp_path: Path) -> None:
    """Runs read configs/ relative to the working directory, which tests move to tmp_path."""
    if not (tmp_path / "configs").exists():
        shutil.copytree(REPO / "configs", tmp_path / "configs")


@pytest.fixture(scope="module")
def market() -> SynthMarket:
    return make_market(seed=11, years=3)


def _cfg(family: str = "pairs_distance", **selection: object) -> RunConfig:
    method = "distance" if family == "pairs_distance" else "engle_granger"
    return RunConfig.model_validate(
        {
            "run": {"name": "t", "strategy_family": family, "data_profile": "prototype"},
            "data": {"provider": "free", "start": "2015-01-01", "end": "2017-12-31"},
            "universe": {"top_n": None, "min_price_usd": None},
            "selection": {
                "method": method,
                "grouping": "none" if method == "distance" else "sic2",
                "top_k": 3,
                **selection,
            },
            "windows": {"step_months": 3},
        }
    )


def test_runs_append_one_row_each(market: SynthMarket, tmp_path: Path) -> None:
    first = run_backtest(_cfg(), data=market.data, results_dir=tmp_path)
    second = run_backtest(_cfg("pairs_coint"), data=market.data, results_dir=tmp_path)
    rows = read_trials(tmp_path / "trial_log.csv")
    assert [row["run_id"] for row in rows] == [first.run_id, second.run_id]
    assert append_trial(first) == first.run_id
    assert len(read_trials(tmp_path / "trial_log.csv")) == 2
    with (tmp_path / "trial_log.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header[:10]) == TRIAL_LOG_COLUMNS and tuple(header) == LOG_COLUMNS
    for row in rows:
        series = pl.read_parquet(tmp_path / row["returns_path"])
        assert series.height == int(row["n_obs"])
        assert {
            "date",
            "ret",
            "ret_gross",
            "ret_employed",
            "cost_trading",
            "cost_borrow",
            "gross_exposure",
            "n_open",
        } <= set(series.columns)
        assert (
            series["ret"] == series["ret_gross"] - series["cost_trading"] - series["cost_borrow"]
        ).all()
        run_dir = tmp_path / "runs" / row["run_id"]
        for name in ("config.yaml", "pairs.parquet", "trades.parquet", "metrics.json"):
            assert (run_dir / name).is_file()
        assert '"data_profile":"prototype"' in row["params_json"]
    validator = pytest.importorskip("overfitlab.triallog")
    issues = validator.validate_trial_log(tmp_path / "trial_log.csv")
    assert [issue for issue in issues if issue.severity == "error"] == []


def test_failed_run_is_logged(market: SynthMarket, tmp_path: Path) -> None:
    short = _cfg().model_copy(
        update={"windows": _cfg().windows.model_copy(update={"formation_months": 60})}
    )
    with pytest.raises(ValueError, match="too short"):
        run_backtest(short, data=market.data, results_dir=tmp_path)
    rows = read_trials(tmp_path / "trial_log.csv")
    assert len(rows) == 1 and rows[0]["n_obs"] == "0" and rows[0]["returns_path"] == ""
    failed = pl.read_csv(tmp_path / "failed_runs.csv")
    assert failed["error"][0].startswith("ValueError")


def test_cli_run_sweep_and_trials(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    write_processed(settings.data_dir / "processed", "prototype")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    config = tmp_path / "configs" / "t.yaml"
    config.write_text(
        "extends: base.yaml\n"
        "run: {name: t, strategy_family: pairs_distance, data_profile: prototype}\n"
        "data: {provider: free, start: 2015-01-01, end: 2017-12-31}\n"
        "universe: {top_n: null, min_price_usd: null}\n"
        "selection: {method: distance, grouping: none, top_k: 3}\n"
        "windows: {step_months: 6}\n"
    )
    grid = tmp_path / "grid.yaml"
    grid.write_text("rules.entry_z: [1.5, 2.0]\n")
    runner = CliRunner()
    ran = runner.invoke(cli.app, ["run", "--config", str(config)])
    assert ran.exit_code == 0, ran.output
    swept = runner.invoke(cli.app, ["sweep", "--config", str(config), "--grid", str(grid)])
    assert swept.exit_code == 0, swept.output
    assert runner.invoke(cli.app, ["trials", "count"]).output.strip() == "3"
    rows = read_trials(settings.results_dir / "trial_log.csv")
    assert sum("sweep" in row["x_note"] for row in rows) == 2


def test_workers_do_not_change_returns(
    market: SynthMarket, settings: Settings, tmp_path: Path
) -> None:
    processed = settings.data_dir / "processed"
    write_processed(processed, "prototype")
    cfg = _cfg().model_copy(
        update={"windows": _cfg().windows.model_copy(update={"step_months": 6})}
    )
    one = run_backtest(cfg, 1, processed=processed, results_dir=tmp_path / "a")
    two = run_backtest(cfg, 2, processed=processed, results_dir=tmp_path / "b")
    assert one.returns.equals(two.returns)
    assert one.trades.drop("run_id").equals(two.trades.drop("run_id"))
