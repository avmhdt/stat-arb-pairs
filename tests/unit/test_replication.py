from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from statarb import cli
from statarb.evaluation import replication
from statarb.evaluation.metrics import monthly, newey_west_mean, summary_metrics, trade_stats
from statarb.settings import Settings

REPO = Path(__file__).resolve().parents[2]
TARGETS = pl.read_csv(REPO / "tests" / "fixtures" / "ggr2006_targets.csv")


def _fake_run(runs: Path, run_id: str, top_k: int, rank_from: int, lag: int, mean: float) -> None:
    folder = runs / run_id
    folder.mkdir(parents=True)
    params = {
        "selection": {"method": "distance", "top_k": top_k, "rank_from": rank_from},
        "windows": {"execution_lag_days": lag},
        "data_profile": "headline",
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(params))
    days = [
        date(1999, 1, 4) + timedelta(days=i)
        for i in range(1500)
        if (date(1999, 1, 4) + timedelta(days=i)).weekday() < 5
    ]
    ret = mean / 21 + 0.0002 * np.sin(np.arange(len(days)))
    pl.DataFrame({"date": days, "ret": ret, "ret_employed": ret * 1.7}).write_parquet(
        folder / "returns.parquet"
    )


def test_targets_file_is_complete() -> None:
    assert TARGETS.height == 15
    assert TARGETS["confirmed"].all()
    assert TARGETS["source"].str.contains("Table").all()
    full = TARGETS.filter(pl.col("period") == replication.FULL_SAMPLE)
    assert full.height == 12


def test_table_and_verdict(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    specs = [
        ("a", 5, 1, 0, 0.005),
        ("b", 20, 1, 0, 0.0015),
        ("c", 20, 101, 0, 0.0002),
        ("d", 5, 1, 1, 0.003),
        ("e", 20, 1, 1, 0.001),
        ("f", 20, 101, 1, 0.0001),
    ]
    for spec in specs:
        _fake_run(runs, *spec)
    loaded = [replication.read_run(runs, rid) for rid, *_ in specs]
    assert {run.portfolio for run in loaded} == {"top5", "top20", "101-120"}
    table = replication.replication_table(loaded, TARGETS)
    assert table.height == 12
    assert table["same_sign"].all()
    assert table["paper_mean_monthly"].null_count() == 0
    result = replication.verdict(table)
    assert result["all_same_sign"] is True
    assert result["no_wait_ordering_holds"] is True
    assert result["pass"] is False  # these fake returns are far from the paper's level
    replication.figure(table, tmp_path / "fig" / "ggr")
    assert (tmp_path / "fig" / "ggr.png").is_file() and (tmp_path / "fig" / "ggr.svg").is_file()
    partial = replication.verdict(table.filter(pl.col("wait") == "no_wait"))
    assert partial["wait_one_day_complete"] is False


def test_rejects_cointegration_runs(tmp_path: Path) -> None:
    folder = tmp_path / "x"
    folder.mkdir()
    (folder / "config.yaml").write_text(yaml.safe_dump({"selection": {"method": "engle_granger"}}))
    with pytest.raises(ValueError, match="distance"):
        replication.read_run(tmp_path, "x")


def test_cli_evaluate_replication(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    runs = settings.results_dir / "runs"
    for spec in [("a", 5, 1, 0, 0.01), ("b", 20, 1, 0, 0.012), ("c", 20, 101, 0, 0.005)]:
        _fake_run(runs, *spec)
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "ggr2006_targets.csv").write_text(
        (REPO / "tests" / "fixtures" / "ggr2006_targets.csv").read_text()
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(
        cli.app,
        ["evaluate", "replication", "--runs", "a,b,c", "--out", "t/ggr.csv", "--figure", "f/ggr"],
    )
    assert result.exit_code == 0, result.output
    assert "pass: False" in result.output
    assert pl.read_csv(tmp_path / "t" / "ggr.csv").height == 6


def test_metrics_helpers() -> None:
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(70)]
    daily = pl.DataFrame({"date": days, "ret": [0.001] * 69 + [-0.01], "windows_active": 2})
    months = monthly(daily)
    assert months.height == 3
    assert months["ret"][0] == pytest.approx(1.001**31 - 1)
    mean, se, t = newey_west_mean(np.array([0.01, 0.02, 0.0, 0.01, 0.015]))
    assert mean == pytest.approx(0.011) and se > 0 and t > 0
    assert np.isnan(newey_west_mean(np.array([1.0]))[0])
    stats = summary_metrics(daily["ret"])
    assert stats["max_drawdown"] == pytest.approx(-0.01)
    assert summary_metrics(pl.Series([0.1]))["n_obs"] == 1.0
    assert trade_stats(pl.DataFrame(), daily, 1e7)["round_trips"] == 0.0
