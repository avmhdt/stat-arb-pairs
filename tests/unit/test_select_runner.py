"""statarb select and statarb report on the synthetic market."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from fixtures.synth import make_market
from statarb import cli
from statarb.config import RunConfig, load_config
from statarb.reports import tables
from statarb.selection import runner
from statarb.settings import Settings
from unit.test_data_layer import write_processed

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg() -> RunConfig:
    raw = load_config(REPO / "configs" / "coint_eg_kalman.yaml").model_dump()
    raw["data"].update(provider="sharadar", start="2015-01-01", end="2017-12-31")
    raw["universe"].update(top_n=None, min_price_usd=None)
    raw["run"]["data_profile"] = "headline"
    return RunConfig.model_validate(raw)


def test_run_selection_writes_pairs(cfg: RunConfig, tmp_path: Path) -> None:
    data = make_market(seed=4, years=3).data
    result = runner.run_selection(cfg, tmp_path, tmp_path / "runs", data=data)
    pairs = pl.read_parquet(result.path / "pairs.parquet")
    assert result.windows == pairs["window_id"].n_unique() == 19
    assert result.rows == pairs.height
    info = json.loads((result.path / runner.SELECTION_INFO).read_text())
    assert info["method"] == "engle_granger" and info["data_profile"] == "prototype"
    assert info["config"]["selection"]["fdr"]["q"] == 0.05
    one = runner.run_selection(cfg, tmp_path, tmp_path / "runs", ["2016-02-01"], data=data)
    assert one.windows == 1
    with pytest.raises(ValueError, match="no such windows"):
        runner.run_selection(cfg, tmp_path, tmp_path / "runs", ["1990-01-01"], data=data)


def test_workers_give_identical_rows(cfg: RunConfig, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_processed(processed, "headline")
    short = cfg.model_copy(update={"windows": cfg.windows.model_copy(update={"step_months": 6})})
    serial = runner.run_selection(short, processed, tmp_path / "a")
    parallel = runner.run_selection(short, processed, tmp_path / "b", workers=2)
    left = pl.read_parquet(serial.path / "pairs.parquet").drop("run_id")
    right = pl.read_parquet(parallel.path / "pairs.parquet").drop("run_id")
    assert left.equals(right)


def test_report_writes_aggregates(cfg: RunConfig, tmp_path: Path) -> None:
    data = make_market(seed=4, years=3).data
    object.__setattr__(data, "data_profile", "headline")
    runs = tmp_path / "runs"
    runner.run_selection(cfg, tmp_path, runs, data=data)
    doc = tmp_path / "doc.md"
    doc.write_text(f"intro\n{tables.START}\nold\n{tables.END}\noutro\n", encoding="utf-8")
    table = tables.write_fdr_outputs(runs, tmp_path / "tables", doc)
    csv = pl.read_csv(tmp_path / "tables" / "fdr_by_window.csv")
    assert csv.height == table.height == 19
    assert not {"leg_y", "leg_x", "security_id", "ticker"} & set(csv.columns)
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old" not in text
    assert "| coint_eg_kalman_sic2 | 2016 |" in text
    with pytest.raises(FileNotFoundError):
        tables.write_fdr_outputs(tmp_path / "empty", tmp_path / "t", doc)
    (tmp_path / "bad.md").write_text("no markers")
    with pytest.raises(ValueError, match="markers"):
        tables.write_fdr_outputs(runs, tmp_path / "tables", tmp_path / "bad.md")


def test_cli_select_and_report(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    write_processed(settings.data_dir / "processed", "headline")
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    runner_ = CliRunner()
    args = [
        "select",
        "--config",
        "configs/coint_eg_kalman.yaml",
        "--set",
        "data.start=2015-01-01",
        "--set",
        "data.end=2017-12-31",
        "--set",
        "universe.top_n=null",
        "--window",
        "2016-02-01",
    ]
    result = runner_.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "1 windows" in result.output
    bad = runner_.invoke(cli.app, [*args[:-1], "1990-01-01"])
    assert bad.exit_code == 2
    doc = tmp_path / "docs.md"
    doc.write_text(f"{tables.START}\n{tables.END}\n")
    reported = runner_.invoke(cli.app, ["report", "--doc", str(doc)])
    assert reported.exit_code == 0, reported.output
    assert (
        runner_.invoke(cli.app, ["report", "--profile", "prototype", "--doc", str(doc)]).exit_code
        == 2
    )
