from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from statarb import __version__, cli
from statarb.smoke import Check
from statarb.tracking import log_failed, log_trial

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_no_args_shows_help() -> None:
    result = runner.invoke(cli.app, [])
    assert "smoke" in result.output


@pytest.mark.parametrize(("status", "code"), [("ok", 0), ("warn", 0), ("fail", 1)])
def test_smoke_exit_code(monkeypatch: pytest.MonkeyPatch, status: str, code: int) -> None:
    monkeypatch.setattr(
        cli, "run_smoke", lambda settings: [Check("a", "ok", "x"), Check("b", status, "y")]
    )
    result = runner.invoke(cli.app, ["smoke"])
    assert result.exit_code == code
    assert f"{status.upper():<5} b" in result.output
    assert "checks passed" in result.output


def _fill_log(results: Path) -> None:
    log = results / "trial_log.csv"
    dates = [date(2021, 1, 1) + timedelta(days=i) for i in range(5)]
    log_trial(log, "pairs_distance", {"k": 1}, dates, [0.01, -0.02, 0.0, 0.01, 0.02])
    log_trial(log, "pairs_coint", {"k": 2}, dates, [0.01, -0.01, 0.0, 0.01, 0.0])
    log_failed(log, "pairs_coint", {"k": 3}, "boom")


def test_trials_count_and_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STATARB_RESULTS_DIR", str(tmp_path / "results"))
    assert runner.invoke(cli.app, ["trials", "count"]).output.strip() == "0"
    _fill_log(tmp_path / "results")
    assert runner.invoke(cli.app, ["trials", "count"]).output.strip() == "3"
    listed = runner.invoke(cli.app, ["trials", "list"]).output
    assert "pairs_coint: 2" in listed
    assert "pairs_distance: 1" in listed
    only = runner.invoke(cli.app, ["trials", "list", "--family", "pairs_coint"]).output
    assert "pairs_distance" not in only
    assert "sharpe=-" in only


def test_trials_bad_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STATARB_RESULTS_DIR", str(tmp_path))
    (tmp_path / "trial_log.csv").write_text("a,b\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["trials", "count"])
    assert result.exit_code == 1
