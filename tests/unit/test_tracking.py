from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from statarb import tracking
from statarb.tracking import (
    LOG_COLUMNS,
    TRIAL_LOG_COLUMNS,
    TrialLogError,
    annual_sharpe,
    canonical_json,
    log_failed,
    log_trial,
    read_trials,
)

PARAMS = {"data_profile": "prototype", "rules": {"entry_z": 2.0}, "top_k": 20}
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _series(n: int = 80) -> tuple[list[date], list[float]]:
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    returns = [0.001 * math.sin(i) + 0.0002 for i in range(n)]
    return dates, returns


def test_columns_match_contract_order() -> None:
    assert TRIAL_LOG_COLUMNS == (
        "run_id",
        "timestamp",
        "git_sha",
        "strategy_family",
        "params_json",
        "start",
        "end",
        "n_obs",
        "sharpe_annual",
        "returns_path",
    )
    assert LOG_COLUMNS[:10] == TRIAL_LOG_COLUMNS
    assert all(name.startswith("x_") for name in LOG_COLUMNS[10:])


def test_canonical_json() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"\\u00e9"],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_log_trial_writes_row_and_series(tmp_path: Path) -> None:
    log = tmp_path / "results" / "trial_log.csv"
    dates, returns = _series()
    run_id = log_trial(log, "pairs_distance", PARAMS, dates, returns, note="unit")
    assert UUID4.match(run_id)
    rows = read_trials(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", row["timestamp"])
    assert re.match(r"^[0-9a-f]{40}(-dirty)?$", row["git_sha"])
    assert json.loads(row["params_json"]) == PARAMS
    assert (row["start"], row["end"], row["n_obs"]) == ("2020-01-01", "2020-03-20", "80")
    assert float(row["sharpe_annual"]) == pytest.approx(annual_sharpe(returns), rel=1e-9)
    assert row["returns_path"] == f"runs/{run_id}/returns.parquet"
    assert row["x_periods_per_year"] == "252"
    table = pq.read_table(log.parent / row["returns_path"])
    assert table.column_names == ["date", "ret"]
    assert table.column("ret").to_pylist() == returns
    raw = log.read_bytes()
    assert b"\r\n" not in raw
    assert raw.startswith(b"run_id,timestamp,")


def test_appends_one_row_per_run(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    dates, returns = _series()
    ids = {log_trial(log, "pairs_coint", PARAMS, dates, returns) for _ in range(3)}
    assert len(ids) == 3
    with log.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.reader(handle))) == 4


def test_failed_run(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    run_id = log_failed(log, "al_pca", PARAMS, "ValueError: boom")
    row = read_trials(log)[0]
    assert row["n_obs"] == "0"
    assert row["sharpe_annual"] == row["returns_path"] == row["start"] == ""
    with (tmp_path / "failed_runs.csv").open(encoding="utf-8", newline="") as handle:
        failed = list(csv.DictReader(handle))
    assert failed == [
        {
            "run_id": run_id,
            "timestamp": row["timestamp"],
            "strategy_family": "al_pca",
            "error": "ValueError: boom",
        }
    ]


def test_flat_series_has_zero_sharpe() -> None:
    assert annual_sharpe([0.0, 0.0, 0.0]) == 0.0


@pytest.mark.parametrize(
    ("dates", "returns", "match"),
    [
        ([date(2020, 1, 1)], [0.1], "two"),
        ([date(2020, 1, 2), date(2020, 1, 1)], [0.1, 0.2], "increasing"),
        ([date(2020, 1, 1), date(2020, 1, 2)], [0.1, math.inf], "finite"),
    ],
)
def test_rejects_bad_series(
    tmp_path: Path, dates: list[date], returns: list[float], match: str
) -> None:
    with pytest.raises(TrialLogError, match=match):
        log_trial(tmp_path / "t.csv", "pairs_distance", PARAMS, dates, returns)


def test_rejects_unknown_family(tmp_path: Path) -> None:
    dates, returns = _series()
    with pytest.raises(TrialLogError, match="strategy_family"):
        log_trial(tmp_path / "t.csv", "Pairs", PARAMS, dates, returns)


def test_rejects_foreign_header(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log.write_text("run_id,timestamp\n", encoding="utf-8")
    dates, returns = _series()
    with pytest.raises(TrialLogError, match="header"):
        log_trial(log, "pairs_distance", PARAMS, dates, returns)
    with pytest.raises(TrialLogError):
        read_trials(log)


def test_read_missing_log(tmp_path: Path) -> None:
    assert read_trials(tmp_path / "none.csv") == []


def test_git_sha_outside_a_repo(tmp_path: Path) -> None:
    assert tracking.git_sha(tmp_path / "trial_log.csv") == "0" * 40 + "-dirty"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_sha_clean_and_dirty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "code.py").write_text("x = 1\n")
    log = repo / "results" / "trial_log.csv"
    log.parent.mkdir()
    log.write_text("header\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    sha = tracking.git_sha(log)
    assert re.match(r"^[0-9a-f]{40}$", sha)
    log.write_text("header\nrow\n")
    assert tracking.git_sha(log) == sha
    (repo / "code.py").write_text("x = 2\n")
    assert tracking.git_sha(log) == f"{sha}-dirty"


def test_output_passes_overfitlab_validator(tmp_path: Path) -> None:
    triallog = pytest.importorskip("overfitlab.triallog")
    log = tmp_path / "results" / "trial_log.csv"
    dates, returns = _series()
    log_trial(log, "pairs_distance", PARAMS, dates, returns)
    log_trial(log, "pairs_distance", PARAMS | {"top_k": 5}, dates, [r * 2 for r in returns])
    log_failed(log, "pairs_distance", PARAMS, "aborted")
    issues = triallog.validate_trial_log(log)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == []
    assert {issue.code for issue in issues} <= {"W_DIRTY", "W_FAILED"}
