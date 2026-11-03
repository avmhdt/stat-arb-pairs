"""Trial log in backtest-overfitting-lab's contract v1.0 (FR-27).

Adapted from the contract's reference writer: the ten required columns in their exact
order, then ``x_`` extensions. Every run appends exactly one row, failed runs included,
so the trial count stays honest. M1 ships the writer; ``append_trial`` on a full run
result arrives with the backtest engine in M4.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TRIAL_LOG_COLUMNS: tuple[str, ...] = (
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
EXTRA_COLUMNS: tuple[str, ...] = ("x_periods_per_year", "x_note")
LOG_COLUMNS = TRIAL_LOG_COLUMNS + EXTRA_COLUMNS
FAILED_COLUMNS: tuple[str, ...] = ("run_id", "timestamp", "strategy_family", "error")
STRATEGY_FAMILIES = frozenset(
    {"pairs_distance", "pairs_coint", "al_pca", "al_etf", "intraday_pairs", "crypto_pairs"}
)


class TrialLogError(ValueError):
    """The trial log or a row for it breaks the contract."""


def canonical_json(params: Mapping[str, object]) -> str:
    """The contract's canonical JSON form of ``params``; NaN and infinity are rejected."""
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def git_sha(log_path: Path) -> str:
    """HEAD's full SHA, suffixed ``-dirty`` if tracked files other than the log changed."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=log_path.parent
        ).stdout

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        log = log_path.resolve().relative_to(root.resolve()).as_posix()
        status = git(
            "status", "--porcelain", "--untracked-files=no", "--", ":/", f":(top,exclude){log}"
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


def _check_header(log_path: Path) -> None:
    with log_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if tuple(header) != LOG_COLUMNS:
        raise TrialLogError(f"{log_path} header {header} differs from {list(LOG_COLUMNS)}")


def _append(path: Path, columns: Sequence[str], row: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new and columns == LOG_COLUMNS:
        _check_header(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="", lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(dict(row))


def _base_row(log_path: Path, family: str, params: Mapping[str, object]) -> dict[str, str]:
    if family not in STRATEGY_FAMILIES:
        raise TrialLogError(f"unknown strategy_family {family!r}")
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(log_path),
        "strategy_family": family,
        "params_json": canonical_json(params),
    }


def annual_sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """``mean / std(ddof=1) * sqrt(q)``; 0.0 for a flat series, as the reference writer does."""
    std = statistics.stdev(returns)
    return statistics.fmean(returns) / std * math.sqrt(periods_per_year) if std > 0 else 0.0


def log_trial(
    log_path: Path,
    family: str,
    params: Mapping[str, object],
    dates: Sequence[date],
    returns: Sequence[float],
    periods_per_year: int = 252,
    note: str = "",
) -> str:
    """Write ``runs/<run_id>/returns.parquet`` next to the log, then append its row.

    Parameters
    ----------
    log_path
        ``results/trial_log.csv``.
    family
        One of ``STRATEGY_FAMILIES``.
    params
        The fully resolved config, including ``data_profile``.
    dates, returns
        Strictly increasing dates and finite net daily returns on committed capital.
    periods_per_year
        ``q`` for the Sharpe ratio; 252 for every family in this repo.
    note
        Optional free text for ``x_note``.

    Returns
    -------
    str
        The new ``run_id``.
    """
    if len(dates) != len(returns) or len(returns) < 2:
        raise TrialLogError("need at least two returns, one per date")
    if any(later <= earlier for earlier, later in itertools.pairwise(dates)):
        raise TrialLogError("dates must be strictly increasing")
    if not all(math.isfinite(value) for value in returns):
        raise TrialLogError("returns must be finite")
    row = _base_row(log_path, family, params)
    relative = f"runs/{row['run_id']}/returns.parquet"
    target = log_path.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"date": pa.array(list(dates), pa.date32()), "ret": pa.array(list(returns), pa.float64())}
    )
    pq.write_table(table, target)
    row |= {
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "n_obs": str(len(returns)),
        "sharpe_annual": f"{annual_sharpe(returns, periods_per_year):.10g}",
        "returns_path": relative,
        "x_periods_per_year": str(periods_per_year),
        "x_note": note,
    }
    _append(log_path, LOG_COLUMNS, row)
    return row["run_id"]


def log_failed(
    log_path: Path,
    family: str,
    params: Mapping[str, object],
    error: str,
    periods_per_year: int = 252,
) -> str:
    """Append an aborted run: ``n_obs = 0``, no Sharpe, no series; details go to failed_runs.csv."""
    row = _base_row(log_path, family, params) | {
        "n_obs": "0",
        "x_periods_per_year": str(periods_per_year),
        "x_note": "failed",
    }
    _append(log_path, LOG_COLUMNS, row)
    _append(
        log_path.parent / "failed_runs.csv",
        FAILED_COLUMNS,
        {key: row.get(key, "") for key in FAILED_COLUMNS[:3]} | {"error": error},
    )
    return row["run_id"]


def read_trials(log_path: Path) -> list[dict[str, str]]:
    """All rows of the trial log, or an empty list if it does not exist yet."""
    if not log_path.exists():
        return []
    _check_header(log_path)
    with log_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
