from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from statarb.evaluation import comparison
from statarb.evaluation.bootstrap import sharpe, sharpe_difference_ci, stationary_indices

REPO = Path(__file__).resolve().parents[2]


def test_stationary_indices_blocks() -> None:
    rng = np.random.default_rng(0)
    index = stationary_indices(10_000, 20.0, rng)
    assert index.min() >= 0 and index.max() < 10_000
    steps = np.diff(index)
    continued = (steps == 1) | (steps == -9_999)
    assert 0.93 <= continued.mean() <= 0.97  # a new block about every 20 steps
    assert stationary_indices(0, 20.0, rng).size == 0


def test_sharpe_difference_ci() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.001, 0.01, 2000)
    b = rng.normal(0.0, 0.01, 2000)
    point, lo, hi = sharpe_difference_ci(a, b, n_boot=300, seed=3)
    assert lo < point < hi
    assert point == pytest.approx(sharpe(a) - sharpe(b))
    assert sharpe(np.zeros(5)) == 0.0
    with pytest.raises(ValueError):
        sharpe_difference_ci(a, b[:-1])


def _run(runs: Path, run_id: str, family: str, flat: bool, lag: int, drift: float) -> None:
    folder = runs / run_id
    folder.mkdir(parents=True)
    params = {
        "run": {"strategy_family": family},
        "costs": {"half_spread": {"model": "flat" if flat else "quotes_then_corwin_schultz"}},
        "windows": {"execution_lag_days": lag},
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(params))
    (folder / "metrics.json").write_text(
        json.dumps({"turnover_annual": 50.0, "avg_holding_days": 4.0})
    )
    days = [
        d for d in (date(2003, 1, 1) + timedelta(days=i) for i in range(2400)) if d.weekday() < 5
    ]
    rng = np.random.default_rng(len(run_id))
    pl.DataFrame({"date": days, "ret": rng.normal(drift, 0.005, len(days))}).write_parquet(
        folder / "returns.parquet"
    )


def test_comparison_tables(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _run(runs, "a", "al_pca", True, 1, 0.0005)
    _run(runs, "bb", "al_etf", True, 1, 0.0002)
    _run(runs, "ccc", "pairs_distance", True, 1, 0.0001)
    _run(runs, "dddd", "pairs_coint", False, 1, 0.0)
    loaded = [comparison.read_run(runs, rid) for rid in ("a", "bb", "ccc", "dddd")]
    summary = comparison.summary(loaded)
    assert set(summary["period"]) == {"al_sample", "post_2007", "all"}
    assert summary.filter(pl.col("family") == "al_pca")["costs"].unique().to_list() == ["flat5"]
    yearly = comparison.by_year(loaded)
    assert yearly["year"].min() == 2003
    diffs = comparison.differences(loaded, n_boot=100)
    assert set(diffs["pairs_family"]) == {"pairs_distance"}
    assert (diffs["ci_low"] <= diffs["ci_high"]).all()
    paper = comparison.against_paper(
        loaded, pl.read_csv(REPO / "tests" / "fixtures" / "al2010_targets.csv")
    )
    assert {"al_pca", "al_etf"} == set(paper["strategy"])
    assert paper["our_first"].min() >= date(2003, 1, 1)
    comparison.figure(yearly, tmp_path / "f" / "al")
    assert (tmp_path / "f" / "al.png").is_file()
