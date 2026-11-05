"""AC-W-11: determinism and one-command reproduction.

Fast tests (CI): a backtest on the synthetic market gives a bit-identical ``returns.parquet``
when run twice and with 1 or 2 workers, and its returns match a committed golden hash; the
study manifest resolves names to complete runs. The slow test runs ``statarb reproduce
--profile prototype`` on a small study and needs the Stooq snapshot (``STOOQ_DIR``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from statarb.config import RunConfig
from statarb.settings import Settings
from statarb.study import StudyError, find_run, load_study, normalized, resolve_all
from statarb.trading.backtest import run_backtest
from unit.test_data_layer import write_processed

REPO = Path(__file__).resolve().parents[2]
GOLDEN = "c2c30413d0558aeb74922bcf"  # first 24 hex of rounded_digest() on the synthetic run


def _cfg() -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"name": "det", "strategy_family": "pairs_coint", "data_profile": "prototype"},
            "data": {"provider": "free", "start": "2015-01-01", "end": "2017-12-31"},
            "universe": {"top_n": None, "min_price_usd": None},
            "selection": {"top_k": 5},
            "model": {"hedge": "static"},
            "windows": {"step_months": 3},
        }
    )


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    folder = tmp_path / "processed"
    write_processed(folder, "prototype")
    return folder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_digest(returns: pl.DataFrame) -> str:
    """Digest of every column rounded to 10 decimals: stable across BLAS and platforms."""
    digest = hashlib.sha256()
    for name in sorted(returns.columns):
        values = returns[name]
        if values.dtype.is_float():
            digest.update(np.round(values.to_numpy(), 10).tobytes())
        else:
            digest.update(str(values.to_list()).encode())
    return digest.hexdigest()


def test_returns_are_bit_identical_across_runs_and_workers(
    processed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _cfg()
    one = run_backtest(cfg, 1, processed=processed, results_dir=tmp_path / "a")
    again = run_backtest(cfg, 1, processed=processed, results_dir=tmp_path / "b")
    many = run_backtest(cfg, 2, processed=processed, results_dir=tmp_path / "c")
    shas = {_sha(r.path / "returns.parquet") for r in (one, again, many)}  # type: ignore[operator]
    assert len(shas) == 1
    assert one.trades.height > 0
    assert rounded_digest(one.returns)[:24] == GOLDEN


def _study(tmp_path: Path) -> Path:
    path = tmp_path / "study.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runs": {
                    "base": {"config": "configs/coint_eg_kalman.yaml", "set": []},
                    "hold": {
                        "config": "configs/coint_eg_kalman.yaml",
                        "set": ["rules.break_policy=hold"],
                    },
                },
                "tables": {"policies": ["base", "hold"], "nested": {"a": ["base"]}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_study_resolves_names_to_complete_runs(
    processed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    study = load_study(_study(tmp_path))
    assert study.names("policies") == ["base", "hold"]
    assert study.names("nested") == ["base"]
    log = tmp_path / "results" / "trial_log.csv"
    runs = tmp_path / "results" / "runs"
    with pytest.raises(StudyError, match="no complete prototype run"):
        resolve_all(study, log, runs, "prototype", "abc")
    cfg = _cfg()
    assert normalized(json.loads(cfg.model_dump_json())) == json.loads(cfg.model_dump_json())
    assert find_run(log, runs, cfg, "prototype", "abc") is None
    result = run_backtest(cfg, 1, processed=processed, results_dir=tmp_path / "results")
    assert find_run(log, runs, cfg, "prototype", "abc") == result.run_id
    assert find_run(log, runs, cfg, "headline", "abc") is None
    assert find_run(log, runs, cfg, "prototype", "other") is None
    other = cfg.model_copy(update={"selection": cfg.selection.model_copy(update={"top_k": 4})})
    assert find_run(log, runs, other, "prototype", "abc") is None
    (runs / result.run_id / "metrics.json").unlink()
    assert find_run(log, runs, cfg, "prototype", "abc") is None


def test_unknown_names_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"runs": {}, "tables": {"x": ["missing"]}}), encoding="utf-8")
    with pytest.raises(StudyError, match="undefined"):
        load_study(path)


def test_the_committed_study_is_valid() -> None:
    study = load_study(REPO / "configs" / "study.yaml")
    assert len(study.runs) >= 28
    for spec in study.runs.values():
        assert (REPO / spec.config).exists(), spec.config


@pytest.mark.slow
@pytest.mark.data
def test_reproduce_prototype_twice_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the local Stooq build: a small study reproduced twice, 2 then 1 workers."""
    from statarb import reproduce

    monkeypatch.setattr(reproduce, "TIMINGS", tmp_path / "run_times.csv")
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    # The autouse fixture hides the developer's .env; this data test reads it on purpose.
    settings = Settings(_env_file=REPO / ".env")
    if not settings.data_dir.is_absolute():
        settings = settings.model_copy(update={"statarb_data_dir": str(REPO / settings.data_dir)})
    if settings.stooq_dir is None:
        pytest.skip("needs the Stooq snapshot (STOOQ_DIR)")
    local = reproduce.profile_settings(settings, "prototype")
    if not (local.data_dir / "processed" / "build_info.json").exists():
        pytest.skip("run `statarb reproduce --profile prototype` once first")
    sets = [
        "run.data_profile=prototype",
        "data.provider=free",
        "data.start=2022-01-01",
        "data.end=2024-12-31",
        "universe.rank_by=dollar_volume",
    ]
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "runs": {"small": {"config": "configs/coint_eg_kalman.yaml", "set": sets}},
                "tables": {},
            }
        ),
        encoding="utf-8",
    )
    results = settings.model_copy(update={"statarb_results_dir": tmp_path / "results"})
    shas = []
    for workers in (2, 1):
        reproduce.reproduce_all(results, "prototype", study, workers, False, echo=lambda s: None)
        run_id = (tmp_path / "results" / "trial_log.csv").read_text().splitlines()[-1].split(",")[0]
        shas.append(_sha(tmp_path / "results" / "runs" / run_id / "returns.parquet"))
    assert shas[0] == shas[1]


def test_reproduce_orchestrates_every_step(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from statarb import reproduce
    from statarb.data import pipeline
    from statarb.reports import study as report_study

    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    local = reproduce.profile_settings(settings, "prototype")
    assert local.data_dir == settings.data_dir / "prototype"
    assert local.statarb_equity_provider == "free"
    write_processed(local.data_dir / "processed", "prototype")
    calls: list[str] = []

    def step(name: str) -> object:
        def run(*args: object, **kwargs: object) -> pipeline.StepReport:
            calls.append(name)
            return pipeline.StepReport([], 0)

        return run

    for name in ("ingest_prices", "ingest_etf", "ingest_sic", "ingest_borrow", "ingest_french"):
        monkeypatch.setattr(pipeline, name, step(name))
    monkeypatch.setattr(reproduce, "needs_build", lambda s: False)
    monkeypatch.setattr(pipeline, "run_dq", lambda *a, **k: [])
    monkeypatch.setattr(report_study, "build_all", lambda study, paths, profile: {"x": "y"})
    sets = [
        "run.data_profile=prototype",
        "data.provider=free",
        "data.start=2015-01-01",
        "data.end=2017-12-31",
        "universe.top_n=null",
        "universe.min_price_usd=null",
        "selection.top_k=5",
        "model.hedge=static",
        "windows.step_months=3",
    ]
    path = tmp_path / "study.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runs": {"small": {"config": "configs/coint_eg_kalman.yaml", "set": sets}},
                "tables": {},
            }
        ),
        encoding="utf-8",
    )
    echoed: list[str] = []
    lines = reproduce.reproduce_all(settings, "prototype", path, 1, False, echo=echoed.append)
    assert calls == ["ingest_prices", "ingest_etf", "ingest_sic", "ingest_borrow", "ingest_french"]
    assert lines == ["report: 1 study runs resolved; tables and figures rebuilt"]
    assert any(line.startswith("run small: ") for line in echoed)
    timings = pl.read_csv(tmp_path / "results" / "run_times.csv")
    assert timings["name"].to_list() == ["small"]
    echoed.clear()
    reproduce.reproduce_all(settings, "prototype", path, 1, True, echo=echoed.append)
    assert "run small: complete run found; reused" in echoed
    with pytest.raises(reproduce.ReproduceError, match="unknown profile"):
        reproduce.profile_settings(settings, "other")


def test_reproduce_stops_on_dq_failure(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from statarb import reproduce
    from statarb.data import dq, pipeline

    monkeypatch.chdir(tmp_path)
    for name in ("ingest_prices", "ingest_etf", "ingest_sic", "ingest_borrow", "ingest_french"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: pipeline.StepReport([], 0))
    monkeypatch.setattr(reproduce, "needs_build", lambda s: False)
    bad = dq.DqResult("DQ-01", "fail", "broken")
    monkeypatch.setattr(pipeline, "run_dq", lambda *a, **k: [bad])
    monkeypatch.setattr(dq, "failed", lambda results: list(results))
    with pytest.raises(reproduce.ReproduceError, match="dq: 1 hard failures"):
        reproduce.reproduce_all(
            settings, "headline", REPO / "configs" / "study.yaml", 1, False, echo=lambda s: None
        )


def test_needs_build(settings: Settings) -> None:
    from statarb import reproduce

    local = reproduce.profile_settings(settings, "headline")
    assert reproduce.needs_build(local)
    write_processed(local.data_dir / "processed", "headline")
    assert reproduce.needs_build(local)  # the fixture's manifest hash is not the raw one


def test_study_figures_are_deterministic(tmp_path: Path) -> None:
    from statarb.reports.study import example_pair_figure, protocol_figure

    for draw in (protocol_figure, example_pair_figure):
        draw(tmp_path / "a")
        first = (tmp_path / "a.svg").read_bytes(), (tmp_path / "a.png").read_bytes()
        draw(tmp_path / "a")
        assert ((tmp_path / "a.svg").read_bytes(), (tmp_path / "a.png").read_bytes()) == first
