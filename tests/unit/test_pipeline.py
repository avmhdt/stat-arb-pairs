"""Ingest, build and DQ end to end on local fakes: no network, no licensed data."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from eqpit.snapshot import open_snapshot
from typer.testing import CliRunner

from statarb import cli
from statarb.data import manifest, pipeline
from statarb.data.providers import eqpit_store
from statarb.settings import Settings

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture
def repo_configs(tmp_path: Path) -> None:
    shutil.copytree(REPO / "configs", tmp_path / "configs")


def _source_snapshots(root: Path) -> None:
    for dataset in eqpit_store.DATASETS:
        with open_snapshot(
            root, "sharadar", dataset, False, source=f"https://x/{dataset}"
        ) as staging:
            (staging / f"{dataset}.parquet").write_bytes(dataset.encode() * 10)


def test_seed_links_and_is_idempotent(settings: Settings, tmp_path: Path) -> None:
    source = tmp_path / "efp" / "raw" / "sharadar"
    _source_snapshots(source)
    first = pipeline.ingest_prices(settings, "sharadar", seed_from=source)
    assert first.downloaded == 0
    assert all(" linked " in line or " copied " in line for line in first.lines)
    entries = manifest.read_manifest(settings.data_dir)
    assert set(entries["dataset"].to_list()) == set(eqpit_store.DATASETS)
    assert entries["path"].str.starts_with("raw/sharadar/").all()
    second = pipeline.ingest_prices(settings, "sharadar", seed_from=source)
    assert all(" kept " in line for line in second.lines)
    assert manifest.verify(settings.data_dir, ["sharadar"]) == []


def test_seed_needs_every_dataset(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        eqpit_store.seed(tmp_path / "empty", tmp_path / "dest")


def test_sharadar_needs_key(settings: Settings) -> None:
    with pytest.raises(pipeline.PipelineError, match="SHARADAR_API_KEY"):
        pipeline.ingest_prices(settings, "sharadar")
    with pytest.raises(pipeline.PipelineError, match="not available"):
        pipeline.ingest_prices(settings, "crsp")
    with pytest.raises(pipeline.PipelineError, match="not available"):
        pipeline.ingest_etf(settings, "tiingo")
    with pytest.raises(pipeline.PipelineError, match="not available"):
        pipeline.ingest_sic(settings, "crsp")


def test_stooq_profile_end_to_end(settings: Settings, repo_configs: None, tmp_path: Path) -> None:
    free = settings.model_copy(update={"statarb_equity_provider": "free"})
    ingest = pipeline.ingest_prices(free, "stooq")
    assert "written" in ingest.lines[0]
    assert "kept" in pipeline.ingest_prices(free, "stooq").lines[0]
    assert "Stooq snapshot" in pipeline.ingest_etf(free, "stooq").lines[0]
    assert "written" in pipeline.ingest_borrow(free, "assumed").lines[0]
    assert "kept" in pipeline.ingest_borrow(free, "assumed").lines[0]
    with pytest.raises(pipeline.PipelineError):
        pipeline.ingest_sic(free.model_copy(update={"sec_user_agent": ""}), "edgar")
    built = pipeline.run_build(free)
    assert any(line.startswith("prices_daily") for line in built.lines)
    processed = free.data_dir / "processed"
    master = pl.read_parquet(processed / "security_master" / "*.parquet")
    assert master["security_id"].str.starts_with("free:").all()
    assert pl.read_parquet(processed / "etf_returns" / "*.parquet")["etf"].unique().to_list() == [
        "SPY"
    ]
    report = tmp_path / "reports" / "dq.md"
    results = pipeline.run_dq(free, "prototype", report)
    assert report.read_text(encoding="utf-8").startswith("# Data-quality report (prototype)")
    statuses = {result.rule: result.status for result in results}
    assert statuses["DQ-13"] == "pass"
    assert statuses["DQ-06"] == "warn"
    again = pipeline.run_build(free, ("borrow",))
    assert again.lines[0].startswith("borrow_rates")


def test_build_needs_raw_data(settings: Settings, repo_configs: None) -> None:
    with pytest.raises(pipeline.PipelineError, match="ingest prices"):
        pipeline.run_build(settings)
    free = settings.model_copy(update={"statarb_equity_provider": "free"})
    with pytest.raises(pipeline.PipelineError, match="ingest prices"):
        pipeline.run_build(free)
    with pytest.raises(pipeline.PipelineError, match="WRDS"):
        pipeline.run_build(settings.model_copy(update={"statarb_equity_provider": "crsp"}))
    with pytest.raises(pipeline.PipelineError, match="STOOQ_DIR"):
        pipeline.ingest_prices(settings.model_copy(update={"stooq_dir": None}), "stooq")
    with pytest.raises(pipeline.PipelineError, match="stooq first"):
        pipeline.ingest_sic(settings, "edgar")


def test_sic_from_sharadar_is_covered(settings: Settings) -> None:
    report = pipeline.ingest_sic(settings, "sharadar")
    assert report.downloaded == 0


def test_cli_commands(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, repo_configs: None
) -> None:
    free = settings.model_copy(update={"statarb_equity_provider": "free"})
    monkeypatch.setattr(cli, "get_settings", lambda: free)
    for args in (
        ["ingest", "prices", "--provider", "stooq", "--start", "1998-01-02"],
        ["ingest", "etf", "--provider", "stooq"],
        ["ingest", "sic", "--provider", "sharadar"],
        ["ingest", "borrow", "--source", "assumed"],
        ["build"],
    ):
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 0, (args, result.output)
        assert "downloaded 0 files" in result.output
    assert runner.invoke(cli.app, ["build", "--tables", "nope"]).exit_code == 2
    assert runner.invoke(cli.app, ["ingest", "prices", "--provider", "crsp"]).exit_code == 2
    checked = runner.invoke(
        cli.app, ["dq", "check", "--profile", "prototype", "--report", "r/dq.md"]
    )
    assert checked.exit_code == 0, checked.output
    assert "hard failures" in checked.output
    failing = runner.invoke(
        cli.app, ["dq", "check", "--profile", "headline", "--report", "r/dq.md"]
    )
    assert failing.exit_code == 1


def test_eqpit_store_is_reused(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    source = tmp_path / "efp" / "raw" / "sharadar"
    _source_snapshots(source)
    pipeline.ingest_prices(settings, "sharadar", seed_from=source)
    calls: list[Path] = []
    monkeypatch.setattr(eqpit_store, "build_store", lambda raw, out: calls.append(out) or {})
    out = pipeline._eqpit_store(settings)
    assert calls == [out]
    snapshots = {
        d: p.name.split("=", 1)[1]
        for d, p in pipeline._latest(pipeline.raw_dir(settings, "sharadar")).items()
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(__import__("json").dumps({"raw_snapshots": snapshots}))
    assert pipeline._eqpit_store(settings) == out
    assert len(calls) == 1
    assert date.today() >= date(2026, 1, 1)


def test_read_store_maps_a_real_eqpit_build(tmp_path: Path) -> None:
    from eqpit.pit.build import build_pit
    from eqpit.providers.synthetic import SyntheticProvider, write_market
    from eqpit.synthetic.generator import SyntheticConfig

    from statarb.data import build, schemas

    raw = tmp_path / "raw"
    write_market(SyntheticConfig(seed=3, n_securities=30, years=2, start_year=2010), raw)
    out = tmp_path / "store"
    build_pit(SyntheticProvider(raw, policy=eqpit_store.POLICY), out, eqpit_store.POLICY)
    store = eqpit_store.read_store(out)
    assert set(eqpit_store.PRICE_COLUMNS) <= set(store.prices.columns)
    assert store.manifest["provider"] == "synthetic"
    tables = build.sharadar_tables(store)
    for name, frame in tables.items():
        schemas.conform(frame, schemas.TABLES[name])
    delisted = tables["security_master"].filter(pl.col("last_date").is_not_null())
    with_ret = tables["prices_daily"].filter(pl.col("ret_delist").is_not_null())
    assert delisted.height == with_ret.height > 0
