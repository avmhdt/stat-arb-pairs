from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from statarb.data import dq, manifest, schemas
from statarb.industry import SicEtfMap, SicRange, load_sic_etf_map
from unit.test_data_layer import write_processed

REPO = Path(__file__).resolve().parents[2]
SYNTH_MAP = SicEtfMap((SicRange(2800, 2899, "XLB"),), {"XLB": "SPY"}, "SPY")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    write_processed(tmp_path / "processed")
    raw = tmp_path / "raw" / "borrow"
    raw.mkdir(parents=True)
    (raw / "assumed_schedule.yaml").write_text("x")
    manifest.record(
        tmp_path,
        [manifest.entry_for(tmp_path, raw / "assumed_schedule.yaml", "borrow", "assumed", "u")],
    )
    return tmp_path


def _by_rule(results: list[dq.DqResult]) -> dict[str, dq.DqResult]:
    return {result.rule: result for result in results}


def test_synthetic_prototype_passes(data_dir: Path) -> None:
    results = _by_rule(dq.run_checks(data_dir, "prototype", SYNTH_MAP, ["borrow"]))
    assert not dq.failed(list(results.values())), [
        r for r in results.values() if r.status == "fail"
    ]
    assert results["DQ-03"].summary.startswith(
        "3 delisted securities, 3 with a delisting return (2 imputed: 1 performance"
    )
    assert results["DQ-04"].status == "pass"
    assert results["DQ-08"].status == "pass"
    assert results["DQ-09"].status == "pass"
    survivors = results["DQ-15"].by_year
    assert survivors[2015]["later_delisted"] == 3
    assert results["DQ-13"].status == "pass"
    assert {"DQ-01", "DQ-02", "DQ-11", "DQ-14"} <= set(results)
    report = dq.render_report(list(results.values()), "prototype")
    assert "| DQ-15 |" in report and "## DQ-15 by year" in report
    assert "syn:" not in report


def test_profile_mismatch_and_missing_build(data_dir: Path, tmp_path: Path) -> None:
    results = _by_rule(dq.run_checks(data_dir, "headline", SYNTH_MAP, ["borrow"]))
    assert results["profile"].status == "fail"
    assert results["DQ-07"].status == "pass"
    empty = dq.run_checks(tmp_path / "nowhere", "headline", SYNTH_MAP, ["borrow"])
    assert empty[0].rule == "build" and empty[0].status == "fail"


def test_failures_are_detected(data_dir: Path) -> None:
    processed = data_dir / "processed"
    prices = pl.read_parquet(processed / "prices_daily" / "**" / "*.parquet")
    broken = prices.with_columns(
        pl.when(pl.col("date") == prices["date"].max())
        .then(pl.col("tr_index") * 2)
        .otherwise(pl.col("tr_index"))
        .alias("tr_index"),
        pl.when(pl.col("security_id") == "syn:000")
        .then(pl.lit(0.9))
        .otherwise(pl.col("ret_total"))
        .alias("ret_total"),
        pl.lit(None, pl.Float64).alias("ret_delist"),
    )
    schemas.write_table(broken, processed, "prices_daily")
    industry = pl.read_parquet(processed / "industry_history" / "*.parquet").with_columns(
        pl.lit(False).alias("is_point_in_time")
    )
    schemas.write_table(industry, processed, "industry_history")
    history = pl.read_parquet(processed / "ticker_history" / "*.parquet").with_columns(
        pl.lit("BARE").alias("security_id")
    )
    schemas.write_table(history, processed, "ticker_history")
    (data_dir / "raw" / "borrow" / "assumed_schedule.yaml").write_text("changed")
    sic_map = load_sic_etf_map(REPO / "configs" / "sic_to_etf.yaml")
    results = _by_rule(dq.run_checks(data_dir, "headline", sic_map, ["borrow"]))
    for rule in ("DQ-03", "DQ-04", "DQ-06", "DQ-07", "DQ-09", "DQ-13"):
        assert results[rule].status == "fail", rule
    assert results["DQ-10"].status == "warn"
    assert (processed / "dq" / "extreme_returns.parquet").exists()
    prototype = _by_rule(dq.run_checks(data_dir, "prototype", sic_map, ["borrow"]))
    for rule in ("DQ-03", "DQ-06", "DQ-07"):
        assert prototype[rule].status == "warn", rule


def test_survivorship_fails_without_delistings(data_dir: Path) -> None:
    processed = data_dir / "processed"
    master = pl.read_parquet(processed / "security_master" / "*.parquet").with_columns(
        pl.lit(None, pl.Date).alias("last_date")
    )
    schemas.write_table(master, processed, "security_master")
    result = dq.check_survivorship(processed, "headline")
    assert result.status == "fail"
    assert dq.check_survivorship(processed, "prototype").status == "warn"
