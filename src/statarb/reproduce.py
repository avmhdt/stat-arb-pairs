"""``statarb reproduce``: data, every study run and the report in one command (FR-29).

Steps: ingest for the profile (existing raw snapshots are kept, so a second run downloads
nothing), build when the processed tables do not match the raw manifest, DQ (any hard
failure stops), every run in ``configs/study.yaml``, then ``statarb report``.

The prototype profile lives under ``<data root>/prototype`` so it never overwrites the
headline build; both write their runs to the same results folder, labelled by profile.
The IBKR borrow collector is forward-only and is not part of reproduction.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from statarb.data import build as build_tables
from statarb.data import dq, manifest, pipeline, schemas
from statarb.settings import Settings

PROVIDERS = {"headline": "sharadar", "prototype": "free"}
STUDIES = {
    "headline": Path("configs") / "study.yaml",
    "prototype": Path("configs") / "study_prototype.yaml",
}
TIMINGS = Path("results") / "run_times.csv"


class ReproduceError(RuntimeError):
    """A step failed; the message names it."""


def profile_settings(settings: Settings, profile: str) -> Settings:
    """Settings for ``profile``: its equity provider and, for the prototype, its own root."""
    if profile not in PROVIDERS:
        raise ReproduceError(f"unknown profile {profile!r}")
    update: dict[str, object] = {"statarb_equity_provider": PROVIDERS[profile]}
    if profile == "prototype":
        update["statarb_data_dir"] = str(settings.data_dir / "prototype")
    return settings.model_copy(update=update)


def ingest(settings: Settings, profile: str) -> list[str]:
    """Every ingest step of the profile; cached raw files are kept."""
    steps: list[tuple[str, Callable[[], pipeline.StepReport]]]
    if profile == "headline":
        steps = [
            ("prices", lambda: pipeline.ingest_prices(settings, "sharadar")),
            ("etf", lambda: pipeline.ingest_etf(settings, "sharadar")),
            ("sic", lambda: pipeline.ingest_sic(settings, "sharadar")),
        ]
    else:
        steps = [
            ("prices", lambda: pipeline.ingest_prices(settings, "stooq")),
            ("etf", lambda: pipeline.ingest_etf(settings, "stooq")),
            ("sic", lambda: pipeline.ingest_sic(settings, "edgar")),
        ]
    steps += [
        ("borrow", lambda: pipeline.ingest_borrow(settings, "assumed")),
        ("french", lambda: pipeline.ingest_french(settings)),
    ]
    lines = []
    for name, step in steps:
        try:
            report = step()
        except pipeline.PipelineError as error:
            raise ReproduceError(f"ingest {name}: {error}") from error
        lines.append(f"ingest {name}: {report.downloaded} files downloaded")
    return lines


def needs_build(settings: Settings) -> bool:
    """True when the processed tables are missing or were built from other raw files."""
    processed = settings.data_dir / "processed"
    info = schemas.read_build_info(processed)
    provider = settings.statarb_equity_provider
    current = manifest.manifest_hash(settings.data_dir, pipeline.SOURCES[provider])
    tables_present = all((processed / name).exists() for name in schemas.TABLES)
    return not tables_present or info.get("manifest_hash") != current


def reproduce_all(
    settings: Settings,
    profile: str,
    study_path: Path,
    workers: int,
    reuse: bool,
    echo: Callable[[str], None] = print,
) -> list[str]:
    """Run every step; returns the summary lines. Timings go to ``results/run_times.csv``."""
    from statarb.reports.study import build_all, paths_for
    from statarb.study import StudyError, find_run, load_study
    from statarb.trading.backtest import run_backtest

    local = profile_settings(settings, profile)
    study = load_study(study_path)
    lines = ingest(local, profile)
    for line in lines:
        echo(line)
    if needs_build(local):
        echo("build: processed tables do not match the raw manifest; rebuilding")
        try:
            pipeline.run_build(local, build_tables.ALL_TABLES)
        except pipeline.PipelineError as error:
            raise ReproduceError(f"build: {error}") from error
    else:
        echo("build: processed tables match the raw manifest; kept")
    report_name = "dq_report.md" if profile == "headline" else f"dq_report_{profile}.md"
    results = pipeline.run_dq(local, profile, Path("reports") / "dq" / report_name)
    failures = dq.failed(results)
    if failures:
        raise ReproduceError(f"dq: {len(failures)} hard failures: {[r.rule for r in failures]}")
    echo(f"dq: {len(results)} rules, no hard failures")
    processed = local.data_dir / "processed"
    manifest_hash = str(schemas.read_build_info(processed).get("manifest_hash", ""))
    log_path = local.results_dir / "trial_log.csv"
    timings = []
    for name, spec in study.runs.items():
        cfg = spec.resolve()
        if reuse and find_run(log_path, local.results_dir / "runs", cfg, profile, manifest_hash):
            echo(f"run {name}: complete run found; reused")
            continue
        start = time.perf_counter()
        result = run_backtest(
            cfg, workers, processed=processed, results_dir=local.results_dir, note=f"study {name}"
        )
        seconds = time.perf_counter() - start
        timings.append((name, result.run_id, workers, round(seconds, 1)))
        echo(f"run {name}: {result.run_id} in {seconds / 60:.1f} min")
    if timings:
        _write_timings(timings, profile)
    paths = paths_for(local.results_dir, local.data_dir, profile)
    try:
        ids = build_all(study, paths, profile)
    except StudyError as error:
        raise ReproduceError(f"report: {error}") from error
    return [f"report: {len(ids)} study runs resolved; tables and figures rebuilt"]


def _write_timings(rows: list[tuple[str, str, int, float]], profile: str) -> None:
    """Append ``name, run_id, workers, seconds, profile`` rows (AC-W-13 run times)."""
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    new = not TIMINGS.exists()
    with TIMINGS.open("a", encoding="utf-8", newline="\n") as handle:
        if new:
            handle.write("name,run_id,workers,seconds,data_profile\n")
        for name, run_id, workers, seconds in rows:
            handle.write(f"{name},{run_id},{workers},{seconds},{profile}\n")
