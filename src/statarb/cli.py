"""The ``statarb`` command line. Commands arrive milestone by milestone (spec section 10)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import typer

from statarb import __version__
from statarb.data import build as build_tables
from statarb.data import dq, pipeline
from statarb.log import configure_logging
from statarb.settings import get_settings
from statarb.smoke import run_smoke
from statarb.tracking import TrialLogError, read_trials

app = typer.Typer(
    help="Distance, cointegration and Avellaneda-Lee statistical arbitrage research.",
    no_args_is_help=True,
    add_completion=False,
)
trials_app = typer.Typer(help="Inspect results/trial_log.csv.", no_args_is_help=True)
app.add_typer(trials_app, name="trials")
ingest_app = typer.Typer(help="Download or copy raw data into the data root.", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")
dq_app = typer.Typer(help="Data-quality checks DQ-01 to DQ-15.", no_args_is_help=True)
app.add_typer(dq_app, name="dq")


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    """Load settings and set up logging for every command."""
    if version:
        typer.echo(__version__)
        raise typer.Exit
    configure_logging(get_settings().statarb_log_level)


@app.command()
def smoke() -> None:
    """Check settings, Sharadar SEP and SFP access (or the local Stooq snapshot) and SEC."""
    checks = run_smoke(get_settings())
    for check in checks:
        typer.echo(f"{check.status.upper():<5} {check.name:<17} {check.detail}")
    failed = sum(check.status == "fail" for check in checks)
    typer.echo(f"smoke: {len(checks) - failed} of {len(checks)} checks passed")
    if failed:
        raise typer.Exit(code=1)


def _load_trials() -> list[dict[str, str]]:
    try:
        return read_trials(get_settings().results_dir / "trial_log.csv")
    except TrialLogError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@trials_app.command("count")
def trials_count() -> None:
    """Print the number of rows in the trial log, the honest trial count."""
    typer.echo(len(_load_trials()))


@trials_app.command("list")
def trials_list(
    family: Annotated[str | None, typer.Option("--family", help="Only this family.")] = None,
) -> None:
    """Print one line per trial, then the count per strategy family."""
    rows = [row for row in _load_trials() if family is None or row["strategy_family"] == family]
    for row in rows:
        typer.echo(
            f"{row['run_id']}  {row['timestamp']}  {row['strategy_family']:<15} "
            f"{row['start'] or '-':>10} {row['end'] or '-':>10} n={row['n_obs']:>6} "
            f"sharpe={row['sharpe_annual'] or '-'}"
        )
    for name, count in sorted(Counter(row["strategy_family"] for row in rows).items()):
        typer.echo(f"{name}: {count}")


def _step(run: Callable[[], pipeline.StepReport]) -> None:
    try:
        report = run()
    except pipeline.PipelineError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    for line in report.lines:
        typer.echo(line)
    typer.echo(f"downloaded {report.downloaded} files")


@ingest_app.command("prices")
def ingest_prices(
    provider: Annotated[
        Literal["sharadar", "stooq", "crsp", "tiingo"], typer.Option("--provider")
    ] = "sharadar",
    start: Annotated[
        str | None, typer.Option("--start", help="Recorded only: bulk files are full history.")
    ] = None,
    end: Annotated[str | None, typer.Option("--end", help="Recorded only.")] = None,
    seed_from: Annotated[
        Path | None,
        typer.Option(
            "--seed-from", help="Link existing eqpit raw snapshots instead of downloading."
        ),
    ] = None,
) -> None:
    """Equity prices, security master and delistings (FR-05)."""
    for value in (start, end):
        if value is not None:
            date.fromisoformat(value)
    _step(lambda: pipeline.ingest_prices(get_settings(), provider, seed_from))


@ingest_app.command("etf")
def ingest_etf(
    provider: Annotated[Literal["sharadar", "stooq"], typer.Option("--provider")] = "sharadar",
) -> None:
    """Sector and market ETF prices for configs/sic_to_etf.yaml."""
    _step(lambda: pipeline.ingest_etf(get_settings(), provider))


@ingest_app.command("sic")
def ingest_sic(
    provider: Annotated[
        Literal["sharadar", "edgar", "crsp"], typer.Option("--provider")
    ] = "sharadar",
) -> None:
    """Industry history (FR-07)."""
    _step(lambda: pipeline.ingest_sic(get_settings(), provider))


@ingest_app.command("borrow")
def ingest_borrow(
    source: Annotated[Literal["assumed", "ibkr"], typer.Option("--source")] = "assumed",
) -> None:
    """The assumed borrow schedule (D-07)."""
    _step(lambda: pipeline.ingest_borrow(get_settings(), source))


@app.command()
def build(
    tables: Annotated[
        str, typer.Option("--tables", help="Comma list of prices, industry, etf, borrow.")
    ] = ",".join(build_tables.ALL_TABLES),
) -> None:
    """Raw to processed tables for STATARB_EQUITY_PROVIDER."""
    chosen = tuple(name.strip() for name in tables.split(",") if name.strip())
    unknown = sorted(set(chosen) - set(build_tables.ALL_TABLES))
    if unknown:
        typer.echo(f"unknown tables {unknown}", err=True)
        raise typer.Exit(code=2)
    _step(lambda: pipeline.run_build(get_settings(), chosen))


@dq_app.command("check")
def dq_check(
    profile: Annotated[Literal["headline", "prototype"], typer.Option("--profile")] = "headline",
    report: Annotated[Path, typer.Option("--report")] = Path("reports/dq/dq_report.md"),
) -> None:
    """Run DQ-01 to DQ-15, write the report, exit 1 on any hard failure."""
    results = pipeline.run_dq(get_settings(), profile, report)
    for result in results:
        typer.echo(f"{result.status.upper():<5} {result.rule:<8} {result.summary}")
    failures = dq.failed(results)
    typer.echo(f"dq: {len(failures)} hard failures; report {report.as_posix()}")
    if failures:
        raise typer.Exit(code=1)


@app.command()
def select(
    config: Annotated[Path, typer.Option("--config", help="Run config YAML.")],
    window: Annotated[
        list[str] | None, typer.Option("--window", help="Only this window (T, ISO date).")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, max=8)] = 1,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="key=value override.")
    ] = None,
) -> None:
    """Select pairs for every window; writes pairs.parquet only (FR-08 to FR-11)."""
    from statarb.config import load_config
    from statarb.selection.runner import run_selection

    settings = get_settings()
    cfg = load_config(config, overrides or [])
    try:
        result = run_selection(
            cfg, settings.data_dir / "processed", settings.results_dir / "runs", window, workers
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"{result.windows} windows, {result.rows:,} pair rows -> {result.path.as_posix()}")


@app.command()
def report(
    profile: Annotated[Literal["headline", "prototype"], typer.Option("--profile")] = "headline",
    doc: Annotated[Path, typer.Option("--doc")] = Path("docs/multiple_testing.md"),
) -> None:
    """Regenerate the aggregate tables and generated doc blocks from results/ (FR-29, FR-30)."""
    from statarb.reports.tables import write_fdr_outputs

    settings = get_settings()
    try:
        table = write_fdr_outputs(
            settings.results_dir / "runs", settings.results_dir / "tables", doc, profile
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(f"fdr_by_window.csv: {table.height} rows; {doc.as_posix()} updated")
