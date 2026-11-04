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
evaluate_app = typer.Typer(help="Evaluations over finished runs.", no_args_is_help=True)
app.add_typer(evaluate_app, name="evaluate")


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


@ingest_app.command("french")
def ingest_french() -> None:
    """Ken French daily short-term reversal and market factors (FR-25)."""
    _step(lambda: pipeline.ingest_french(get_settings()))


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
    study: Annotated[Path, typer.Option("--study", help="Study manifest.")] = Path(
        "configs/study.yaml"
    ),
    doc: Annotated[Path, typer.Option("--doc")] = Path("docs/multiple_testing.md"),
) -> None:
    """Rebuild every table and figure the memo cites from the study's runs (FR-29, FR-30)."""
    from statarb.reports.study import Paths, build_all
    from statarb.study import StudyError, load_study

    settings = get_settings()
    paths = Paths(
        settings.results_dir,
        settings.data_dir / "processed",
        settings.data_dir / "raw",
        doc=doc,
    )
    try:
        ids = build_all(load_study(study), paths, profile)
    except (StudyError, FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(
        f"{len(ids)} runs resolved ({paths.tables.as_posix()}/study_runs.csv); tables in "
        f"{paths.tables.as_posix()}, figures in {paths.figures.as_posix()}"
    )


@app.command()
def reproduce(
    profile: Annotated[Literal["headline", "prototype"], typer.Option("--profile")] = "headline",
    study: Annotated[Path, typer.Option("--study", help="Study manifest.")] = Path(
        "configs/study.yaml"
    ),
    workers: Annotated[int, typer.Option("--workers", min=1, max=8)] = 3,
    reuse: Annotated[
        bool, typer.Option("--reuse", help="Skip runs already complete for this data.")
    ] = False,
) -> None:
    """Ingest (cached files are kept), build, DQ, every study run, then the report (FR-29)."""
    from statarb.reproduce import ReproduceError, reproduce_all

    try:
        lines = reproduce_all(get_settings(), profile, study, workers, reuse, echo=typer.echo)
    except ReproduceError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    for line in lines:
        typer.echo(line)


def _run_one(config: Path, overrides: list[str], workers: int, note: str = "") -> None:
    from statarb.config import load_config
    from statarb.trading.backtest import run_backtest

    settings = get_settings()
    cfg = load_config(config, overrides)
    result = run_backtest(
        cfg,
        workers,
        processed=settings.data_dir / "processed",
        results_dir=settings.results_dir,
        note=note,
    )
    m = result.metrics
    typer.echo(
        f"{result.run_id} {cfg.run.strategy_family} {result.data_profile} windows={result.windows} "
        f"days={result.returns.height} sharpe={m.get('sharpe_annual', float('nan')):.3f} "
        f"mean={m.get('mean_annual', float('nan')):.4f} "
        f"pairs/window={m['pairs_traded_per_window']:.2f}"
    )


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", help="Run config YAML.")],
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="key=value override.")
    ] = None,
    workers: Annotated[int, typer.Option("--workers", min=1, max=8)] = 1,
) -> None:
    """Select, trade and evaluate every window; append one trial-log row (FR-27)."""
    _run_one(config, overrides or [], workers)


@app.command()
def sweep(
    config: Annotated[Path, typer.Option("--config", help="Run config YAML.")],
    grid: Annotated[Path, typer.Option("--grid", help="YAML: key -> list of values.")],
    workers: Annotated[int, typer.Option("--workers", min=1, max=8)] = 1,
) -> None:
    """One run, and one trial-log row, per grid member.

    The grid maps keys to lists of values (their product), and may hold ``members``, a list
    of override sets, each crossed with that product.
    """
    from statarb.config import grid_members

    spec = yaml_load(grid)
    for overrides in grid_members(spec):
        note = "sweep " + grid.as_posix() + ": " + ", ".join(overrides)
        _run_one(config, overrides, workers, note)


def yaml_load(path: Path) -> dict[str, object]:
    """A YAML mapping from ``path``."""
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        typer.echo(f"{path} is not a mapping", err=True)
        raise typer.Exit(code=2)
    return loaded


@evaluate_app.command("replication")
def evaluate_replication(
    runs: Annotated[str, typer.Option("--runs", help="Comma-separated run ids of the GGR runs.")],
    out: Annotated[Path, typer.Option("--out")] = Path("results/tables/ggr_replication.csv"),
    figure_path: Annotated[Path, typer.Option("--figure")] = Path(
        "reports/figures/ggr_replication"
    ),
) -> None:
    """GGR (2006) replication table, bounce table and figure (FR-20, AC-H-01)."""
    from statarb.reports.study import Paths, replication_tables

    settings = get_settings()
    paths = Paths(settings.results_dir, settings.data_dir / "processed", settings.data_dir / "raw")
    ids = [rid.strip() for rid in runs.split(",") if rid.strip()]
    result = replication_tables(paths, ids, paths.processed, out, figure_path)
    for key, value in result.items():
        typer.echo(f"{key}: {value}")


@evaluate_app.command("al-comparison")
def evaluate_al_comparison(
    runs: Annotated[
        str, typer.Option("--runs", help="Comma-separated run ids: A-L and pairs runs.")
    ],
) -> None:
    """Avellaneda-Lee against simple pairs (FR-18, FR-19, AC-H-03)."""
    from statarb.reports.study import Paths, al_tables

    settings = get_settings()
    paths = Paths(settings.results_dir, settings.data_dir / "processed", settings.data_dir / "raw")
    al_tables(paths, [rid.strip() for rid in runs.split(",") if rid.strip()])
    typer.echo(f"wrote al_vs_pairs*.csv and al_vs_paper.csv to {paths.tables.as_posix()}")


def _specs(runs: str) -> list[str]:
    return [item.strip() for item in runs.split(",") if item.strip()]


RunsOption = Annotated[
    str, typer.Option("--runs", help="Comma-separated run ids, optionally label=run_id.")
]


@evaluate_app.command("decay")
def evaluate_decay(runs: RunsOption) -> None:
    """Sub-period returns, rolling 36-month Sharpe and the time trend (FR-21, AC-H-02)."""
    from statarb.reports import economics

    table = economics.decay(get_settings().results_dir / "runs", _specs(runs))
    typer.echo(f"decay_by_period.csv: {table.height} rows")


@evaluate_app.command("persistence")
def evaluate_persistence(runs: RunsOption) -> None:
    """Out-of-sample cointegration of selected pairs against a matched baseline (FR-22)."""
    from statarb.reports import economics

    settings = get_settings()
    table = economics.persistence_tables(
        settings.results_dir / "runs", settings.data_dir / "processed", _specs(runs)
    )
    typer.echo(f"oos_persistence.csv: {table.height} rows")


@evaluate_app.command("breakeven")
def evaluate_breakeven(runs: RunsOption) -> None:
    """Break-even cost per side with bootstrap intervals and borrow sensitivity (FR-23)."""
    from statarb.reports import economics

    table = economics.break_even_tables(get_settings().results_dir / "runs", _specs(runs))
    typer.echo(f"break_even.csv: {table.height} rows")


@evaluate_app.command("capacity")
def evaluate_capacity(runs: RunsOption) -> None:
    """Net Sharpe and return against AUM, last five years (FR-24)."""
    from statarb.reports import economics

    table = economics.capacity_tables(get_settings().results_dir / "runs", _specs(runs))
    typer.echo(f"capacity.csv: {table.height} rows")


@evaluate_app.command("crowding")
def evaluate_crowding(runs: RunsOption) -> None:
    """Reversal-factor exposure, event drawdowns, overlap, concentration, trends (FR-25)."""
    from statarb.reports import economics

    settings = get_settings()
    factors = settings.data_dir / "raw" / "french" / "factors.parquet"
    table = economics.crowding_tables(settings.results_dir / "runs", factors, _specs(runs))
    typer.echo(f"crowding.csv: {table.height} rows")


@evaluate_app.command("policies")
def evaluate_policies(runs: RunsOption) -> None:
    """Cointegration-break policies on the same windows (FR-26, AC-H-08)."""
    from statarb.reports import economics

    table = economics.policy_table(get_settings().results_dir / "runs", _specs(runs))
    typer.echo(f"break_policies.csv: {table.height} rows")


@evaluate_app.command("sizing")
def evaluate_sizing(runs: RunsOption) -> None:
    """Sizing schemes by half-life bucket (FR-31, AC-H-09)."""
    from statarb.reports import economics

    table = economics.sizing_table(get_settings().results_dir / "runs", _specs(runs))
    typer.echo(f"sizing_half_life.csv: {table.height} rows")


@evaluate_app.command("deflated-sharpe")
def evaluate_deflated_sharpe(
    headline: Annotated[str, typer.Option("--headline", help="Comma-separated family=run_id.")],
) -> None:
    """PSR and DSR per family from the trial log (FR-28, AC-H-10)."""
    from statarb.reports import economics

    chosen = dict(item.split("=", 1) for item in _specs(headline))
    table = economics.dsr_table(get_settings().results_dir / "trial_log.csv", chosen)
    typer.echo(f"deflated_sharpe.csv: {table.height} rows")


@evaluate_app.command("survivorship")
def evaluate_survivorship(
    pairs: Annotated[str, typer.Option("--pairs", help="Comma-separated full_run:survivors_run.")],
) -> None:
    """Full against survivors-only universe returns (AC-W-09)."""
    from statarb.reports import economics

    couples = [tuple(item.split(":", 1)) for item in _specs(pairs)]
    table = economics.survivorship_table(get_settings().results_dir / "runs", couples)  # type: ignore[arg-type]
    typer.echo(f"survivorship_bias.csv: {table.height} rows")


@evaluate_app.command("borrow")
def evaluate_borrow() -> None:
    """The latest IBKR snapshot's fees by size bucket against the assumed schedule (FR-34)."""
    from statarb.data.build import borrow_rates
    from statarb.reports import economics

    settings = get_settings()
    snapshots = sorted((settings.data_dir / "raw" / "ibkr").glob("snapshot=*"))
    if not snapshots:
        typer.echo("no IBKR snapshot; run statarb ingest borrow --source ibkr", err=True)
        raise typer.Exit(code=2)
    table = economics.borrow_table(
        snapshots[-1],
        settings.data_dir / "processed",
        borrow_rates(Path("configs/borrow_schedule.yaml")),
    )
    typer.echo(f"borrow_assumed_vs_ibkr.csv: {table.height} rows")
