"""The ``statarb`` command line. Commands arrive milestone by milestone (spec section 10)."""

from __future__ import annotations

from collections import Counter
from typing import Annotated

import typer

from statarb import __version__
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
