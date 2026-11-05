"""Avellaneda-Lee against simple pairs on the same dates, universe and costs (AC-H-03).

Each run is labelled by family, cost setting (A-L's flat 5 bp per side, or this project's
model) and execution lag. Periods: the A-L sample (from the first trading day to 2007),
2008 onward, and the whole run. A-L runs get a Sharpe difference against every pairs run
with the same cost setting and lag, with a paired stationary-bootstrap 95% interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from statarb.evaluation.bootstrap import sharpe, sharpe_difference_ci

TARGETS = Path("tests") / "fixtures" / "al2010_targets.csv"
PERIODS = {
    "al_sample": (date(1990, 1, 1), date(2007, 12, 31)),
    "post_2007": (date(2008, 1, 1), date(2100, 1, 1)),
    "all": (date(1990, 1, 1), date(2100, 1, 1)),
}
AL_FAMILIES = ("al_pca", "al_etf")


@dataclass(frozen=True)
class LabelledRun:
    """A run with its comparison labels."""

    run_id: str
    family: str
    costs: str
    lag: int
    returns: pl.DataFrame
    metrics: dict[str, float]


def read_run(runs: Path, run_id: str) -> LabelledRun:
    """Load a run and label it from its saved config."""
    folder = runs / run_id
    params = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    costs = "flat5" if params["costs"]["half_spread"]["model"] == "flat" else "model"
    return LabelledRun(
        run_id,
        params["run"]["strategy_family"],
        costs,
        int(params["windows"]["execution_lag_days"]),
        pl.read_parquet(folder / "returns.parquet").select("date", "ret").sort("date"),
        json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
    )


def _slice(frame: pl.DataFrame, period: str) -> pl.DataFrame:
    start, end = PERIODS[period]
    return frame.filter(pl.col("date").is_between(start, end))


def summary(runs: list[LabelledRun]) -> pl.DataFrame:
    """Annual return, Sharpe, turnover and holding period by run and period."""
    rows = []
    for run in runs:
        for period in PERIODS:
            part = _slice(run.returns, period)
            if part.height < 2:
                continue
            values = part["ret"].to_numpy()
            rows.append(
                {
                    "family": run.family,
                    "costs": run.costs,
                    "lag": run.lag,
                    "period": period,
                    "first": part["date"][0],
                    "last": part["date"][-1],
                    "annual_return": float(values.mean() * 252),
                    "sharpe": sharpe(values),
                    "turnover_annual": run.metrics.get("turnover_annual", float("nan")),
                    "avg_holding_days": run.metrics.get("avg_holding_days", float("nan")),
                    "run_id": run.run_id,
                }
            )
    return pl.DataFrame(rows)


def by_year(runs: list[LabelledRun]) -> pl.DataFrame:
    """Sharpe and return per calendar year for every run."""
    frames = []
    for run in runs:
        yearly = (
            run.returns.group_by(pl.col("date").dt.year().alias("year"), maintain_order=True)
            .agg(
                (pl.col("ret").mean() * 252).alias("annual_return"),
                (pl.col("ret").mean() / pl.col("ret").std() * np.sqrt(252)).alias("sharpe"),
            )
            .with_columns(
                pl.lit(run.family).alias("family"),
                pl.lit(run.costs).alias("costs"),
                pl.lit(run.lag).alias("lag"),
            )
        )
        frames.append(yearly)
    return (
        pl.concat(frames)
        .select("family", "costs", "lag", "year", "annual_return", "sharpe")
        .sort("family", "costs", "lag", "year")
    )


def differences(runs: list[LabelledRun], seed: int = 20261030, n_boot: int = 2000) -> pl.DataFrame:
    """A-L Sharpe minus pairs Sharpe on common days, per period, with a bootstrap CI."""
    rows = []
    for al in (r for r in runs if r.family in AL_FAMILIES):
        for pairs in (
            r
            for r in runs
            if r.family not in AL_FAMILIES and r.costs == al.costs and r.lag == al.lag
        ):
            joined = al.returns.join(pairs.returns, on="date", suffix="_pairs")
            for period in PERIODS:
                part = _slice(joined, period)
                if part.height < 60:
                    continue
                point, lo, hi = sharpe_difference_ci(
                    part["ret"].to_numpy(), part["ret_pairs"].to_numpy(), n_boot=n_boot, seed=seed
                )
                rows.append(
                    {
                        "al_family": al.family,
                        "pairs_family": pairs.family,
                        "costs": al.costs,
                        "lag": al.lag,
                        "period": period,
                        "days": part.height,
                        "sharpe_diff": point,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return pl.DataFrame(rows)


def against_paper(runs: list[LabelledRun], targets: pl.DataFrame) -> pl.DataFrame:
    """Our Sharpe over each target's period (clipped to our data) beside A-L's value."""
    rows = []
    for target in targets.iter_rows(named=True):
        start = date.fromisoformat(str(target["period_start"]))
        end = date.fromisoformat(str(target["period_end"]))
        for run in (r for r in runs if r.family == target["strategy"] and r.costs == "flat5"):
            part = run.returns.filter(pl.col("date").is_between(start, end))
            if part.height < 60:
                continue
            ours = sharpe(part["ret"].to_numpy())
            rows.append(
                {
                    "strategy": run.family,
                    "lag": run.lag,
                    "paper_period": f"{start.year}-{end.year}",
                    "our_first": part["date"][0],
                    "our_last": part["date"][-1],
                    "paper_sharpe": float(target["sharpe"]),
                    "our_sharpe": ours,
                    "same_sign": bool(np.sign(ours) == np.sign(float(target["sharpe"]))),
                    "source": target["source"],
                }
            )
    return pl.DataFrame(rows)


def figure(by_year_table: pl.DataFrame, path: Path) -> None:
    """Sharpe by year: A-L variants and pairs, at A-L's 5 bp; PNG and SVG at 300 dpi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    subset = by_year_table.filter((pl.col("costs") == "flat5") & (pl.col("lag") == 1))
    for (family,), part in subset.group_by(["family"], maintain_order=True):
        ax.plot(
            part["year"].to_list(),
            part["sharpe"].to_list(),
            marker="o",
            markersize=3,
            label=str(family),
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Sharpe ratio (annualised)")
    ax.set_title(
        "Avellaneda-Lee vs simple pairs, A-L universe (cap >= USD 1bn), 5 bp per side, one-day lag"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    from statarb.reports.figures import save

    save(fig, path)
