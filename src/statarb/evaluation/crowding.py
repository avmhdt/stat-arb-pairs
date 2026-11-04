"""Crowding (FR-25, AC-H-07).

- correlation and beta of monthly strategy returns to Ken French's short-term reversal
  factor and the market, over the whole run and rolling 36 months;
- drawdowns over 6-10 August 2007 and March 2020 in units of the strategy's trailing daily
  volatility (60 sessions before the event);
- overlap (Jaccard) of the names traded by each method, by year;
- share of traded gross notional in the 10 most-traded names;
- trends in the selected pairs' average half-life and in entries per pair (divergence
  frequency), by year.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from statarb.evaluation.economics import Run
from statarb.evaluation.metrics import monthly

EVENTS = {
    "aug_2007": (date(2007, 8, 6), date(2007, 8, 10)),
    "mar_2020": (date(2020, 3, 1), date(2020, 3, 31)),
}
TRAILING_VOL_DAYS = 60


def factor_exposure(run: Run, factors: pl.DataFrame) -> tuple[dict[str, float], pl.DataFrame]:
    """Full-sample and rolling 36-month correlation and beta to reversal and market factors.

    ``factors`` holds daily ``date, st_rev, mkt_rf`` in decimals.
    """
    strat = monthly(run.returns.select("date", "ret"))
    fac = factors.sort("date")
    monthly_f = fac.group_by(
        pl.col("date").dt.truncate("1mo").alias("month"), maintain_order=True
    ).agg(
        ((1 + pl.col("st_rev")).product() - 1).alias("st_rev"),
        ((1 + pl.col("mkt_rf")).product() - 1).alias("mkt_rf"),
    )
    joined = strat.join(monthly_f, on="month").sort("month")
    out: dict[str, float] = {"months": float(joined.height)}
    for name in ("st_rev", "mkt_rf"):
        x, y = joined[name].to_numpy(), joined["ret"].to_numpy()
        out[f"corr_{name}"] = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else math.nan
        out[f"beta_{name}"] = (
            float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)) if x.size > 2 else math.nan
        )
    rolling = joined.with_columns(
        pl.rolling_corr("ret", "st_rev", window_size=36).alias("corr_st_rev_36m"),
        (pl.rolling_cov("ret", "st_rev", window_size=36) / pl.col("st_rev").rolling_var(36)).alias(
            "beta_st_rev_36m"
        ),
        pl.lit(run.label).alias("run"),
    )
    return out, rolling.select("run", "month", "corr_st_rev_36m", "beta_st_rev_36m")


def event_drawdowns(run: Run) -> dict[str, float]:
    """Cumulative return over each event window, in units of trailing daily volatility."""
    ret = run.returns.select("date", "ret").sort("date")
    out = {}
    for name, (start, end) in EVENTS.items():
        before = ret.filter(pl.col("date") < start).tail(TRAILING_VOL_DAYS)["ret"].to_numpy()
        during = ret.filter(pl.col("date").is_between(start, end))["ret"].to_numpy()
        if during.size == 0 or before.size < 20:
            out[f"{name}_return"] = math.nan
            out[f"{name}_in_trailing_vol"] = math.nan
            continue
        total = float(np.prod(1 + during) - 1)
        out[f"{name}_return"] = total
        out[f"{name}_in_trailing_vol"] = (
            total / float(before.std(ddof=1)) if before.std(ddof=1) > 0 else math.nan
        )
    return out


def traded_names(run: Run) -> pl.DataFrame:
    """``year, security_id`` of names entered (hedge instruments excluded)."""
    trades = run.trades.filter(pl.col("reason") == "entry")
    return trades.select(pl.col("date_exec").dt.year().alias("year"), "security_id").unique()


def overlap(runs: list[Run]) -> pl.DataFrame:
    """Jaccard overlap of traded names between each pair of runs, by year."""
    names = {run.label: traded_names(run) for run in runs}
    rows = []
    labels = list(names)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for year in sorted(set(names[a]["year"].to_list()) & set(names[b]["year"].to_list())):
                sa = set(names[a].filter(pl.col("year") == year)["security_id"].to_list())
                sb = set(names[b].filter(pl.col("year") == year)["security_id"].to_list())
                union = sa | sb
                rows.append(
                    {
                        "run_a": a,
                        "run_b": b,
                        "year": year,
                        "jaccard": len(sa & sb) / len(union) if union else math.nan,
                    }
                )
    return pl.DataFrame(rows)


def concentration(run: Run, top: int = 10) -> float:
    """Share of traded gross notional (entries) in the ``top`` most-traded names."""
    entries = run.trades.filter(pl.col("reason") == "entry")
    if entries.is_empty():
        return math.nan
    by_name = (
        entries.group_by("security_id")
        .agg(pl.col("notional_usd").abs().sum().alias("gross"))
        .sort("gross", descending=True)
    )
    total = float(by_name["gross"].to_numpy().sum())
    return float(by_name.head(top)["gross"].to_numpy().sum()) / total if total > 0 else math.nan


def pair_trends(run: Run, pairs: pl.DataFrame) -> pl.DataFrame:
    """Average half-life of the selected pairs and entries per selected pair, by year."""
    chosen = pairs.filter(pl.col("selected")).with_columns(
        pl.col("window_id").str.slice(0, 4).cast(pl.Int32).alias("year")
    )
    half = chosen.group_by("year").agg(
        pl.col("ou_half_life_days").mean().alias("mean_half_life_days"), pl.len().alias("pairs")
    )
    entries = (
        run.trades.filter(pl.col("reason") == "entry")
        .select("window_id", "pair_id", "date_exec")
        .unique()
    )
    per_year = (
        entries.with_columns(pl.col("window_id").str.slice(0, 4).cast(pl.Int32).alias("year"))
        .group_by("year")
        .len()
        .rename({"len": "entries"})
    )
    return (
        half.join(per_year, on="year", how="left")
        .with_columns(
            (pl.col("entries").fill_null(0) / pl.col("pairs")).alias("entries_per_pair"),
            pl.lit(run.label).alias("run"),
        )
        .sort("year")
    )
