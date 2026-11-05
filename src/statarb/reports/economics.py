"""M6 tables and figures: decay, persistence, break-even, capacity, crowding, break policies,
sizing, deflated Sharpe, survivorship bias and borrow (AC-H-02, AC-H-04 to AC-H-10, AC-W-09).

Each function reads runs from ``results/runs`` (and processed data where it must) and writes
aggregate CSVs to ``results/tables`` and figures to ``reports/figures``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import yaml

from statarb.config import RunConfig
from statarb.data.market import load_market_data
from statarb.evaluation import crowding as crowd
from statarb.evaluation import dsr as dsr_mod
from statarb.evaluation import economics as econ
from statarb.evaluation.persistence import persistence
from statarb.reports import figures

TABLES = Path("results") / "tables"
FIGURES = Path("reports") / "figures"


def _load(runs_dir: Path, specs: list[str]) -> list[econ.Run]:
    """``label=run_id`` or bare run ids."""
    out = []
    for spec in specs:
        label, _, rid = spec.rpartition("=")
        out.append(econ.read_run(runs_dir, rid.strip(), label.strip() or None))
    return out


def decay(
    runs_dir: Path, specs: list[str], tables: Path = TABLES, figs: Path = FIGURES
) -> pl.DataFrame:
    """``decay_by_period.csv``, ``decay_trend.csv`` and ``ggr_decay`` (rolling 36 months)."""
    periods, rolls, trends = [], [], []
    for run in _load(runs_dir, specs):
        table, roll, summary = econ.decay(run)
        periods.append(table)
        rolls.append(roll)
        trends.append({"run": run.label, **summary})
    out = pl.concat(periods)
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "decay_by_period.csv", float_precision=6)
    pl.DataFrame(trends).write_csv(tables / "decay_trend.csv", float_precision=6)
    figures.lines(
        pl.concat(rolls),
        "month",
        "sharpe_36m",
        "run",
        figs / "ggr_decay",
        "Rolling 36-month Sharpe ratio of monthly returns",
        "Sharpe (annualised)",
    )
    return out


def persistence_tables(
    runs_dir: Path, processed: Path, specs: list[str], tables: Path = TABLES, figs: Path = FIGURES
) -> pl.DataFrame:
    """``oos_persistence.csv`` and its figure."""
    frames = []
    for spec in specs:
        label, _, rid = spec.rpartition("=")
        folder = runs_dir / rid
        params = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        method = label or params["selection"]["method"]
        cfg = RunConfig.model_validate(
            {k: v for k, v in params.items() if k not in ("data_profile", "manifest_hash")}
        )
        data = load_market_data(
            processed,
            cfg.data.start,
            cfg.data.end,
            ["tr_index"],
            tuple(cfg.universe.security_types),
        )
        pairs = pl.read_parquet(folder / "pairs.parquet")
        _, table = persistence(
            data, pairs, method, seed=cfg.run.seed, trading_months=cfg.windows.trading_months
        )
        frames.append(table)
    out = pl.concat(frames, how="diagonal")
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "oos_persistence.csv", float_precision=6)
    yearly = out.filter((pl.col("year") > 0) & (pl.col("horizon") == "p_trading")).with_columns(
        (pl.col("method") + " " + pl.col("sample")).alias("series")
    )
    figures.lines(
        yearly,
        "year",
        "share_p_lt_05",
        "series",
        figs / "oos_persistence",
        "Share of pairs with EG p < 0.05 in the trading window",
        "share",
        hline=0.05,
    )
    return out


def break_even_tables(
    runs_dir: Path, specs: list[str], tables: Path = TABLES, figs: Path = FIGURES
) -> pl.DataFrame:
    """``break_even.csv`` and ``break_even`` (all-period, borrow x1)."""
    out = pl.concat([econ.break_even(run) for run in _load(runs_dir, specs)])
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "break_even.csv", float_precision=6)
    view = out.filter((pl.col("period") == "all") & (pl.col("borrow_multiplier") == 1.0))
    figures.bars(
        view["run"].to_list(),
        {
            "break-even (bps per side)": view["break_even_bps"].to_list(),
            "modelled cost": view["modelled_bps"].to_list(),
        },
        figs / "break_even",
        "Break-even cost per side, 95% bootstrap interval, borrow x1",
        "basis points",
        {"break-even (bps per side)": (view["ci_low"].to_list(), view["ci_high"].to_list())},
    )
    return out


def capacity_tables(
    runs_dir: Path, specs: list[str], tables: Path = TABLES, figs: Path = FIGURES
) -> pl.DataFrame:
    """``capacity.csv``, ``capacity_summary.csv`` and ``capacity``."""
    grids, sums = [], []
    for run in _load(runs_dir, specs):
        grid, summary = econ.capacity(run)
        grids.append(grid)
        sums.append(summary)
    grid, summary = pl.concat(grids), pl.concat(sums)
    tables.mkdir(parents=True, exist_ok=True)
    grid.write_csv(tables / "capacity.csv", float_precision=6)
    summary.write_csv(tables / "capacity_summary.csv", float_precision=6)
    figures.lines(
        grid.filter(pl.col("y_coeff") == 1.0).with_columns(
            (pl.col("run") + " (" + pl.col("basis").str.replace("_", " ") + ")").alias("series")
        ),
        "aum_usd",
        "net_sharpe",
        "series",
        figs / "capacity",
        "Sharpe against AUM, last five years, Y = 1: all costs and impact only",
        "net Sharpe",
        logx=True,
    )
    return grid


def crowding_tables(
    runs_dir: Path,
    factors_path: Path,
    specs: list[str],
    tables: Path = TABLES,
    figs: Path = FIGURES,
) -> pl.DataFrame:
    """``crowding.csv`` plus rolling, overlap and trend tables and figures."""
    factors = pl.read_parquet(factors_path)
    runs = _load(runs_dir, specs)
    rows, rolls, trends = [], [], []
    for run in runs:
        exposure, roll = crowd.factor_exposure(run, factors)
        row = {
            "run": run.label,
            **exposure,
            **crowd.event_drawdowns(run),
            "top10_share_of_traded_gross": crowd.concentration(run),
        }
        rows.append(row)
        rolls.append(roll)
        pairs_path = runs_dir / run.run_id / "pairs.parquet"
        if run.family.startswith("pairs") and pairs_path.exists():
            trends.append(crowd.pair_trends(run, pl.read_parquet(pairs_path)))
    out = pl.DataFrame(rows)
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "crowding.csv", float_precision=6)
    pl.concat(rolls).write_csv(tables / "crowding_rolling.csv", float_precision=6)
    lap = crowd.overlap(runs)
    lap.write_csv(tables / "crowding_overlap.csv", float_precision=6)
    if lap.height:
        lap.group_by("run_a", "run_b", maintain_order=True).agg(
            pl.len().alias("years"),
            pl.col("jaccard").mean().alias("mean_jaccard"),
            pl.col("jaccard").max().alias("max_jaccard"),
        ).write_csv(tables / "crowding_overlap_summary.csv", float_precision=6)
    if trends:
        trend = pl.concat(trends)
        trend.write_csv(tables / "crowding_trends.csv", float_precision=6)
        figures.lines(
            trend,
            "year",
            "mean_half_life_days",
            "run",
            figs / "crowding_half_life",
            "Average half-life of selected pairs by formation year",
            "sessions",
            hline=None,
        )
        figures.lines(
            trend,
            "year",
            "entries_per_pair",
            "run",
            figs / "crowding_divergence",
            "Entries per selected pair (divergence frequency)",
            "entries per pair",
            hline=None,
        )
    figures.lines(
        pl.concat(rolls),
        "month",
        "beta_st_rev_36m",
        "run",
        figs / "crowding_reversal_beta",
        "Rolling 36-month beta to the short-term reversal factor",
        "beta",
    )
    return out


def policy_table(runs_dir: Path, specs: list[str], tables: Path = TABLES) -> pl.DataFrame:
    """``break_policies.csv``."""
    out = econ.policies(_load(runs_dir, specs))
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "break_policies.csv", float_precision=6)
    return out


def sizing_table(
    runs_dir: Path, specs: list[str], tables: Path = TABLES, figs: Path = FIGURES
) -> pl.DataFrame:
    """``sizing_half_life.csv`` and ``sizing_half_life``."""
    out = econ.sizing(_load(runs_dir, specs))
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "sizing_half_life.csv", float_precision=6)
    view = out.filter(pl.col("half_life_bucket") != "all").with_columns(
        (pl.col("family") + " " + pl.col("scheme")).alias("series")
    )
    buckets = [b for b in ("1-5", "5-15", "15-30", "30-63") if b in set(view["half_life_bucket"])]
    series: dict[str, Sequence[float]] = {}
    for (name,), part in view.group_by(["series"], maintain_order=True):
        values = dict(part.select("half_life_bucket", "pnl_per_capital_day_bps").iter_rows())
        series[str(name)] = [values.get(b, math.nan) for b in buckets]
    figures.bars(
        buckets,
        series,
        figs / "sizing_half_life",
        "Net P&L per capital-day by half-life bucket",
        "bps per capital-day",
    )
    return out


def dsr_table(log_path: Path, headline: dict[str, str], tables: Path = TABLES) -> pl.DataFrame:
    """``deflated_sharpe.csv``."""
    out = dsr_mod.family_table(log_path, headline)
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "deflated_sharpe.csv", float_precision=6)
    return out


def survivorship_table(
    runs_dir: Path, pairs_of_runs: list[tuple[str, str]], tables: Path = TABLES
) -> pl.DataFrame:
    """``survivorship_bias.csv``: (full, survivors-only) run id pairs."""
    rows = [
        econ.survivorship(econ.read_run(runs_dir, a), econ.read_run(runs_dir, b))
        for a, b in pairs_of_runs
    ]
    out = pl.DataFrame(rows)
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "survivorship_bias.csv", float_precision=6)
    return out


def borrow_table(
    snapshot: Path, processed: Path, schedule: pl.DataFrame, tables: Path = TABLES
) -> pl.DataFrame:
    """``borrow_assumed_vs_ibkr.csv``: today's IBKR fees by size bucket against D-07."""
    from statarb.data.providers.ibkr import compare

    master = pl.read_parquet(processed / "security_master" / "*.parquet").filter(
        pl.col("last_date").is_null() & (pl.col("security_type") == "common")
    )
    scan = pl.scan_parquet(processed / "prices_daily" / "**" / "*.parquet", hive_partitioning=False)
    last = scan.select(pl.col("date").max()).collect().item()
    caps = scan.filter(pl.col("date") == last).select("security_id", "mktcap_usd").collect()
    latest = (
        caps.join(master.select("security_id", "ticker"), on="security_id")
        .drop_nulls("mktcap_usd")
        .sort("mktcap_usd", descending=True)
        .with_row_index("rank", offset=1)
        .select("ticker", "rank")
    )
    out = compare(pl.read_parquet(snapshot / "usa.parquet"), latest, schedule).with_columns(
        pl.lit(snapshot.name.split("=")[1]).alias("snapshot"), pl.lit(str(last)).alias("caps_as_of")
    )
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "borrow_assumed_vs_ibkr.csv", float_precision=6)
    return out


def kalman_table(runs_dir: Path, ids: list[str], tables: Path = TABLES) -> pl.DataFrame:
    """``kalman_delta.csv``: the hedge model and Kalman delta against results (FR-12)."""
    from statarb.evaluation.bootstrap import sharpe

    rows = []
    for rid in ids:
        run = econ.read_run(runs_dir, rid)
        model = run.params["model"]
        ret = run.returns["ret"].to_numpy()
        trips = run.roundtrips
        rows.append(
            {
                "hedge": model["hedge"],  # type: ignore[index]
                "delta": model["kalman"]["delta"] if model["hedge"] == "kalman" else None,  # type: ignore[index]
                "sharpe": sharpe(ret),
                "return_annual": float(ret.mean() * 252),
                "trades": trips.height,
                "mean_holding_days": econ._mean(trips["holding_days"]) if trips.height else None,
                "mean_trade_return": econ._mean(trips["ret_on_slot"]) if trips.height else None,
                "run_id": rid,
            }
        )
    out = pl.DataFrame(rows, infer_schema_length=None)
    tables.mkdir(parents=True, exist_ok=True)
    out.write_csv(tables / "kalman_delta.csv", float_precision=6)
    return out
