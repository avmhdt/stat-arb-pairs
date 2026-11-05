"""GGR (2006) replication table (FR-20, AC-H-01).

Each GGR run is one portfolio (top 5, top 20 or pairs 101-120) under one execution rule
(no wait, or wait one day). Daily returns are compounded to calendar months; means carry
Newey-West t-statistics with six lags, as in the paper. Targets come from
``tests/fixtures/ggr2006_targets.csv``. On Sharadar data the comparison covers the
1998-2002 overlap only and is labelled partial (D-03).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from statarb.costs import half_spread
from statarb.data.market import MarketData
from statarb.evaluation.metrics import monthly, newey_west_mean

TARGETS = Path("tests") / "fixtures" / "ggr2006_targets.csv"
FULL_SAMPLE = "1963-07/2002-12"
TOLERANCE = 0.30


@dataclass(frozen=True)
class GgrRun:
    """One replication run read back from ``results/runs/<run_id>``."""

    run_id: str
    portfolio: str
    wait: str
    data_profile: str
    returns: pl.DataFrame


def portfolio_name(top_k: int, rank_from: int) -> str:
    """``top5``, ``top20`` or ``101-120`` style names."""
    if rank_from == 1:
        return f"top{top_k}"
    return f"{rank_from}-{rank_from + top_k - 1}"


def read_run(runs: Path, run_id: str) -> GgrRun:
    """Load a run's resolved config and daily returns."""
    folder = runs / run_id
    params = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    selection = params["selection"]
    if selection["method"] != "distance":
        raise ValueError(f"{run_id} is not a distance run")
    wait = "no_wait" if params["windows"]["execution_lag_days"] == 0 else "wait_one_day"
    return GgrRun(
        run_id,
        portfolio_name(selection["top_k"], selection["rank_from"]),
        wait,
        str(params["data_profile"]),
        pl.read_parquet(folder / "returns.parquet"),
    )


def run_rows(run: GgrRun) -> list[dict[str, object]]:
    """Two rows per run: committed and fully invested (employed) capital."""
    rows: list[dict[str, object]] = []
    for capital, column in (("committed", "ret"), ("fully_invested", "ret_employed")):
        months = monthly(run.returns.select("date", column), column)
        mean, se, t = newey_west_mean(months[column].to_numpy())
        dates = months["month"].to_list()
        first, last = dates[0], dates[-1]
        rows.append(
            {
                "portfolio": run.portfolio,
                "wait": run.wait,
                "capital": capital,
                "period": f"{first:%Y-%m}/{last:%Y-%m}",
                "months": months.height,
                "mean_monthly": mean,
                "se_newey_west": se,
                "t_stat": t,
                "data_profile": run.data_profile,
                "run_id": run.run_id,
            }
        )
    return rows


def replication_table(runs: list[GgrRun], targets: pl.DataFrame) -> pl.DataFrame:
    """Ours beside the paper's Table 1 values, with the sign and ratio for each cell."""
    ours = pl.DataFrame([row for run in runs for row in run_rows(run)])
    paper = targets.filter(pl.col("period") == FULL_SAMPLE).select(
        "portfolio",
        "wait",
        "capital",
        pl.col("mean_monthly").alias("paper_mean_monthly"),
        pl.col("t_stat").alias("paper_t_stat"),
    )
    late = targets.filter(pl.col("period") != FULL_SAMPLE).select(
        "portfolio", "wait", "capital", pl.col("mean_monthly").alias("paper_1989_2002_mean_monthly")
    )
    return (
        ours.join(paper, on=["portfolio", "wait", "capital"], how="left")
        .join(late, on=["portfolio", "wait", "capital"], how="left")
        .with_columns(
            (pl.col("mean_monthly") / pl.col("paper_mean_monthly")).alias("ratio_to_paper"),
            (np.sign(pl.col("mean_monthly")) == np.sign(pl.col("paper_mean_monthly"))).alias(
                "same_sign"
            ),
            pl.lit("partial: Sharadar 1998-2002 overlap (D-03)").alias("label"),
        )
        .sort("wait", "capital", "portfolio")
    )


def verdict(table: pl.DataFrame) -> dict[str, object]:
    """AC-H-01's three checks, per execution rule and overall."""
    out: dict[str, object] = {"all_same_sign": bool(table["same_sign"].all())}
    passed = bool(out["all_same_sign"])
    for wait in ("no_wait", "wait_one_day"):
        invested = {
            row["portfolio"]: row
            for row in table.filter(
                (pl.col("wait") == wait) & (pl.col("capital") == "fully_invested")
            ).iter_rows(named=True)
        }
        if not {"top5", "top20", "101-120"} <= set(invested):
            out[f"{wait}_complete"] = False
            passed = False
            continue
        top20 = invested["top20"]
        within = abs(float(top20["ratio_to_paper"]) - 1.0) <= TOLERANCE
        significant = float(top20["t_stat"]) > 2.0
        ordered = (
            min(invested["top5"]["mean_monthly"], top20["mean_monthly"])
            > invested["101-120"]["mean_monthly"]
        )
        out[f"{wait}_top20_within_30pct"] = within
        out[f"{wait}_top20_t_gt_2"] = significant
        out[f"{wait}_ordering_holds"] = bool(ordered)
        passed = passed and within and significant and bool(ordered)
    out["pass"] = passed
    return out


def figure(table: pl.DataFrame, path: Path) -> None:
    """Bars of ours against the paper's Table 1, per cell; PNG and SVG at 300 dpi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table.sort("wait", "capital", "portfolio")
    labels = [
        f"{r['portfolio']}\n{r['capital'].replace('_', ' ')}\n{r['wait'].replace('_', ' ')}"
        for r in rows.iter_rows(named=True)
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.2, rows["paper_mean_monthly"].to_numpy() * 100, 0.4, label="GGR 2006, 1963-2002")
    ax.bar(
        x + 0.2, rows["mean_monthly"].to_numpy() * 100, 0.4, label="This repo, Sharadar 1999-2002"
    )
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Mean monthly excess return (%)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("GGR replication (partial: 1998-2002 overlap only, gross of costs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    from statarb.reports.figures import save

    save(fig, path)


def pair_volatility(runs: Path, run: GgrRun, data: MarketData) -> dict[str, float]:
    """How volatile a run's traded pairs were in their formation windows.

    Per selected pair: the formation spread sigma (normalised price difference), the mean
    daily return volatility of its two legs over the formation window, and the legs' mean
    Corwin-Schultz half-spread over the last 20 formation sessions, a proxy for the bid-ask
    bounce that waiting a day removes. Also the round trips per pair.
    """
    pairs = pl.read_parquet(runs / run.run_id / "pairs.parquet").filter(pl.col("selected"))
    trades = pl.read_parquet(runs / run.run_id / "trades.parquet")
    prices = data.prices
    vols: list[float] = []
    spreads: list[float] = []
    for window_id, start, end, leg_y, leg_x in pairs.select(
        "window_id", "formation_start", "formation_end", "leg_y", "leg_x"
    ).iter_rows():
        rows = prices.filter(
            pl.col("date").is_between(start, end, closed="left")
            & pl.col("security_id").is_in([leg_y, leg_x])
        )
        for _, leg in rows.sort("date").group_by("security_id", maintain_order=True):
            vols.append(float(leg["ret_total"].drop_nulls().std() or math.nan))  # type: ignore[arg-type]
            tail = leg.tail(21)
            spreads.append(
                half_spread(
                    tail["high"].to_numpy(),
                    tail["low"].to_numpy(),
                    None,
                    None,
                    "quotes_then_corwin_schultz",
                    0.0,
                )
            )
        del window_id
    entries = trades.filter(pl.col("reason") == "entry").height / 2
    return {
        "pairs": float(pairs.height),
        "spread_sigma_mean": float(pairs["spread_sigma"].mean()),  # type: ignore[arg-type]
        "leg_daily_vol_mean": float(np.nanmean(vols)) if vols else math.nan,
        "leg_cs_half_spread_bps_mean": float(np.nanmean(spreads)) * 1e4 if spreads else math.nan,
        "round_trips_per_pair": entries / max(pairs.height, 1),
    }


def bounce_table(
    runs: Path, loaded: list[GgrRun], table: pl.DataFrame, data: MarketData
) -> pl.DataFrame:
    """Per portfolio: no-wait and wait-one-day means side by side, the drop, and pair volatility.

    GGR read the drop from waiting a day as bid-ask bounce. If more volatile pairs carry more
    bounce, the drop should be larger for them, in basis points and as a share.
    """
    rows = []
    for portfolio in ("top5", "top20", "101-120"):
        wait = next(
            run for run in loaded if run.portfolio == portfolio and run.wait == "wait_one_day"
        )
        record: dict[str, object] = {"portfolio": portfolio}
        for capital in ("fully_invested", "committed"):
            cells = table.filter(
                (pl.col("portfolio") == portfolio) & (pl.col("capital") == capital)
            )
            no_wait = float(cells.filter(pl.col("wait") == "no_wait")["mean_monthly"][0])
            waited = float(cells.filter(pl.col("wait") == "wait_one_day")["mean_monthly"][0])
            record |= {
                f"{capital}_no_wait": no_wait,
                f"{capital}_wait_one_day": waited,
                f"{capital}_drop_bps": (no_wait - waited) * 1e4,
                f"{capital}_drop_share": (no_wait - waited) / no_wait if no_wait else math.nan,
            }
        record |= pair_volatility(runs, wait, data)
        rows.append(record)
    return pl.DataFrame(rows)
