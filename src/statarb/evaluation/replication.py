"""GGR (2006) replication table (FR-20, AC-H-01).

Each GGR run is one portfolio (top 5, top 20 or pairs 101-120) under one execution rule
(no wait, or wait one day). Daily returns are compounded to calendar months; means carry
Newey-West t-statistics with six lags, as in the paper. Targets come from
``tests/fixtures/ggr2006_targets.csv``. On Sharadar data the comparison covers the
1998-2002 overlap only and is labelled partial (D-03).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
