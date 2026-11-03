"""Cointegration selection with FDR control (FR-09 to FR-11), and selection over windows.

Per window: every candidate pair in its group is tested by Engle-Granger in both orderings
and the larger p-value kept, which makes the test order-invariant. BH (primary) and BY are
applied to the p-values of all tested pairs, so the family size m is the number tested.
Discoveries are then filtered by OU half-life, optionally confirmed by Johansen, ranked by
p-value, and the first ``top_k`` traded with each stock in at most one pair.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from statarb.industry import SicEtfMap
from statarb.models.ou import fit_ou
from statarb.selection.candidates import candidate_pairs
from statarb.selection.cointegration import engle_granger_batch, johansen_lag, johansen_trace
from statarb.selection.distance import select_distance
from statarb.selection.fdr import fdr_reject
from statarb.selection.pairs import PairSet, conform_tested, greedy_unique, params_from
from statarb.selection.panel import formation_panel

if TYPE_CHECKING:
    from statarb.config import SelectionConfig
    from statarb.windows import FormationView, Window


FITTED_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "leg_y": pl.String,
    "leg_x": pl.String,
    "beta0": pl.Float64,
    "alpha0": pl.Float64,
    "spread_mu": pl.Float64,
    "spread_sigma": pl.Float64,
    "ou_kappa": pl.Float64,
    "ou_half_life_days": pl.Float64,
    "ou_sigma_eq": pl.Float64,
    "johansen_trace_r0": pl.Float64,
    "johansen_cv95_r0": pl.Float64,
}
"""Columns fitted on FDR discoveries only; empty when a window has none."""


def _hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    (beta, alpha), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta), float(alpha)


def select_cointegration(
    view: FormationView,
    cfg: SelectionConfig,
    sic_map: SicEtfMap | None = None,
    run_id: str | None = None,
) -> PairSet:
    """Engle-Granger plus FDR selection for one window; ``tested`` holds every candidate."""
    candidates = candidate_pairs(view, cfg.grouping, sic_map)
    ids = sorted(set(candidates["leg_y"].to_list()) | set(candidates["leg_x"].to_list()))
    logp = np.log(formation_panel(view, ids))
    usable = {name for name, ok in zip(ids, ~np.isnan(logp).any(axis=0), strict=True) if ok}
    column = {name: i for i, name in enumerate(ids)}
    tested = candidates.filter(
        pl.col("leg_y").is_in(list(usable)) & pl.col("leg_x").is_in(list(usable))
    )
    index = np.array(
        [[column[a], column[b]] for a, b in tested.select("leg_y", "leg_x").iter_rows()],
        dtype=np.int64,
    ).reshape(-1, 2)
    yx = engle_granger_batch(logp, index, cfg.eg.autolag, cfg.eg.maxlag)
    xy = engle_granger_batch(logp, index[:, ::-1], cfg.eg.autolag, cfg.eg.maxlag)
    pvalue = np.maximum(yx[:, 1], xy[:, 1])
    q = cfg.fdr.q
    tested = tested.with_columns(
        pl.Series("eg_pvalue_yx", yx[:, 1]),
        pl.Series("eg_pvalue_xy", xy[:, 1]),
        pl.Series("eg_pvalue", pvalue),
        pl.lit(cfg.fdr.primary).alias("fdr_method"),
        pl.lit(q).alias("fdr_q"),
        pl.lit(tested.height, pl.Int64).alias("fdr_m"),
        pl.Series("fdr_reject", fdr_reject(pvalue, q, cfg.fdr.primary)),
        pl.Series("fdr_reject_by", fdr_reject(pvalue, q, "by")),
    )
    records = []
    for row in tested.filter(pl.col("fdr_reject")).iter_rows(named=True):
        y, x = logp[:, column[row["leg_y"]]], logp[:, column[row["leg_x"]]]
        beta, alpha = _hedge(y, x)
        spread = y - alpha - beta * x
        ou = fit_ou(spread)
        record = {
            "leg_y": row["leg_y"],
            "leg_x": row["leg_x"],
            "beta0": beta,
            "alpha0": alpha,
            "spread_mu": float(spread.mean()),
            "spread_sigma": float(spread.std(ddof=1)),
            "ou_kappa": None if ou is None else ou.kappa,
            "ou_half_life_days": None if ou is None else ou.half_life_days,
            "ou_sigma_eq": None if ou is None else ou.sigma_eq,
            "johansen_trace_r0": None,
            "johansen_cv95_r0": None,
        }
        in_band = (
            ou is not None and cfg.half_life_days.min <= ou.half_life_days <= cfg.half_life_days.max
        )
        if cfg.johansen_confirm.enabled and in_band:
            pair = np.column_stack([y, x])
            lag = johansen_lag(pair, cfg.johansen_confirm.max_k_ar_diff)
            trace, cv95 = johansen_trace(pair, cfg.johansen_confirm.det_order, lag)
            record["johansen_trace_r0"], record["johansen_cv95_r0"] = trace, cv95
        records.append(record)
    fitted = pl.DataFrame(records, schema=FITTED_SCHEMA)
    tested = tested.join(fitted, on=["leg_y", "leg_x"], how="left")
    eligible = pl.col("fdr_reject") & pl.col("ou_half_life_days").is_between(
        cfg.half_life_days.min, cfg.half_life_days.max
    )
    if cfg.johansen_confirm.enabled:
        eligible = eligible & (pl.col("johansen_trace_r0") > pl.col("johansen_cv95_r0"))
    ranked = (
        tested.filter(eligible.fill_null(False))
        .sort("eg_pvalue", "leg_y", "leg_x")
        .select("leg_y", "leg_x")
    )
    order = list(ranked.iter_rows())
    depth = cfg.rank_from + cfg.top_k - 1
    taken = [order[i] for i in greedy_unique(order, depth, cfg.max_pairs_per_stock)]
    ranks = pl.DataFrame(
        {
            "leg_y": [a for a, _ in taken],
            "leg_x": [b for _, b in taken],
            "rank": list(range(1, len(taken) + 1)),
        },
        schema={"leg_y": pl.String, "leg_x": pl.String, "rank": pl.Int32},
    )
    tested = tested.join(ranks, on=["leg_y", "leg_x"], how="left").with_columns(
        pl.col("rank").is_between(cfg.rank_from, depth).fill_null(False).alias("selected")
    )
    frame = conform_tested(tested, run_id or str(uuid.uuid4()), view.window)
    return PairSet(view.window, params_from(frame), frame)


def select_window(
    view: FormationView,
    cfg: SelectionConfig,
    sic_map: SicEtfMap | None = None,
    run_id: str | None = None,
) -> PairSet:
    """Dispatch on ``cfg.method``."""
    if cfg.method == "distance":
        return select_distance(
            view,
            top_k=cfg.top_k,
            max_pairs_per_stock=cfg.max_pairs_per_stock,
            grouping=cfg.grouping,
            rank_from=cfg.rank_from,
            store_top=cfg.store_top,
            sic_map=sic_map,
            run_id=run_id,
        )
    return select_cointegration(view, cfg, sic_map, run_id)


def fdr_summary(tested: pl.DataFrame, raw_alpha: float = 0.05) -> pl.DataFrame:
    """Per-window family size and discoveries for FR-30's table: aggregates only."""
    return (
        tested.group_by("window_id", maintain_order=True)
        .agg(
            pl.col("fdr_m").first().alias("m"),
            (pl.col("eg_pvalue") < raw_alpha).sum().alias("raw_p_lt_05"),
            pl.col("fdr_reject").sum().alias("bh_discoveries"),
            pl.col("fdr_reject_by").sum().alias("by_discoveries"),
            (pl.col("fdr_reject") & pl.col("ou_half_life_days").is_not_null())
            .sum()
            .alias("ou_fitted"),
            pl.col("selected").sum().alias("traded"),
        )
        .with_columns(
            (pl.col("m") * raw_alpha).round(4).alias("expected_false_raw"),
            (pl.col("bh_discoveries") * tested["fdr_q"].first())
            .round(4)
            .alias("bh_expected_false_max"),
        )
        .sort("window_id")
    )


def windows_between(windows: Sequence[Window], first: str | None, last: str | None) -> list[Window]:
    """Windows whose id lies in ``[first, last]``."""
    return [
        w
        for w in windows
        if (first is None or w.window_id >= first) and (last is None or w.window_id <= last)
    ]
