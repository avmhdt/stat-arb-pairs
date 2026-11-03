"""The frozen output of selection: the only object that crosses ``T`` (spec Figure 1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from statarb.windows import Window

PAIRS_SELECTED: Mapping[str, pl.DataType | type[pl.DataType]] = {
    "run_id": pl.String,
    "window_id": pl.String,
    "formation_start": pl.Date,
    "formation_end": pl.Date,
    "leg_y": pl.String,
    "leg_x": pl.String,
    "group": pl.String,
    "ssd": pl.Float64,
    "eg_pvalue_yx": pl.Float64,
    "eg_pvalue_xy": pl.Float64,
    "eg_pvalue": pl.Float64,
    "johansen_trace_r0": pl.Float64,
    "johansen_cv95_r0": pl.Float64,
    "fdr_method": pl.String,
    "fdr_q": pl.Float64,
    "fdr_m": pl.Int64,
    "fdr_reject": pl.Boolean,
    # Beyond the spec's columns: the also-reported procedure's decision (D-05).
    "fdr_reject_by": pl.Boolean,
    "beta0": pl.Float64,
    "alpha0": pl.Float64,
    "spread_mu": pl.Float64,
    "spread_sigma": pl.Float64,
    "ou_kappa": pl.Float64,
    "ou_half_life_days": pl.Float64,
    "ou_sigma_eq": pl.Float64,
    "selected": pl.Boolean,
    "rank": pl.Int32,
}


@dataclass(frozen=True, slots=True)
class PairParams:
    """One traded pair frozen at ``T``; spreads in log prices (coint) or normalised prices."""

    leg_y: str
    leg_x: str
    beta0: float
    alpha0: float
    spread_mu: float
    spread_sigma: float
    ou_kappa: float | None
    ou_half_life_days: float | None
    eg_pvalue: float | None
    fdr_reject: bool | None
    rank: int


@dataclass(frozen=True, slots=True)
class PairSet:
    """The selected pairs of one window and every candidate row behind them."""

    window: Window
    pairs: tuple[PairParams, ...]
    tested: pl.DataFrame  # schema PAIRS_SELECTED


def conform_tested(frame: pl.DataFrame, run_id: str, window: Window) -> pl.DataFrame:
    """Fill run and window columns and cast to ``PAIRS_SELECTED``."""
    filled = frame.with_columns(
        pl.lit(run_id).alias("run_id"),
        pl.lit(window.window_id).alias("window_id"),
        pl.lit(window.formation_start).alias("formation_start"),
        pl.lit(window.formation_end).alias("formation_end"),
    )
    return filled.select(
        pl.col(name).cast(dtype)
        if name in filled.columns
        else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in PAIRS_SELECTED.items()
    )


def params_from(tested: pl.DataFrame) -> tuple[PairParams, ...]:
    """Selected rows as ``PairParams``, in rank order."""
    rows = tested.filter(pl.col("selected")).sort("rank")
    return tuple(
        PairParams(
            leg_y=row["leg_y"],
            leg_x=row["leg_x"],
            beta0=row["beta0"],
            alpha0=row["alpha0"],
            spread_mu=row["spread_mu"],
            spread_sigma=row["spread_sigma"],
            ou_kappa=row["ou_kappa"],
            ou_half_life_days=row["ou_half_life_days"],
            eg_pvalue=row["eg_pvalue"],
            fdr_reject=row["fdr_reject"],
            rank=row["rank"],
        )
        for row in rows.iter_rows(named=True)
    )


def greedy_unique(
    ordered: list[tuple[str, str]], limit: int, max_per_stock: int | None
) -> list[int]:
    """Indexes into ``ordered`` taken in order, each stock in at most ``max_per_stock`` pairs."""
    used: dict[str, int] = {}
    taken: list[int] = []
    for index, (leg_y, leg_x) in enumerate(ordered):
        if len(taken) >= limit:
            break
        if max_per_stock is not None and (
            used.get(leg_y, 0) >= max_per_stock or used.get(leg_x, 0) >= max_per_stock
        ):
            continue
        used[leg_y] = used.get(leg_y, 0) + 1
        used[leg_x] = used.get(leg_x, 0) + 1
        taken.append(index)
    return taken
