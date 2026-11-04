"""Out-of-sample cointegration persistence (FR-22, AC-H-04).

For every selected pair, Engle-Granger (both orderings, the larger p-value, as in selection)
is rerun on the trading window and on the 12 months after it. The baseline is a matched
random sample of candidates that were tested but not selected, drawn from the same groups
in the same windows, one per selected pair. Shares of p < 0.05 come with Wilson intervals;
under no cointegration the share would be about the test's size.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from statarb.data.market import MarketData
from statarb.selection.cointegration import engle_granger_batch

ALPHA = 0.05


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def matched_baseline(pairs: pl.DataFrame, seed: int) -> pl.DataFrame:
    """One unselected candidate per selected pair, same window and group."""
    rng = np.random.default_rng(seed)
    picks = []
    for (window_id, group), part in pairs.group_by(["window_id", "group"], maintain_order=True):
        chosen = part.filter(pl.col("selected"))
        others = part.filter(~pl.col("selected"))
        if chosen.is_empty() or others.is_empty():
            continue
        take = rng.choice(others.height, size=min(chosen.height, others.height), replace=False)
        picks.append(others[np.sort(take).tolist()])
        del window_id, group
    return pl.concat(picks) if picks else pairs.clear()


def _testable(y: np.ndarray, x: np.ndarray) -> bool:
    """Finite, non-constant legs that are not one series twice (the test divides by zero)."""
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        return False
    if np.ptp(np.diff(y)) == 0 or np.ptp(np.diff(x)) == 0:
        return False
    return bool(abs(np.corrcoef(y, x)[0, 1]) < 1 - 1e-10)


def _pvalues(
    data: MarketData, rows: pl.DataFrame, starts: list[date], ends: list[date]
) -> np.ndarray:
    """Engle-Granger p-value (larger of both orderings) of each pair over its own span."""
    out = np.full(rows.height, np.nan)
    prices = data.prices.select("date", "security_id", "tr_index")
    spans: dict[tuple[date, date], list[int]] = {}
    for i, key in enumerate(zip(starts, ends, strict=True)):
        spans.setdefault(key, []).append(i)
    for (start, end), index in spans.items():
        part = rows[index]
        ids = sorted(set(part["leg_y"].to_list()) | set(part["leg_x"].to_list()))
        window = prices.filter(
            pl.col("date").is_between(start, end, closed="left") & pl.col("security_id").is_in(ids)
        )
        if window.is_empty():
            continue
        wide = window.pivot(on="security_id", index="date", values="tr_index").sort("date")
        present = [name for name in ids if name in wide.columns]
        panel = np.log(wide.select(pl.col(present).forward_fill()).to_numpy().astype(np.float64))
        column = {name: j for j, name in enumerate(present)}
        if panel.shape[0] < 40:
            continue
        good = []
        pairs = []
        for k, (leg_y, leg_x) in enumerate(part.select("leg_y", "leg_x").iter_rows()):
            if leg_y in column and leg_x in column:
                cy, cx = column[leg_y], column[leg_x]
                if _testable(panel[:, cy], panel[:, cx]):
                    good.append(k)
                    pairs.append((cy, cx))
        if not pairs:
            continue
        idx = np.array(pairs, dtype=np.int64)
        p = np.maximum(
            engle_granger_batch(panel, idx)[:, 1], engle_granger_batch(panel, idx[:, ::-1])[:, 1]
        )
        for k, value in zip(good, p, strict=True):
            out[index[k]] = value
    return out


def persistence(
    data: MarketData, pairs: pl.DataFrame, method: str, seed: int = 0, trading_months: int = 6
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-pair p-values in the trading window and the next 12 months, and the share table."""
    selected = pairs.filter(pl.col("selected")).with_columns(pl.lit("selected").alias("sample"))
    baseline = matched_baseline(pairs, seed).with_columns(pl.lit("baseline").alias("sample"))
    rows = pl.concat([selected, baseline], how="vertical")
    trading_start = [date.fromisoformat(w) for w in rows["window_id"].to_list()]
    trading_end = [_add_months(day, trading_months) for day in trading_start]
    after_end = [_add_months(day, 12) for day in trading_end]
    rows = rows.with_columns(
        pl.Series("p_trading", _pvalues(data, rows, trading_start, trading_end)),
        pl.Series("p_next_12m", _pvalues(data, rows, trading_end, after_end)),
        pl.lit(method).alias("method"),
        pl.col("window_id").str.slice(0, 4).cast(pl.Int32).alias("year"),
    )
    shares = []
    for horizon in ("p_trading", "p_next_12m"):
        for key, part in rows.group_by(["method", "sample", "year"], maintain_order=True):
            values = part[horizon].drop_nulls().drop_nans()
            hits = int((values < ALPHA).sum())
            shares.append(
                {
                    "method": key[0],
                    "sample": key[1],
                    "year": key[2],
                    "horizon": horizon,
                    "pairs": values.len(),
                    "share_p_lt_05": hits / values.len() if values.len() else math.nan,
                }
            )
        for key, part in rows.group_by(["method", "sample"], maintain_order=True):
            values = part[horizon].drop_nulls().drop_nans()
            hits = int((values < ALPHA).sum())
            lo, hi = wilson(hits, values.len())
            shares.append(
                {
                    "method": key[0],
                    "sample": key[1],
                    "year": 0,
                    "horizon": horizon,
                    "pairs": values.len(),
                    "share_p_lt_05": hits / values.len() if values.len() else math.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    table = pl.DataFrame(shares, infer_schema_length=None).sort(
        "method", "horizon", "sample", "year"
    )
    return rows.select("method", "sample", "year", "window_id", "p_trading", "p_next_12m"), table
