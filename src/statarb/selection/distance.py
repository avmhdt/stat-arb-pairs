"""Distance selection (GGR 2006, FR-08).

Total-return indices are normalised to 1 on the first formation session; pairs are ranked
by the sum of squared differences of the normalised series, computed for all pairs at once
with ``|a - b|^2 = |a|^2 + |b|^2 - 2 a.b``. Each stock enters at most
``max_pairs_per_stock`` pairs, taken in SSD order.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from statarb.industry import Grouping, SicEtfMap, group_keys
from statarb.models.ou import fit_ou
from statarb.selection.pairs import PairSet, conform_tested, greedy_unique, params_from
from statarb.selection.panel import formation_panel

if TYPE_CHECKING:
    from statarb.windows import FormationView

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def distance_ssd(norm_prices: F64) -> F64:
    """``(n_stocks, n_stocks)`` SSD between columns of ``norm_prices`` (sessions x stocks)."""
    prices = np.asarray(norm_prices, dtype=np.float64)
    norms = np.einsum("ij,ij->j", prices, prices)
    ssd: F64 = norms[:, None] + norms[None, :] - 2.0 * (prices.T @ prices)
    np.maximum(ssd, 0.0, out=ssd)
    np.fill_diagonal(ssd, 0.0)
    return ssd


def ordered_pairs(ssd: F64, allowed: npt.NDArray[np.bool_], want: int) -> tuple[I64, I64]:
    """Upper-triangle pairs the mask allows, the ``want`` smallest SSD first, ties by index."""
    rows, cols = np.triu_indices(ssd.shape[0], k=1)
    keep = allowed[rows, cols]
    rows, cols = rows[keep], cols[keep]
    values = ssd[rows, cols]
    if values.size > want:
        part = np.argpartition(values, want - 1)[:want]
        rows, cols, values = rows[part], cols[part], values[part]
    order = np.lexsort((cols, rows, values))
    return rows[order].astype(np.int64), cols[order].astype(np.int64)


def _groups(view: FormationView, grouping: Grouping, sic_map: SicEtfMap | None) -> pl.DataFrame:
    universe = view.universe().select("security_id")
    if grouping == "none":
        return universe.with_columns(pl.lit("all").alias("group"))
    return group_keys(view.industry(), grouping, sic_map).join(
        universe, on="security_id", how="semi"
    )


def select_distance(
    view: FormationView,
    top_k: int = 20,
    max_pairs_per_stock: int | None = 1,
    grouping: Grouping = "none",
    rank_from: int = 1,
    store_top: int = 200,
    sic_map: SicEtfMap | None = None,
    run_id: str | None = None,
) -> PairSet:
    """Rank pairs by SSD and select ranks ``rank_from`` to ``rank_from + top_k - 1``.

    ``tested`` keeps the ``max(store_top, rank_from + top_k - 1)`` best-ranked pairs, not
    all of them: with every eligible stock that is millions of rows per window. ``fdr_m``
    records how many pairs were compared.
    """
    groups = _groups(view, grouping, sic_map)
    ids = sorted(groups["security_id"].to_list())
    panel = formation_panel(view, ids)
    usable = ~np.isnan(panel).any(axis=0) if ids else np.zeros(0, dtype=bool)
    ids = [name for name, ok in zip(ids, usable, strict=True) if ok]
    norm = panel[:, usable] / panel[0, usable]
    ssd = distance_ssd(norm)
    group_of = dict(groups.iter_rows())
    labels = np.array([group_of[name] for name in ids], dtype=object)
    allowed = labels[:, None] == labels[None, :]
    compared = int(allowed[np.triu_indices(len(ids), k=1)].sum())
    depth = max(store_top, rank_from + top_k - 1)
    rows, cols = ordered_pairs(ssd, allowed, min(compared, max(50 * depth, 20_000)))
    names = [(ids[i], ids[j]) for i, j in zip(rows, cols, strict=True)]
    records = []
    for rank, index in enumerate(greedy_unique(names, depth, max_pairs_per_stock), start=1):
        i, j = int(rows[index]), int(cols[index])
        spread = norm[:, i] - norm[:, j]
        ou = fit_ou(spread)
        records.append(
            {
                "leg_y": ids[i],
                "leg_x": ids[j],
                "group": str(labels[i]),
                "ssd": float(ssd[i, j]),
                "fdr_m": compared,
                "beta0": 1.0,
                "alpha0": 0.0,
                "spread_mu": float(spread.mean()),
                "spread_sigma": float(spread.std(ddof=1)),
                "ou_kappa": None if ou is None else ou.kappa,
                "ou_half_life_days": None if ou is None else ou.half_life_days,
                "ou_sigma_eq": None if ou is None else ou.sigma_eq,
                "selected": rank_from <= rank < rank_from + top_k,
                "rank": rank,
            }
        )
    frame = pl.DataFrame(records, infer_schema_length=None)
    tested = conform_tested(frame, run_id or str(uuid.uuid4()), view.window)
    return PairSet(view.window, params_from(tested), tested)
