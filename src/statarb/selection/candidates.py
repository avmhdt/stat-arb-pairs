"""Candidate pairs within point-in-time industry groups (FR-07, D-06)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from statarb.industry import Grouping, SicEtfMap, group_keys

if TYPE_CHECKING:
    from statarb.windows import FormationView

CANDIDATE_COLUMNS = ("leg_y", "leg_x", "group")


def candidate_pairs(
    view: FormationView,
    grouping: Grouping,
    sic_map: SicEtfMap | None = None,
) -> pl.DataFrame:
    """Every unordered pair of eligible securities sharing a group at ``T - 1``.

    Parameters
    ----------
    view
        The formation view; its universe and SIC codes as of ``T - 1`` are used.
    grouping
        ``none`` (all pairs), ``sic2``, ``sic3`` or ``etf_sector``.
    sic_map
        Required for ``etf_sector``.

    Returns
    -------
    polars.DataFrame
        ``leg_y, leg_x, group`` with ``leg_y < leg_x``, sorted. The row count is the
        family size m used by FDR control (FR-11).
    """
    members = view.universe().select("security_id")
    groups = group_keys(view.industry(), grouping, sic_map).join(
        members, on="security_id", how="semi"
    )
    left = groups.rename({"security_id": "leg_y"})
    right = groups.rename({"security_id": "leg_x"})
    return (
        left.join(right, on="group", how="inner")
        .filter(pl.col("leg_y") < pl.col("leg_x"))
        .select(CANDIDATE_COLUMNS)
        .sort("group", "leg_y", "leg_x")
    )
