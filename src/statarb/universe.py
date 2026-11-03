"""Point-in-time universe at each formation date (FR-04, DQ-02, DQ-05, D-06).

Everything here comes from a ``FormationView``, so nothing dated on or after ``T`` can reach
it. Securities that later delist stay in: the view does not know about them, apart from
the deliberate ``survivors_only`` comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from statarb.config import UniverseConfig
    from statarb.windows import FormationView

UNIVERSE_COLUMNS = ("security_id", "rank", "mktcap_usd", "dollar_volume", "traded_frac", "close")


def traded_expr() -> pl.Expr:
    """A traded day: a positive-volume close that is not a quote midpoint (DQ-05)."""
    return (
        pl.col("close").is_not_null()
        & (pl.col("close") > 0)
        & (pl.col("volume").fill_null(0.0) > 0)
        & ~pl.col("close_is_mid").fill_null(False)
    )


def eligible_universe(view: FormationView, cfg: UniverseConfig) -> pl.DataFrame:
    """Eligible securities at ``T`` ranked by size, largest first.

    Rules, all on formation data:

    - listed on ``T - 1`` as one of ``cfg.security_types`` on one of ``cfg.exchanges``;
    - traded on every formation session if ``require_full_history`` (GGR), else on at least
      ``min_traded_frac`` of them;
    - close on ``T - 1`` at least ``min_price_usd`` and size at least ``min_mktcap_usd``,
      when set;
    - the ``top_n`` by ``rank_by`` on ``T - 1``: market cap, or the median formation dollar
      volume where no share counts exist (prototype data).

    Returns
    -------
    polars.DataFrame
        ``security_id, rank, mktcap_usd, dollar_volume, traded_frac, close``, by rank.
    """
    sessions = view.sessions()
    last = view.last_session()
    listed = view.master().filter(
        pl.col("security_type").is_in(list(cfg.security_types)),
        pl.col("exchange").is_in(list(cfg.exchanges)),
    )
    prices = view.prices(["close", "volume", "close_is_mid", "mktcap_usd"]).join(
        listed.select("security_id"), on="security_id", how="semi"
    )
    stats = (
        prices.sort("security_id", "date")
        .group_by("security_id", maintain_order=True)
        .agg(
            traded_expr().sum().alias("n_traded"),
            pl.col("date").last().alias("last_date"),
            pl.col("close").last().alias("close"),
            pl.col("mktcap_usd").last().alias("mktcap_usd"),
            (pl.col("close") * pl.col("volume")).median().alias("dollar_volume"),
        )
        .filter(pl.col("last_date") == last)
        .with_columns((pl.col("n_traded") / len(sessions)).alias("traded_frac"))
    )
    if cfg.require_full_history:
        stats = stats.filter(pl.col("n_traded") == len(sessions))
    else:
        stats = stats.filter(pl.col("traded_frac") >= cfg.min_traded_frac)
    if cfg.min_price_usd is not None:
        stats = stats.filter(pl.col("close") >= cfg.min_price_usd)
    if cfg.min_mktcap_usd is not None:
        stats = stats.filter(pl.col("mktcap_usd") >= cfg.min_mktcap_usd)
    if cfg.survivors_only:
        stats = stats.filter(~pl.col("security_id").is_in(list(view.delisted_ids())))
    key = cfg.rank_by
    ranked = (
        stats.filter(pl.col(key).is_not_null() & (pl.col(key) > 0))
        .sort([key, "security_id"], descending=[True, False])
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("rank"))
    )
    if cfg.top_n is not None:
        ranked = ranked.filter(pl.col("rank") <= cfg.top_n)
    return ranked.select(UNIVERSE_COLUMNS)
