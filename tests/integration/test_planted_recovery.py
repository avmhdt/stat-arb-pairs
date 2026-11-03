"""Spec section 11: EG + BH recovers planted pairs, recall >= 8/10 and FDP <= 0.10 on average."""

from __future__ import annotations

import polars as pl

from fixtures.synth import make_market
from statarb.config import SelectionConfig, UniverseConfig
from statarb.selection.select import select_window
from statarb.windows import FormationView, rolling_windows


def test_eg_bh_recovers_planted_pairs() -> None:
    recalls, fdps = [], []
    for seed in (1, 2, 3, 4, 5):
        market = make_market(seed=seed, years=3)
        window = rolling_windows(market.data.sessions, 12, 6, 1)[0]
        view = FormationView(market.data, window, UniverseConfig(top_n=None, min_price_usd=None))
        tested = select_window(view, SelectionConfig()).tested
        planted = {tuple(sorted((p.leg_y, p.leg_x))) for p in market.planted}
        found = {
            tuple(row)
            for row in tested.filter(pl.col("fdr_reject")).select("leg_y", "leg_x").iter_rows()
        }
        recalls.append(len(found & planted) / len(planted))
        fdps.append(len(found - planted) / max(len(found), 1))
    assert sum(recalls) / len(recalls) >= 0.8, recalls
    assert sum(fdps) / len(fdps) <= 0.10, fdps
