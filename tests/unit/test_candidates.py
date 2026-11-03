from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fixtures.synth import SynthMarket, make_market
from statarb.config import UniverseConfig
from statarb.industry import group_keys, load_sic_etf_map
from statarb.selection.candidates import candidate_pairs
from statarb.windows import FormationView, rolling_windows

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def market() -> SynthMarket:
    return make_market(seed=5)


def _view(market: SynthMarket, cfg: UniverseConfig, index: int = 0) -> FormationView:
    windows = rolling_windows(market.data.sessions, 12, 6, 1)
    return FormationView(market.data, windows[index], cfg)


def test_universe_ranks_by_size(market: SynthMarket) -> None:
    universe = _view(market, UniverseConfig(top_n=10, min_price_usd=None)).universe()
    assert universe.height == 10
    assert universe["rank"].to_list() == list(range(1, 11))
    caps = universe["mktcap_usd"].to_list()
    assert caps == sorted(caps, reverse=True)


def test_full_history_rule_drops_thin_names(market: SynthMarket) -> None:
    loose = _view(market, UniverseConfig(top_n=None, min_price_usd=None)).universe()
    strict = _view(
        market, UniverseConfig(top_n=None, min_price_usd=None, require_full_history=True)
    ).universe()
    assert market.thin in loose["security_id"].to_list()
    assert market.thin not in strict["security_id"].to_list()


def test_delisted_names_stay_until_they_delist(market: SynthMarket) -> None:
    windows = rolling_windows(market.data.sessions, 12, 6, 1)
    cfg = UniverseConfig(top_n=None, min_price_usd=None)
    first = FormationView(market.data, windows[0], cfg).universe()["security_id"].to_list()
    assert set(market.delisted) <= set(first)
    survivors = FormationView(
        market.data, windows[0], cfg.model_copy(update={"survivors_only": True})
    ).universe()
    assert not set(market.delisted) & set(survivors["security_id"].to_list())
    last = FormationView(market.data, windows[-1], cfg).universe()["security_id"].to_list()
    assert not set(market.delisted) & set(last)


def test_price_size_and_type_filters(market: SynthMarket) -> None:
    rich = _view(market, UniverseConfig(top_n=None, min_price_usd=1e6)).universe()
    assert rich.is_empty()
    big = _view(market, UniverseConfig(top_n=None, min_price_usd=None, min_mktcap_usd=1e11))
    assert big.universe()["mktcap_usd"].min() is None or big.universe()["mktcap_usd"].min() >= 1e11  # type: ignore[operator]
    etfs = _view(market, UniverseConfig(security_types=("etf",), top_n=None)).universe()
    assert etfs.is_empty()
    by_volume = _view(market, UniverseConfig(rank_by="dollar_volume", top_n=5)).universe()
    volumes = by_volume["dollar_volume"].to_list()
    assert volumes == sorted(volumes, reverse=True)


def test_group_keys() -> None:
    industry = pl.DataFrame({"security_id": ["a", "b", "c", "d"], "sic": [2834, 2899, 7372, None]})
    assert group_keys(industry, "sic2")["group"].to_list() == ["28", "28", "73"]
    assert group_keys(industry, "sic3")["group"].to_list() == ["283", "289", "737"]
    assert group_keys(industry, "none")["group"].to_list() == ["all"] * 4
    sic_map = load_sic_etf_map(REPO / "configs" / "sic_to_etf.yaml")
    sectors = group_keys(industry, "etf_sector", sic_map)
    assert sectors["group"].to_list() == ["XLV", "XLB", "XLK"]
    with pytest.raises(ValueError, match="map"):
        group_keys(industry, "etf_sector")


def test_candidates_within_sic2(market: SynthMarket) -> None:
    view = _view(market, UniverseConfig(top_n=None, min_price_usd=None))
    pairs = candidate_pairs(view, "sic2")
    assert (pairs["leg_y"] < pairs["leg_x"]).all()
    sic = dict(view.industry().select("security_id", "sic").iter_rows())
    for leg_y, leg_x, group in pairs.iter_rows():
        assert sic[leg_y] // 100 == sic[leg_x] // 100 == int(group)
    members = view.universe().height
    assert candidate_pairs(view, "none").height == members * (members - 1) // 2
    planted = {
        (p.leg_x, p.leg_y) if p.leg_x < p.leg_y else (p.leg_y, p.leg_x) for p in market.planted
    }
    assert planted <= set(zip(pairs["leg_y"], pairs["leg_x"], strict=True))


def test_sic_etf_map() -> None:
    sic_map = load_sic_etf_map(REPO / "configs" / "sic_to_etf.yaml")
    assert sic_map.etf_for(1311) == "XLE"
    assert sic_map.etf_for(6798) == "XLRE"
    assert sic_map.etf_for(4813) == "XLC"
    assert sic_map.etf_for(None) is None
    assert sic_map.etf_for(9999) is None
    assert "SPY" in sic_map.etfs
    from datetime import date

    inception = {"XLC": date(2018, 6, 19), "XLK": date(1998, 12, 22)}
    assert sic_map.usable("XLC", inception, date(2010, 1, 1)) == "XLK"
    assert sic_map.usable("XLC", inception, date(2020, 1, 1)) == "XLC"
    assert sic_map.usable("XLK", inception, date(1998, 1, 1)) is None


def test_bad_sic_range(tmp_path: Path) -> None:
    path = tmp_path / "map.yaml"
    path.write_text("market: SPY\nranges:\n  - [200, 100, XLB]\n")
    with pytest.raises(ValueError, match="empty"):
        load_sic_etf_map(path)
