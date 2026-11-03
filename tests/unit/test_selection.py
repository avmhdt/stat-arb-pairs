from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fixtures.synth import SynthMarket, make_market
from statarb.config import SelectionConfig, UniverseConfig
from statarb.models.ou import OUFailure, ar1_fit, fit_ou, fit_ou_detail
from statarb.selection.distance import distance_ssd, ordered_pairs, select_distance
from statarb.selection.fdr import expected_false_raw, fdr_reject
from statarb.selection.pairs import PAIRS_SELECTED, greedy_unique
from statarb.selection.select import fdr_summary, select_window, windows_between
from statarb.windows import FormationView, rolling_windows

UNIVERSE = UniverseConfig(top_n=None, min_price_usd=None)


@pytest.fixture(scope="module")
def market() -> SynthMarket:
    return make_market(seed=11)


def _view(market: SynthMarket, cfg: UniverseConfig = UNIVERSE, index: int = 0) -> FormationView:
    return FormationView(market.data, rolling_windows(market.data.sessions, 12, 6, 1)[index], cfg)


def test_fdr_hand_case() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert fdr_reject(p, 0.05, "bh").tolist() == [True, True] + [False] * 8
    assert fdr_reject(p[::-1], 0.05, "bh").tolist() == ([False] * 8 + [True, True])
    assert fdr_reject(np.array([0.001, np.nan]), 0.05, "bh").tolist() == [True, False]
    assert fdr_reject(np.array([]), 0.05, "by").size == 0
    with pytest.raises(ValueError, match="unknown"):
        fdr_reject(p, 0.05, "holm")  # type: ignore[arg-type]
    assert expected_false_raw(20_000) == 1000.0


def test_ou_mapping() -> None:
    rng = np.random.default_rng(0)
    b, mu, sd = 0.9, 0.5, 0.01
    x = np.empty(20_000)
    x[0] = mu
    for t in range(1, x.size):
        x[t] = mu + b * (x[t - 1] - mu) + rng.normal(0, sd)
    fitted = fit_ou(x)
    assert fitted is not None
    assert fitted.half_life_days == pytest.approx(math.log(2) / -math.log(b), rel=0.05)
    assert fitted.mu == pytest.approx(mu, abs=0.01)
    assert fitted.sigma_eq == pytest.approx(sd / math.sqrt(1 - b * b), rel=0.05)
    assert fitted.kappa == pytest.approx(-math.log(b) * 252, rel=0.05)
    assert fitted.sigma == pytest.approx(fitted.sigma_eq * math.sqrt(2 * fitted.kappa))
    mle = fit_ou(x, method="mle")
    assert mle is not None and mle.half_life_days == pytest.approx(fitted.half_life_days, rel=0.01)


def test_ou_failures() -> None:
    walk = np.cumsum(np.random.default_rng(1).normal(0.001, 0.01, 500)) + 5.0
    assert fit_ou(np.exp(walk)) is None or fit_ou(np.exp(walk)).half_life_days > 50  # type: ignore[union-attr]
    flip = np.array([1.0, -1.0] * 50) + np.random.default_rng(2).normal(0, 0.01, 100)
    assert isinstance(fit_ou_detail(flip), OUFailure)
    assert isinstance(ar1_fit(np.array([1.0, 2.0])), OUFailure)
    assert isinstance(fit_ou_detail(np.ones(10)), OUFailure)
    assert fit_ou(np.array([1.0, np.nan, 2.0, 3.0, 4.0])) is None


def test_ssd_identity_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    prices = np.cumprod(1 + rng.normal(0, 0.01, (60, 8)), axis=0)
    brute = ((prices[:, :, None] - prices[:, None, :]) ** 2).sum(axis=0)
    assert np.allclose(distance_ssd(prices), brute, rtol=1e-10, atol=1e-12)


def test_ordered_pairs_and_greedy() -> None:
    ssd = np.array([[0, 3, 1, 5], [3, 0, 2, 4], [1, 2, 0, 6], [5, 4, 6, 0]], dtype=float)
    allowed = np.ones((4, 4), dtype=bool)
    rows, cols = ordered_pairs(ssd, allowed, 3)
    assert list(zip(rows.tolist(), cols.tolist(), strict=True)) == [(0, 2), (1, 2), (0, 1)]
    names = [("a", "c"), ("b", "c"), ("a", "b"), ("b", "d")]
    assert greedy_unique(names, 5, 1) == [0, 3]
    assert greedy_unique(names, 2, None) == [0, 1]


def test_select_distance_on_synthetic(market: SynthMarket) -> None:
    pairs = select_distance(_view(market), top_k=5, store_top=12, run_id="r")
    assert [p.rank for p in pairs.pairs] == [1, 2, 3, 4, 5]
    legs = [leg for p in pairs.pairs for leg in (p.leg_y, p.leg_x)]
    assert len(legs) == len(set(legs))
    assert pairs.tested.height == 12
    assert pairs.tested.columns == list(PAIRS_SELECTED)
    assert pairs.tested["run_id"].unique().to_list() == ["r"]
    m = pairs.tested["fdr_m"][0]
    n = len(set(pairs.tested["leg_y"]) | set(pairs.tested["leg_x"]))
    assert m >= n
    later = select_distance(_view(market), top_k=3, rank_from=4)
    assert [p.rank for p in later.pairs] == [4, 5, 6]
    grouped = select_distance(_view(market), top_k=5, grouping="sic2")
    assert grouped.tested["fdr_m"][0] < m
    assert all(len(g) == 2 for g in grouped.tested["group"].to_list())


def test_select_cointegration_on_synthetic(market: SynthMarket) -> None:
    cfg = SelectionConfig(top_k=5)
    pairs = select_window(_view(market), cfg, run_id="r")
    tested = pairs.tested
    assert tested.columns == list(PAIRS_SELECTED)
    assert tested["fdr_m"].unique().to_list() == [tested.height]
    selected = tested.filter(pl.col("selected"))
    assert selected.height == len(pairs.pairs) <= 5
    assert selected["fdr_reject"].all()
    assert selected["ou_half_life_days"].is_between(1, 63).all()
    assert (selected["johansen_trace_r0"] > selected["johansen_cv95_r0"]).all()
    larger = tested.select(pl.max_horizontal("eg_pvalue_yx", "eg_pvalue_xy")).to_series()
    assert (tested["eg_pvalue"] == larger).all()
    assert tested.filter(pl.col("fdr_reject_by"))["fdr_reject"].all()
    no_johansen = select_window(
        _view(market),
        cfg.model_copy(
            update={"johansen_confirm": cfg.johansen_confirm.model_copy(update={"enabled": False})}
        ),
    )
    assert no_johansen.tested["johansen_trace_r0"].null_count() == no_johansen.tested.height
    summary = fdr_summary(tested)
    assert summary.row(0, named=True)["m"] == tested.height
    assert summary.row(0, named=True)["traded"] == len(pairs.pairs)
    assert "security_id" not in summary.columns and "leg_y" not in summary.columns


def test_selection_without_discoveries(market: SynthMarket) -> None:
    strict = SelectionConfig(fdr={"q": 1e-12})  # type: ignore[arg-type]
    pairs = select_window(_view(market), strict)
    assert pairs.pairs == ()
    assert pairs.tested["ou_half_life_days"].null_count() == pairs.tested.height


def test_windows_between(market: SynthMarket) -> None:
    windows = rolling_windows(market.data.sessions, 12, 6, 1)
    chosen = windows_between(windows, windows[2].window_id, windows[4].window_id)
    assert [w.window_id for w in chosen] == [w.window_id for w in windows[2:5]]
    assert windows_between(windows, None, None) == windows
