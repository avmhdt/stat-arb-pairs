"""FR-03 / AC-W-01: no information from the trading window reaches selection.

M2 runs the harness on a dummy selector that touches every formation accessor. Each strategy
family joins this file before it is used (spec section 16).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fixtures.synth import SynthMarket, make_market
from integration.leakage import PERTURBATIONS, check_no_lookahead, perturb
from statarb.config import UniverseConfig
from statarb.selection.candidates import candidate_pairs
from statarb.windows import FormationView, Window, rolling_windows

UNIVERSE = UniverseConfig(top_n=40, min_price_usd=1.0)


def dummy_selector(view: FormationView) -> tuple[object, ...]:
    """Top pairs by SSD within SIC2, with a formation OLS fit; reads every accessor."""
    candidates = candidate_pairs(view, "sic2")
    prices = view.prices(["tr_index", "close"])
    wide = prices.pivot(on="security_id", index="date", values="tr_index").sort("date")
    first = wide.drop("date").row(0)
    names = wide.columns[1:]
    norm = {name: wide[name].to_numpy() / first[i] for i, name in enumerate(names)}
    scored = []
    for leg_y, leg_x, _ in candidates.iter_rows():
        a, b = norm[leg_y], norm[leg_x]
        scored.append((float(np.sum((a - b) ** 2)), leg_y, leg_x))
    scored.sort()
    fitted = []
    for ssd, leg_y, leg_x in scored[:5]:
        y, x = np.log(norm[leg_y]), np.log(norm[leg_x])
        design = np.column_stack([np.ones_like(x), x])
        (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
        spread = y - alpha - beta * x
        fitted.append((leg_y, leg_x, ssd, float(beta), float(spread.mean()), float(spread.std())))
    etf = view.etf_returns()
    rows = view.query("prices", view.window.formation_start, view.window.formation_end)
    return (
        tuple(view.universe()["security_id"].to_list()),
        tuple(fitted),
        float(etf["ret_total"].sum()),
        rows.height,
        tuple(view.master()["ticker"].to_list()),
    )


def leaky_selector(view: FormationView) -> tuple[object, ...]:
    """Reaches round the view to the full panel: the harness must catch it."""
    everything = view._data.prices  # deliberately leaky fixture
    return (float(everything["close"].sum()),)


@pytest.fixture(scope="module")
def market() -> SynthMarket:
    return make_market(seed=11)


def _windows(market: SynthMarket) -> list[Window]:
    windows = rolling_windows(market.data.sessions, 12, 6, 1)
    return [windows[0], windows[len(windows) // 2], windows[-1]]


def test_trading_window_perturbation_does_not_change_selection(market: SynthMarket) -> None:
    compared = check_no_lookahead(dummy_selector, market.data, _windows(market), UNIVERSE)
    assert compared == 3 * len(PERTURBATIONS)


def test_dummy_selector_finds_planted_pairs(market: SynthMarket) -> None:
    result = dummy_selector(FormationView(market.data, _windows(market)[0], UNIVERSE))
    assert len(result[1]) == 5  # type: ignore[arg-type]


def test_harness_catches_a_leaky_selector(market: SynthMarket) -> None:
    with pytest.raises(AssertionError, match="perturbation changed"):
        check_no_lookahead(leaky_selector, market.data, _windows(market), UNIVERSE)


@pytest.mark.parametrize("how", PERTURBATIONS)
def test_perturbations_change_the_future(market: SynthMarket, how: str) -> None:
    window = _windows(market)[1]
    noisy = perturb(market.data, window, how, seed=3)
    t = window.formation_end
    before = market.data.prices.filter(pl.col("date") < t)
    assert noisy.prices.filter(pl.col("date") < t).equals(before)
    if how == "future_industry":
        assert noisy.industry.height == 2 * market.data.industry.height
    else:
        after = market.data.prices.filter(pl.col("date") >= t)["close"]
        assert not noisy.prices.filter(pl.col("date") >= t)["close"].equals(after)


def test_unknown_perturbation(market: SynthMarket) -> None:
    with pytest.raises(ValueError, match="unknown"):
        perturb(market.data, _windows(market)[0], "bogus", 0)


def _frozen(pairs: object) -> tuple[object, ...]:
    from statarb.selection.pairs import PairSet

    assert isinstance(pairs, PairSet)
    tested = pairs.tested.drop("run_id")
    return (pairs.pairs, tuple(tested.columns), tuple(map(tuple, tested.rows())))


def distance_family(view: FormationView) -> tuple[object, ...]:
    """pairs_distance selection exactly as M3 runs it."""
    from statarb.selection.distance import select_distance

    return _frozen(select_distance(view, top_k=5, store_top=20, run_id="x"))


def coint_family(view: FormationView) -> tuple[object, ...]:
    """pairs_coint selection: EG both orderings, BH and BY, half-life, Johansen."""
    from statarb.config import SelectionConfig
    from statarb.selection.select import select_window

    return _frozen(select_window(view, SelectionConfig(top_k=5), run_id="x"))


@pytest.mark.parametrize(
    "family", [distance_family, coint_family], ids=["pairs_distance", "pairs_coint"]
)
def test_pairs_selection_is_unchanged_by_trading_data(market: SynthMarket, family) -> None:  # type: ignore[no-untyped-def]
    universe = UniverseConfig(top_n=None, min_price_usd=None)
    assert check_no_lookahead(family, market.data, _windows(market), universe) == 3 * len(
        PERTURBATIONS
    )


def _frozen_family(method: str):  # type: ignore[no-untyped-def]
    """The whole frozen window: selection, Kalman state through formation, leg cost inputs."""
    from pathlib import Path

    from statarb.config import RunConfig
    from statarb.trading.engine import borrow_schedule, freeze

    family = "pairs_distance" if method == "distance" else "pairs_coint"
    cfg = RunConfig.model_validate(
        {
            "run": {"name": "leak", "strategy_family": family, "data_profile": "prototype"},
            "data": {"provider": "free"},
            "universe": {"top_n": None, "min_price_usd": None},
            "selection": {
                "method": method,
                "grouping": "none" if method == "distance" else "sic2",
                "top_k": 5,
            },
        }
    )
    schedule = borrow_schedule(
        Path(__file__).resolve().parents[2] / "configs" / "borrow_schedule.yaml"
    )

    def selector(view: FormationView) -> tuple[object, ...]:
        frozen = freeze(view, cfg, schedule, run_id="x")
        tested = frozen.pairset.tested.drop("run_id")
        return (frozen.pairs, tuple(sorted(frozen.legs.items())), tuple(map(tuple, tested.rows())))

    return selector


@pytest.mark.parametrize(
    "method", ["distance", "engle_granger"], ids=["pairs_distance", "pairs_coint"]
)
def test_frozen_window_is_unchanged_by_trading_data(market: SynthMarket, method: str) -> None:
    universe = UniverseConfig(top_n=None, min_price_usd=None)
    selector = _frozen_family(method)
    assert check_no_lookahead(selector, market.data, _windows(market), universe) == 3 * len(
        PERTURBATIONS
    )
