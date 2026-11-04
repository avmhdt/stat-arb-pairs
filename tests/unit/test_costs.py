"""AC-W-08: twelve hand-computed cost examples, each to 1e-12."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from statarb.costs import (
    CostParams,
    LegCostInputs,
    borrow_accrual,
    borrow_cost,
    corwin_schultz,
    half_spread,
    participation,
    participation_scale,
    trade_cost,
    trade_costs,
)
from statarb.trading.sizing import leg_weights, pair_scales

LEG = LegCostInputs(
    half_spread=0.0005, adv_usd=50_000_000.0, sigma_daily=0.02, fee_rate_annual=0.01
)
PARAMS = CostParams(commission_bps=1.0, impact_y=1.0, aum_usd=10_000_000.0)


def test_01_commission_one_bp() -> None:
    assert trade_cost(0.05, LEG, PARAMS)[0] == pytest.approx(0.05 * 1e-4, abs=1e-12)


def test_02_commission_is_side_symmetric_and_linear() -> None:
    double = CostParams(commission_bps=2.0, impact_on=False)
    assert trade_cost(-0.05, LEG, double)[0] == pytest.approx(
        2 * trade_cost(0.05, LEG, PARAMS)[0], abs=1e-12
    )


def test_03_half_spread_cost() -> None:
    assert trade_cost(0.05, LEG, PARAMS)[1] == pytest.approx(0.05 * 0.0005, abs=1e-12)


def test_04_square_root_impact() -> None:
    # q = 0.05 * 10m = 500k USD; q / ADV = 0.01; impact = 0.05 * 1.0 * 0.02 * 0.1 = 1e-4.
    assert trade_cost(0.05, LEG, PARAMS)[2] == pytest.approx(1e-4, abs=1e-12)
    assert participation(0.05, LEG, PARAMS) == pytest.approx(0.01, abs=1e-12)


def test_05_impact_off_and_zero_adv() -> None:
    assert trade_cost(0.05, LEG, CostParams(impact_on=False))[2] == 0.0
    dry = LegCostInputs(0.0005, 0.0, 0.02, 0.01)
    assert trade_cost(0.05, dry, PARAMS)[2] == 0.0
    assert participation(0.05, dry, PARAMS) == math.inf


def test_06_borrow_one_weekday() -> None:
    # 0.025 short at 1%/yr for one day ACT/360.
    assert borrow_cost(-0.025, 0.01, date(2024, 3, 5), date(2024, 3, 6)) == pytest.approx(
        0.025 * 0.01 / 360, abs=1e-12
    )


def test_07_borrow_over_a_weekend_pays_three_days() -> None:
    friday, monday = date(2024, 3, 8), date(2024, 3, 11)
    assert borrow_cost(-0.025, 0.05, friday, monday) == pytest.approx(
        0.025 * 0.05 * 3 / 360, abs=1e-12
    )


def test_08_borrow_multiplier() -> None:
    once = borrow_cost(-0.025, 0.0025, date(2024, 3, 5), date(2024, 3, 6))
    assert borrow_cost(-0.025, 0.0025, date(2024, 3, 5), date(2024, 3, 6), 4.0) == pytest.approx(
        4 * once, abs=1e-12
    )


def test_09_corwin_schultz_two_days() -> None:
    high, low = np.array([10.5, 10.6]), np.array([10.0, 10.1])
    beta = math.log(10.5 / 10.0) ** 2 + math.log(10.6 / 10.1) ** 2
    gamma = math.log(10.6 / 10.0) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (math.sqrt(2) - 1) * math.sqrt(beta) / k - math.sqrt(gamma / k)
    expected = max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0.0)
    assert corwin_schultz(high, low)[0] == pytest.approx(expected, abs=1e-12)


def test_10_half_spread_floor_and_models() -> None:
    flat_range = np.array([10.0, 10.0, 10.0])
    assert half_spread(
        flat_range, flat_range, None, None, "quotes_then_corwin_schultz", 2.0
    ) == pytest.approx(2e-4, abs=1e-12)
    assert half_spread(
        flat_range, flat_range, None, None, "flat", 2.0, flat_bps=5.0
    ) == pytest.approx(5e-4, abs=1e-12)
    assert half_spread(flat_range, flat_range, None, None, "none", 2.0) == 0.0


def test_11_half_spread_from_quotes() -> None:
    bid, ask = np.array([9.99, 19.98]), np.array([10.01, 20.02])
    expected = ((0.02 / 20.0) + (0.04 / 40.0)) / 2
    got = half_spread(
        np.array([1.0, 1.0]), np.array([1.0, 1.0]), bid, ask, "quotes_then_corwin_schultz", 0.0
    )
    assert got == pytest.approx(expected, abs=1e-12)


def test_12_frame_interfaces() -> None:
    trades = pl.DataFrame({"security_id": ["a", "b"], "notional_usd": [500_000.0, -250_000.0]})
    spreads = pl.DataFrame({"security_id": ["a", "b"], "half_spread": [0.0005, 0.001]})
    adv = pl.DataFrame(
        {"security_id": ["a", "b"], "adv_usd": [50e6, 25e6], "sigma_daily": [0.02, 0.01]}
    )
    out = trade_costs(trades, spreads, adv, PARAMS)
    assert out["cost_commission"].to_list() == pytest.approx([50.0, 25.0], abs=1e-12)
    assert out["cost_spread"].to_list() == pytest.approx([250.0, 250.0], abs=1e-12)
    assert out["cost_impact"].to_list() == pytest.approx([1000.0, 250.0], abs=1e-9)
    positions = pl.DataFrame(
        {
            "security_id": ["a", "b"],
            "date": [date(2024, 3, 11)] * 2,
            "prev_date": [date(2024, 3, 8)] * 2,
            "notional_usd": [100.0, -360_000.0],
        }
    )
    fees = pl.DataFrame({"security_id": ["a", "b"], "fee_rate_annual": [0.01, 0.01]})
    assert borrow_accrual(positions, fees)["cost_borrow"].to_list() == pytest.approx(
        [0.0, 30.0], abs=1e-12
    )


def test_sizing() -> None:
    assert leg_weights(1, 1.0, 0.1, True) == (0.05, -0.05)
    assert leg_weights(-1, 1.5, 0.1, False) == pytest.approx((-0.04, 0.06))
    assert leg_weights(0, 1.5, 0.1, False) == (0.0, 0.0)
    assert pair_scales([10.0, 5.0], [0.1, 0.1], "equal_gross") == [1.0, 1.0]
    assert pair_scales([10.0, 5.0], [0.1, 0.1], "inverse_half_life") == pytest.approx(
        [2 / 3, 4 / 3]
    )
    assert pair_scales([None, 5.0], [0.1, 0.2], "vol_target") == pytest.approx([4 / 3, 2 / 3])
    assert pair_scales([None, None], [0.1, 0.2], "inverse_half_life") == [1.0, 1.0]
    assert pair_scales([], [], "vol_target") == []


def test_participation_cap_scales_the_larger_leg_to_the_limit() -> None:
    thin = LegCostInputs(0.0005, 2_000_000.0, 0.02, 0.01)
    # 0.05 of 10m = 500k: 1% of the liquid leg's ADV, 25% of the thin leg's.
    scale = participation_scale((0.05, -0.05), (LEG, thin), PARAMS)
    assert scale == pytest.approx(0.4)
    assert participation(0.05 * scale, thin, PARAMS) == pytest.approx(0.10)
    assert participation_scale((0.001, -0.001), (LEG, thin), PARAMS) == 1.0
    dry = LegCostInputs(0.0005, 0.0, 0.02, 0.01)
    assert participation_scale((0.01,), (dry,), PARAMS) == 0.0
    off = CostParams(impact_on=False, aum_usd=10_000_000.0)
    assert participation_scale((0.05, -0.05), (LEG, thin), off) == 1.0
