"""AC-W-08: the state machine, table-driven, plus property checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from statarb.trading.rules import (
    ENTRY,
    EXIT,
    STOP,
    TIME_STOP,
    WINDOW_END,
    max_holding_days,
    position_state_machine,
    state_machine,
)

N = math.nan
CASES = [
    # name, z, exit_mode, max_holding, positions, reasons
    (
        "short spread on high z, exit at cross",
        [0, 2.1, 1.0, 0.1, -0.2, 0],
        "cross_zero",
        None,
        [0, -1, -1, -1, 0, 0],
        [0, ENTRY, 0, 0, EXIT, 0],
    ),
    (
        "long spread on low z",
        [0, -2.5, -1.0, 0.0, 1.0, 0],
        "cross_zero",
        None,
        [0, 1, 1, 0, 0, 0],
        [0, ENTRY, 0, EXIT, 0, 0],
    ),
    (
        "band exit",
        [0, 2.2, 1.0, 0.4, 0.3, 0],
        "band",
        None,
        [0, -1, -1, 0, 0, 0],
        [0, ENTRY, 0, EXIT, 0, 0],
    ),
    (
        "stop, then no re-entry",
        [0, 2.5, 4.2, 2.5, 0, 2.6, 0],
        "cross_zero",
        None,
        [0, -1, 0, 0, 0, 0, 0],
        [0, ENTRY, STOP, 0, 0, 0, 0],
    ),
    (
        "re-entry after exit",
        [2.1, -0.1, 2.3, -0.5, 0],
        "cross_zero",
        None,
        [-1, 0, -1, 0, 0],
        [ENTRY, EXIT, ENTRY, EXIT, 0],
    ),
    (
        "no entry when already past stop",
        [0, 4.5, 3.0, 0],
        "cross_zero",
        None,
        [0, 0, -1, 0],
        [0, 0, ENTRY, EXIT],
    ),
    (
        "time stop after max holding",
        [2.0, 1.9, 1.8, 1.7, 1.6, 0],
        "cross_zero",
        3,
        [-1, -1, -1, 0, 0, 0],
        [ENTRY, 0, 0, TIME_STOP, 0, 0],
    ),
    (
        "cross from far side exits only",
        [-2.2, 2.4, 2.4, 0],
        "cross_zero",
        None,
        [1, 0, -1, 0],
        [ENTRY, EXIT, ENTRY, EXIT],
    ),
    (
        "NaN holds the state",
        [2.2, N, N, -0.1, 0],
        "cross_zero",
        None,
        [-1, -1, -1, 0, 0],
        [ENTRY, 0, 0, EXIT, 0],
    ),
    (
        "window end closes",
        [0, 2.3, 1.5, 1.2],
        "cross_zero",
        None,
        [0, -1, -1, 0],
        [0, ENTRY, 0, WINDOW_END],
    ),
    ("entry exactly at threshold", [2.0, 0.0], "cross_zero", None, [-1, 0], [ENTRY, EXIT]),
    (
        "below threshold never enters",
        [1.99, -1.99, 1.5, 0],
        "cross_zero",
        None,
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ),
]


@pytest.mark.parametrize(
    ("name", "z", "mode", "hold", "positions", "reasons"), CASES, ids=[c[0] for c in CASES]
)
def test_transitions(
    name: str, z: list[float], mode: str, hold: int | None, positions: list[int], reasons: list[int]
) -> None:
    got, why = state_machine(np.array(z, dtype=float), 2.0, 0.5, 4.0, mode, hold)  # type: ignore[arg-type]
    assert got.tolist() == positions, name
    assert why.tolist() == reasons, name


def test_spec_interface_and_validation() -> None:
    z = np.array([0, 2.5, 0.0])
    assert position_state_machine(z, 2.0, 0.5, 4.0, "cross_zero", None).tolist() == [0, -1, 0]
    assert state_machine(z, 2.0, 0.5, None, "cross_zero", None)[0].tolist() == [0, -1, 0]
    with pytest.raises(ValueError, match="exit_mode"):
        state_machine(z, 2.0, 0.5, 4.0, "trailing", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entry < stop"):
        state_machine(z, 4.0, 0.5, 2.0, "band", None)
    assert max_holding_days(10.0, 3.0) == 30
    assert max_holding_days(0.2, 3.0) == 1
    assert max_holding_days(None, 3.0) is None
    assert max_holding_days(5.0, None) is None


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.one_of(st.floats(-6, 6), st.just(math.nan)), min_size=1, max_size=80),
    st.sampled_from(["cross_zero", "band"]),
    st.one_of(st.none(), st.integers(1, 10)),
)
def test_never_flips_and_ends_flat(z: list[float], mode: str, hold: int | None) -> None:
    positions, _ = state_machine(np.array(z), 2.0, 0.5, 4.0, mode, hold)  # type: ignore[arg-type]
    assert positions[-1] == 0
    steps = np.diff(positions.astype(int))
    assert np.all(np.abs(steps) <= 1)
