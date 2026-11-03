"""Property tests: SSD symmetric and non-negative; BH monotone in q and a superset of BY."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from statarb.selection.distance import distance_ssd
from statarb.selection.fdr import fdr_reject

prices = arrays(
    np.float64,
    st.tuples(st.integers(2, 40), st.integers(2, 12)),
    elements=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False),
)
pvalues = arrays(np.float64, st.integers(1, 300), elements=st.floats(0.0, 1.0, allow_nan=False))


@settings(max_examples=80, deadline=None)
@given(prices)
def test_ssd_symmetric_non_negative(panel: np.ndarray) -> None:
    ssd = distance_ssd(panel)
    assert np.all(ssd >= 0)
    assert np.array_equal(ssd, ssd.T)
    assert np.all(np.diag(ssd) == 0)


@settings(max_examples=150, deadline=None)
@given(pvalues, st.floats(0.001, 0.3), st.floats(0.001, 0.3))
def test_bh_monotone_in_q_and_contains_by(p: np.ndarray, q1: float, q2: float) -> None:
    low, high = sorted((q1, q2))
    bh_low, bh_high = fdr_reject(p, low, "bh"), fdr_reject(p, high, "bh")
    assert np.all(bh_high[bh_low])
    assert np.all(bh_low[fdr_reject(p, low, "by")])
