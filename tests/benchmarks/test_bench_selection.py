"""NFR-01 benchmarks on synthetic data (local only, marker ``slow``).

Run with ``uv run pytest -m slow tests/benchmarks --benchmark-json reports/benchmarks.json``.
Targets: 499,500 SSDs for 1,000 stocks in under 5 s; 20,000 Engle-Granger tests in under
60 s, single process.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from statarb.selection.cointegration import engle_granger_batch
from statarb.selection.distance import distance_ssd, ordered_pairs

pytestmark = pytest.mark.slow


def test_ssd_1000_stocks(benchmark: Any) -> None:
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + rng.normal(0, 0.01, (252, 1000)), axis=0)
    allowed = np.ones((1000, 1000), dtype=bool)

    def run() -> None:
        ordered_pairs(distance_ssd(prices), allowed, 20_000)

    benchmark.pedantic(run, rounds=5, warmup_rounds=1)
    assert benchmark.stats.stats.max < 5.0


def test_engle_granger_20000_pairs(benchmark: Any) -> None:
    rng = np.random.default_rng(1)
    logp = np.cumsum(rng.normal(0, 0.01, (252, 1000)), axis=0)
    pairs = rng.integers(0, 1000, (20_000, 2))
    pairs[:, 1] = (pairs[:, 0] + 1 + pairs[:, 1] % 999) % 1000
    engle_granger_batch(logp, pairs[:10])  # compile first

    def run() -> None:
        engle_granger_batch(logp, pairs)
        engle_granger_batch(logp, pairs[:, ::-1])

    benchmark.pedantic(run, rounds=3, warmup_rounds=0)
    assert benchmark.stats.stats.max < 60.0
