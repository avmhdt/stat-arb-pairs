"""AC-W-04: BH and BY match statsmodels exactly and control the realised FDR."""

from __future__ import annotations

import math

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from statarb.selection.cointegration import engle_granger_batch
from statarb.selection.fdr import fdr_reject


@pytest.mark.parametrize("q", [0.05, 0.10])
def test_matches_multipletests_on_1000_vectors(q: float) -> None:
    rng = np.random.default_rng(99)
    for _ in range(1000):
        m = int(rng.integers(1, 400))
        p = rng.beta(0.3, 1.0, m) if rng.random() < 0.5 else rng.uniform(0, 1, m)
        if rng.random() < 0.2:
            p = np.round(p, 3)  # ties
        for method, name in (("bh", "fdr_bh"), ("by", "fdr_by")):
            expected = multipletests(p, alpha=q, method=name)[0]
            assert np.array_equal(fdr_reject(p, q, method), expected)  # type: ignore[arg-type]


def test_realised_fdr_by_simulation() -> None:
    """200 families of 1,000 independent null pairs and 50 cointegrated ones (n = 252).

    The p-values are the ones selection feeds to FDR: the larger of the two orderings.
    """
    rng = np.random.default_rng(2026)
    n, nulls, real, q = 252, 1000, 50, 0.05
    b = 2.0 ** (-1.0 / 5.0)
    fdp = {"bh": [], "by": []}
    for _ in range(200):
        walks = np.cumsum(rng.normal(0, 0.01, (n, 2 * nulls + real)), axis=0)
        shocks = rng.normal(0, 0.01, (n, real))
        spread = np.empty((n, real))
        spread[0] = shocks[0] / math.sqrt(1 - b * b)
        for t in range(1, n):
            spread[t] = b * spread[t - 1] + shocks[t]
        y = 1.1 * walks[:, 2 * nulls :] + spread
        logp = np.column_stack([walks, y])
        pairs = np.vstack(
            [
                np.arange(2 * nulls).reshape(nulls, 2),
                np.column_stack(
                    [
                        np.arange(2 * nulls + real, 2 * nulls + 2 * real),
                        np.arange(2 * nulls, 2 * nulls + real),
                    ]
                ),
            ]
        )
        p = np.maximum(
            engle_granger_batch(logp, pairs)[:, 1],
            engle_granger_batch(logp, pairs[:, ::-1])[:, 1],
        )
        for method in ("bh", "by"):
            reject = fdr_reject(p, q, method)  # type: ignore[arg-type]
            false = int(reject[:nulls].sum())
            fdp[method].append(false / max(int(reject.sum()), 1))
    assert float(np.mean(fdp["bh"])) <= q + 0.01
    assert float(np.mean(fdp["by"])) <= q
