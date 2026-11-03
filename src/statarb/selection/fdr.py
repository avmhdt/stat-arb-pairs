"""Benjamini-Hochberg and Benjamini-Yekutieli step-up procedures (FR-11, D-05).

The family is every candidate pair tested in a window, not the ones that survive a filter
chosen after seeing p-values (spec section 16). Decisions match
``statsmodels.stats.multitest.multipletests(..., method="fdr_bh" | "fdr_by")`` exactly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def fdr_reject(pvalues: F64, q: float, method: Literal["bh", "by"]) -> npt.NDArray[np.bool_]:
    """Reject decisions in the input order.

    BH is valid under independence and positive regression dependence. BY divides the
    level by ``sum(1/i)`` and is valid under any dependence, which matters because pairs
    that share a leg are dependent. A NaN p-value (an untestable pair) counts in the family
    and is never rejected.
    """
    p = np.where(np.isnan(np.asarray(pvalues, dtype=np.float64)), 1.0, pvalues)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=np.bool_)
    order = np.argsort(p)
    ecdf = np.arange(1, m + 1) / float(m)
    if method == "by":
        ecdf = ecdf / np.sum(1.0 / np.arange(1, m + 1))
    elif method != "bh":
        raise ValueError(f"unknown FDR method {method!r}")
    below = p[order] <= ecdf * q
    sorted_reject = np.zeros(m, dtype=np.bool_)
    if below.any():
        sorted_reject[: int(np.nonzero(below)[0].max()) + 1] = True
    reject = np.empty(m, dtype=np.bool_)
    reject[order] = sorted_reject
    return reject


def expected_false_raw(m: int, alpha: float = 0.05) -> float:
    """False rejections expected at raw ``alpha`` if every pair in the family were null."""
    return alpha * m
