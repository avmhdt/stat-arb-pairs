"""Stationary bootstrap (Politis and Romano 1994) for dependent daily returns."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def stationary_indices(
    n: int, mean_block: float, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """One resample's indices: blocks of geometric length with mean ``mean_block``, wrapping."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    restart = rng.random(n) < 1.0 / mean_block
    restart[0] = True
    first = np.nonzero(restart)[0]
    block = np.cumsum(restart) - 1
    offset = np.arange(n) - first[block]
    starts = rng.integers(0, n, first.size)
    return ((starts[block] + offset) % n).astype(np.int64)


def sharpe(values: F64, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio, 0 for a flat series."""
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)) / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0


def sharpe_difference_ci(
    a: F64, b: F64, n_boot: int = 2000, block: float = 20.0, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Sharpe of ``a`` minus Sharpe of ``b`` on the same days, with a paired bootstrap CI."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("series must be aligned")
    point = sharpe(x) - sharpe(y)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        index = stationary_indices(x.size, block, rng)
        draws[i] = sharpe(x[index]) - sharpe(y[index])
    tail = (1.0 - level) / 2.0
    return point, float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))
