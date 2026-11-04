"""Probabilistic and deflated Sharpe ratios from the trial log (FR-28, AC-H-10).

Bailey and Lopez de Prado (2014): with per-period Sharpe ``SR`` over ``n`` observations,
skewness ``g3`` and non-excess kurtosis ``g4``,

    PSR(SR*) = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)),

and the deflated Sharpe ratio is PSR at the expected maximum Sharpe of ``N`` trials with
variance ``V`` across them,

    SR0 = sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))).

N counts every row of the family, failed runs included; V uses the valid trials' series
(the backtest-overfitting-lab contract).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def psr(sr: float, n_obs: int, skewness: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio, per-period inputs, ``kurt`` non-excess."""
    denom = 1.0 - skewness * sr + (kurt - 1.0) / 4.0 * sr * sr
    if n_obs < 2 or denom <= 0:
        return math.nan
    return float(norm.cdf((sr - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """SR0: the expected maximum per-period Sharpe of ``n_trials`` skill-free trials."""
    if n_trials < 2:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(max(var_trial_sr, 0.0)) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, var_trial_sr: float
) -> float:
    """Spec interface: PSR against the expected maximum of ``n_trials`` trials."""
    return psr(sr, n_obs, skew, kurt, expected_max_sharpe(n_trials, var_trial_sr))


def family_table(log_path: Path, headline: dict[str, str]) -> pl.DataFrame:
    """PSR and DSR for each family's headline run, with N and V from the trial log.

    ``headline`` maps family to the run id the memo reports.
    """
    log = pl.read_csv(
        log_path,
        infer_schema_length=None,
        schema_overrides={"sharpe_annual": pl.Float64, "n_obs": pl.Int64},
    )
    rows = []
    for family, part in log.group_by("strategy_family", maintain_order=True):
        name = str(family[0])
        series = {}
        for run_id, path in (
            part.filter(pl.col("n_obs") > 1).select("run_id", "returns_path").iter_rows()
        ):
            file = log_path.parent / str(path)
            if file.exists():
                series[run_id] = pl.read_parquet(file, columns=["ret"])["ret"].to_numpy()
        srs = np.array([s.mean() / s.std(ddof=1) for s in series.values() if s.std(ddof=1) > 0])
        n_trials = part.height
        var = float(srs.var(ddof=1)) if srs.size >= 2 else math.nan
        chosen = headline.get(name)
        if chosen is None or chosen not in series:
            continue
        values = series[chosen]
        sr = float(values.mean() / values.std(ddof=1))
        g3 = float(skew(values))
        g4 = float(kurtosis(values, fisher=False))
        sr0 = expected_max_sharpe(n_trials, var if math.isfinite(var) else 1.0 / (values.size - 1))
        rows.append(
            {
                "strategy_family": name,
                "run_id": chosen,
                "n_trials": n_trials,
                "valid_trials": int(srs.size),
                "var_trial_sr_annual": var * 252 if math.isfinite(var) else math.nan,
                "sharpe_annual": sr * math.sqrt(252),
                "n_obs": values.size,
                "skew": g3,
                "kurtosis": g4,
                "psr_vs_zero": psr(sr, values.size, g3, g4, 0.0),
                "sr0_annual": sr0 * math.sqrt(252),
                "dsr": psr(sr, values.size, g3, g4, sr0),
            }
        )
    return pl.DataFrame(rows)
