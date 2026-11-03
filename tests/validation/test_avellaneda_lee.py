"""AC-W-14: PCA residual s-scores recover planted OU names; the strategy earns on synthetic data."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from fixtures.synth_al import AlMarket, make_al_market
from statarb.config import RunConfig
from statarb.models.residuals import eigenportfolios, regression_residuals, s_scores
from statarb.trading.backtest import run_backtest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def market() -> AlMarket:
    return make_al_market()


def _cfg(family: str = "al_pca") -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"name": "al", "strategy_family": family, "data_profile": "prototype"},
            "data": {"provider": "free", "start": "2015-01-01", "end": "2017-12-31"},
            "universe": {"top_n": None, "min_price_usd": None, "min_mktcap_usd": 1e9},
            "windows": {
                "formation_months": 12,
                "trading_months": 1,
                "step_months": 1,
                "execution_lag_days": 1,
            },
            "al": {"factors": "pca" if family == "al_pca" else "etf", "n_factors": 3},
            "costs": {
                "commission_bps": 0.0,
                "half_spread": {"model": "none"},
                "impact": {"model": "none"},
                "borrow": {"multiplier": 0.0},
            },
        }
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC-W-14 asks for a rank correlation >= 0.6 between estimated and true kappa (25/yr "
        "for 30% of names). At A-L's 60-day window it is about 0.23 from PCA residuals; from "
        "the true residuals, with no PCA at all, it is about 0.47 (0.68 at 120 days, 0.79 at "
        "252), so the 60-day estimate cannot separate kappa = 25 from a random walk. Owner "
        "ruling 2026-11-03: accepted as a finding; threshold and window unchanged."
    ),
)
def test_pca_kappa_recovers_planted_names(market: AlMarket) -> None:
    prices = market.data.prices.pivot(on="security_id", index="date", values="ret_total").sort(
        "date"
    )
    ids = prices.columns[1:]
    returns = prices.drop("date").to_numpy()
    truth = np.array([market.true_kappa[sid] for sid in ids])
    correlations = []
    for end in range(300, returns.shape[0], 20):
        window = returns[end - 252 : end]
        weights, _ = eigenportfolios(window, 3)
        recent = window[-60:]
        resid, _ = regression_residuals(recent, recent @ weights)
        _, kappa = s_scores(resid, kappa_min=0.0)
        correlations.append(spearmanr(np.nan_to_num(kappa), truth).statistic)
    assert float(np.mean(correlations)) >= 0.6


def test_strategy_earns_gross_on_synthetic_data(market: AlMarket, tmp_path: Path) -> None:
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    result = run_backtest(_cfg(), data=market.data, results_dir=tmp_path)
    assert result.metrics["gross_sharpe_annual"] > 1.0
    assert result.trades.height > 0
    assert (
        result.returns["ret"]
        == result.returns["ret_gross"]
        - result.returns["cost_trading"]
        - result.returns["cost_borrow"]
    ).all()
    etf = run_backtest(_cfg("al_etf"), data=market.data, results_dir=tmp_path)
    assert etf.trades["security_id"].is_in(["SPY", "XLK"]).any()
