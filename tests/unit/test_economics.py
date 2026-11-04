"""M6 evaluation: decay, break-even, capacity, survivorship, policies, sizing, persistence
statistics, deflated Sharpe, crowding, and the French and IBKR parsers."""

from __future__ import annotations

import io
import math
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from statarb.data.providers import french, ibkr
from statarb.evaluation import crowding, dsr, economics
from statarb.evaluation.persistence import matched_baseline, wilson
from statarb.reports import economics as report


def _days(start: date, n: int) -> list[date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _params(family: str = "pairs_coint", policy: str = "stop_only", scheme: str = "equal") -> dict:
    return {
        "run": {"strategy_family": family, "name": "t"},
        "costs": {"aum_usd": 1e7, "impact": {"y_coeff": 1.0}},
        "rules": {"break_policy": policy},
        "sizing": {"scheme": scheme},
    }


def _run(
    seed: int = 0,
    n: int = 2600,
    start: date = date(2003, 1, 1),
    drift: float = 4e-4,
    family: str = "pairs_coint",
    policy: str = "stop_only",
    scheme: str = "equal",
    vol: float = 0.004,
) -> economics.Run:
    rng = np.random.default_rng(seed)
    days = _days(start, n)
    gross = drift + rng.normal(0, vol, n)
    turnover = np.full(n, 0.2)
    trading = turnover * 5e-4
    impact = turnover * 1e-4
    borrow = np.full(n, 2e-5)
    returns = pl.DataFrame(
        {
            "date": days,
            "ret_gross": gross,
            "cost_trading": trading,
            "cost_borrow": borrow,
            "ret": gross - trading - borrow,
            "turnover": turnover,
            "cost_impact": impact,
            "windows_active": np.full(n, 6.0),
        }
    )
    ids = [f"S{i}" for i in range(30)]
    trades = pl.DataFrame(
        {
            "date_exec": [days[i] for i in range(0, n, 10)],
            "security_id": [ids[k % 30] for k in range(len(range(0, n, 10)))],
            "reason": ["entry"] * len(range(0, n, 10)),
            "notional_usd": [1e5 * (1 + (k % 30)) for k in range(len(range(0, n, 10)))],
            "participation": np.full(len(range(0, n, 10)), 0.002),
            "cost_impact": np.full(len(range(0, n, 10)), 10.0),
            "window_id": [days[i].replace(day=1).isoformat() for i in range(0, n, 10)],
            "pair_id": [f"P{i % 7}" for i in range(0, n, 10)],
        }
    )
    m = 200
    trips = pl.DataFrame(
        {
            "half_life_days": rng.uniform(1, 63, m),
            "holding_days": rng.integers(1, 40, m).astype(np.int32),
            "slot_capital": np.full(m, 1e5),
            "pnl": rng.normal(200, 1000, m),
            "ret_on_slot": rng.normal(0.002, 0.01, m),
            "exit_reason": ["break" if i % 5 == 0 else "exit" for i in range(m)],
        }
    )
    return economics.Run(
        f"run-{seed}-{policy}-{scheme}",
        _params(family, policy, scheme),
        returns,
        trades,
        trips,
        f"{family}-{seed}",
    )


def test_decay_recovers_a_planted_trend() -> None:
    run = _run(n=5000, start=date(1999, 1, 1), drift=0.0)
    months = economics.monthly(run.returns.select("date", "ret"))
    del months
    t = np.arange(run.returns.height)
    falling = run.returns.with_columns(
        pl.Series(
            "ret", 1.2e-3 * (1 - t / t.size) + np.random.default_rng(1).normal(0, 2e-3, t.size)
        )
    )
    table, roll, summary = economics.decay(
        economics.Run(run.run_id, run.params, falling, run.trades, run.roundtrips, "x")
    )
    assert set(table["period"]) >= {"2003-2009", "2010-2017"}
    assert summary["trend_per_decade"] < 0
    assert summary["trend_t_newey_west"] < -3
    assert summary["post_2002_ci_low"] < summary["post_2002_mean_monthly"]
    assert roll["sharpe_36m"].drop_nulls().len() == roll.height - 35


def test_newey_west_slope_matches_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 1, 120)) * 0.1 + rng.normal(0, 1, 120)
    x = np.arange(120) / 120.0
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    ours = economics._newey_west_slope_t(design, y - design @ coef, coef)
    fit = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False})
    assert ours == pytest.approx(fit.tvalues[1], rel=1e-9)


def test_break_even_is_edge_over_turnover() -> None:
    run = _run()
    table = economics.break_even(run, n_boot=200)
    row = table.filter((pl.col("period") == "all") & (pl.col("borrow_multiplier") == 1.0)).row(
        0, named=True
    )
    r = run.returns
    expected = (r["ret_gross"] - r["cost_borrow"]).mean() / r["turnover"].mean() * 1e4
    assert row["break_even_bps"] == pytest.approx(expected)
    assert row["ci_low"] < row["break_even_bps"] < row["ci_high"]
    assert row["modelled_bps"] == pytest.approx(5.0)
    zero = table.filter((pl.col("period") == "all") & (pl.col("borrow_multiplier") == 0.0))
    assert zero["break_even_bps"][0] > row["break_even_bps"]


def test_daily_turnover_falls_back_to_trades() -> None:
    run = _run()
    old = economics.Run(
        run.run_id,
        run.params,
        run.returns.drop("turnover", "cost_impact"),
        run.trades,
        run.roundtrips,
        "x",
    )
    flows = economics.daily_turnover(old)
    first = run.trades.row(0, named=True)
    day = flows.filter(pl.col("date") == first["date_exec"]).row(0, named=True)
    assert day["turnover"] == pytest.approx(first["notional_usd"] / 1e7 / 6.0)
    assert flows["turnover"].sum() > 0


def test_capacity_degrades_with_aum_and_impact() -> None:
    grid, summary = economics.capacity(_run(drift=2e-4, vol=5e-4))
    at_y1 = grid.filter((pl.col("y_coeff") == 1.0) & (pl.col("basis") == "all_costs")).sort(
        "aum_usd"
    )
    assert (np.diff(at_y1["net_sharpe"].to_numpy()) < 0).all()
    big = (
        grid.filter((pl.col("aum_usd") == 1e9) & (pl.col("basis") == "all_costs"))
        .sort("y_coeff")["net_sharpe"]
        .to_numpy()
    )
    assert (np.diff(big) < 0).all()
    row = summary.filter((pl.col("y_coeff") == 1.0) & (pl.col("basis") == "all_costs")).row(
        0, named=True
    )
    spread_free = summary.filter(
        (pl.col("y_coeff") == 1.0) & (pl.col("basis") == "impact_only")
    ).row(0, named=True)
    assert spread_free["aum_zero_return_usd"] > row["aum_zero_return_usd"]
    assert 1e7 < row["aum_half_sharpe_usd"] < row["aum_zero_return_usd"]
    assert grid.filter(pl.col("aum_usd") == 1e9)["share_trades_over_cap"][0] == 1.0
    assert grid.filter(pl.col("aum_usd") == 1e6)["share_trades_over_cap"][0] == 0.0
    _, losing = economics.capacity(_run(drift=-2e-4, vol=5e-4))
    losing = losing.filter(pl.col("basis") == "all_costs")
    assert losing["aum_zero_return_usd"].is_nan().all()
    assert losing["aum_half_sharpe_usd"].is_nan().all()


def test_survivorship_difference() -> None:
    full, surv = _run(0), _run(0)
    shifted = surv.returns.with_columns(pl.col("ret") + 1e-4)
    out = economics.survivorship(
        full, economics.Run("s", surv.params, shifted, surv.trades, surv.roundtrips, "s")
    )
    assert out["difference_annual"] == pytest.approx(1e-4 * 252)


def test_policies_and_sizing_tables() -> None:
    runs = [_run(i, policy=p) for i, p in enumerate(["hold", "stop_only", "close_on_break"])]
    table = economics.policies(runs)
    assert table["policy"].to_list() == ["hold", "stop_only", "close_on_break"]
    assert (table["break_exits"] == 40).all()
    assert (table["cvar5_trade_return"] < table["mean_trade_return"]).all()
    sizing = economics.sizing([_run(0, scheme="equal"), _run(1, scheme="inverse_half_life")])
    assert sizing.height == 10
    every = sizing.filter(pl.col("half_life_bucket") == "all")
    parts = sizing.filter(pl.col("half_life_bucket") != "all")
    assert parts.group_by("scheme").agg(pl.col("trades").sum())["trades"].to_list() == [200, 200]
    assert every["trades"].to_list() == [200, 200]


def test_cvar() -> None:
    values = np.arange(100, dtype=np.float64)
    assert economics.cvar(values, 0.05) == pytest.approx(np.arange(5).mean() + 0.0, abs=0.2)
    assert math.isnan(economics.cvar(np.zeros(0)))


def test_wilson_interval() -> None:
    lo, hi = wilson(5, 100)
    assert lo == pytest.approx(0.02154, abs=1e-4)
    assert hi == pytest.approx(0.11175, abs=1e-4)
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_matched_baseline_draws_within_groups() -> None:
    pairs = pl.DataFrame(
        {
            "window_id": ["2010-01-01"] * 6 + ["2010-07-01"] * 4,
            "group": ["a", "a", "a", "b", "b", "b", "a", "a", "a", "a"],
            "selected": [True, False, False, True, True, False, True, False, False, False],
            "leg_y": list("abcdefghij"),
            "leg_x": list("ABCDEFGHIJ"),
        }
    )
    base = matched_baseline(pairs, seed=0)
    assert not base["selected"].any()
    counts = base.group_by("window_id", "group").len().sort("window_id", "group")
    assert counts["len"].to_list() == [1, 1, 1]


def test_psr_and_dsr_known_example() -> None:
    # Bailey and Lopez de Prado (2014), section 5: SR 2.5 annual, 1,250 days, skew -3,
    # kurtosis 10, 100 trials, V = 0.5 annualised; DSR about 0.90.
    periods = 250
    sr = 2.5 / math.sqrt(periods)
    value = dsr.deflated_sharpe(sr, 1250, -3.0, 10.0, 100, 0.5 / periods)
    assert value == pytest.approx(0.9004, abs=1e-4)
    assert dsr.psr(sr, 1250, -3.0, 10.0) > value
    assert dsr.expected_max_sharpe(1, 1.0) == 0.0
    assert math.isnan(dsr.psr(0.1, 1, 0.0, 3.0))


def test_dsr_family_table(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    (tmp_path / "runs").mkdir()
    for i in range(6):
        path = f"runs/r{i}.parquet"
        pl.DataFrame({"ret": rng.normal(2e-4 * i, 0.01, 500)}).write_parquet(tmp_path / path)
        rows.append(
            {
                "run_id": f"r{i}",
                "strategy_family": "pairs_coint",
                "n_obs": 500,
                "sharpe_annual": 0.1,
                "returns_path": path,
            }
        )
    rows.append(
        {
            "run_id": "dead",
            "strategy_family": "pairs_coint",
            "n_obs": 0,
            "sharpe_annual": None,
            "returns_path": None,
        }
    )
    pl.DataFrame(rows).write_csv(tmp_path / "trial_log.csv")
    table = dsr.family_table(tmp_path / "trial_log.csv", {"pairs_coint": "r5"})
    row = table.row(0, named=True)
    assert row["n_trials"] == 7
    assert row["valid_trials"] == 6
    assert row["dsr"] < row["psr_vs_zero"]


def _factors(run: economics.Run, beta: float) -> pl.DataFrame:
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 0.006, run.returns.height)
    return run.returns.select(
        "date",
        ((pl.col("ret") - 4e-4) / beta + pl.Series(noise) * 0.0).alias("st_rev"),
        pl.Series("mkt_rf", noise),
    )


def test_crowding_exposure_events_overlap_concentration() -> None:
    run = _run(n=5000, start=date(2001, 1, 1))
    exposure, rolling = crowding.factor_exposure(run, _factors(run, 2.0))
    assert exposure["corr_st_rev"] > 0.99
    assert exposure["beta_st_rev"] == pytest.approx(2.0, rel=0.05)
    assert abs(exposure["corr_mkt_rf"]) < 0.2
    assert rolling["beta_st_rev_36m"].drop_nulls().len() > 0
    events = crowding.event_drawdowns(run)
    assert math.isfinite(events["aug_2007_in_trailing_vol"])
    assert math.isnan(events["mar_2020_return"])
    other = _run(1, n=5000, start=date(2001, 1, 1))
    lap = crowding.overlap([run, other])
    assert (lap["jaccard"] == 1.0).all()
    share = crowding.concentration(run)
    weights = np.arange(1, 31, dtype=float)
    assert share == pytest.approx(weights[-10:].sum() / weights.sum(), rel=0.05)


def test_pair_trends() -> None:
    run = _run(n=600, start=date(2010, 1, 1))
    pairs = pl.DataFrame(
        {
            "window_id": ["2010-01-01", "2010-01-01", "2011-01-01"],
            "selected": [True, True, True],
            "ou_half_life_days": [10.0, 20.0, 30.0],
        }
    )
    trend = crowding.pair_trends(run, pairs)
    assert trend.filter(pl.col("year") == 2010)["mean_half_life_days"][0] == 15.0
    assert (trend["entries_per_pair"] > 0).all()


def test_french_parse_and_zip() -> None:
    text = (
        "This file was created by CMPT_ST_RETS\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.10,   -0.25,   -0.27,    0.009\n"
        "19260702,    0.45,   -0.33,   -0.06,    0.009\n"
        "\n Annual Factors\n,Mkt-RF\n1927, 20.0\n"
    )
    frame = french.parse_daily(text)
    assert frame.height == 2
    assert frame["Mkt-RF"][1] == pytest.approx(0.0045)
    assert frame["date"][0] == date(1926, 7, 1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("x.CSV", text)
    assert french.read_zip(buffer.getvalue()) == text
    with pytest.raises(ValueError, match="header"):
        french.parse_daily("nothing here\n")


def test_french_ingest_from_cache(tmp_path: Path) -> None:
    def bundle(text: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("a.CSV", text)
        return buffer.getvalue()

    (tmp_path / french.FILES["st_rev"]).write_bytes(
        bundle(",ST_Rev\n20200102, 1.0\n20200103, -2.0\n")
    )
    (tmp_path / french.FILES["ff3"]).write_bytes(
        bundle(",Mkt-RF,SMB,HML,RF\n20200102, 0.5, 0, 0, 0.01\n20200103, 0.1, 0, 0, 0.01\n")
    )
    out, fetched = french.ingest(tmp_path)
    frame = pl.read_parquet(out)
    assert fetched == 0
    assert frame.columns == ["date", "st_rev", "mkt_rf", "rf"]
    assert frame["st_rev"].to_list() == pytest.approx([0.01, -0.02])


def test_ibkr_parse_and_compare() -> None:
    text = (
        "#BOF|2026.11.03|09:00:00\n"
        "#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|\n"
        "AAA|USD|A Inc|1|US1|3.5|0.25|>10000000|\n"
        "BBB|USD|B Inc|2|US2|-10.0|15.0|5000|\n"
        "CCC|CAD|C Inc|3|CA3|3.0|0.5|100|\n"
        "#EOF|3\n"
    )
    snap = ibkr.parse(text)
    assert snap.height == 3
    assert snap["fee_rate_annual"].to_list() == pytest.approx([0.0025, 0.15, 0.005])
    assert snap["available_shares"][0] == 10_000_000
    latest = pl.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "rank": [1, 1200, 2]})
    schedule = pl.DataFrame(
        {"bucket": ["top500", "top1000", "rest"], "fee_rate_annual": [0.0025, 0.01, 0.05]}
    )
    table = ibkr.compare(snap, latest, schedule)
    assert table["bucket"].to_list() == ["rest", "top500"]
    assert table.filter(pl.col("bucket") == "rest")["mean_over_assumed"][0] == pytest.approx(3.0)


def _write_run(root: Path, run: economics.Run) -> str:
    folder = root / run.run_id
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(run.params), encoding="utf-8")
    run.returns.write_parquet(folder / "returns.parquet")
    run.trades.write_parquet(folder / "trades.parquet")
    run.roundtrips.write_parquet(folder / "roundtrips.parquet")
    (folder / "metrics.json").write_text("{}", encoding="utf-8")
    return run.run_id


def test_incomplete_run_is_refused(tmp_path: Path) -> None:
    rid = _write_run(tmp_path, _run(0, n=300))
    (tmp_path / rid / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        economics.read_run(tmp_path, rid)


def test_reports_write_tables_and_figures(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    tables, figs = tmp_path / "tables", tmp_path / "figs"
    a = _write_run(runs, _run(0, n=5000, start=date(2001, 1, 1), policy="hold"))
    b = _write_run(runs, _run(1, n=5000, start=date(2001, 1, 1), scheme="inverse_half_life"))
    specs = [f"one={a}", b]
    assert report.decay(runs, specs, tables, figs).height > 0
    assert report.break_even_tables(runs, specs, tables, figs).height > 0
    assert report.capacity_tables(runs, specs, tables, figs).height == 72
    assert report.policy_table(runs, specs, tables).height == 2
    assert report.sizing_table(runs, specs, tables, figs).height == 10
    assert report.survivorship_table(runs, [(a, b)], tables).height == 1
    factors = tmp_path / "factors.parquet"
    run = economics.read_run(runs, a)
    _factors(run, 2.0).write_parquet(factors)
    assert report.crowding_tables(runs, factors, specs, tables, figs).height == 2
    for name in (
        "decay_by_period",
        "decay_trend",
        "break_even",
        "capacity",
        "capacity_summary",
        "break_policies",
        "sizing_half_life",
        "survivorship_bias",
        "crowding",
        "crowding_overlap",
    ):
        assert (tables / f"{name}.csv").exists(), name
    for name in (
        "ggr_decay",
        "break_even",
        "capacity",
        "sizing_half_life",
        "crowding_reversal_beta",
    ):
        assert (figs / f"{name}.png").exists(), name
        assert (figs / f"{name}.svg").exists(), name


def test_persistence_separates_cointegrated_from_random_pairs() -> None:
    from statarb.data.market import MarketData
    from statarb.evaluation.persistence import persistence

    rng = np.random.default_rng(4)
    days = _days(date(2010, 1, 1), 600)
    frames = []
    pairs = []
    for k in range(20):
        x = np.cumsum(rng.normal(0, 0.01, len(days)))
        spread = np.zeros(len(days))
        for t in range(1, len(days)):
            spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.004)
        y = x + spread if k < 10 else np.cumsum(rng.normal(0, 0.01, len(days)))
        for name, path in ((f"Y{k}", y), (f"X{k}", x)):
            frames.append(
                pl.DataFrame({"date": days, "security_id": name, "tr_index": np.exp(path)})
            )
        pairs.append(
            {
                "window_id": "2010-07-01",
                "group": "g",
                "leg_y": f"Y{k}",
                "leg_x": f"X{k}",
                "selected": k < 10,
            }
        )
    prices = pl.concat(frames)
    empty = pl.DataFrame()
    data = MarketData(prices, empty, empty, empty, empty, empty)
    rows, table = persistence(data, pl.DataFrame(pairs), "coint", trading_months=6)
    overall = table.filter((pl.col("year") == 0) & (pl.col("horizon") == "p_trading"))
    share = dict(overall.select("sample", "share_p_lt_05").iter_rows())
    assert share["selected"] >= 0.9
    assert share["baseline"] <= 0.3
    assert rows.height == 20
    assert rows["p_next_12m"].drop_nans().len() == 20


def test_degenerate_pairs_are_not_tested() -> None:
    from statarb.evaluation.persistence import _testable

    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 0.01, 100))
    y = np.cumsum(rng.normal(0, 0.01, 100))
    assert _testable(y, x)
    assert not _testable(x.copy(), x)
    assert not _testable(np.full(100, 1.5), x)
    assert not _testable(np.where(np.arange(100) == 3, np.nan, y), x)


@pytest.mark.parametrize(
    ("sr", "n_obs", "skewness", "kurt", "n_trials", "var"),
    [(2.5 / math.sqrt(250), 1250, -3.0, 10.0, 100, 0.5 / 250), (0.03, 6791, 0.4, 9.0, 37, 4e-4)],
)
def test_dsr_matches_backtest_overfitting_lab(
    sr: float, n_obs: int, skewness: float, kurt: float, n_trials: int, var: float
) -> None:
    deflated = pytest.importorskip("overfitlab.deflated")
    theirs = deflated.deflated_sharpe_ratio(
        sr, n_obs, n_trials=n_trials, var_trials=var, skew=skewness, kurtosis=kurt
    )
    ours = dsr.deflated_sharpe(sr, n_obs, skewness, kurt, n_trials, var)
    assert ours == pytest.approx(theirs.dsr, abs=1e-9)
    assert dsr.expected_max_sharpe(n_trials, var) == pytest.approx(theirs.sr0, abs=1e-12)
