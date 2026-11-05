# Findings for the research memo

Collected as milestones land; the README memo (M7) draws on these. Every number comes from a file under `results/tables/` named beside it.

## GGR (2006) replication, Sharadar 1998-2002 (AC-H-01)

Source: `results/tables/ggr_replication.csv`, `results/tables/ggr_bounce.csv`, `reports/figures/ggr_replication.png`. Partial replication: trading months 1999-01 to 2002-12 only (D-03), gross of costs as in the paper.

- **What matches.** Every portfolio, capital basis and execution rule has the paper's sign. The top-20 fully invested portfolio earns 1.01% a month without waiting (paper 1.44%) and 0.70% with the one-day wait (paper 0.90%): ratios 0.70 and 0.78, inside the +/-30% band, with Newey-West t-statistics of 3.0 and 2.8.
- **What does not.** The paper's ordering, top 5 and top 20 above pairs 101-120, fails: pairs 101-120 earn 2.42% a month without waiting and 1.84% with it, two to two-and-a-third times the paper's values and well above the top pairs. Accepted as a finding by the owner (2026-11-03); the check stays a strict expected failure in `tests/unit/test_ggr_verdict.py`.
- **GGR saw the same late in their sample.** Their Table 8 for 1989-2002 (wait one day, fully invested) has the top 5 at 0.217% a month, below pairs 101-120 at 0.327%, with the top 20 at 0.375%. The full-sample ordering is an average over a period in which it did not always hold.
- **Bid-ask bounce does not explain it.** Pairs 101-120 are the most volatile (formation spread sigma 0.044 against 0.027 for the top 20 and 0.021 for the top 5; mean leg daily volatility 2.3% against 1.5% and 1.8%; Corwin-Schultz half-spread 41 bp against 28 bp and 32 bp). Waiting a day costs them the most in basis points (58 bp a month against 32 bp and 11 bp), but no more as a share (24% against 31% for the top 20), and after waiting they still earn 1.84% a month against 0.70%. More volatile pairs diverge further and more often (1.9 round trips per pair against 1.7 and 1.6), and over 1999-2002 that paid.

## Multiple testing (FR-30)

Source: `results/tables/fdr_by_window.csv`, `docs/multiple_testing.md`. Over 319 headline windows, 7.1M Engle-Granger tests give 1,012 BH discoveries and 288 BY discoveries; about 1.3 pairs a window survive the half-life band and Johansen confirmation, against `top_k = 20`.

## Test properties at n = 252 (AC-W-05, AC-W-07)

Owner ruling 2026-11-03: accepted as findings, thresholds unchanged. Engle-Granger rejects 6.55% to 7.3% of independent random-walk pairs at 5% and has power of about 0.5 at a 10-day half-life. The OU half-life has a median signed error of -3.9% and a median absolute error of 12.2%, against a Cramer-Rao floor near 12%.

## Kalman hedge at the spec's delta

With delta = 1e-4 on log prices, the filter absorbs most of the spread and its z-score is underdispersed, so the cointegration strategy rarely reaches |z| >= 2. Not tuned, and no delta sweep was run: the M6 grids vary break policies and sizing at the spec's delta.

## Avellaneda-Lee against simple pairs (AC-H-03)

Source: `results/tables/al_vs_pairs.csv`, `al_vs_pairs_by_year.csv`, `al_vs_pairs_sharpe_diff.csv`, `al_vs_paper.csv`, `reports/figures/al_vs_pairs.png`. Universe: market cap at least USD 1bn at each T; trading days 1999-01 to 2025-12 for every strategy.

- **A-L's own setup reproduces their headline.** Trading at the signal day's close with 5 bp per side, as A-L do, the 15-factor PCA strategy has a Sharpe ratio of 1.42 over 1999-2007, against their 1.44 for 1997-2007. The ETF variant reaches 0.84 against their 1.1 (which used synthetic sector indices before 2002). Over 2003-2007 both are weaker than A-L report (0.15 against 0.9 for PCA, 0.23 against 0.6 for ETFs), with the same sign.
- **Waiting a day removes most of it.** With this project's one-day execution lag (D-13), the PCA Sharpe falls to 0.51 over 1999-2007 and turns negative over 2003-2007 (-0.74); the ETF variant to 0.36 and -0.12. The s-score signal works on a horizon of days, so most of its edge sits in the first day.
- **The project's cost model is fatal for it.** A-L turn their capital over about 270 times a year (holding about 13 trading days). At Corwin-Schultz half-spreads with the 2 bp floor plus square-root impact, both variants lose essentially everything (Sharpe -8 for PCA, -4 for ETFs, 1999-2025). Simple pairs turn over 0.3 (cointegration) to 13 (distance) times a year and lose far less. (Recomputed in M6 after the 10% ADV cap was enforced: PCA -7.95, ETFs -4.21, distance pairs -1.90, cointegration pairs -0.61; the cap changed every model-cost Sharpe here by less than 0.03.)
- **At A-L's 5 bp, A-L and simple pairs cannot be told apart.** Every Sharpe difference between an A-L variant and a pairs strategy with the same costs and lag has a 95% stationary-bootstrap interval that spans zero (for example PCA minus distance, 1999-2025: -0.27, interval -0.77 to 0.19). At the project's cost model A-L is worse than both pairs strategies, with intervals well below zero.

## Avellaneda-Lee kappa recovery (AC-W-14)

On a synthetic three-factor market with OU residuals at kappa = 25 a year in 30% of the names, the rank correlation between estimated and true kappa is about 0.23 from 60-day PCA residuals, against the AC's 0.6. From the true residuals, with no PCA at all, a 60-day AR(1) reaches only about 0.47 (0.68 at 120 days, 0.79 at 252, close to the ceiling for a 30/70 split): 60 days cannot tell kappa = 25 from a random walk. Averaging each name's estimate over the sample gives 0.59 across ten seeds (0.47 to 0.70). The strategy part of AC-W-14 passes (gross Sharpe above 1 on the synthetic market). Accepted as a finding by the owner (2026-11-03); the 60-day window and the threshold are unchanged.

## GGR decay, 1999-2025 (AC-H-02)

Source: `results/tables/decay_by_period.csv`, `decay_trend.csv`, `reports/figures/ggr_decay.png`. GGR's distance rule (top 20, one-day wait, committed capital, gross of costs) over every trading month from 1999-01 to 2025-12; the cointegration headline (EG-Kalman, net of the project's costs) beside it. Sharadar starts in 1998, so FR-21's 1962-1988 period is empty and 1989-2002 holds 1999-2002 only.

- **GGR's edge shrank but did not vanish.** Mean monthly return 0.45% in 1999-2002 (Newey-West t 3.9, Sharpe 1.58), 0.19% in 2003-2009 (t 2.2, Sharpe 0.97), 0.06% in 2010-2017 (t 1.2) and 0.16% in 2018-2025 (t 1.3). After 2002 the mean is 0.13% a month, 95% interval 0.03% to 0.24%: positive, about a third of the 1999-2002 level, and gross of costs.
- **The time trend is negative but not significant:** -0.08 percentage points a month per decade, Newey-West t -1.09. Most of the fall is a step after 2002, not a steady slope.
- **Same direction as Do and Faff (2010):** GGR returns fell after the 1990s, with the strategy still earning in the turbulent 2000-2002 and 2007-2009 markets (here 2003-2009 is the strongest post-2002 period).
- The cointegration strategy is flat to negative after costs in every sub-period (post-2002 mean -0.005% a month, interval -0.010% to 0.001%).

## Out-of-sample cointegration (AC-H-04)

Source: `results/tables/oos_persistence.csv`, `reports/figures/oos_persistence.png`. Engle-Granger (both orderings, the larger p-value) on each selected pair's six-month trading window and on the 12 months after it, against one matched unselected candidate per selected pair.

- **Selected cointegrated pairs do not stay cointegrated.** In the trading window 5.9% of the 421 FDR-selected pairs have p < 0.05 (Wilson 95% interval 4.1% to 8.6%), against 5.0% for the matched baseline and the 5% null rate; 6.4% in the next 12 months (baseline 4.2%). Selection at q = 0.05 on 12 months of formation predicts essentially nothing about the next six.
- **Distance pairs persist slightly more:** 7.6% of 6,346 selected pairs (7.0% to 8.3%) against 4.9% for unselected pairs ranked 21-200; 7.4% in the next 12 months. Statistically above the baseline, economically small.
- Both are limited by the test's power on 126 sessions (AC-W-05: about 0.5 even at 252 sessions and a 10-day half-life), so a low share is partly the test, but the selected and baseline shares are the same test on the same windows.

## Cointegration breaks (AC-H-08)

Source: `results/tables/break_policies.csv`. Four policies on the cointegration headline's identical windows and selections; a break is a rolling 60-session Engle-Granger p-value above 0.20.

| Policy | Annual return | Sharpe | Trades | Mean trade | 5% CVaR per trade | Mean holding days |
|---|---|---|---|---|---|---|
| hold | -0.060% | -0.33 | 196 | -0.96% | -14.7% | 6.0 |
| stop_only (default) | -0.059% | -0.33 | 190 | -0.95% | -14.7% | 6.0 |
| close_on_break | -0.004% | -0.08 | 22 | -0.52% | -5.4% | 2.5 |
| close_and_blacklist | -0.002% | -0.06 | 9 | -0.54% | -5.6% | 2.4 |

- **Recommended: close on a break.** It cuts the tail loss per trade from -14.7% to -5.4% and the mean loss per trade by about half. Its better Sharpe comes mostly from trading much less (22 trades against 190): a flagged day opens nothing, and at 60 sessions and p > 0.20 the test flags enough days to block most entries. Blacklisting adds little over closing.
- The 4-sigma stop almost never binds before the time stop or the window end: `hold` and `stop_only` differ by six trades.

## Survivorship bias (AC-W-09)

Source: `results/tables/survivorship_bias.csv`; `pytest -m data tests/data/test_survivorship.py` passes (every formation year's universe holds later-delisted names). Same configs, universe restricted to names still listed at the end of 2025.

- GGR distance (gross): survivors-only earns 0.21 percentage points a year more (2.39% against 2.18%; Sharpe 0.68 against 0.52), the expected direction.
- Cointegration (net): survivors-only loses 0.02 points a year more (-0.08% against -0.06%; Sharpe -0.49 against -0.33). The restricted universe changes which pairs pass FDR in each window (1.9 a window against 1.3), so the comparison is of two different small selections more than of delisting returns.

## Borrow: assumed against IBKR (FR-34)

Source: `results/tables/borrow_assumed_vs_ibkr.csv`; IBKR's US short-stock file of 2026-11-03 matched to listed common stocks ranked by market cap on 2026-10-30.

- Top 500: median 0.28%, mean 0.55% a year against the assumed 0.25% (mean 2.2 times the assumption).
- 501-1,000: median 0.36%, mean 0.53% against the assumed 1% (half).
- The rest: median 0.54%, 90th percentile 18.9%, mean 11.0% against the assumed 5%: a heavy right tail of hard-to-borrow names the flat rate misses.
- One snapshot from 2026 says nothing about borrow in 1999-2025; it bounds how wrong the assumed schedule might be today.

## Break-even cost (AC-H-05)

Source: `results/tables/break_even.csv`, `reports/figures/break_even.png`. Mean gross return less borrow, over mean one-way turnover, in bps per side; 95% stationary-bootstrap intervals (block 20, 2,000 resamples); 1999-2025 unless noted.

| Strategy | Break-even, borrow x1 | 95% interval | Modelled cost per side | Turnover a year |
|---|---|---|---|---|
| GGR distance | 2.2 | -4.8 to 9.2 | 47 | 9.7 |
| EG-Kalman | 36.8 | -9.7 to 84.9 | 86 | 0.12 |
| A-L PCA | 3.7 | 1.7 to 5.7 | 44 | 272 |
| A-L ETF | 4.6 | 1.9 to 7.4 | 38 | 304 |

- **Every strategy breaks even far below what it would pay.** The modelled cost per side is 8 to 21 times the break-even for GGR and A-L; for the cointegration strategy it is 2.3 times, with a break-even interval that includes zero.
- **Borrow matters for GGR, not for A-L's hedged books.** GGR's break-even is 10.2 bp with no borrow, 2.2 at the assumed schedule, -5.9 at twice it; A-L PCA's goes from 5.3 to 3.7 to 2.1.
- **Decay in costs, too.** GGR's break-even is 17 bp in 1999-2002 and negative after 2009; A-L PCA's 12.7 bp in 1999-2002 and 1.5 to 3.3 bp after.
- The assumed borrow schedule against IBKR's 2026 file is in the borrow section below.

## Capacity (AC-H-06)

Source: `results/tables/capacity.csv`, `capacity_summary.csv`, `reports/figures/capacity.png`. Last five years (2021-2025); entries capped at 10% of ADV in the runs themselves (USD 10m); square-root impact rescaled with AUM.

- **With the full cost model there is no capacity to measure:** every strategy loses money at zero impact over 2021-2025 (net Sharpe at zero impact: GGR -1.13, EG-Kalman -0.57, A-L PCA -6.7, A-L ETF -3.0), so neither the half-Sharpe nor the zero-return AUM exists.
- **Impact alone (gross less borrow and impact, no commission or spread):** A-L PCA has a 0.59 Sharpe at zero impact, halves at USD 0.34m and reaches zero return at USD 1.3m (Y = 1; at Y = 0.5 USD 1.3m and 5.4m; at Y = 2 below USD 0.1m, the grid's floor, and USD 0.34m). A-L ETF: 0.16, USD 0.13m and 0.50m. GGR and EG-Kalman are already negative at zero impact on this basis in 2021-2025.
- **The ADV cap binds early for GGR:** at USD 10m 15% of GGR's trades would exceed 10% of ADV, at USD 100m 66%, at USD 1bn 98%; for the A-L strategies (stocks above USD 1bn) 3% at USD 100m and 41 to 46% at USD 1bn.

## Crowding (AC-H-07)

Source: `results/tables/crowding.csv`, `crowding_rolling.csv`, `crowding_overlap.csv`, `crowding_trends.csv`, `reports/figures/crowding_*.png`. Monthly net returns against Ken French's short-term reversal factor and Mkt-RF (Kenneth R. French Data Library).

- **Reversal exposure is modest.** Correlation with the reversal factor: GGR 0.27, EG-Kalman 0.19, A-L PCA 0.18, A-L ETF 0.22; betas 0.03, 0.002, 0.24 and 0.37. The A-L books also carry a market beta of 0.18 to 0.24 net of their hedges.
- **August 2007:** over 6-10 August GGR earned 0.10% (0.7 trailing daily volatilities), A-L PCA 0.81% (2.0) and A-L ETF 5.4% (8.5); no strategy here lost in the quant unwind week. The cointegration strategy held nothing.
- **March 2020** was the bad month: GGR lost 3.5% (36 trailing volatilities, its trailing volatility being very low), A-L PCA 6.1% (9.3); A-L ETF gained 4.5%; the cointegration strategy held nothing.
- **The methods trade different names.** Yearly Jaccard overlap of traded names: GGR against either A-L variant 0.085, GGR against EG-Kalman 0.006, EG-Kalman against A-L 0.007; the two A-L variants 0.998 (same universe). Concentration: the 10 most-traded names carry 11% of GGR's traded gross, 24% of EG-Kalman's (few pairs) and under 1% of A-L's.
- **Trends:** GGR's selected pairs have a formation half-life of about 6 to 8 sessions in most years (16 to 19 in 2004-2006) and enter 1.3 to 1.9 times per pair a year, peaking in 2007-2008 (2.0, 2.4) and 2020 (2.2). No secular trend in either.

## Sizing by half-life (AC-H-09)

Source: `results/tables/sizing_half_life.csv`, `reports/figures/sizing_half_life.png`. Equal gross, inverse half-life (1 / formation half-life) and volatility targeting (1 / spread sigma) on identical selections, net of costs.

- **Sizing does not rescue either strategy.** GGR distance: Sharpe -2.01, -2.01 and -2.10; net P&L per capital-day -3.0, -3.1 and -3.4 bp. EG-Kalman: -0.33, -0.32 and -0.35.
- **Faster pairs are barely more profitable per capital-day here.** For GGR at equal gross, net P&L per capital-day is -2.9 bp for half-lives of 1-5 sessions and -3.1 to -3.3 bp for 5-63; inverse half-life gives -2.8 bp for 1-5 and -7.3 bp for the 16 trades at 30-63; volatility targeting -3.2 bp for 1-5 and -3.0 bp for 5-15; the selected cointegrated pairs almost all have half-lives under 5 sessions (186 of 190 trades), so the buckets above are nearly empty. Inverse-half-life sizing moves capital toward the 1-5 bucket and changes little because that bucket already holds most trades.

## Trial count and deflated Sharpe (AC-H-10)

Source: `results/tables/deflated_sharpe.csv` (`uv run statarb evaluate deflated-sharpe`, rebuilt by `statarb report`); N counts every trial-log row of the family, failed, aborted, superseded, reproduction and prototype runs included (82 in all on 2026-11-05). The implementation matches backtest-overfitting-lab's `deflated_sharpe_ratio` to 1e-9 (`tests/unit/test_economics.py`).

| Family | Trials | Headline Sharpe | PSR vs 0 | SR0 (annual) | DSR |
|---|---|---|---|---|---|
| pairs_distance | 33 | -2.01 | 0.00 | 3.75 | 0.00 |
| pairs_coint | 31 | -0.33 | 0.04 | 0.64 | 0.00 |
| al_pca | 9 | -7.95 | 0.00 | 6.76 | 0.00 |
| al_etf | 9 | -4.21 | 0.00 | 3.70 | 0.00 |

Headline runs are net of this project's costs with the one-day lag. None has a positive Sharpe, so deflation changes no conclusion; the expected maximum of the trials is large because the families mix gross and net runs.

## Kalman delta and the static hedge (M7)

Source: `results/tables/kalman_delta.csv`. The cointegration headline on identical selections with four hedges.

| Hedge | Sharpe | Trades | Mean holding days | Mean trade |
|---|---|---|---|---|
| Static formation OLS | -1.42 | 1,261 | 8.3 | -1.03% |
| Kalman, delta 1e-5 | -0.43 | 614 | 7.3 | -0.59% |
| Kalman, delta 1e-4 (spec) | -0.33 | 190 | 6.0 | -0.95% |
| Kalman, delta 1e-3 | -0.42 | 35 | 4.6 | -2.75% |

A larger delta lets beta absorb more of the spread, so z reaches 2 less often and trades fall from 1,261 (static) to 35. Fewer trades lose less in total, but no setting makes money; the spec's delta is kept rather than the best of four chosen on the test data.

## Reproduction (AC-W-11)

`uv run statarb reproduce --profile headline` on 2026-11-04 downloaded nothing, kept the build (its manifest matched), passed DQ, reran all 28 study runs in 2 h 1 min with 3 workers and rebuilt every table. Each rerun's daily returns are identical, column for column, to the run it replaced, including runs first made in M4 and M5 under earlier engine code; only run ids changed in the tables. A second call with `--reuse` reused those 28 and added the three hedge runs above.

The prototype profile (Stooq, survivorship-biased, 2010-2025, `results/tables/prototype/`) was reproduced twice, with 3 workers and with 1: identical `returns.parquet` sha256 for both study runs.
