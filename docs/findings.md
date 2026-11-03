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

With delta = 1e-4 on log prices, the filter absorbs most of the spread and its z-score is underdispersed, so the cointegration strategy rarely reaches |z| >= 2. Not tuned; delta is examined in the M6 sweeps.

## Avellaneda-Lee against simple pairs (AC-H-03)

Source: `results/tables/al_vs_pairs.csv`, `al_vs_pairs_by_year.csv`, `al_vs_pairs_sharpe_diff.csv`, `al_vs_paper.csv`, `reports/figures/al_vs_pairs.png`. Universe: market cap at least USD 1bn at each T; trading days 1999-01 to 2025-12 for every strategy.

- **A-L's own setup reproduces their headline.** Trading at the signal day's close with 5 bp per side, as A-L do, the 15-factor PCA strategy has a Sharpe ratio of 1.42 over 1999-2007, against their 1.44 for 1997-2007. The ETF variant reaches 0.84 against their 1.1 (which used synthetic sector indices before 2002). Over 2003-2007 both are weaker than A-L report (0.15 against 0.9 for PCA, 0.23 against 0.6 for ETFs), with the same sign.
- **Waiting a day removes most of it.** With this project's one-day execution lag (D-13), the PCA Sharpe falls to 0.51 over 1999-2007 and turns negative over 2003-2007 (-0.74); the ETF variant to 0.36 and -0.12. The s-score signal works on a horizon of days, so most of its edge sits in the first day.
- **The project's cost model is fatal for it.** A-L turn their capital over about 270 times a year (holding about 13 trading days). At Corwin-Schultz half-spreads with the 2 bp floor plus square-root impact, both variants lose essentially everything (Sharpe -8 for PCA, -4 for ETFs, 1999-2025). Simple pairs turn over 0.3 (cointegration) to 13 (distance) times a year and lose far less.
- **At A-L's 5 bp, A-L and simple pairs cannot be told apart.** Every Sharpe difference between an A-L variant and a pairs strategy with the same costs and lag has a 95% stationary-bootstrap interval that spans zero (for example PCA minus distance, 1999-2025: -0.27, interval -0.77 to 0.19). At the project's cost model A-L is worse than both pairs strategies, with intervals well below zero.

## Avellaneda-Lee kappa recovery (AC-W-14)

On a synthetic three-factor market with OU residuals at kappa = 25 a year in 30% of the names, the rank correlation between estimated and true kappa is about 0.23 from 60-day PCA residuals, against the AC's 0.6. From the true residuals, with no PCA at all, a 60-day AR(1) reaches only about 0.47 (0.68 at 120 days, 0.79 at 252, close to the ceiling for a 30/70 split): 60 days cannot tell kappa = 25 from a random walk. Averaging each name's estimate over the sample gives 0.59 across ten seeds (0.47 to 0.70). The strategy part of AC-W-14 passes (gross Sharpe above 1 on the synthetic market). Accepted as a finding by the owner (2026-11-03); the 60-day window and the threshold are unchanged.
