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
