# Statistical arbitrage, done properly

> The classic pairs and residual-reversion trade, with the rigor that separates it from a tutorial.

**Status:** planning

| | |
|---|---|
| Track | Quant Researcher |
| Estimated effort | 4–6 weeks |
| Stack | Python (statsmodels, pykalman or hand-written filters) |
| Data cost | Free → Academic |
| Strongest for | Stat-arb pods, entry-level QR |

## Requirements

- [ ] Formation and trading windows kept strictly apart (for example 12 months formation, 6 months trading, rolling).
- [ ] Pair selection by distance and by cointegration (Engle–Granger, Johansen), with false-discovery-rate control.
- [ ] Kalman-filter hedge ratios, an Ornstein–Uhlenbeck fit and half-life, and z-score entry, exit and stop rules.
- [ ] A step up to residual stat arb using PCA or ETF-factor residuals (Avellaneda and Lee 2010).
- [ ] Costs: commissions, spread and short-borrow fees.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [Tiingo or Stooq daily](https://www.tiingo.com) | Free | Equity and ETF history for prototyping. |
| [Alpaca market data](https://alpaca.markets) | Free tier | Minute bars (IEX feed on the free plan) for intraday versions. |
| [Sharadar or CRSP](https://data.nasdaq.com) | Paid | A survivorship-free universe, needed for credible final results. |
| [SEC EDGAR (SIC codes)](https://www.sec.gov/edgar/sec-api-documentation) | Free | Industry grouping for candidate pairs. |
| [Binance public data](https://data.binance.vision) | Free | Crypto pairs as a second market. |

## Success criteria

### Works (correct and reproducible)

- [ ] A test shows no information from the trading window leaks into pair selection.
- [ ] Multiple-testing control on cointegration p-values is documented.

### Impresses hirers

- [ ] A replication of Gatev, Goetzmann and Rouwenhorst (2006) showing how returns have decayed since.
- [ ] A replication of Avellaneda–Lee compared with simple pairs.
- [ ] The share of pairs that stay cointegrated out of sample, and the break-even transaction cost.
- [ ] A capacity and crowding discussion backed by numbers.

## Questions this project must answer

- Cointegration broke during your trading window. What do you do?
- Why does half-life matter for position sizing?
- How is this different from what every quant ran in 2007?
