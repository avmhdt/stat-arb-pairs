# Methodology

Every formula the code uses, where it lives, and its source. Choices the spec left open are in `docs/decisions.md`; the multiple-testing argument is in `docs/multiple_testing.md`.

Notation: `t` indexes trading sessions; `P_t` is a total-return index (prices adjusted for splits and dividends, delisting returns applied); `p_t = ln P_t`; one year is 252 sessions.

## 1. Windows and the information boundary

A window has a formation period of 12 months ending at `T` (the first session of a month) and a trading period of the next 6 months; a new window starts every month, so six trade at once (`src/statarb/windows.py`). Selection, model fitting, cost inputs and the universe use only rows dated before `T`, through `FormationView`, which raises `LookAheadError` on any query past `T`. Trading reads rows from `T` on through `TradingView`, which exposes only the frozen pair parameters. Signals at the close of `t` trade at the close of `t + 1` (D-13). The leakage tests perturb every post-`T` row (noise, permutation, NaN, future industry codes) and require identical selections (`tests/integration/test_no_lookahead.py`, AC-W-01, AC-W-02).

The universe at `T` is point in time and survivorship-free: names that later delist are included until they do, and their delisting returns are applied (Shumway 1997; `docs/data.md`). Cointegration runs take the 1,000 largest NYSE, AMEX and NASDAQ common stocks by market cap at `T - 1` that traded on at least 95% of formation sessions and closed at USD 5 or more. GGR runs, as in the paper, take every common stock with a price on every formation session. A-L runs and the pairs compared with them take stocks with market cap of at least USD 1bn.

## 2. Pair selection

### Distance (Gatev, Goetzmann and Rouwenhorst 2006)

Normalise each stock's total-return index to 1 on the first formation session, `n_t = P_t / P_0`. For every pair, `SSD = sum_t (n^y_t - n^x_t)^2` over formation; pairs are ranked by SSD and taken from rank `rank_from` with each stock used once. The spread's formation standard deviation `sigma_d` (ddof 1) scales the trading signal `z_t = (n^y_t - n^x_t) / sigma_d`, with both indices re-based to 1 on the first trading session.

### Engle-Granger (Engle and Granger 1987; MacKinnon 2010)

Regress `p^y_t = c + b p^x_t + u_t` by least squares (QR), then run an ADF regression on the residuals,

    Delta u_t = rho u_{t-1} + sum_{i=1}^{k} phi_i Delta u_{t-i} + e_t,

with `k` chosen by AIC up to `maxlag = floor(12 (n / 100)^{1/4})` (15 at `n = 252`). The statistic is the t-ratio of `rho`; the p-value is MacKinnon's (2010) response-surface approximation for two variables with a constant, as in statsmodels `coint`. Both orderings are tested and the larger p-value is kept, so the result does not depend on which leg is called `y`. The Numba batch implementation matches statsmodels to 2.4e-13 (`src/statarb/selection/cointegration.py`).

### Johansen confirmation (Johansen 1991)

On FDR discoveries only, the trace statistic for rank 0 against rank 1 or more, VECM lag in differences by AIC (at most 5), constant in the cointegrating relation (`deterministic="co"`), compared with the 95% critical value.

### False discovery rate (Benjamini and Hochberg 1995; Benjamini and Yekutieli 2001)

The family is every candidate pair tested in the window. With sorted p-values `p_(1) <= ... <= p_(m)`, BH rejects the hypotheses up to the largest `i` with `p_(i) <= i q / m`; BY replaces `q` with `q / sum_{j=1}^{m} 1/j`, valid under any dependence. `q = 0.05`. Why BH holds approximately when pairs share a leg, and why not Bonferroni, is in `docs/multiple_testing.md`.

### Filters after FDR

The OU half-life (section 3) must lie in [1, 63] sessions, then Johansen must confirm; the remaining pairs are ranked by p-value and each stock is used once, up to `top_k = 20`.

## 3. Ornstein-Uhlenbeck fit and half-life

The spread `s_t = p^y_t - b p^x_t` is modelled as `ds = kappa (mu - s) dt + sigma dW`. Its exact discretisation is an AR(1), `s_{t+1} = a + phi s_t + eta_t`, so from the least-squares fit

    kappa = -ln(phi) / dt,  mu = a / (1 - phi),  sigma_eq = sd(eta) / sqrt(1 - phi^2),
    half-life = ln 2 / kappa,

with `dt = 1/252` and the half-life reported in sessions (`src/statarb/models/ou.py`; exact maximum likelihood is the alternative, Elliott, van der Hoek and Malcolm 2005). The half-life matters for sizing because a spread that halves in `h` sessions returns its expected profit over about `h` sessions: capital tied up per unit of expected P&L scales with `h`, so `1 / h` sizing equalises expected P&L per capital-day (tested in `results/tables/sizing_half_life.csv`).

## 4. Kalman hedge ratio (Chan 2013)

State `theta_t = (alpha_t, beta_t)` follows a random walk with covariance `W = delta / (1 - delta) I`; the observation is `p^y_t = alpha_t + beta_t p^x_t + v_t`, `var(v) = V`. Each step:

    R = P + W,  F = (1, p^x_t),  e_t = p^y_t - F theta,  Q_t = F R F' + V,
    K = R F' / Q_t,  theta <- theta + K e_t,  P <- R - K F R.

The filter starts from the formation OLS coefficients, their covariance and residual variance (`V`), runs through formation, and its final state is frozen at `T` and resumed on trading data; it never smooths (`src/statarb/models/kalman.py`). The signal is `z_t = e_t / sqrt(Q_t)`.

**Choosing delta.** `delta` sets how fast beta may move: `delta -> 0` is static OLS; large `delta` lets beta absorb the spread. The spec fixes `delta = 1e-4`; on log prices near 4, `W (1 + p_x^2)` is about 1.7e-3 against `V` near 6e-4, so `Q` is dominated by state noise and `z` is underdispersed (it rarely reaches 2). `results/tables/kalman_delta.csv` compares the static hedge and `delta` = 1e-5, 1e-4, 1e-3 on identical selections. A rolling OLS hedge was not used: it is a crude filter with a hard window, its beta jumps when an observation leaves the window, and the formation-to-trading hand-off needs the trailing formation data the frozen Kalman state already summarises.

## 5. Trading rules and sizing

Open at `|z| >= 2` (short the spread when `z > 0`), close when `z` crosses 0, stop at `|z| >= 4` (`stop_only`, the default), a time stop at 3 half-lives, and close at the window end, on a delisting or after 5 stale sessions (`src/statarb/trading/rules.py`). No entry while `|z|` is already past the stop.

**Break policies (FR-26).** A causal rolling Engle-Granger p-value on the last 60 sessions (formation tail included) flags a break when above 0.20. `hold` ignores breaks and the stop; `close_on_break` closes on a flagged day and opens nothing on one; `close_and_blacklist` also bars the pair for the rest of the window.

**Sizing.** Each window commits capital 1, split into `top_k` slots of `1 / top_k`; a pair gets `slot x gross_per_pair x scale`, with `scale = 1` (equal gross), proportional to `1 / half-life`, or to `1 / sigma_spread` (volatility target), normalised to average 1 across the window's pairs. Distance pairs are dollar neutral; cointegration pairs hold `beta` dollars of `x` per dollar of `y` (by `1 / (1 + |beta|)` shares of gross).

## 6. Costs (FR-16)

Per side, as a fraction of traded notional `q`:

- **Commission:** 1 bp.
- **Half-spread:** quoted where available, else Corwin and Schultz (2012) from daily highs and lows over the last 20 formation sessions, floored at 2 bp. For days `t, t + 1`: `beta = ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2`, `gamma = ln(max H / min L)^2`, `alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))`, spread `= 2 (e^alpha - 1) / (1 + e^alpha)`, negatives set to 0.
- **Impact:** square root (Almgren, Thum, Hauptmann and Li 2005), `Y sigma_daily sqrt(q / ADV20)` with `Y = 1`. Entries are scaled so no leg trades more than 10% of ADV20 (`max_participation`).
- **Borrow:** `|short notional| x fee x days / 360` (ACT/360), fee by size bucket at `T` (top 500 by market cap 0.25%, 501-1,000 1%, the rest 5% a year; D-07), compared with an IBKR snapshot in `results/tables/borrow_assumed_vs_ibkr.csv`.

## 7. Accounting (FR-17)

Daily P&L per window is marked at the close; return on committed capital is P&L over the window's capital at the previous close, and `ret = ret_gross - cost_trading - cost_borrow` exactly. The run's daily return averages the windows active that day (GGR's overlapping portfolios). Return on employed capital divides by the slot capital of pairs open that day. Turnover is one-way traded notional over committed capital.

## 8. Avellaneda-Lee residual strategy (Avellaneda and Lee 2010)

Every day, from the trailing 252 sessions of returns of stocks with market cap at least USD 1bn at `T - 1`:

1. **Factors.** PCA variant: standardise returns, take the top 15 eigenvectors of their correlation matrix by SVD, and form eigenportfolios with weights `v_i / sigma_i`. ETF variant: the stock's sector ETF (DQ-07 fallback chain ending at SPY).
2. **Residuals.** Regress each stock's last 60 daily returns on the factor returns with an intercept; cumulate the residuals, `X_k = sum_{j<=k} eps_j`.
3. **OU.** AR(1) on `X`: `kappa = -252 ln b`, `m = a / (1 - b)`, `sigma_eq = sd(zeta) / sqrt(1 - b^2)`; reject the fit when `kappa <= 252/30 = 8.4` (mean reversion slower than half the window). Means are centred across stocks (A-L eq. 18).
4. **Signal.** `s = (X_T - m) / sigma_eq`; open short above 1.25 and long below -1.25, close a short below 0.75 and a long above -0.50.
5. **Sizing.** A new position gets `2 / max(m, 100)` of current equity, `m` the recent number of positions per side; the factor exposure is hedged with SPY (PCA) or the sector ETF, netted into one book per instrument.

## 9. Statistics

- **Newey-West** standard errors with a Bartlett kernel and 6 lags (GGR) for mean returns and the decay trend slope.
- **Stationary bootstrap** (Politis and Romano 1994): blocks of geometric length with mean 20 days, 2,000 resamples, for the break-even cost and Sharpe-difference intervals.
- **Wilson interval** for shares: `(p + z^2/2n +/- z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)`, `z = 1.96`.
- **Probabilistic and deflated Sharpe** (Bailey and Lopez de Prado 2014): with per-period Sharpe `SR`, `n` observations, skewness `g3` and kurtosis `g4`, `PSR(SR*) = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1) SR^2 / 4))`; the DSR is `PSR(SR0)` with `SR0 = sqrt(V) ((1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)))`, `N` every trial of the family in `results/trial_log.csv` and `V` the variance of their Sharpe ratios. Matches backtest-overfitting-lab to 1e-9.

## 10. Economics evaluations (M6)

- **Decay:** monthly returns by sub-period and a regression on months elapsed with a Newey-West t-statistic.
- **Out-of-sample persistence:** Engle-Granger on each selected pair's trading window and the next 12 months, against one matched unselected candidate per selected pair from the same window and group.
- **Break-even cost per side:** `mean(gross - k x borrow) / mean(turnover)`, `k` in {0, 1, 2, 4}.
- **Capacity:** each trade's impact scaled by `sqrt(AUM / AUM_run) x Y / Y_run` over the last five years, on all costs and on gross less borrow and impact only; the share of trades whose scaled participation would exceed 10% of ADV.
- **Crowding:** correlation and beta of monthly returns to Ken French's short-term reversal factor (full and rolling 36 months), drawdowns over 6-10 August 2007 and March 2020 in trailing 60-day volatility units (Khandani and Lo 2011), Jaccard overlap of traded names, top-10 concentration, half-life and entry-rate trends.

## References

Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). Direct Estimation of Equity Market Impact. Risk 18(7).
Avellaneda, M., Lee, J.-H. (2010). Statistical Arbitrage in the US Equities Market. Quantitative Finance 10(7), 761-782.
Bailey, D. H., Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management 40(5), 94-107.
Benjamini, Y., Hochberg, Y. (1995). Controlling the False Discovery Rate. Journal of the Royal Statistical Society B 57(1), 289-300.
Benjamini, Y., Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple Testing under Dependency. Annals of Statistics 29(4), 1165-1188.
Chan, E. P. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.
Corwin, S. A., Schultz, P. (2012). A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices. Journal of Finance 67(2), 719-760.
Do, B., Faff, R. (2010). Does Simple Pairs Trading Still Work? Financial Analysts Journal 66(4), 83-95.
Elliott, R. J., van der Hoek, J., Malcolm, W. P. (2005). Pairs Trading. Quantitative Finance 5(3), 271-276.
Engle, R. F., Granger, C. W. J. (1987). Co-integration and Error Correction. Econometrica 55(2), 251-276.
Gatev, E., Goetzmann, W. N., Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. Review of Financial Studies 19(3), 797-827.
Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models. Econometrica 59(6), 1551-1580.
Khandani, A. E., Lo, A. W. (2011). What Happened to the Quants in August 2007? Journal of Financial Markets 14(1), 1-46.
MacKinnon, J. G. (2010). Critical Values for Cointegration Tests. Queen's Economics Department Working Paper 1227.
Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association 89(428), 1303-1313.
Shumway, T. (1997). The Delisting Bias in CRSP Data. Journal of Finance 52(1), 327-340.
Kenneth R. French, Data Library (short-term reversal factor, Fama/French factors), mba.tuck.dartmouth.edu.
