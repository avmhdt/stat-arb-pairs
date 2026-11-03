# Decisions

Owner decisions copied from the portfolio-wide `DECISIONS.md` (recorded 2026-10-31, updated 2026-11-03), then the build agents' choices by milestone. `DECISIONS.md` overrides `SPEC.html`, and the spec overrides `README.md`. API keys never go in this file.

## Owner decisions

### Plan

- This is project 8 of 15, built in the third batch (2026-11-03) with `llm-text-alpha`, `event-driven-backtester` and `regime-portfolio-construction`. No fixed date.
- **Merging:** agents push feature branches and open PRs, one milestone per PR. PRs merge automatically once CI is green, with `gh pr merge --auto --squash` run as `avmhdt`. Branch protection and auto-merge are turned on once CI exists (M1).

### Identity and access

- **GitHub:** `gh` active account `avmhdt`, admin on the repo; Actions enabled (P-02).
- **SEC (P-04):** `SEC_USER_AGENT="avmhdt josessc.js@gmail.com"`.
- **WRDS (P-05):** no access. CRSP stays off, so history starts in 1998.

### Data (D-01, D-02, P-06, P-12)

- **Sharadar:** Full History Bundle bought directly from Sharadar, not through Nasdaq Data Link. Base URL `https://api.sharadar.com/v1.0`, key in `SHARADAR_API_KEY`, sent in the `x-api-key` header so it stays out of URLs and logs. Tables `stocks` (SEP), `funds` (SFP), `fundamentals` (SF1), `tickers`, `actions`; bulk zips with `years=full`. `actions` has no delisting-return field.
- **Licence:** Personal Use License. Sharadar confirmed in writing on 2026-10-31 that public code, aggregate results, job applications, aggregate cross-checks and hosted coding tools are allowed. Publishing is aggregate-only, with no security-level data.
- **Stooq:** the owner's local snapshot at `E:\trading-strategy-brainstorm\data\stooq` (data through 2026-10-30) is the labelled fallback and cross-check. Results from it carry `data_profile=prototype`.
- **Not approved:** Alpaca SIP and Tiingo. No other paid data (USD 0 beyond the held Sharadar subscription).

### Spec decisions

| ID | Ruling |
|---|---|
| D-01 | Sharadar SEP and SFP primary; Stooq the labelled fallback |
| D-02 | Sharadar already held; nothing else approved |
| D-03 | No WRDS, so the Sharadar row applies: GGR replicated on the 1998–2002 overlap only, decay measured 2003–2025 |
| D-04 to D-08 | Spec defaults stand. D-07's borrow schedule is 0.25%/yr for the top 500 by market cap, 1.0%/yr for ranks 501–1,000, 5.0%/yr otherwise, the schedule equity-factor-pipeline also uses |
| D-09 | Crypto variant skipped |
| D-10 | Intraday variant skipped |
| D-11 to D-14 | Spec defaults stand: reference hardware in `docs/hardware.md`, no docs site, signal at close t executed at close t+1, MIT licence |

### Machine

- Data stays on `D:` (native Windows `./data`) or WSL ext4, never `C:`, and never in git. Free space is checked at each milestone.
- Live recorders share this PC: downloads are capped at 10 MB/s, one table at a time; heavy runs use at most 8 threads. Never restart WSL, reboot, or edit `.wslconfig`.

## Agent decisions

| Milestone | Item | Choice | Why |
|---|---|---|---|
| M1 | `.gitignore` | Appended `logs/`, `results/runs/` (spec §6) and two negations, `!src/statarb/data/` and `!tests/data/` | The existing `data/` rule matches any folder named `data`, so it would hide the spec's own `src/statarb/data/` package and `tests/data/` folder. The rule itself is unchanged |
| M1 | Coverage data file | `.pytest_cache/coverage` | Keeps `.coverage` out of `git status` without a further `.gitignore` change |
| M1 | Sharadar key check tickers | SEP and TICKERS on `GME`, SFP on `XLE` | The API answers AAPL and MSFT (SEP) and SPY (SFP) without a key, so a check on those passes with no key. Probed 2026-11-03: `funds?ticker=XLE` and `stocks?ticker=GME` return 403 without a key. TICKERS needs no key at all, so its check only proves the table is reachable |
| M1 | Smoke without a key | Index the Stooq snapshot, write the index to `data/raw/_smoke/`, read `SPY.US`. With a key, Stooq is skipped | The spec's "local Stooq index if the key is absent" |
| M1 | `SEC_USER_AGENT` check | Needs a name and an email; surrounding quotes are stripped | SEC fair-access policy asks for both |
| M1 | C: guard | Refuses a data root on `C:`, or on `/mnt/c` under WSL | Same drive under both systems |
| M1 | Trial-log writer | The contract's reference writer, adapted: ten required columns, then `x_periods_per_year` (252) and `x_note`. Failed runs also go to `results/failed_runs.csv` with `run_id, timestamp, strategy_family, error` | Contract v1.0 and FR-27 |
| M1 | Contract check | `overfitlab` is a dev dependency from its public git repo, pinned by `uv.lock`. A unit test runs its validator on the writer's output, and CI validates `results/trial_log.csv` once it exists | `overfitlab` is not on PyPI; this makes the CI check real rather than skipped |
| M1 | Licensed-data guard | `scripts/check_tracked.py`, in pre-commit and in CI on every tracked file. Beyond the NFR-07 rules it rejects any security-level column (`security_id`, `ticker`, `permno`, `permaticker`) in `results/tables/` CSVs, daily or not | Aggregate tables never need one, and the stricter rule is simpler to check |
| M1 | `uv.lock` size | Exempt by name from the 500 kB limit (it is about 600 kB) | The spec commits the lockfile, and it holds no data |
| M1 | CLI | M1 ships `smoke` and `trials {count,list}`. The other commands in spec §10 arrive with the milestones that implement them | No placeholder commands that do nothing |
| M2 | Sharadar path | Through `eqpit` (FR-35): `ingest prices` fetches eqpit raw snapshots (TICKERS, ACTIONS, SEP, SF1, EDGAR); `build` runs eqpit's `build_pit` into `interim/eqpit/` and maps securities, prices and delistings onto this repo's schemas | eqpit already handles Sharadar's split-adjusted fields, dividends, share basis and point-in-time SIC, checked through equity-factor-pipeline M2 to M7 |
| M2 | eqpit policy | Delisting imputation -30% for performance-related reasons on every exchange, 0 otherwise; share types `common` and `reit` count as common | DQ-03 as written. eqpit's default -55% on Nasdaq is a different rule. CRSP share codes 10 and 11 include REITs |
| M2 | Reading the eqpit store | Prices through `PitStore.labels().daily_prices`; securities and delistings from the store's own folders | `PitStore` has no full-history reader for those two tables |
| M2 | Raw data on this PC | Seeded with `--seed-from` from equity-factor-pipeline's raw snapshots (2026-11-02), hard-linked on D: and verified against their sha256s; SFP downloaded (337 MB at 10 MB/s) | Same immutable files, no 3 GB re-download while the crypto capture runs |
| M2 | SFP | Own bulk ingest into an eqpit-style snapshot; ETF `ret_total` from the `closeadj` ratio, `close` and `volume` restated to the unadjusted basis | eqpit does not read SFP |
| M2 | `etf_returns` columns | Adds `close` and `volume` to the spec's four | The A-L ETF hedge leg needs them for spread and impact costs (M5) |
| M2 | Delisting return placement | On the last price row on or before the delisting date | Sharadar's delisting date often falls after the last trade |
| M2 | Traded day (DQ-05) | Positive volume, positive close, not a midpoint | Sharadar has no midpoint closes; a zero-volume row is a non-trading day |
| M2 | Windows | `T` is the first session of a month; formation is the `F` calendar months before it, trading the `H` months from it; only complete windows. The last window ends the day after the last session | Simplest reading of FR-01 that matches GGR's monthly overlapping portfolios |
| M2 | GGR universe | `require_full_history: true` with `min_price_usd: null` | DQ-05 gives GGR the full-history rule and no price floor |
| M2 | Prototype ranking | Median formation dollar volume where no share counts exist | Stooq has no shares outstanding |
| M2 | DQ-10 | Counts by year in the committed report; the row-level list in `data/processed/dq/extreme_returns.parquet` | The spec asks for a listing, and the committed report must stay aggregate. 42,138 rows are beyond 50%; 35,539 of them close under USD 5, below the headline price floor |
| M2 | DQ-15 universe | Common stocks priced on each year's first session, and the top 1,000 of them by market cap; the final year is reported but not failed | A year at the data's end cannot have later delistings yet |
| M2 | SIC to ETF map | `configs/sic_to_etf.yaml`: SIC ranges to SPDR sector ETFs, with fallbacks before inception | An approximation, since GICS sectors are not a function of SIC; it serves grouping and the A-L ETF variant |
| M2 | Leakage harness | `tests/integration/leakage.py`: noise, permutation, NaN replacement of every price on or after `T`, and a SIC change and ETF returns change from `T`; a leaky selector must fail it | FR-03; each strategy family joins it before use |
| M3 | Engle-Granger | Own Numba code reproducing `statsmodels.tsa.stattools.coint(y, x, trend="c")` with an explicit `maxlag`; the AIC search scores every lag length from one QR of the largest design, then reruns OLS at the chosen lag | FR-09, NFR-04. Measured: statistic within 2.4e-13 relative and p-value within 1.2e-15 of statsmodels on 500 pairs; 20,000 pairs in under 1 s |
| M3 | ADF maxlag | `floor(12 (n/100)^(1/4))` = 15 at n = 252, passed explicitly | The spec's floor. statsmodels' own default is the ceiling (16) |
| M3 | Order invariance | Both orderings tested; `eg_pvalue` is the larger p-value, and it feeds FDR | Spec section 7(b). The rule is conservative: on 20,000 independent random-walk pairs it rejects 2.0% at 5% |
| M3 | FDR family | Every candidate pair in the window's groups with both legs priced from the window's first session; a pair whose leg starts mid-window cannot be tested and is not in the family. A NaN p-value counts in m and is never rejected | FR-11, spec section 16 |
| M3 | Order of filters | BH (primary) on all tested pairs, then the OU half-life band [1, 63], then Johansen confirmation at 95% (lag in differences by AIC, at most 5, `deterministic="co"`), then rank by p-value and take each stock once, up to `top_k` | FR-10, FR-11. Johansen runs only on discoveries, so it never feeds FDR |
| M3 | Distance rows kept | The `max(store_top, rank_from + top_k - 1)` best-ranked pairs per window (200 by default); `fdr_m` holds how many pairs were compared | All-pairs rows would be millions per window; GGR needs ranks up to 120 |
| M3 | Distance spread sigma | Sample standard deviation (`ddof = 1`) of the normalised price difference over formation | GGR's "historical standard deviation" |
| M3 | AC-W-07 reading | "Median relative error of half-life <= 10%" read as the median signed error (the estimator's bias). Measured on 1,000 paths: median signed -3.9%, median absolute 12.2%, 95.8% within +/-35%, AR(1) and MLE median gap under 2% | The Cramer-Rao bound puts the median absolute error near 12% at 1,000 steps and half-life 10, so the absolute reading cannot be met by any estimator. Flagged for the owner |
| M3 | AC-W-04 simulation | Realised FDR measured on the p-values selection uses (the larger of the two orderings) | That is the family FDR sees. With one ordering's p-values instead, realised FDR was 9.6% (BH) and 12.2% (BY): MacKinnon's asymptotic p-values are too small in the far tail at n = 252 (2.4 times nominal at 0.05%) |
| M3 | AC-W-05 size and power | Not met; held for the owner. Size on independent random walks with one ordering and AIC lags: 7.3% on the AC's 2,000 pairs, 6.55% on 20,000 (band 3.5% to 6.5%). Power at a 10-day half-life, n = 252: about 0.5 (target 0.90), the same in statsmodels' `coint`. Both stay in `tests/validation/test_cointegration.py` as strict expected failures; thresholds unchanged | Properties of the Engle-Granger test at this sample size, not of this implementation |
| M3 | Synthetic planted half-lives | 1 to 3.25 sessions | The spec leaves them open. Recovery (section 11: recall >= 8/10, FDP <= 0.10) checks that clearly cointegrated pairs are found; weaker ones are the power tests' job |
| M3 | Finding: few FDR discoveries | Headline 1999 to 2025: 7.1M tests in 319 windows, BH 1,012 discoveries, BY 288, 422 traded, about 1.3 a window against `top_k = 20`. Reported in `docs/multiple_testing.md` and `results/tables/fdr_by_window.csv`; q unchanged | Risk R-03; spec section 16 forbids loosening q after seeing results |
| M3 | `statarb report` | At M3 it writes `results/tables/fdr_by_window.csv` from the newest headline cointegration selection per config and regenerates the marked block in `docs/multiple_testing.md` | AC-W-03; later milestones add their tables |
| M3 | `statarb select` output | `results/runs/select-<uuid>/pairs.parquet` and `selection.json` (config, data profile, manifest hash, git SHA); git-ignored | The rows name securities |
| M3 | Benchmarks | `reports/benchmarks.json` without the host name | NFR-01 on the reference PC: SSD for 1,000 stocks 0.03 s (target 5 s); 40,000 Engle-Granger tests 1.26 s (target 60 s for 20,000) |
