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
