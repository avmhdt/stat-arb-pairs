# Data

Raw and licensed data live under the data root and never enter git. Only code, configs, aggregate tables and figures are committed.

## Profiles

| Profile | Provider | Survivorship | SIC | Share counts | Use |
|---|---|---|---|---|---|
| `headline` | Sharadar Full History Bundle (D-01) through equity-factor-pipeline's `eqpit` package (FR-35) | Free: delisted names stay in | Point in time, from TICKERS and the `sicchangefrom` actions | From SF1 filings, as of each filing's EDGAR acceptance time | Every headline result |
| `prototype` | The owner's local Stooq snapshot (through 2026-10-30) plus EDGAR current SIC | Biased: only names in the snapshot | Current only, `is_point_in_time = false` (DQ-06) | None, so universes rank by dollar volume | Cross-checks; every output is labelled prototype |

No WRDS access exists, so CRSP is not used and history starts in 1998 (D-03). Tiingo and Alpaca SIP are not approved.

## Where data lives

The data root is `STATARB_DATA_DIR`, else `DATA_ROOT`, else `./data`. On native Windows it is the repo's git-ignored `./data` on `D:`; under WSL it is `~/data/stat-arb-pairs` on ext4. `C:` is refused by `statarb smoke`.

```
data/
  raw/
    manifest.parquet                    one row per raw file: source, url, retrieved_at, sha256, bytes, rows
    sharadar/<DATASET>/snapshot=<id>/   eqpit snapshots: TICKERS, ACTIONS, SEP, SF1, EDGAR, plus SFP
    stooq/snapshot=20261030/            index.parquet, us_daily.parquet (prototype)
    edgar/                              company_tickers.parquet, sic_current.parquet (prototype)
    borrow/assumed_schedule.yaml        copy of configs/borrow_schedule.yaml
  interim/eqpit/                        eqpit's point-in-time store, rebuilt when raw snapshot ids change
  processed/                            this project's tables (below) and build_info.json
    dq/extreme_returns.parquet          DQ-10's row-level list, local only
```

## Commands

```bash
uv run statarb ingest prices --provider sharadar
uv run statarb ingest etf --provider sharadar
uv run statarb ingest sic --provider sharadar
uv run statarb ingest borrow --source assumed
uv run statarb build
uv run statarb dq check --profile headline
```

`ingest prices --provider sharadar --seed-from <eqpit raw root>` hard-links another checkout's immutable eqpit snapshots instead of downloading them (copying when the volumes differ), then verifies every file against its sha256 manifest. Downloads run one table at a time, capped at 10 MB/s. A second run of any ingest command downloads nothing. The key is sent only in the `x-api-key` header, and the signed storage link is followed without it.

## Processed tables

Schemas are in `src/statarb/data/schemas.py` and checked on every write. Daily tables key on the exchange date and `security_id`; validity intervals are inclusive, with a null `valid_to` open.

| Table | Rows (headline, 2026-11-03) | Notes |
|---|---|---|
| `security_master` | 21,028 | `shr:<permaticker>`; `last_date` set only for delisted names |
| `ticker_history` | 38,451 | Ticker, type and exchange windows from eqpit |
| `industry_history` | 38,272 | Point-in-time SIC windows |
| `prices_daily` | 45,517,618 | Unadjusted OHLC and volume; total return; delisting return on the last price row; market cap |
| `etf_returns` | 75,214 | SPDR sector ETFs and SPY from SFP |
| `borrow_rates` | 3 | The assumed schedule (D-07) |

## Point-in-time rules applied in the build

- **Prices (DQ-04).** `ret_total` is eqpit's total return from unadjusted closes, split ratios and cash distributions, with the vendor's return on split and spin-off days. `tr_index` compounds it from 1.0 on each security's first date. Sharadar's own `close`, `open`, `high`, `low` and `volume` are split-adjusted; eqpit restates them to the unadjusted basis with `closeunadj / close`.
- **Delistings (DQ-03).** Sharadar has no delisting return (`delisted.value` is a market cap). Every one is imputed: -30% when the delisting is performance-related (regulatory or voluntary delisting, or bankruptcy, after Shumway 1997), 0 otherwise, with `ret_delist_imputed = true`. The return sits on the security's last price row on or before the delisting date. Of 14,706 delistings, 4,764 are imputed at -30% and 9,942 at 0.
- **Market cap.** Shares come from SF1 filings, visible from the close of the trading day after EDGAR accepted each filing (eqpit's default lag), so a formation date never sees a later filing.
- **SIC (DQ-06).** Today's SIC is walked back through `sicchangefrom` actions, so each window holds the code in force then.
- **ETFs (DQ-07).** `ret_total` is the ratio of SFP's `closeadj`. An ETF serves as a factor only once it has 252 sessions of history; `configs/sic_to_etf.yaml` names the fallback before then (XLC to XLK, XLRE to XLF, the rest to SPY).
- **Universe (DQ-02, DQ-05).** Built from formation rows only. A traded day is a positive-volume close that is not a quote midpoint. Delisted names stay in every window they qualified for.

## Licensing

Sharadar is under the owner's Personal Use License. Sharadar confirmed in writing on 2026-10-31 that public code, aggregate results and job-application use are allowed. Nothing security-level is published: `results/tables/` and `reports/` hold aggregates only, and CI rejects tracked Parquet files and security-level columns in committed tables (NFR-07). The DQ report under `reports/dq/` holds counts only. Stooq and EDGAR data are never redistributed either.
