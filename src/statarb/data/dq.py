"""``statarb dq check``: data-quality rules DQ-01 to DQ-15 (FR-06).

Rules about data are checked on the processed tables. Rules about code (DQ-01, DQ-02,
DQ-11, DQ-14) are enforced by the views and the trading engine and proven by tests; the
report names the test. The committed report holds aggregate counts only; row-level lists
(DQ-10) go to ``<data root>/processed/dq/`` and stay out of git.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from statarb.data import manifest, schemas
from statarb.industry import SicEtfMap

Status = Literal["pass", "warn", "fail", "info"]
ID_PREFIXES = ("crsp:", "shr:", "free:", "syn:")
EXTREME_RETURN = 0.5
TR_TOLERANCE = 1e-9
ETF_MIN_SESSIONS = 252


@dataclass(frozen=True)
class DqResult:
    """Outcome of one rule. ``by_year`` holds per-year counts for the report."""

    rule: str
    status: Status
    summary: str
    by_year: dict[int, dict[str, int]] = field(default_factory=dict)


def _scan_prices(processed: Path) -> pl.LazyFrame:
    return pl.scan_parquet(processed / "prices_daily" / "**" / "*.parquet", hive_partitioning=False)


def _read(processed: Path, table: str) -> pl.DataFrame:
    return pl.read_parquet(processed / table / "*.parquet")


def check_code_rules() -> list[DqResult]:
    """Rules enforced by code rather than data."""
    return [
        DqResult(
            "DQ-01",
            "info",
            "FormationView refuses dates >= T; tests/integration/test_no_lookahead.py",
        ),
        DqResult(
            "DQ-02",
            "info",
            "universe from formation rows only (dates <= T-1); test_candidates.py",
        ),
        DqResult("DQ-11", "info", "carry-forward limit applied by the trading engine (M4)"),
        DqResult("DQ-14", "info", "one-day execution lag applied by the trading engine (M4)"),
    ]


def check_delistings(processed: Path, profile: str) -> DqResult:
    """DQ-03: every delisted security carries a delisting return on its last price row."""
    master = _read(processed, "security_master")
    delisted = master.filter(pl.col("last_date").is_not_null())
    with_ret = (
        _scan_prices(processed)
        .filter(pl.col("ret_delist").is_not_null())
        .select("security_id", "ret_delist", "ret_delist_imputed")
        .collect()
    )
    covered = delisted.join(with_ret, on="security_id", how="inner")
    imputed_rows = covered.filter(pl.col("ret_delist_imputed").fill_null(False))
    imputed = imputed_rows.height
    performance = int((imputed_rows["ret_delist"] < 0).sum())
    missing = delisted.height - covered.height
    status: Status = "pass"
    if missing:
        status = "fail" if profile == "headline" else "warn"
    if profile == "prototype" and delisted.height and not with_ret.height:
        status = "warn"
    return DqResult(
        "DQ-03",
        status,
        f"{delisted.height:,} delisted securities, {covered.height:,} with a delisting return "
        f"({imputed:,} imputed: {performance:,} performance-related at -30%, "
        f"{imputed - performance:,} at 0), {missing:,} without",
    )


def check_total_return(processed: Path) -> DqResult:
    """DQ-04: the total-return index compounds ``ret_total`` exactly."""
    mismatches = (
        _scan_prices(processed)
        .select("security_id", "date", "ret_total", "tr_index")
        .sort("security_id", "date")
        .with_columns(
            (pl.col("tr_index") / pl.col("tr_index").shift(1).over("security_id") - 1.0).alias(
                "implied"
            )
        )
        .filter(pl.col("implied").is_not_null())
        .filter(
            (pl.col("implied") - pl.col("ret_total").fill_null(0.0)).abs()
            > TR_TOLERANCE * (1.0 + pl.col("ret_total").fill_null(0.0).abs())
        )
        .select(pl.len())
        .collect()
        .item()
    )
    return DqResult(
        "DQ-04",
        "pass" if mismatches == 0 else "fail",
        f"{mismatches:,} rows where tr_index does not compound ret_total",
    )


def check_traded_days(processed: Path) -> DqResult:
    """DQ-05: counts behind the eligibility rule (midpoint closes, zero-volume days)."""
    counts = (
        _scan_prices(processed)
        .select(
            pl.len().alias("rows"),
            pl.col("close_is_mid").fill_null(False).sum().alias("mid"),
            (pl.col("volume").fill_null(0.0) <= 0).sum().alias("zero_volume"),
        )
        .collect()
        .row(0, named=True)
    )
    return DqResult(
        "DQ-05",
        "info",
        f"{counts['rows']:,} rows, {counts['mid']:,} midpoint closes, "
        f"{counts['zero_volume']:,} zero-volume days",
    )


def check_industry(processed: Path, profile: str) -> DqResult:
    """DQ-06: SIC history is point in time on the headline profile."""
    industry = _read(processed, "industry_history")
    not_pit = int((~industry["is_point_in_time"]).sum())
    status: Status = "pass"
    if not_pit:
        status = "fail" if profile == "headline" else "warn"
    if industry.is_empty():
        status = "fail" if profile == "headline" else "warn"
    return DqResult(
        "DQ-06",
        status,
        f"{industry['security_id'].n_unique():,} securities with SIC, "
        f"{not_pit:,} rows not point in time",
    )


def check_etfs(processed: Path, sic_map: SicEtfMap, profile: str) -> DqResult:
    """DQ-07: every mapped ETF is present, with its inception and first usable date."""
    etf = _read(processed, "etf_returns")
    present = etf.group_by("etf").agg(
        pl.col("inception_date").first(), pl.col("date").sort().alias("dates")
    )
    found = set(present["etf"].to_list())
    missing = sorted(set(sic_map.etfs) - found)
    usable = []
    for name, inception, dates in present.sort("etf").iter_rows():
        first_usable = dates[ETF_MIN_SESSIONS] if len(dates) > ETF_MIN_SESSIONS else None
        usable.append(f"{name} {inception} (usable {first_usable})")
    status: Status = "pass" if not missing else ("fail" if profile == "headline" else "warn")
    return DqResult("DQ-07", status, f"missing {missing or 'none'}; " + "; ".join(usable))


def check_dates(processed: Path) -> DqResult:
    """DQ-08: daily tables key on an exchange date, not a timestamp."""
    prices = _scan_prices(processed).collect_schema()["date"]
    etf = _read(processed, "etf_returns").schema["date"]
    ok = prices == pl.Date and etf == pl.Date
    return DqResult("DQ-08", "pass" if ok else "fail", f"prices date {prices}, etf date {etf}")


def check_ids(processed: Path) -> DqResult:
    """DQ-09: ids are permanent and prefixed; reused tickers are counted."""
    history = _read(processed, "ticker_history")
    bad = history.filter(
        ~pl.any_horizontal(pl.col("security_id").str.starts_with(p) for p in ID_PREFIXES)
    )
    reused = (
        history.group_by("ticker")
        .agg(pl.col("security_id").n_unique())
        .filter(pl.col("security_id") > 1)
    )
    return DqResult(
        "DQ-09",
        "fail" if bad.height else "pass",
        f"{bad.height:,} ids without a source prefix; "
        f"{reused.height:,} tickers used by more than one security",
    )


def check_extreme_returns(processed: Path) -> DqResult:
    """DQ-10: daily |ret| above 50% are listed locally, counted by year, never dropped."""
    rows = (
        _scan_prices(processed)
        .filter(pl.col("ret_total").abs() > EXTREME_RETURN)
        .select("date", "security_id", "ret_total", "close")
        .collect()
        .sort("date", "security_id")
    )
    target = processed / "dq"
    target.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(target / "extreme_returns.parquet")
    by_year = {
        int(year): {"extreme_returns": int(count)}
        for year, count in rows.group_by(pl.col("date").dt.year().alias("y"))
        .len()
        .sort("y")
        .iter_rows()
    }
    return DqResult(
        "DQ-10",
        "warn" if rows.height else "pass",
        f"{rows.height:,} daily returns beyond +/-50%, "
        "listed in processed/dq/extreme_returns.parquet",
        by_year,
    )


def check_borrow(processed: Path) -> DqResult:
    """DQ-12: history uses the assumed schedule; IBKR snapshots are forward-only."""
    borrow = _read(processed, "borrow_rates")
    sources = sorted(set(borrow["source"].to_list()))
    ibkr = borrow.filter(pl.col("source") == "ibkr")
    first_ibkr = str(ibkr["date"].min()) if ibkr.height else "none"
    return DqResult(
        "DQ-12",
        "pass" if set(sources) <= {"assumed", "ibkr"} else "fail",
        f"sources {sources}; first IBKR snapshot {first_ibkr}",
    )


def check_manifest(data_dir: Path, sources: Sequence[str]) -> DqResult:
    """DQ-13: every raw file is in the manifest and still matches its sha256."""
    frame = manifest.read_manifest(data_dir).filter(pl.col("source").is_in(list(sources)))
    problems = manifest.verify(data_dir, sources)
    status: Status = "fail" if problems or frame.is_empty() else "pass"
    detail = "; ".join(problems[:5]) if problems else "all match"
    return DqResult(
        "DQ-13",
        status,
        f"{frame.height:,} files, {int(frame['bytes'].sum() or 0) / 1e9:.2f} GB "
        f"from {list(sources)}: {detail}",
    )


def check_survivorship(processed: Path, profile: str, top_n: int = 1000) -> DqResult:
    """DQ-15: each year's universe holds securities that later delist.

    For each year the universe is the common stocks priced on the year's first session,
    and the top ``top_n`` of them by market cap where caps exist.
    """
    master = _read(processed, "security_master").filter(pl.col("security_type") == "common")
    prices = (
        _scan_prices(processed)
        .select("date", "security_id", "mktcap_usd")
        .join(master.lazy().select("security_id", "last_date"), on="security_id")
    )
    firsts = (
        _scan_prices(processed)
        .select("date")
        .unique()
        .group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.col("date").min())
        .collect()
        .sort("year")
    )
    rows = prices.join(firsts.lazy(), on="date").collect()
    by_year: dict[int, dict[str, int]] = {}
    for (year,), part in rows.group_by("year", maintain_order=False):
        start: date = part["date"][0]
        later = pl.col("last_date").is_not_null() & (pl.col("last_date") > start)
        ranked = (
            part.filter(pl.col("mktcap_usd").is_not_null())
            .sort("mktcap_usd", descending=True)
            .head(top_n)
        )
        by_year[int(year)] = {
            "universe": part.height,
            "later_delisted": int(part.select(later.sum()).item()),
            f"top{top_n}_later_delisted": int(ranked.select(later.sum()).item())
            if ranked.height
            else 0,
        }
    by_year = dict(sorted(by_year.items()))
    zero_years = [year for year, counts in by_year.items() if counts["later_delisted"] == 0]
    # The final year cannot have later delistings yet; it is reported but not failed.
    zero_years = [year for year in zero_years if year != max(by_year, default=0)]
    status: Status = "pass"
    if zero_years or not by_year:
        status = "fail" if profile == "headline" else "warn"
    return DqResult(
        "DQ-15",
        status,
        f"{len(by_year)} years; years with no later-delisted names: {zero_years or 'none'}",
        by_year,
    )


def run_checks(
    data_dir: Path, profile: str, sic_map: SicEtfMap, sources: Sequence[str]
) -> list[DqResult]:
    """Run every rule on ``<data_dir>/processed`` and return results in rule order."""
    processed = data_dir / "processed"
    info = schemas.read_build_info(processed)
    if not info:
        return [
            DqResult("build", "fail", f"no processed tables under {processed}; run statarb build")
        ]
    built = str(info.get("data_profile"))
    results = [
        DqResult(
            "profile",
            "pass" if built == profile else "fail",
            f"tables built as {built}, checked as {profile}; "
            f"manifest {str(info.get('manifest_hash'))[:12]}",
        ),
        *check_code_rules(),
        check_delistings(processed, profile),
        check_total_return(processed),
        check_traded_days(processed),
        check_industry(processed, profile),
        check_etfs(processed, sic_map, profile),
        check_dates(processed),
        check_ids(processed),
        check_extreme_returns(processed),
        check_borrow(processed),
        check_manifest(data_dir, sources),
        check_survivorship(processed, profile),
    ]
    order = {name: i for i, name in enumerate(["profile", *(f"DQ-{n:02d}" for n in range(1, 16))])}
    return sorted(results, key=lambda result: order.get(result.rule, 99))


def render_report(results: Sequence[DqResult], profile: str) -> str:
    """Markdown report with aggregate counts only."""
    lines = [
        f"# Data-quality report ({profile})",
        "",
        "Generated by `statarb dq check`. Aggregate counts only; no security-level rows.",
        "",
        "| Rule | Status | Result |",
        "|---|---|---|",
    ]
    lines += [f"| {r.rule} | {r.status} | {r.summary.replace('|', '/')} |" for r in results]
    for result in results:
        if not result.by_year:
            continue
        columns = sorted({key for counts in result.by_year.values() for key in counts})
        lines += ["", f"## {result.rule} by year", "", "| Year | " + " | ".join(columns) + " |"]
        lines.append("|---" * (len(columns) + 1) + "|")
        for year, counts in result.by_year.items():
            lines.append(
                f"| {year} | " + " | ".join(f"{counts.get(c, 0):,}" for c in columns) + " |"
            )
    return "\n".join(lines) + "\n"


def failed(results: Sequence[DqResult]) -> list[DqResult]:
    """The hard failures."""
    return [result for result in results if result.status == "fail"]
