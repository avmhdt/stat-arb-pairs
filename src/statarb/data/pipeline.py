"""Ingest, build and DQ steps behind the CLI, one function per command.

Layout under the data root (section 7d):

- ``raw/sharadar/<DATASET>/snapshot=<id>/``: eqpit snapshots, plus SFP;
- ``raw/stooq/snapshot=<date>/``, ``raw/edgar/``, ``raw/borrow/``;
- ``interim/eqpit/``: eqpit's point-in-time store, rebuilt when its raw snapshots change;
- ``processed/<table>/``: this project's canonical tables.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog

from statarb.data import build, dq, manifest, schemas
from statarb.data.providers import edgar, eqpit_store, sharadar, stooq
from statarb.industry import load_sic_etf_map
from statarb.settings import Settings, SettingsError

log = structlog.get_logger(__name__)

CONFIGS = Path("configs")
SIC_MAP = CONFIGS / "sic_to_etf.yaml"
BORROW_SCHEDULE = CONFIGS / "borrow_schedule.yaml"
PROFILES = {"sharadar": "headline", "free": "prototype"}
SOURCES = {"sharadar": ("sharadar", "borrow"), "free": ("stooq", "edgar", "borrow")}


class PipelineError(RuntimeError):
    """A step cannot run with the current settings or data."""


@dataclass(frozen=True)
class StepReport:
    """What a step did: lines for the console and how many files it fetched or wrote."""

    lines: list[str]
    downloaded: int


def raw_dir(settings: Settings, source: str) -> Path:
    """``<data root>/raw/<source>``."""
    return settings.data_dir / "raw" / source


def _key(settings: Settings) -> str:
    if not settings.has_sharadar_key:
        raise PipelineError("SHARADAR_API_KEY is not set (P-06)")
    return settings.sharadar_api_key.get_secret_value().strip()


def ingest_prices(settings: Settings, provider: str, seed_from: Path | None = None) -> StepReport:
    """Raw equity prices, master and delisting data for ``provider``."""
    if provider == "sharadar":
        root = raw_dir(settings, "sharadar")
        if seed_from is not None:
            results = eqpit_store.seed(seed_from, root)
        else:
            results = eqpit_store.pull(root, _key(settings))
        manifest.record(settings.data_dir, eqpit_store.manifest_entries(settings.data_dir, root))
        fetched = sum(result.files for result in results if result.action == "downloaded")
        lines = [
            f"{r.dataset:<8} {r.action:<10} {r.snapshot.name} ({r.files} files)" for r in results
        ]
        return StepReport(lines, fetched)
    if provider == "stooq":
        if settings.stooq_dir is None:
            raise PipelineError("STOOQ_DIR is not set (P-12)")
        target, wrote = stooq.ingest_snapshot(settings.stooq_dir, raw_dir(settings, "stooq"))
        entries = [
            manifest.entry_for(
                settings.data_dir,
                target / name,
                "stooq",
                name.removesuffix(".parquet"),
                f"file:{settings.stooq_dir.as_posix()} (snapshot {stooq.SNAPSHOT_DATE})",
                rows=pl.scan_parquet(target / name).select(pl.len()).collect().item(),
            )
            for name in ("index.parquet", "us_daily.parquet")
            if wrote or not manifest.is_recorded(settings.data_dir, target / name)
        ]
        if entries:
            manifest.record(settings.data_dir, entries)
        return StepReport([f"stooq    {'written' if wrote else 'kept':<10} {target.name}"], 0)
    raise PipelineError(
        f"provider {provider!r} is not available: no WRDS access, Tiingo not approved"
    )


def ingest_etf(settings: Settings, provider: str) -> StepReport:
    """Sector and market ETF prices for ``configs/sic_to_etf.yaml``."""
    if provider == "sharadar":
        root = raw_dir(settings, "sharadar")
        snapshot, fetched = sharadar.pull_funds(root, _key(settings))
        entries = eqpit_store.manifest_entries(settings.data_dir, root, [sharadar.SFP_DATASET])
        manifest.record(settings.data_dir, entries)
        action = "downloaded" if fetched else "kept"
        return StepReport(
            [f"SFP      {action:<10} {snapshot.name}"], len(entries) if fetched else 0
        )
    if provider == "stooq":
        return StepReport(["ETFs come with the Stooq snapshot (ingest prices --provider stooq)"], 0)
    raise PipelineError(f"provider {provider!r} is not available for ETFs")


def ingest_sic(settings: Settings, provider: str) -> StepReport:
    """Industry codes: Sharadar's come with TICKERS and ACTIONS; EDGAR's are current only."""
    if provider == "sharadar":
        return StepReport(["SIC history comes from TICKERS and ACTIONS (ingest prices)"], 0)
    if provider != "edgar":
        raise PipelineError(f"provider {provider!r} is not available for SIC codes")
    index = raw_dir(settings, "stooq") / f"snapshot={stooq.SNAPSHOT_DATE:%Y%m%d}" / "index.parquet"
    if not index.exists():
        raise PipelineError("run ingest prices --provider stooq first")
    symbols = pl.read_parquet(index).filter(pl.col("security_type") == "stock")["symbol"]
    target = raw_dir(settings, "edgar")
    try:
        client = edgar.EdgarClient(settings.sec_user_agent)
    except SettingsError as error:
        raise PipelineError(str(error)) from error
    with client:  # pragma: no cover - network
        tickers = edgar.company_tickers(client)
        target.mkdir(parents=True, exist_ok=True)
        tickers.write_parquet(target / "company_tickers.parquet")
        ciks = tickers.filter(pl.col("ticker").is_in(symbols.to_list()))["cik"].unique().to_list()
        fetched = edgar.ingest_current_sic(client, ciks, target / "sic_current.parquet")
    manifest.record(
        settings.data_dir,
        [
            manifest.entry_for(settings.data_dir, target / name, "edgar", stem, url)
            for name, stem, url in (
                ("company_tickers.parquet", "company_tickers", edgar.COMPANY_TICKERS_URL),
                ("sic_current.parquet", "sic_current", edgar.SUBMISSIONS_URL),
            )
        ],
    )
    return StepReport([f"edgar    {fetched} submissions fetched for {len(ciks)} CIKs"], fetched)


def ingest_borrow(settings: Settings, source: str) -> StepReport:
    """Materialise the assumed schedule (D-07). IBKR snapshots arrive with FR-34 in M6."""
    if source != "assumed":
        raise PipelineError("the IBKR borrow snapshot collector arrives in M6 (FR-34)")
    target = raw_dir(settings, "borrow") / "assumed_schedule.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = not target.exists() or target.read_bytes() != BORROW_SCHEDULE.read_bytes()
    if changed:
        shutil.copyfile(BORROW_SCHEDULE, target)
    manifest.record(
        settings.data_dir,
        [
            manifest.entry_for(
                settings.data_dir, target, "borrow", "assumed", f"file:{BORROW_SCHEDULE.as_posix()}"
            )
        ],
    )
    return StepReport([f"borrow   {'written' if changed else 'kept':<10} {target.name}"], 0)


def _eqpit_store(settings: Settings) -> Path:
    """The eqpit store, rebuilt only when its raw snapshot ids changed."""
    raw = raw_dir(settings, "sharadar")
    out = settings.data_dir / "interim" / "eqpit"
    wanted = {name: snap.name.split("=", 1)[1] for name, snap in _latest(raw).items()}
    built = out / "manifest.json"
    if (
        built.exists()
        and json.loads(built.read_text(encoding="utf-8")).get("raw_snapshots") == wanted
    ):
        return out
    log.info("eqpit_build", out=str(out))
    eqpit_store.build_store(raw, out)
    return out


def _latest(raw: Path) -> dict[str, Path]:
    from eqpit.snapshot import list_snapshots

    found = {name: list_snapshots(raw, name) for name in eqpit_store.DATASETS}
    missing = [name for name, snaps in found.items() if not snaps]
    if missing:
        raise PipelineError(f"no raw Sharadar snapshots for {missing}; run ingest prices")
    return {name: snaps[-1] for name, snaps in found.items()}


def run_build(settings: Settings, tables: tuple[str, ...] = build.ALL_TABLES) -> StepReport:
    """Raw to processed for the configured provider (``STATARB_EQUITY_PROVIDER``)."""
    provider = settings.statarb_equity_provider
    if provider not in PROFILES:
        raise PipelineError(f"provider {provider!r} is not available: no WRDS access")
    processed = settings.data_dir / "processed"
    sic_map = load_sic_etf_map(SIC_MAP)
    out: dict[str, pl.DataFrame] = {}
    if provider == "sharadar":
        if {"prices", "industry"} & set(tables):
            tables_from_store = build.sharadar_tables(
                eqpit_store.read_store(_eqpit_store(settings))
            )
            out.update(tables_from_store)
        if "etf" in tables:
            out["etf_returns"] = sharadar.etf_returns_from_sfp(
                build.read_sfp(raw_dir(settings, "sharadar")), build.etf_list(sic_map)
            )
    else:
        snapshot = raw_dir(settings, "stooq") / f"snapshot={stooq.SNAPSHOT_DATE:%Y%m%d}"
        if not snapshot.exists():
            raise PipelineError("run ingest prices --provider stooq first")
        daily = pl.read_parquet(snapshot / "us_daily.parquet")
        index = pl.read_parquet(snapshot / "index.parquet")
        if {"prices", "industry"} & set(tables):
            out.update(build.stooq_tables(daily, index, _edgar_sic(settings)))
        if "etf" in tables:
            out["etf_returns"] = build.stooq_etf_returns(daily, build.etf_list(sic_map))
    if "borrow" in tables:
        out["borrow_rates"] = build.borrow_rates(BORROW_SCHEDULE)
    counts = build.write_tables(processed, out)
    old = schemas.read_build_info(processed).get("row_counts", {})
    info = build.build_info(
        settings.data_dir, PROFILES[provider], provider, {**old, **counts}, SOURCES[provider]
    )
    info["built_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    schemas.write_build_info(processed, info)
    lines = [f"{name:<17} {rows:>12,} rows" for name, rows in counts.items()]
    lines.append(f"profile {info['data_profile']}, manifest {str(info['manifest_hash'])[:12]}")
    return StepReport(lines, 0)


def _edgar_sic(settings: Settings) -> pl.DataFrame | None:
    folder = raw_dir(settings, "edgar")
    if not (folder / "sic_current.parquet").exists():
        return None
    tickers = pl.read_parquet(folder / "company_tickers.parquet")
    sic = pl.read_parquet(folder / "sic_current.parquet")
    return tickers.join(sic, on="cik").select(pl.col("ticker").alias("symbol"), "sic")


def run_dq(settings: Settings, profile: str, report: Path) -> list[dq.DqResult]:
    """Run every DQ rule and write the aggregate report."""
    provider = {"headline": "sharadar", "prototype": "free"}[profile]
    results = dq.run_checks(
        settings.data_dir, profile, load_sic_etf_map(SIC_MAP), SOURCES[provider]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(dq.render_report(results, profile), encoding="utf-8", newline="\n")
    return results
