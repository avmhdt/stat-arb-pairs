"""Sharadar through equity-factor-pipeline's ``eqpit`` package (FR-35).

eqpit owns the Sharadar traps (split-adjusted OHLC and dividends, shares on the latest
split basis, ``delisted.value`` being a market cap) and the point-in-time security windows.
This module only fetches or links eqpit raw snapshots, builds an eqpit store under this
repo's data root with this project's policy, and reads it back through ``PitStore``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path

import polars as pl
import structlog
from eqpit.pit.build import build_pit, read_pit_manifest
from eqpit.pit.policy import PitPolicy
from eqpit.providers.sharadar import SharadarClient, SharadarProvider
from eqpit.snapshot import list_snapshots, read_manifest, verify_snapshot

from statarb.data.manifest import ManifestEntry, publish

log = structlog.get_logger(__name__)

DATASETS = ("TICKERS", "ACTIONS", "SEP", "SF1", "EDGAR")
"""What eqpit's Sharadar build reads. SF1 and EDGAR give point-in-time share counts."""

MAX_BYTES_PER_SECOND = 10_000_000

POLICY = PitPolicy(
    imputation_nyse_amex=-0.30,
    imputation_nasdaq=-0.30,
    imputation_other=0.0,
    share_types=("common", "reit"),
    horizons_months=(1,),
)
"""DQ-03: -30% for a performance-related delisting with no return (Shumway 1997), on any
exchange, and 0 otherwise. eqpit's default of -55% on Nasdaq is not this project's rule."""


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """What ingest did for one dataset."""

    dataset: str
    snapshot: Path
    action: str  # "kept", "linked", "copied" or "downloaded"
    files: int


def _link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
        return "copied"
    return "linked"


def seed(
    source_root: Path, dest_root: Path, datasets: Sequence[str] = DATASETS
) -> list[SnapshotResult]:
    """Reuse another checkout's immutable eqpit snapshots without the network.

    Each dataset's latest snapshot under ``source_root`` is hard-linked (or copied, across
    volumes) into ``dest_root`` under the same snapshot id, then verified file by file
    against its own sha256 manifest. A snapshot already present and intact is kept.
    """
    results: list[SnapshotResult] = []
    for dataset in datasets:
        found = list_snapshots(source_root, dataset)
        if not found:
            raise FileNotFoundError(f"no {dataset} snapshot under {source_root}")
        source = found[-1]
        target = dest_root / dataset / source.name
        manifest = read_manifest(source)
        if target.exists():
            verify_snapshot(target)
            results.append(SnapshotResult(dataset, target, "kept", 0))
            continue
        staging = target.with_name(f".seed-{source.name}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        actions = {
            _link_or_copy(source / name, staging / name)
            for name in [*manifest.files, "manifest.json"]
        }
        verify_snapshot(staging)
        publish(staging, target)
        action = "copied" if "copied" in actions else "linked"
        results.append(SnapshotResult(dataset, target, action, len(manifest.files)))
        log.info("sharadar_seed", dataset=dataset, snapshot=source.name, action=action)
    return results


def pull(
    dest_root: Path,
    api_key: str,
    datasets: Sequence[str] = DATASETS,
    refresh: bool = False,
) -> list[SnapshotResult]:
    """Download datasets that have no snapshot yet, one table at a time at 10 MB/s.

    With ``refresh`` every dataset gets a new snapshot. EDGAR acceptance times come from
    SEC's bulk submissions file and need ``SEC_USER_AGENT``.
    """
    results: list[SnapshotResult] = []
    provider = SharadarProvider(dest_root, policy=POLICY)
    today = date.today()
    with SharadarClient(
        api_key, max_bytes_per_second=MAX_BYTES_PER_SECOND
    ) as client:  # pragma: no cover - network
        for dataset in datasets:
            existing = list_snapshots(dest_root, dataset)
            if existing and not refresh:
                results.append(SnapshotResult(dataset, existing[-1], "kept", 0))
                continue
            manifests = provider.pull([dataset], date(1990, 1, 1), today, dest_root, client=client)
            target = dest_root / dataset / f"snapshot={manifests[-1].snapshot_id}"
            results.append(SnapshotResult(dataset, target, "downloaded", len(manifests[-1].files)))
    return results


def manifest_entries(
    data_dir: Path, raw_root: Path, datasets: Sequence[str] = DATASETS
) -> list[ManifestEntry]:
    """Manifest rows for the latest snapshot of each dataset, from eqpit's own sha256s."""
    entries: list[ManifestEntry] = []
    for dataset in datasets:
        found = list_snapshots(raw_root, dataset)
        if not found:
            continue
        snapshot = found[-1]
        manifest = read_manifest(snapshot)
        for name, sha in sorted(manifest.files.items()):
            file = snapshot / name
            entries.append(
                ManifestEntry(
                    source="sharadar",
                    dataset=dataset,
                    path=file.resolve().relative_to(data_dir.resolve()).as_posix(),
                    url=manifest.source,
                    retrieved_at=manifest.created_at.astimezone(UTC),
                    sha256=sha,
                    bytes=file.stat().st_size,
                    rows=manifest.row_counts.get(name),
                )
            )
    return entries


def build_store(raw_root: Path, out: Path) -> dict[str, object]:
    """Build eqpit's point-in-time tables from the raw snapshots; returns its manifest."""
    build_pit(SharadarProvider(raw_root, policy=POLICY), out, POLICY)
    return read_pit_manifest(out)


@dataclass(frozen=True)
class StoreTables:
    """The three eqpit tables this project maps onto its own schemas."""

    securities: pl.DataFrame
    prices: pl.DataFrame
    delistings: pl.DataFrame
    manifest: dict[str, object]


PRICE_COLUMNS = (
    "security_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret",
    "shares_out",
    "mcap",
    "price_flag",
)


def read_store(root: Path) -> StoreTables:
    """Read securities, daily prices and delistings of an eqpit store.

    Prices come through ``PitStore``'s label store, the full-history read it offers. It has
    no full-history reader for securities or delistings, so those two tables are read from
    the store's own folders, as ``PitStore`` itself does.
    """
    from eqpit.store import PitStore

    store = PitStore.open(root)
    securities = pl.read_parquet(root / "securities" / "**" / "*.parquet")
    ids = securities["security_id"].unique().sort().to_list()
    first = date(1900, 1, 1)
    prices = store.labels().daily_prices(PRICE_COLUMNS, ids, first)
    delistings = pl.read_parquet(root / "delistings" / "**" / "*.parquet")
    return StoreTables(securities, prices, delistings, read_pit_manifest(root))
