"""``data/raw/manifest.parquet``: one row per raw file (DQ-13, NFR-12).

Ingest re-runs skip a file whose recorded sha256 still matches, so a second run downloads
nothing. The manifest hash, a digest of every recorded (path, sha256), identifies the data
behind each processed table and run.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

MANIFEST_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "source": pl.String,
    "dataset": pl.String,
    "path": pl.String,
    "url": pl.String,
    "retrieved_at": pl.Datetime("us", "UTC"),
    "sha256": pl.String,
    "bytes": pl.Int64,
    "rows": pl.Int64,
}
_CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One raw file. ``path`` is POSIX, relative to the data root."""

    source: str
    dataset: str
    path: str
    url: str
    retrieved_at: datetime
    sha256: str
    bytes: int
    rows: int | None


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def publish(staging: Path, target: Path, attempts: int = 5, wait: float = 0.2) -> None:
    """Rename a finished staging folder into place.

    On Windows a scanner or indexer can hold a fresh folder open for a moment, so a
    ``PermissionError`` is retried a few times before it is raised.
    """
    for attempt in range(attempts):
        try:
            staging.rename(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait * (attempt + 1))


def manifest_path(data_dir: Path) -> Path:
    """Where the manifest lives under the data root."""
    return data_dir / "raw" / "manifest.parquet"


def read_manifest(data_dir: Path) -> pl.DataFrame:
    """The manifest, or an empty frame with its schema."""
    path = manifest_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=MANIFEST_SCHEMA)
    return pl.read_parquet(path)


def entry_for(
    data_dir: Path,
    file: Path,
    source: str,
    dataset: str,
    url: str,
    rows: int | None = None,
    retrieved_at: datetime | None = None,
    sha256: str | None = None,
) -> ManifestEntry:
    """Describe ``file``; its sha256 is computed unless one is passed in."""
    return ManifestEntry(
        source=source,
        dataset=dataset,
        path=file.resolve().relative_to(data_dir.resolve()).as_posix(),
        url=url,
        retrieved_at=retrieved_at or datetime.now(UTC),
        sha256=sha256 or sha256_file(file),
        bytes=file.stat().st_size,
        rows=rows,
    )


def record(data_dir: Path, entries: Iterable[ManifestEntry]) -> pl.DataFrame:
    """Upsert entries by ``path`` and rewrite the manifest, sorted by path."""
    new = pl.DataFrame([asdict(entry) for entry in entries], schema=MANIFEST_SCHEMA)
    old = read_manifest(data_dir)
    merged = (
        pl.concat([old.filter(~pl.col("path").is_in(new["path"].to_list())), new])
        .sort("path")
        .rechunk()
    )
    target = manifest_path(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(target, compression="zstd")
    return merged


def is_recorded(data_dir: Path, file: Path) -> bool:
    """Whether ``file`` exists with the size and sha256 the manifest holds for it."""
    if not file.exists():
        return False
    relative = file.resolve().relative_to(data_dir.resolve()).as_posix()
    rows = read_manifest(data_dir).filter(pl.col("path") == relative)
    if rows.is_empty() or int(rows["bytes"][0]) != file.stat().st_size:
        return False
    return str(rows["sha256"][0]) == sha256_file(file)


def manifest_hash(data_dir: Path, sources: Iterable[str] | None = None) -> str:
    """sha256 over the sorted ``path:sha256`` lines of the chosen sources."""
    frame = read_manifest(data_dir)
    if sources is not None:
        frame = frame.filter(pl.col("source").is_in(list(sources)))
    digest = hashlib.sha256()
    for path, sha in frame.sort("path").select("path", "sha256").iter_rows():
        digest.update(f"{path}:{sha}\n".encode())
    return digest.hexdigest()


def verify(data_dir: Path, sources: Iterable[str] | None = None) -> list[str]:
    """Problems with recorded files: missing, resized or changed (DQ-13)."""
    frame = read_manifest(data_dir)
    if sources is not None:
        frame = frame.filter(pl.col("source").is_in(list(sources)))
    problems: list[str] = []
    for path, sha, size in frame.select("path", "sha256", "bytes").iter_rows():
        file = data_dir / path
        if not file.exists():
            problems.append(f"{path}: missing")
        elif file.stat().st_size != size:
            problems.append(f"{path}: size changed")
        elif sha256_file(file) != sha:
            problems.append(f"{path}: sha256 changed")
    return problems
