"""Licensed-data guard (NFR-07), used by pre-commit on staged files and by CI on tracked ones.

Rejects: anything under data/; *.parquet outside tests/fixtures/; *.csv outside
tests/fixtures/ and results/; any file over 500 kB except uv.lock, which the spec commits;
any CSV under results/tables/ with a security_id or ticker column, since aggregate tables
never carry security-level rows.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 500_000
SIZE_EXEMPT = frozenset({"uv.lock"})
SECURITY_COLUMNS = frozenset({"security_id", "ticker", "permno", "permaticker"})


def _starts(path: PurePosixPath, *prefix: str) -> bool:
    return path.parts[: len(prefix)] == prefix


def problems(names: list[str], root: Path = Path()) -> list[str]:
    """Return one message per rejected path; paths are relative to ``root``."""
    found: list[str] = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        local = root / name
        suffix = path.suffix.lower()
        in_fixtures = _starts(path, "tests", "fixtures")
        if _starts(path, "data"):
            found.append(f"{name}: nothing under data/ may be committed")
        elif suffix == ".parquet" and not in_fixtures:
            found.append(f"{name}: Parquet files are allowed only under tests/fixtures/")
        elif suffix == ".csv" and not (in_fixtures or _starts(path, "results")):
            found.append(f"{name}: CSV files are allowed only under tests/fixtures/ and results/")
        if name not in SIZE_EXEMPT and local.is_file() and local.stat().st_size > MAX_BYTES:
            found.append(f"{name}: over 500 kB")
        if suffix == ".csv" and _starts(path, "results", "tables") and local.is_file():
            with local.open(encoding="utf-8", newline="") as handle:
                header = {column.strip().lower() for column in next(csv.reader(handle), [])}
            if header & SECURITY_COLUMNS:
                found.append(f"{name}: results/tables/ must not hold security-level columns")
    return found


def main(argv: list[str]) -> int:
    """Print each problem and return 1 if there is any."""
    found = problems(argv)
    for message in found:
        print(message, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
