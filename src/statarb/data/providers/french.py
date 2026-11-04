"""Kenneth French Data Library: daily short-term reversal and market factors (FR-25).

Cite: Kenneth R. French, Data Library, mba.tuck.dartmouth.edu. Files stay under the data
root and are never redistributed.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import polars as pl

from statarb.data.http import get_with_retries

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "st_rev": "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
}


def parse_daily(text: str) -> pl.DataFrame:
    """Rows of a French daily CSV: ``YYYYMMDD,v1,v2,...`` in percent, other lines skipped."""
    header: list[str] | None = None
    rows: list[list[str]] = []
    for raw in text.splitlines():
        cells = [cell.strip() for cell in raw.split(",")]
        if header is None and len(cells) > 1 and cells[0] == "" and all(cells[1:]):
            header = cells[1:]
            continue
        if (
            header is not None
            and cells[0].isdigit()
            and len(cells[0]) == 8
            and len(cells) == len(header) + 1
        ):
            rows.append(cells)
    if header is None:
        raise ValueError("no header row found")
    frame = pl.DataFrame(rows, schema=["date", *header], orient="row")
    return frame.select(
        pl.col("date").str.strptime(pl.Date, "%Y%m%d"),
        *(pl.col(name).cast(pl.Float64) / 100.0 for name in header),
    )


def read_zip(payload: bytes) -> str:
    """The single CSV inside a French zip."""
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        return bundle.read(bundle.namelist()[0]).decode("latin-1")


def ingest(raw_root: Path, transport: httpx.BaseTransport | None = None) -> tuple[Path, int]:
    """Download both zips once and write ``factors.parquet``: ``date, st_rev, mkt_rf, rf``.

    Returns the parquet path and the number of files downloaded (0 when cached).
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    fetched = 0
    with httpx.Client(
        transport=transport, timeout=httpx.Timeout(60.0), follow_redirects=True
    ) as client:
        for name in FILES.values():
            target = raw_root / name
            if target.exists():
                continue
            target.write_bytes(
                get_with_retries(client, BASE + name).content
            )  # pragma: no cover - network
            fetched += 1  # pragma: no cover - network
    reversal = parse_daily(read_zip((raw_root / FILES["st_rev"]).read_bytes()))
    market = parse_daily(read_zip((raw_root / FILES["ff3"]).read_bytes()))
    rev_col = next(c for c in reversal.columns if c.lower().startswith("st"))
    factors = reversal.select("date", pl.col(rev_col).alias("st_rev")).join(
        market.select("date", pl.col("Mkt-RF").alias("mkt_rf"), pl.col("RF").alias("rf")), on="date"
    )
    out = raw_root / "factors.parquet"
    factors.sort("date").write_parquet(out)
    return out, fetched
