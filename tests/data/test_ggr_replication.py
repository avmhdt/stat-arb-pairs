"""AC-H-01 on local headline results: the GGR replication table exists and is complete.

Run after ``statarb sweep --config configs/ggr_distance.yaml --grid
configs/grids/ggr_replication.yaml`` and ``statarb evaluate replication``. It checks
completeness and labels; the pass/fail verdict itself is reported, not asserted, because a
miss is a finding to explain (spec section 2, non-goals).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

TABLE = Path(__file__).resolve().parents[2] / "results" / "tables" / "ggr_replication.csv"
pytestmark = pytest.mark.data


def test_replication_table_is_complete() -> None:
    if not TABLE.exists():
        pytest.skip("run the GGR sweep and statarb evaluate replication first")
    table = pl.read_csv(TABLE)
    assert table.height == 12
    assert set(table["portfolio"]) == {"top5", "top20", "101-120"}
    assert set(table["wait"]) == {"no_wait", "wait_one_day"}
    assert (table["data_profile"] == "headline").all()
    assert table["label"].str.starts_with("partial").all()
    verdict = json.loads(TABLE.with_suffix(".json").read_text(encoding="utf-8"))
    assert {"pass", "all_same_sign"} <= set(verdict)
