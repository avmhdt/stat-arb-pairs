"""AC-W-03: the multiple-testing doc and its generated table exist and are not empty."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from statarb.reports.tables import END, START

REPO = Path(__file__).resolve().parents[2]


def test_multiple_testing_doc_and_table() -> None:
    doc = (REPO / "docs" / "multiple_testing.md").read_text(encoding="utf-8")
    for heading in ("## The family", "## Why FDR", "## Dependence", "## The table"):
        assert heading in doc
    block = doc.split(START, 1)[1].split(END, 1)[0]
    assert "| config | year |" in block
    table = pl.read_csv(REPO / "results" / "tables" / "fdr_by_window.csv")
    assert table.height > 0
    expected = {
        "window_id",
        "m",
        "raw_p_lt_05",
        "bh_discoveries",
        "by_discoveries",
        "expected_false_raw",
    }
    assert expected <= set(table.columns)
    assert (table["data_profile"] == "headline").all()
    assert (table["by_discoveries"] <= table["bh_discoveries"]).all()
