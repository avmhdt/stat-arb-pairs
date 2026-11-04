"""AC-H-11: every number the README's results quote appears in a generated table.

The checked region runs from ``<!-- numbers:start -->`` to ``<!-- numbers:end -->``
(Results and What failed). A quoted number matches a table cell when the cell, read as a
fraction, a percentage, or in millions, rounds to it at the quoted precision; the sign may
be carried by words ("loses 4.4%"). Integers under 100 (parameters, counts of things like
"20 pairs") and years are not checked; larger integers must match a cell or a label.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "results" / "tables"
NUMBER = re.compile(r"(?<![\w.\-])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.\-])-?\d+(?:\.\d+)?")
SCALES = (1.0, 100.0, 1e-6)


def region(text: str) -> str:
    start, end = "<!-- numbers:start -->", "<!-- numbers:end -->"
    assert start in text and end in text, "README lacks the checked-numbers markers"
    body = text.split(start, 1)[1].split(end, 1)[0]
    body = re.sub(r"`[^`]*`", " ", body)  # file names and code
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)  # figures
    return re.sub(r"(?<=\d)-(?=\d)", " ", body)  # ranges: "101-120" is two numbers


def quoted(body: str) -> list[str]:
    tokens = []
    for match in NUMBER.finditer(body):
        token = match.group(0)
        tail = body[match.end() : match.end() + 1]
        if tail.isalpha() and tail not in "m":  # "M6", "2nd", but keep "1.3m"
            continue
        value = float(token.replace(",", ""))
        is_int = "." not in token
        if is_int and (abs(value) < 100 or 1900 <= abs(value) <= 2100):
            continue
        tokens.append(token)
    return tokens


def table_cells() -> tuple[list[float], set[str]]:
    numbers: list[float] = []
    labels: set[str] = set()
    for path in TABLES.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                for cell in row:
                    try:
                        value = float(cell)
                    except ValueError:
                        labels.add(cell)
                        continue
                    if math.isfinite(value):
                        numbers.append(value)
    return numbers, labels


def matches(token: str, numbers: list[float], labels: set[str]) -> bool:
    text = token.replace(",", "")
    target = abs(float(text))
    decimals = len(text.split(".")[1]) if "." in text else 0
    if decimals == 0 and any(text.lstrip("-") in label for label in labels):
        return True
    digits = -1 if decimals == 0 and text.endswith("0") else decimals
    for value in numbers:
        for scale in SCALES:
            if round(abs(value) * scale, digits) == round(target, digits):
                return True
    return False


def test_readme_numbers_appear_in_tables() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    numbers, labels = table_cells()
    tokens = quoted(region(text))
    assert len(tokens) > 40
    missing = [token for token in tokens if not matches(token, numbers, labels)]
    assert not missing, f"README numbers not found in results/tables: {missing}"


@pytest.mark.parametrize(
    ("token", "ok"),
    [("1.01", True), ("2.42", True), ("0.34", True), ("12345.678", False)],
)
def test_matcher(token: str, ok: bool) -> None:
    numbers = [0.010113, 0.024211, 335981.8]
    assert matches(token, numbers, set()) is ok


def test_region_and_tokens() -> None:
    body = region(
        "x <!-- numbers:start --> 1.5% in 2007 for 20 pairs, `a 9.9` 101-120 <!-- numbers:end -->"
    )
    assert quoted(body) == ["1.5", "101", "120"]
