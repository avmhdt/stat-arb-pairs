from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from statarb.config import apply_overrides, deep_merge, load_config

REPO = Path(__file__).resolve().parents[2]


def test_base_config_loads() -> None:
    cfg = load_config(REPO / "configs" / "base.yaml")
    assert cfg.data.start == date(1998, 1, 2)
    assert cfg.universe.top_n == 1000
    assert cfg.windows.formation_months == 12
    assert cfg.run.data_profile == "headline"


def test_extends_and_overrides(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text((REPO / "configs" / "base.yaml").read_text())
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "run: {name: child, strategy_family: pairs_distance}\n"
        "universe: {top_n: null}\n"
    )
    cfg = load_config(child, ["windows.trading_months=3", "universe.exchanges=[NYSE]"])
    assert cfg.run.name == "child"
    assert cfg.run.seed == 20261030
    assert cfg.universe.top_n is None
    assert cfg.universe.min_price_usd == 5.0
    assert cfg.windows.trading_months == 3
    assert cfg.universe.exchanges == ("NYSE",)


def test_bad_configs(tmp_path: Path) -> None:
    loop = tmp_path / "loop.yaml"
    loop.write_text("extends: loop.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(loop)
    listed = tmp_path / "list.yaml"
    listed.write_text("- 1\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(listed)
    typo = tmp_path / "typo.yaml"
    typo.write_text("run: {name: x, strategy_family: pairs_coint}\nuniverse: {topn: 3}\n")
    with pytest.raises(ValidationError):
        load_config(typo)
    free = tmp_path / "free.yaml"
    free.write_text("run: {name: x, strategy_family: pairs_coint}\ndata: {provider: free}\n")
    with pytest.raises(ValidationError, match="prototype"):
        load_config(free)
    assert load_config(free, ["run.data_profile=prototype"]).run.data_profile == "prototype"
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["novalue"])


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 3}, "d": {"e": 1}}) == {
        "a": {"b": 3, "c": 2},
        "d": {"e": 1},
    }
