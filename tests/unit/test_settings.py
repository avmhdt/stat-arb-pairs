from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from statarb.settings import (
    Settings,
    SettingsError,
    get_settings,
    is_on_c_drive,
    validate_user_agent,
)

REPO = Path(__file__).resolve().parents[2]


def test_env_example_parses_to_defaults(tmp_path: Path) -> None:
    shutil.copy(REPO / ".env.example", tmp_path / ".env")
    settings = get_settings()
    assert settings.sec_user_agent == ""
    assert not settings.has_sharadar_key
    assert settings.statarb_equity_provider == "sharadar"
    assert settings.stooq_dir == Path("/mnt/e/trading-strategy-brainstorm/data/stooq")
    assert settings.alpaca_data_feed == "iex"
    assert settings.statarb_borrow_source == "assumed"
    assert settings.data_root == "~/data/stat-arb-pairs"
    assert settings.results_dir == Path("results")
    assert settings.statarb_log_level == "INFO"


def test_env_example_lists_every_setting() -> None:
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    names = {line.split("=", 1)[0] for line in text.splitlines() if "=" in line and line[0] != "#"}
    assert names == {name.upper() for name in Settings.model_fields}


def test_data_dir_precedence() -> None:
    assert Settings(_env_file=None).data_dir == Path("data")
    assert Settings(_env_file=None, data_root="~/x").data_dir == Path("~/x").expanduser()
    both = Settings(_env_file=None, data_root="~/x", statarb_data_dir="/d/y")
    assert both.data_dir == Path("/d/y")


def test_empty_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOOQ_DIR", "")
    monkeypatch.setenv("STATARB_RESULTS_DIR", "")
    settings = get_settings()
    assert settings.stooq_dir is None
    assert settings.results_dir == Path("results")


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARADAR_API_KEY", "not-a-real-key-123")
    settings = get_settings()
    assert settings.has_sharadar_key
    assert "not-a-real-key-123" not in repr(settings)
    assert "not-a-real-key-123" not in str(settings)


def test_blank_key_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARADAR_API_KEY", "   ")
    assert not get_settings().has_sharadar_key


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Jane Doe jane@example.com", "Jane Doe jane@example.com"),
        ('"avmhdt josessc.js@gmail.com"', "avmhdt josessc.js@gmail.com"),
        ("  'A b@c.org'  ", "A b@c.org"),
    ],
)
def test_user_agent_accepted(raw: str, clean: str) -> None:
    assert validate_user_agent(raw) == clean


@pytest.mark.parametrize("raw", ["", "   ", "Jane Doe", "jane@example.com", "Jane jane@example"])
def test_user_agent_rejected(raw: str) -> None:
    with pytest.raises(SettingsError):
        validate_user_agent(raw)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive letters")
def test_c_drive_detection_windows() -> None:
    assert is_on_c_drive(Path("C:/data"))
    assert is_on_c_drive(Path("c:/Users/x"))
    assert not is_on_c_drive(Path("D:/trading-strategy-brainstorm/stat-arb-pairs/data"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX paths")
def test_c_drive_detection_wsl() -> None:
    assert is_on_c_drive(Path("/mnt/c/Users/x"))
    assert is_on_c_drive(Path("/mnt/c"))
    assert not is_on_c_drive(Path("/mnt/d/trading-strategy-brainstorm"))
    assert not is_on_c_drive(Path("/home/x/data/stat-arb-pairs"))
