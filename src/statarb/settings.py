"""Typed environment settings, read from the process environment and ``.env``.

Every variable in ``.env.example`` has a field here. Secrets are ``SecretStr``, so the
settings repr masks them and they never reach a log line (NFR-06).
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EquityProvider = Literal["sharadar", "crsp", "free"]
LogLevel = Literal["DEBUG", "INFO", "WARNING"]

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class SettingsError(ValueError):
    """A setting is missing or unusable for the command that needs it."""


class Settings(BaseSettings):
    """Environment settings for every command."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sec_user_agent: str = ""
    statarb_equity_provider: EquityProvider = "sharadar"
    wrds_username: str = ""
    sharadar_api_key: SecretStr = SecretStr("")
    stooq_dir: Path | None = None
    tiingo_api_key: SecretStr = SecretStr("")
    alpaca_api_key_id: SecretStr = SecretStr("")
    alpaca_api_secret_key: SecretStr = SecretStr("")
    alpaca_data_feed: Literal["iex", "sip"] = "iex"
    statarb_borrow_source: Literal["assumed", "ibkr"] = "assumed"
    data_root: str = ""
    statarb_data_dir: str = ""
    statarb_results_dir: Path = Path("results")
    statarb_log_level: LogLevel = "INFO"

    @field_validator("stooq_dir", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("statarb_results_dir", mode="before")
    @classmethod
    def _empty_results_is_default(cls, value: object) -> object:
        return "results" if value == "" else value

    @property
    def data_dir(self) -> Path:
        """Root for ``raw/``, ``interim/`` and ``processed/``.

        ``STATARB_DATA_DIR`` wins, then ``DATA_ROOT`` with ``~`` expanded, then ``./data``.
        """
        if self.statarb_data_dir:
            return Path(self.statarb_data_dir).expanduser()
        if self.data_root:
            return Path(self.data_root).expanduser()
        return Path("data")

    @property
    def results_dir(self) -> Path:
        """Root for ``trial_log.csv`` and ``runs/``."""
        return self.statarb_results_dir.expanduser()

    @property
    def has_sharadar_key(self) -> bool:
        """Whether a Sharadar key is configured. The key itself is never exposed here."""
        return bool(self.sharadar_api_key.get_secret_value().strip())


def validate_user_agent(user_agent: str) -> str:
    """Return the SEC User-Agent stripped of quotes, or raise if it lacks a contact email.

    Parameters
    ----------
    user_agent
        The ``SEC_USER_AGENT`` value, ``"Full Name email@example.com"`` (P-04).

    Returns
    -------
    str
        The cleaned value, ready for the ``User-Agent`` header.
    """
    candidate = user_agent.strip().strip('"').strip("'").strip()
    if not candidate:
        raise SettingsError("SEC_USER_AGENT is not set (P-04)")
    if not _EMAIL.search(candidate):
        raise SettingsError("SEC_USER_AGENT must contain a contact email address")
    if len(candidate.split()) < 2:
        raise SettingsError("SEC_USER_AGENT must hold a name and a contact email")
    return candidate


def is_on_c_drive(path: Path) -> bool:
    """Whether ``path`` resolves to the Windows ``C:`` drive, where no project data may live.

    Under WSL the same drive is mounted at ``/mnt/c``.
    """
    resolved = path.expanduser().resolve()
    if resolved.as_posix().lower().startswith("/mnt/c/") or resolved.as_posix() == "/mnt/c":
        return True
    return PureWindowsPath(str(resolved)).drive.upper() == "C:"


def get_settings() -> Settings:
    """Load settings fresh, so tests and long runs see the current environment."""
    return Settings()
