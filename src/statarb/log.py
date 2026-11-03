"""structlog setup: JSON lines, to stderr by default or to ``logs/<run_id>.jsonl`` (NFR-08)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

import structlog


def configure_logging(level: str = "INFO", path: Path | None = None) -> None:
    """Route structlog output as JSON lines at ``level``.

    Parameters
    ----------
    level
        ``DEBUG``, ``INFO`` or ``WARNING``.
    path
        Optional ``.jsonl`` file; stderr when omitted, so stdout stays for command output.
    """
    stream: TextIO
    if path is None:
        stream = sys.stderr
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(stream),
        cache_logger_on_first_use=False,
    )
