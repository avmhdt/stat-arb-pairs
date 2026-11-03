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
    stream: TextIO | None = None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")

    def factory(*_: object) -> structlog.PrintLogger:
        # Resolve stderr at use, not at setup: a test runner may swap it out and close it.
        return structlog.PrintLogger(stream if stream is not None else sys.stderr)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )
