"""Sharadar's direct API (D-01). The key travels in the ``x-api-key`` header, never the URL.

Only the access check lives here in M1; bulk downloads arrive with the data layer in M2.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import httpx

from statarb.data.http import DEFAULT_TIMEOUT, get_with_retries

SHARADAR_BASE_URL = "https://api.sharadar.com/v1.0"

# The API answers a few tickers (AAPL and MSFT in SEP, SPY in SFP) without a key, and
# TICKERS needs no key at all. A key check must therefore ask for tickers outside the
# free tier, or a missing key would pass (equity-factor-pipeline, decisions M1).
KEY_CHECKS: dict[str, tuple[str, str]] = {
    "SEP": ("stocks", "GME"),
    "SFP": ("funds", "XLE"),
    "TICKERS": ("tickers", "GME"),
}


@dataclass(frozen=True, slots=True)
class AccessResult:
    """Outcome of a one-row query on one table."""

    dataset: str
    table: str
    ticker: str
    rows: int
    columns: tuple[str, ...]


def sharadar_client(api_key: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Return a client for the API with the key in its headers."""
    if not api_key:
        raise ValueError("SHARADAR_API_KEY is not set")
    return httpx.Client(
        base_url=SHARADAR_BASE_URL,
        headers={"x-api-key": api_key},
        transport=transport,
        timeout=DEFAULT_TIMEOUT,
    )


def check_access(
    dataset: str,
    api_key: str,
    transport: httpx.BaseTransport | None = None,
) -> AccessResult:
    """Fetch one row of ``dataset`` (``SEP``, ``SFP`` or ``TICKERS``) for its check ticker.

    Raises ``httpx.HTTPStatusError`` on a refused key (HTTP 403 ``Exceeds free tier``).
    """
    table, ticker = KEY_CHECKS[dataset]
    with sharadar_client(api_key, transport) as client:
        response = get_with_retries(
            client, f"/data/{table}", params={"ticker": ticker, "limit": 1}, attempts=3
        )
    reader = csv.reader(io.StringIO(response.text))
    header = next(reader, [])
    rows = sum(1 for row in reader if row)
    return AccessResult(dataset, table, ticker, rows, tuple(header))
