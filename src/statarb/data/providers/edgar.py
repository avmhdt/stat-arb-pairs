"""SEC EDGAR access with the fair-access User-Agent and a request-rate cap (P-04, NFR-11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import polars as pl

from statarb.data.http import DEFAULT_TIMEOUT, RateLimiter, get_with_retries
from statarb.settings import validate_user_agent

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# SEC's published limit is 10 requests per second; stay below it.
MAX_REQUESTS_PER_SECOND = 8.0


class EdgarClient:
    """Small EDGAR client. One instance shares one rate limiter across calls.

    Parameters
    ----------
    user_agent
        ``SEC_USER_AGENT``; validated for a name and contact email.
    transport
        httpx transport override, used by tests.
    """

    def __init__(
        self,
        user_agent: str,
        transport: httpx.BaseTransport | None = None,
        rate: float = MAX_REQUESTS_PER_SECOND,
    ) -> None:
        agent = validate_user_agent(user_agent)
        self._client = httpx.Client(
            headers={"User-Agent": agent, "Accept-Encoding": "gzip, deflate"},
            transport=transport,
            timeout=DEFAULT_TIMEOUT,
        )
        self._limiter = RateLimiter(rate)

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def get_json(self, url: str) -> Any:
        """Parsed JSON body of ``url``."""
        return get_with_retries(self._client, url, limiter=self._limiter).json()

    def submissions(self, cik: int | str) -> dict[str, Any]:
        """Return the submissions JSON for ``cik``, which carries the current ``sic``."""
        response = get_with_retries(
            self._client, SUBMISSIONS_URL.format(cik=str(cik)), limiter=self._limiter
        )
        return cast(dict[str, Any], response.json())


def company_tickers(client: EdgarClient) -> pl.DataFrame:
    """SEC's current ticker to CIK map: ``ticker, cik, title``."""
    payload = client.get_json(COMPANY_TICKERS_URL)
    rows = list(payload.values()) if isinstance(payload, dict) else payload
    return pl.DataFrame(
        {
            "ticker": [str(row["ticker"]).upper() for row in rows],
            "cik": [int(row["cik_str"]) for row in rows],
            "title": [str(row["title"]) for row in rows],
        }
    ).unique(subset=["ticker"], keep="first", maintain_order=True)


SIC_SCHEMA = {"cik": pl.Int64, "sic": pl.Int32, "sic_description": pl.String}


def ingest_current_sic(client: EdgarClient, ciks: list[int], target: Path, batch: int = 200) -> int:
    """Fetch current SIC codes for ``ciks`` into ``target``, resuming where it stopped.

    EDGAR gives today's SIC only, so these rows are never point in time (DQ-06). Returns how
    many CIKs were fetched in this call.
    """
    done = pl.read_parquet(target) if target.exists() else pl.DataFrame(schema=SIC_SCHEMA)
    todo = sorted(set(ciks) - set(done["cik"].to_list()))
    fetched: list[dict[str, object]] = []
    for count, cik in enumerate(todo, start=1):
        payload = client.submissions(cik)
        sic = str(payload.get("sic") or "")
        fetched.append(
            {
                "cik": cik,
                "sic": int(sic) if sic.isdigit() else None,
                "sic_description": payload.get("sicDescription"),
            }
        )
        if count % batch == 0 or count == len(todo):
            done = pl.concat([done, pl.DataFrame(fetched, schema=SIC_SCHEMA)]).sort("cik")
            target.parent.mkdir(parents=True, exist_ok=True)
            done.write_parquet(target, compression="zstd")
            fetched = []
    return len(todo)
