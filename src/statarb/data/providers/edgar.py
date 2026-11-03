"""SEC EDGAR access with the fair-access User-Agent and a request-rate cap (P-04, NFR-11)."""

from __future__ import annotations

from typing import Any, cast

import httpx

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

    def submissions(self, cik: int | str) -> dict[str, Any]:
        """Return the submissions JSON for ``cik``, which carries the current ``sic``."""
        response = get_with_retries(
            self._client, SUBMISSIONS_URL.format(cik=str(cik)), limiter=self._limiter
        )
        return cast(dict[str, Any], response.json())
