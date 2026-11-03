"""HTTP plumbing shared by the providers: rate limiting and retries (NFR-11)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT = httpx.Timeout(30.0)
# Seconds of the first backoff wait; tests set it to 0.
BACKOFF_SECONDS = 0.5


class RateLimiter:
    """Allow at most ``rate`` calls per second by spacing calls evenly.

    Parameters
    ----------
    rate
        Calls per second; must be positive.
    clock, sleep
        Injected for tests.
    """

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next call is allowed."""
        with self._lock:
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self._interval


def is_retryable(error: BaseException) -> bool:
    """Retry on transport errors and on 429/5xx responses, never on other client errors."""
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRY_STATUS
    return isinstance(error, httpx.TransportError)


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
    limiter: RateLimiter | None = None,
    attempts: int = 5,
    max_wait: float = 30.0,
) -> httpx.Response:
    """GET ``url`` with exponential backoff, raising ``httpx.HTTPStatusError`` on failure.

    Parameters
    ----------
    client
        The client carrying base URL and headers. Credentials stay in its headers.
    url
        Absolute URL or a path relative to the client's base URL.
    params
        Query parameters. Never put a secret here: they end up in the URL.
    limiter
        Optional rate limiter, applied before every attempt.
    attempts, max_wait
        Retry budget and the cap on a single backoff wait, in seconds.
    """

    @retry(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=BACKOFF_SECONDS, max=max_wait),
        reraise=True,
    )
    def _once() -> httpx.Response:
        if limiter is not None:
            limiter.wait()
        response = client.get(url, params=params)
        response.raise_for_status()
        return response

    return _once()
