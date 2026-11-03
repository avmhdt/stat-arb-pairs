from __future__ import annotations

import httpx
import pytest

from statarb.data.http import RateLimiter, get_with_retries, is_retryable


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(4.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.slept == pytest.approx([0.25, 0.25])
    clock.now += 10.0
    limiter.wait()
    assert len(clock.slept) == 2


def test_rate_limiter_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0)


def _client(statuses: list[int], seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses.pop(0), text="ok")

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x.test")


def test_retries_on_server_errors_then_succeeds() -> None:
    seen: list[httpx.Request] = []
    with _client([503, 429, 200], seen) as client:
        response = get_with_retries(client, "/a", params={"q": 1}, max_wait=0)
    assert response.status_code == 200
    assert len(seen) == 3
    assert seen[0].url.params["q"] == "1"


def test_no_retry_on_client_error() -> None:
    seen: list[httpx.Request] = []
    with _client([403, 200], seen) as client, pytest.raises(httpx.HTTPStatusError):
        get_with_retries(client, "/a", max_wait=0)
    assert len(seen) == 1


def test_gives_up_after_attempts() -> None:
    seen: list[httpx.Request] = []
    with _client([500, 500, 500], seen) as client, pytest.raises(httpx.HTTPStatusError):
        get_with_retries(client, "/a", attempts=3, max_wait=0)
    assert len(seen) == 3


def test_limiter_is_applied() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock, sleep=clock.sleep)
    seen: list[httpx.Request] = []
    with _client([200, 200], seen) as client:
        get_with_retries(client, "/a", limiter=limiter)
        get_with_retries(client, "/b", limiter=limiter)
    assert clock.slept == pytest.approx([0.5])


def test_retryable_classification() -> None:
    request = httpx.Request("GET", "https://x.test")
    assert is_retryable(httpx.ConnectError("down", request=request))
    for code, expected in [(502, True), (404, False)]:
        error = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(code, request=request)
        )
        assert is_retryable(error) is expected
    assert not is_retryable(ValueError("no"))
