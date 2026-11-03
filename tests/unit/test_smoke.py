from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl
import pytest
from pydantic import SecretStr

from statarb.settings import Settings
from statarb.smoke import Check, run_smoke, smoke_dir


def _never_c(_: Path) -> bool:
    return False


def _transport(sec_status: int = 200, sharadar_status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data.sec.gov":
            return httpx.Response(sec_status, json={"name": "Apple Inc.", "sic": "3571"})
        if request.url.host == "api.sharadar.com":
            return httpx.Response(sharadar_status, text="ticker,date\nGME,2026-10-30\n")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _status(checks: list[Check], name: str) -> str:
    return next(check.status for check in checks if check.name == name)


def test_smoke_without_key_uses_stooq(settings: Settings) -> None:
    checks = run_smoke(settings, transport=_transport(), on_c=_never_c)
    names = [check.name for check in checks]
    assert names == [
        "sec_user_agent",
        "data_dir",
        "disk_free",
        "sharadar_api_key",
        "stooq_index",
        "stooq_read",
        "sec_submissions",
    ]
    assert _status(checks, "sharadar_api_key") == "warn"
    assert all(check.status != "fail" for check in checks)
    out = smoke_dir(settings)
    assert pl.read_parquet(out / "stooq_index.parquet").height == 3
    assert (out / "sec_submissions_CIK0000320193.json").is_file()
    written = {path.relative_to(settings.data_dir) for path in settings.data_dir.rglob("*")}
    assert all(path.parts[:2] == ("raw", "_smoke") for path in written if path.is_file())


def test_smoke_with_key_checks_sharadar(settings: Settings) -> None:
    keyed = settings.model_copy(update={"sharadar_api_key": SecretStr("fake-key")})
    checks = run_smoke(keyed, transport=_transport(), on_c=_never_c)
    for name in ("sharadar_sep", "sharadar_sfp", "sharadar_tickers"):
        assert _status(checks, name) == "ok"
    assert "stooq_index" not in {check.name for check in checks}
    assert "fake-key" not in " ".join(check.detail for check in checks)


def test_smoke_reports_refused_key(settings: Settings) -> None:
    keyed = settings.model_copy(update={"sharadar_api_key": SecretStr("fake-key")})
    checks = run_smoke(keyed, transport=_transport(sharadar_status=403), on_c=_never_c)
    assert _status(checks, "sharadar_sep") == "fail"
    assert next(c.detail for c in checks if c.name == "sharadar_sep") == "HTTP 403"


def test_smoke_reports_network_error(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    keyed = settings.model_copy(update={"sharadar_api_key": SecretStr("fake-key")})
    checks = run_smoke(keyed, transport=httpx.MockTransport(handler), on_c=_never_c)
    assert _status(checks, "sharadar_sep") == "fail"
    assert _status(checks, "sec_submissions") == "fail"


def test_smoke_sec_http_error(settings: Settings) -> None:
    checks = run_smoke(settings, transport=_transport(sec_status=403), on_c=_never_c)
    assert _status(checks, "sec_submissions") == "fail"


def test_smoke_sec_without_sic(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "X"})

    checks = run_smoke(settings, transport=httpx.MockTransport(handler), on_c=_never_c)
    assert _status(checks, "sec_submissions") == "fail"


def test_smoke_missing_user_agent(settings: Settings) -> None:
    bare = settings.model_copy(update={"sec_user_agent": ""})
    checks = run_smoke(bare, transport=_transport(), on_c=_never_c)
    assert _status(checks, "sec_user_agent") == "fail"
    assert _status(checks, "sec_submissions") == "fail"


def test_smoke_refuses_c_drive(settings: Settings) -> None:
    checks = run_smoke(settings, transport=_transport(), on_c=lambda _: True)
    assert [check.name for check in checks] == ["sec_user_agent", "data_dir"]
    assert checks[1].status == "fail"
    assert not settings.data_dir.exists()


def test_smoke_unwritable_data_dir(settings: Settings, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    blocked = settings.model_copy(update={"statarb_data_dir": str(blocker / "data")})
    checks = run_smoke(blocked, transport=_transport(), on_c=_never_c)
    assert checks[1].status == "fail"


@pytest.mark.parametrize("problem", ["unset", "missing", "no_spy", "bad_spy"])
def test_smoke_stooq_problems(
    settings: Settings, stooq_dir: Path, tmp_path: Path, problem: str
) -> None:
    spy = stooq_dir / "d_us_txt/data/daily/us/nyse etfs/2/spy.us.txt"
    update: dict[str, object] = {}
    if problem == "unset":
        update["stooq_dir"] = None
    elif problem == "missing":
        update["stooq_dir"] = tmp_path / "nowhere"
    elif problem == "no_spy":
        spy.unlink()
    else:
        spy.write_text("garbage\n")
    checks = run_smoke(settings.model_copy(update=update), transport=_transport(), on_c=_never_c)
    assert any(check.status == "fail" for check in checks if check.name.startswith("stooq"))
