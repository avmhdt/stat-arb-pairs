from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from fixtures.synth import make_market
from statarb.data import build, manifest, schemas
from statarb.data.market import load_market_data

REPO = Path(__file__).resolve().parents[2]


def write_processed(processed: Path, profile: str = "prototype") -> None:
    """The synthetic market as processed tables."""
    data = make_market(seed=2, years=3).data
    tables = {
        "security_master": data.master,
        "ticker_history": data.ticker_history,
        "industry_history": data.industry,
        "prices_daily": data.prices,
        "etf_returns": data.etf_returns,
        "borrow_rates": data.borrow,
    }
    build.write_tables(processed, tables)
    schemas.write_build_info(processed, {"data_profile": profile, "manifest_hash": "abc"})


def test_manifest_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "x"
    raw.mkdir(parents=True)
    file = raw / "a.csv"
    file.write_text("a,b\n1,2\n")
    entry = manifest.entry_for(tmp_path, file, "stooq", "a", "file:x", rows=1)
    assert entry.path == "raw/x/a.csv"
    assert entry.sha256 == manifest.sha256_file(file)
    assert not manifest.is_recorded(tmp_path, file)
    manifest.record(tmp_path, [entry])
    assert manifest.is_recorded(tmp_path, file)
    first = manifest.manifest_hash(tmp_path)
    manifest.record(tmp_path, [entry])
    assert manifest.read_manifest(tmp_path).height == 1
    assert manifest.manifest_hash(tmp_path) == first
    assert manifest.manifest_hash(tmp_path, ["other"]) != first
    assert manifest.verify(tmp_path) == []
    file.write_text("a,b\n1,3\n")
    assert manifest.verify(tmp_path) == ["raw/x/a.csv: sha256 changed"]
    assert not manifest.is_recorded(tmp_path, file)
    file.write_text("short")
    assert manifest.verify(tmp_path) == ["raw/x/a.csv: size changed"]
    file.unlink()
    assert manifest.verify(tmp_path, ["stooq"]) == ["raw/x/a.csv: missing"]
    assert not manifest.is_recorded(tmp_path, file)


def test_conform_rules() -> None:
    frame = pl.DataFrame({"etf": ["SPY"], "date": [date(2020, 1, 2)], "ret_total": [0.01]})
    shaped = schemas.conform(frame, schemas.ETF_RETURNS)
    assert shaped.columns == list(schemas.ETF_RETURNS)
    assert shaped["close"].null_count() == 1
    with pytest.raises(schemas.SchemaError, match="missing key"):
        schemas.conform(frame.drop("etf"), schemas.ETF_RETURNS)
    with pytest.raises(schemas.SchemaError, match="outside"):
        schemas.conform(frame.with_columns(pl.lit(1).alias("junk")), schemas.ETF_RETURNS)
    with pytest.raises(schemas.SchemaError, match="null keys"):
        schemas.conform(
            frame.with_columns(pl.lit(None, pl.String).alias("etf")), schemas.ETF_RETURNS
        )


def test_write_and_load_processed(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_processed(processed)
    years = sorted(path.name for path in (processed / "prices_daily").iterdir())
    assert years == ["year=2015", "year=2016", "year=2017"]
    write_processed(processed)
    assert len(list((processed / "prices_daily").rglob("*.parquet"))) == 3
    data = load_market_data(processed, date(2016, 1, 1), date(2016, 12, 31), ["close"], ("common",))
    assert data.prices.columns == ["date", "security_id", "close"]
    assert data.sessions[0] == date(2016, 1, 1)
    assert data.data_profile == "prototype"
    assert data.manifest_hash == "abc"
    with pytest.raises(KeyError):
        data.table("nope")
    assert schemas.read_build_info(tmp_path) == {}


def test_attach_delistings_on_last_row() -> None:
    prices = pl.DataFrame(
        {
            "security_id": ["a", "a", "a", "b"],
            "date": [date(2020, 1, d) for d in (2, 3, 6)] + [date(2020, 1, 2)],
        }
    )
    delistings = pl.DataFrame(
        {
            "security_id": ["a", "c"],
            "delist_date": [date(2020, 1, 4), date(2020, 1, 4)],
            "dlret": [-0.3, 0.1],
            "imputed": [True, False],
        }
    )
    out = build.attach_delistings(prices, delistings).sort("security_id", "date")
    assert out["ret_delist"].to_list() == [None, -0.3, None, None]
    assert out["ret_delist_imputed"].to_list() == [None, True, None, None]


def _eqpit_like() -> object:
    class Store:
        securities = pl.DataFrame(
            {
                "security_id": ["SHR:1", "SHR:1", "SHR:2"],
                "ticker": ["OLD", "NEW", "B"],
                "name": ["One", "One", "Two"],
                "exchange": ["NYSE", "NYSE", "NASDAQ"],
                "share_type": ["common", "common", "adr"],
                "sic": [2834, 2834, None],
                "valid_from": [date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 1)],
                "valid_to": [date(2020, 1, 2), None, None],
            },
            schema_overrides={"sic": pl.Int16},
        )
        prices = pl.DataFrame(
            {
                "security_id": ["SHR:1"] * 3 + ["SHR:2"] * 2,
                "date": [date(2020, 1, d) for d in (2, 3, 6)]
                + [date(2020, 1, 2), date(2020, 1, 3)],
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": [10.0, 11.0, 12.1, 5.0, 5.5],
                "volume": 100.0,
                "ret": [None, 0.1, 0.1, None, 0.1],
                "shares_out": 1e6,
                "mcap": [1e7, 1.1e7, 1.21e7, 5e6, 5.5e6],
                "price_flag": pl.Series([0, 1, 0, 0, 0], dtype=pl.UInt8),
            }
        )
        delistings = pl.DataFrame(
            {
                "security_id": ["SHR:2"],
                "delist_date": [date(2020, 1, 5)],
                "dlret": [-0.3],
                "imputation_rule": ["shumway_nasdaq"],
            }
        )

    return Store()


def test_sharadar_mapping() -> None:
    tables = build.sharadar_tables(_eqpit_like())
    master = tables["security_master"].sort("security_id")
    assert master["security_id"].to_list() == ["shr:1", "shr:2"]
    assert master["ticker"].to_list() == ["NEW", "B"]
    assert master["security_type"].to_list() == ["common", "other"]
    assert master["last_date"].to_list() == [None, date(2020, 1, 3)]
    prices = tables["prices_daily"].sort("security_id", "date")
    assert prices["close_is_mid"].to_list() == [False, True, False, False, False]
    assert prices["tr_index"].to_list() == pytest.approx([1.0, 1.1, 1.21, 1.0, 1.1])
    assert prices["ret_delist"].to_list() == [None, None, None, None, -0.3]
    assert prices["ret_delist_imputed"].to_list()[-1] is True
    assert tables["industry_history"].height == 2
    assert tables["ticker_history"].height == 3
    for name, frame in tables.items():
        schemas.conform(frame, schemas.TABLES[name])


def test_stooq_mapping() -> None:
    daily = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * 3 + ["SPY.US"] * 3 + ["OLD.US"],
            "date": [date(2020, 1, d) for d in (2, 3, 6)] * 2 + [date(2020, 1, 2)],
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": [10.0, 11.0, 12.1, 100.0, 101.0, 102.01, 5.0],
            "volume": 10.0,
        }
    )
    index = pl.DataFrame(
        {
            "ticker": ["AAA.US", "SPY.US", "OLD.US"],
            "symbol": ["AAA", "SPY", "OLD"],
            "exchange": ["NYSE", "NYSE", "NASDAQ"],
            "security_type": ["stock", "etf", "stock"],
            "path": ["a", "b", "c"],
        }
    )
    sic = pl.DataFrame({"symbol": ["AAA"], "sic": [3571]})
    tables = build.stooq_tables(daily, index, sic)
    master = tables["security_master"]
    assert master["security_id"].to_list() == [
        "free:AAA:2020-01-02",
        "free:OLD:2020-01-02",
        "free:SPY:2020-01-02",
    ]
    assert master["security_type"].to_list() == ["common", "common", "etf"]
    assert master["last_date"].to_list() == [None, date(2020, 1, 2), None]
    assert tables["industry_history"]["is_point_in_time"].to_list() == [False]
    assert "mktcap_usd" not in tables["prices_daily"].columns
    for name, frame in tables.items():
        schemas.conform(frame, schemas.TABLES[name])
    assert build.stooq_tables(daily, index, None)["industry_history"].is_empty()
    etf = build.stooq_etf_returns(daily, ["SPY", "XLK"])
    assert etf["etf"].unique().to_list() == ["SPY"]
    assert etf["ret_total"].to_list()[1] == pytest.approx(0.01)


def test_borrow_schedule() -> None:
    rows = build.borrow_rates(REPO / "configs" / "borrow_schedule.yaml")
    assert rows["bucket"].to_list() == ["top500", "top1000", "rest"]
    assert rows["fee_rate_annual"].to_list() == [0.0025, 0.01, 0.05]
    assert rows["source"].unique().to_list() == ["assumed"]
    schemas.conform(rows, schemas.BORROW_RATES, schemas.KEYS["borrow_rates"])


def test_etf_returns_from_sfp() -> None:
    from statarb.data.providers.sharadar import etf_returns_from_sfp

    sfp = pl.DataFrame(
        {
            "ticker": ["XLK", "XLK", "XLK", "QQQ"],
            "date": [date(2020, 1, d) for d in (2, 3, 6)] + [date(2020, 1, 2)],
            "close": [50.0, 51.0, 52.0, 1.0],
            "closeadj": [100.0, 102.0, 104.04, 1.0],
            "closeunadj": [100.0, 102.0, 104.0, 1.0],
            "volume": [10.0, 20.0, 30.0, 1.0],
        }
    ).lazy()
    rows = etf_returns_from_sfp(sfp, ["XLK"])
    assert rows["ret_total"].to_list()[1:] == pytest.approx([0.02, 0.02])
    assert rows["close"].to_list() == [100.0, 102.0, 104.0]
    assert rows["volume"].to_list() == pytest.approx([5.0, 10.0, 15.0])
    assert rows["inception_date"].unique().to_list() == [date(2020, 1, 2)]


def test_sfp_to_parquet(tmp_path: Path) -> None:
    import zipfile

    from statarb.data.providers.sharadar import sfp_to_parquet

    archive = tmp_path / "funds.csv.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "SHARADAR_SFP.csv",
            "ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated\n"
            "XLK,2020-01-02,1,1,1,1,10,1,1,2020-01-03\n",
        )
    assert sfp_to_parquet(archive, tmp_path / "SFP.parquet") == 1
    frame = pl.read_parquet(tmp_path / "SFP.parquet")
    assert frame.schema["date"] == pl.Date
    assert frame.schema["volume"] == pl.Float64
    assert not (tmp_path / ".extract-SFP").exists()


def test_build_info_written(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "borrow"
    raw.mkdir(parents=True)
    (raw / "x.yaml").write_text("x")
    manifest.record(
        tmp_path,
        [
            manifest.entry_for(
                tmp_path,
                raw / "x.yaml",
                "borrow",
                "x",
                "u",
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    )
    info = build.build_info(tmp_path, "headline", "sharadar", {"a": 1}, ["borrow"])
    assert info["manifest_hash"] == manifest.manifest_hash(tmp_path, ["borrow"])
    assert info["data_profile"] == "headline"


def test_publish_retries_permission_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir()
    real = Path.rename
    calls = []

    def flaky(self: Path, other: Path) -> Path:
        calls.append(other)
        if len(calls) < 3:
            raise PermissionError("held open")
        return real(self, other)

    monkeypatch.setattr(Path, "rename", flaky)
    manifest.publish(staging, target, wait=0.0)
    assert target.is_dir() and len(calls) == 3
    calls.clear()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(Path, "rename", lambda self, o: (_ for _ in ()).throw(PermissionError("x")))
    with pytest.raises(PermissionError):
        manifest.publish(other, tmp_path / "never", attempts=2, wait=0.0)
