"""Contract checks for the data boundary; all price inputs here are synthetic."""

import hashlib
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest


def data_module():
    """A missing implementation should be an explicit contract-test failure."""
    try:
        from src import data
    except ImportError:
        pytest.fail("src.data must implement the frozen-snapshot data boundary")
    return data


@pytest.fixture
def config():
    return data_module().SnapshotConfig(
        start="2024-01-02",
        as_of="2024-01-03",
        stocks=("AAA.NS", "BBB.NS"),
        benchmark="^NSEI",
    )


@pytest.fixture
def prices():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for ticker in ("AAA.NS", "BBB.NS", "^NSEI"):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": 100.0,
                    "high": 103.0,
                    "low": 99.0,
                    "close": 102.0,
                    "volume": 0.0 if ticker == "^NSEI" else 1000.0,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_accepts_valid_aligned_prices_with_zero_index_volume(config, prices):
    data_module().validate_prices(prices, config, min_sessions=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0.0, -1.0])
def test_rejects_missing_or_nonpositive_prices(config, prices, bad_value):
    prices.loc[0, "close"] = bad_value
    with pytest.raises(data_module().DataValidationError, match="price"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_rejects_duplicate_stock_session(config, prices):
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(data_module().DataValidationError, match="Duplicate"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_rejects_impossible_ohlc_range(config, prices):
    prices.loc[0, "high"] = 101.0
    with pytest.raises(data_module().DataValidationError, match="OHLC"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_rejects_missing_stock_session_instead_of_filling(config, prices):
    prices = prices.drop(index=0)
    with pytest.raises(data_module().DataValidationError, match="calendar"):
        data_module().validate_prices(prices, config, min_sessions=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
def test_rejects_missing_or_negative_volume(config, prices, bad_value):
    prices.loc[0, "volume"] = bad_value
    with pytest.raises(data_module().DataValidationError, match="volume"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_accepts_zero_stock_volume_without_inventing_prices(config, prices):
    prices.loc[0, "volume"] = 0.0
    data_module().validate_prices(prices, config, min_sessions=2)


def test_rejects_a_snapshot_too_short_for_the_research(config, prices):
    with pytest.raises(data_module().DataValidationError, match="sessions"):
        data_module().validate_prices(prices, config, min_sessions=3)


def test_rejects_an_unrequested_ticker(config, prices):
    prices.loc[0, "ticker"] = "UNKNOWN.NS"
    with pytest.raises(data_module().DataValidationError, match="ticker"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_rejects_bars_outside_the_frozen_request(config, prices):
    prices.loc[prices.date == pd.Timestamp("2024-01-03"), "date"] = pd.Timestamp(
        "2024-01-04"
    )
    with pytest.raises(data_module().DataValidationError, match="requested"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_conservative_cutoff_rejects_the_current_india_date():
    today = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
    with pytest.raises(data_module().DataValidationError, match="completed"):
        data_module().SnapshotConfig(start="2020-01-01", as_of=today)


def test_normalizes_yahoo_columns_without_filling_a_missing_price(config, prices):
    data = data_module()
    yahoo_columns = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends", "stock_splits": "Stock Splits",
    }
    frames = {
        ticker: group.set_index("date")[list(yahoo_columns)].rename(columns=yahoo_columns)
        for ticker, group in prices.groupby("ticker")
    }
    raw = pd.concat(frames, axis=1)
    raw.loc[pd.Timestamp("2024-01-03"), ("AAA.NS", "Close")] = np.nan
    normalized = data.normalize_download(raw, config.tickers)
    missing = normalized.loc[
        (normalized.ticker == "AAA.NS")
        & (normalized.date == pd.Timestamp("2024-01-03")), "close"
    ]
    assert missing.isna().all()
    with pytest.raises(data.DataValidationError, match="price"):
        data.validate_prices(normalized, config, min_sessions=2)


def test_snapshot_round_trip_works_without_yfinance(config, prices, tmp_path, monkeypatch):
    data = data_module()
    destination = tmp_path / "snapshot"
    data.write_snapshot(prices, config, destination, min_sessions=2)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    loaded, manifest = data.load_snapshot(destination)
    pd.testing.assert_frame_equal(loaded, prices, check_dtype=False)
    assert manifest["sha256"] == hashlib.sha256(
        (destination / "prices.csv").read_bytes()
    ).hexdigest()
    assert manifest["rows"] == len(prices)
    assert manifest["sessions_per_ticker"] == {"AAA.NS": 2, "BBB.NS": 2, "^NSEI": 2}


def test_snapshot_loader_rejects_modified_price_file(config, prices, tmp_path):
    data = data_module()
    destination = tmp_path / "snapshot"
    data.write_snapshot(prices, config, destination, min_sessions=2)
    csv_path = destination / "prices.csv"
    csv_path.write_bytes(csv_path.read_bytes().replace(b"102", b"104", 1))
    with pytest.raises(data.DataValidationError, match="checksum"):
        data.load_snapshot(destination)


def test_snapshot_writer_never_overwrites_frozen_data(config, prices, tmp_path):
    data = data_module()
    destination = tmp_path / "snapshot"
    data.write_snapshot(prices, config, destination, min_sessions=2)
    with pytest.raises(data.DataValidationError, match="exists"):
        data.write_snapshot(prices, config, destination, min_sessions=2)


def test_download_requests_explicit_adjustments_and_exclusive_end(
    config, prices, tmp_path, monkeypatch
):
    """Provider calls are the one boundary stubbed to keep tests offline."""
    from types import SimpleNamespace

    data = data_module()
    columns = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends", "stock_splits": "Stock Splits",
    }
    raw = pd.concat(
        {
            ticker: group.set_index("date")[list(columns)].rename(columns=columns)
            for ticker, group in prices.groupby("ticker")
        }, axis=1,
    )
    observed = {}

    def fake_download(**kwargs):
        observed.update(kwargs)
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))
    data.download_snapshot(config, tmp_path / "snapshot", min_sessions=2)
    assert observed["start"] == "2024-01-02"
    assert observed["end"] == "2024-01-04"
    assert observed["auto_adjust"] is True
    assert observed["back_adjust"] is False
    assert observed["repair"] is False
    assert observed["keepna"] is True
    assert observed["actions"] is True
    assert observed["interval"] == "1d"


def test_rejects_a_request_without_distinct_stocks(config):
    with pytest.raises(data_module().DataValidationError, match="distinct"):
        replace(config, stocks=("AAA.NS", "AAA.NS"))


def test_rejects_an_official_market_holiday_in_accepted_prices(config, prices):
    config = replace(config, start="2026-01-14", as_of="2026-01-16")
    prices["date"] = prices["date"].map(
        {pd.Timestamp("2024-01-02"): pd.Timestamp("2026-01-15"),
         pd.Timestamp("2024-01-03"): pd.Timestamp("2026-01-16")}
    )
    with pytest.raises(data_module().DataValidationError, match="non-trading"):
        data_module().validate_prices(prices, config, min_sessions=2)


def test_download_audits_official_holiday_rows_instead_of_using_them(
    config, prices, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    data = data_module()
    config = replace(config, start="2026-01-14", as_of="2026-01-16")
    prices["date"] = prices["date"].map(
        {pd.Timestamp("2024-01-02"): pd.Timestamp("2026-01-14"),
         pd.Timestamp("2024-01-03"): pd.Timestamp("2026-01-16")}
    )
    holiday = prices.loc[prices.date == pd.Timestamp("2026-01-14")].copy()
    holiday["date"] = pd.Timestamp("2026-01-15")
    holiday[["open", "high", "low", "close"]] = 100.0
    holiday["volume"] = 0.0
    holiday.loc[holiday.ticker == "^NSEI", ["open", "high", "low", "close"]] = np.nan
    observed = pd.concat([prices, holiday], ignore_index=True)
    columns = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends", "stock_splits": "Stock Splits",
    }
    raw = pd.concat(
        {ticker: group.set_index("date")[list(columns)].rename(columns=columns)
         for ticker, group in observed.groupby("ticker")}, axis=1,
    ).sort_index()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda **kwargs: raw))
    destination = tmp_path / "snapshot"
    manifest = data.download_snapshot(config, destination, min_sessions=2)
    loaded, _ = data.load_snapshot(destination)
    assert len(loaded) == len(prices)
    assert pd.Timestamp("2026-01-15") not in set(loaded.date)
    audit = pd.read_csv(destination / "excluded_non_sessions.csv")
    assert len(audit) == len(config.tickers)
    assert audit.loc[audit.ticker == "^NSEI", "close"].isna().all()
    assert manifest["excluded_non_sessions"][0]["date"] == "2026-01-15"
    assert manifest["excluded_non_sessions"][0]["rows"] == len(config.tickers)
    assert manifest["excluded_non_sessions"][0]["source_url"] == (
        "https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf"
    )
    audit_path = destination / "excluded_non_sessions.csv"
    audit_path.write_bytes(audit_path.read_bytes() + b"\n")
    with pytest.raises(data.DataValidationError, match="checksum"):
        data.load_snapshot(destination)
