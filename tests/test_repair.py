"""Primary exchange replacements are bounded and preserve the original cache."""

import hashlib

import pandas as pd
import pytest

from src.data import DataValidationError, SnapshotConfig, load_snapshot, write_snapshot


def repair_module():
    try:
        from src import repair
    except ImportError:
        pytest.fail("src.repair must produce an audited immutable primary-source correction")
    return repair


@pytest.fixture
def repair_inputs():
    config = SnapshotConfig("2025-03-17", "2025-03-19", stocks=("AAA.NS", "BBB.NS"))
    rows = []
    for ticker in config.tickers:
        for day, close in zip(pd.to_datetime(["2025-03-17", "2025-03-18", "2025-03-19"]), [50.0, 50.0, 55.0]):
            zero = ticker == "AAA.NS" and day == pd.Timestamp("2025-03-18")
            rows.append({"date": day, "ticker": ticker, "open": close, "high": close if zero else close+1,
                         "low": close if zero else close-1, "close": close,
                         "volume": 0 if zero or ticker == "^NSEI" else 1000,
                         "dividends": 0.0, "stock_splits": 0.0})
    prices = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    primary = pd.DataFrame([
        {"SYMBOL": symbol, "SERIES": "EQ", "DATE1": "18-Mar-2025", "PREV_CLOSE": 100.0,
         "OPEN_PRICE": 102.0, "HIGH_PRICE": 108.0, "LOW_PRICE": 100.0, "CLOSE_PRICE": 106.0, "TTL_TRD_QNTY": 2000}
        for symbol in ("AAA", "BBB")
    ])
    after = pd.DataFrame([
        {"SYMBOL": symbol, "SERIES": "EQ", "DATE1": "19-Mar-2025", "CLOSE_PRICE": 110.0}
        for symbol in ("AAA", "BBB")
    ])
    return prices, config, primary, after


def test_primary_ohlc_uses_verified_adjustment_scale_and_preserves_all_other_rows(repair_inputs):
    prices, config, primary, after = repair_inputs
    result, audit = repair_module().correct_placeholder_bars(prices, config, primary, after)
    changed = prices.ticker.eq("AAA.NS") & prices.date.eq("2025-03-18")
    row = result.loc[changed].iloc[0]
    assert (row.open, row.high, row.low, row.close, row.volume) == (51, 54, 50, 53, 2000)
    pd.testing.assert_frame_equal(prices.loc[~changed], result.loc[~changed])
    assert len(audit) == 1 and audit[0]["adjustment_scale"] == 0.5
    assert audit[0]["original"]["volume"] == 0 and audit[0]["corrected"]["close"] == 53
    assert audit[0]["scale_relative_discrepancy"] == 0


@pytest.mark.parametrize("mutation", ["wrong_date", "missing_stock", "zero_primary_volume", "inconsistent_scale", "corporate_action", "not_flat"])
def test_ambiguous_or_unverified_correction_is_rejected(repair_inputs, mutation):
    prices, config, primary, after = repair_inputs
    if mutation == "wrong_date": primary.loc[0, "DATE1"] = "17-Mar-2025"
    elif mutation == "missing_stock": primary = primary.iloc[1:]
    elif mutation == "zero_primary_volume": primary.loc[0, "TTL_TRD_QNTY"] = 0
    elif mutation == "inconsistent_scale": after.loc[0, "CLOSE_PRICE"] = 150
    elif mutation == "corporate_action": prices.loc[prices.date.eq("2025-03-18") & prices.ticker.eq("AAA.NS"), "dividends"] = 1
    else: prices.loc[prices.date.eq("2025-03-18") & prices.ticker.eq("AAA.NS"), "high"] += 1
    with pytest.raises(DataValidationError):
        repair_module().correct_placeholder_bars(prices, config, primary, after)


def test_corrected_snapshot_keeps_source_and_verifies_primary_audit_bytes(repair_inputs, tmp_path):
    prices, config, primary, after = repair_inputs
    source, output, audit_dir = tmp_path / "source", tmp_path / "corrected", tmp_path / "primary"
    audit_dir.mkdir()
    original = write_snapshot(prices, config, source, min_sessions=2)
    primary.to_csv(audit_dir / "sec_bhavdata_full_18032025.csv", index=False)
    after.to_csv(audit_dir / "sec_bhavdata_full_19032025.csv", index=False)
    repair_module().create_corrected_snapshot(source, audit_dir, output)
    actual, manifest = load_snapshot(output)
    assert manifest["correction"]["source_sha256"] == original["sha256"]
    assert manifest["correction"]["row_count"] == 1
    assert hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest() == original["sha256"]
    assert actual.loc[actual.ticker.eq("AAA.NS") & actual.date.eq("2025-03-18"), "close"].iloc[0] == 53
    path = output / "corrections.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataValidationError, match="audit checksum"):
        load_snapshot(output)
    with pytest.raises(DataValidationError, match="exists"):
        repair_module().create_corrected_snapshot(source, audit_dir, output)
