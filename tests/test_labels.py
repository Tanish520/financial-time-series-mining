"""Temporal label tests use deliberately synthetic bars, never market claims."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, SnapshotConfig, write_snapshot


def labels_module():
    try:
        from src import labels
    except ImportError:
        pytest.fail("src.labels must implement causal features and next-session targets")
    return labels


@pytest.fixture
def config():
    return SnapshotConfig(
        start="2024-01-04", as_of="2024-01-09", stocks=("AAA.NS", "BBB.NS")
    )


@pytest.fixture
def prices():
    sessions = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    bars = {
        "AAA.NS": ([100.0, 100.0, 110.0, 98.0], [100.0, 101.0, 99.0, 100.0]),
        "BBB.NS": ([200.0, 198.0, 198.0, 200.0], [200.0, 198.0, 201.0, 200.0]),
        "^NSEI": ([1000.0] * 4, [1000.0] * 4),
    }
    rows = []
    for ticker, (opens, closes) in bars.items():
        for session, opening, closing in zip(sessions, opens, closes):
            rows.append({
                "date": session, "ticker": ticker, "open": opening,
                "high": max(opening, closing) + 1.0,
                "low": min(opening, closing) - 1.0, "close": closing,
                "volume": 0.0 if ticker == "^NSEI" else 1000.0,
                "dividends": 0.0, "stock_splits": 0.0,
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_thresholds_are_strict_and_missing_is_not_neutral():
    values = pd.Series([-0.0051, -0.005, 0.0, 0.005, 0.0051, np.nan], name="return")
    actual = labels_module().label_returns(values, theta=0.005)
    expected = pd.Series(["D", "N", "N", "N", "U", pd.NA], name="return", dtype="string")
    pd.testing.assert_series_equal(actual, expected)


def test_division_roundoff_at_an_exact_boundary_is_neutral():
    values = pd.Series([100.5 / 100 - 1, 99.5 / 100 - 1])
    assert labels_module().label_returns(values, 0.005).tolist() == ["N", "N"]


def test_meaningful_values_just_outside_the_boundary_are_directional():
    values = pd.Series([0.005 + 1e-10, -0.005 - 1e-10])
    assert labels_module().label_returns(values, 0.005).tolist() == ["U", "D"]


@pytest.mark.parametrize("theta", [0.0, -0.005, np.nan, np.inf, 1.0])
def test_rejects_invalid_thresholds(theta):
    with pytest.raises(DataValidationError, match="theta"):
        labels_module().label_returns(pd.Series([0.0]), theta)


@pytest.mark.parametrize("bad_return", [np.inf, -np.inf])
def test_rejects_infinite_returns(bad_return):
    with pytest.raises(DataValidationError, match="finite"):
        labels_module().label_returns(pd.Series([bad_return]), 0.005)


def test_first_feature_is_unknown_and_benchmark_is_excluded(config, prices):
    tables = labels_module().build_label_tables(prices, config, theta=0.005)
    assert set(tables.features.ticker) == set(config.stocks)
    assert len(tables.features) == 8
    first = tables.features.loc[tables.features.feature_session == pd.Timestamp("2024-01-04")]
    assert first[["feature_return", "feature_label", "feature_item"]].isna().all().all()
    pd.testing.assert_series_equal(
        tables.features.feature_available_session, tables.features.feature_session,
        check_names=False,
    )


def test_features_are_per_stock_and_targets_use_next_open(config, prices):
    tables = labels_module().build_label_tables(prices, config, theta=0.005)
    rows = tables.samples.loc[tables.samples.feature_session == pd.Timestamp("2024-01-05")]
    aaa = rows.loc[rows.ticker == "AAA.NS"].iloc[0]
    bbb = rows.loc[rows.ticker == "BBB.NS"].iloc[0]
    assert aaa.feature_return == pytest.approx(0.01)
    assert aaa.feature_item == "AAA_U"
    assert aaa.target_return == pytest.approx(99 / 110 - 1)
    assert aaa.target_item == "NEXT_AAA_D"
    assert bbb.feature_return == pytest.approx(-0.01)
    assert bbb.feature_item == "BBB_D"
    assert bbb.target_return == pytest.approx(201 / 198 - 1)
    assert bbb.target_item == "NEXT_BBB_U"


def test_next_session_skips_the_weekend_and_sets_outcome_availability(config, prices):
    samples = labels_module().build_label_tables(prices, config, theta=0.005).samples
    friday = samples.loc[samples.feature_session == pd.Timestamp("2024-01-05")]
    assert (friday.execution_session == pd.Timestamp("2024-01-08")).all()
    assert (friday.outcome_available_session == friday.execution_session).all()
    assert (samples.feature_session < samples.execution_session).all()
    assert len(samples) == 4


def test_final_feature_can_be_used_but_its_target_is_unknown(config, prices):
    tables = labels_module().build_label_tables(prices, config, theta=0.005)
    last = pd.Timestamp("2024-01-09")
    features = tables.features.loc[tables.features.feature_session == last]
    targets = tables.targets.loc[tables.targets.feature_session == last]
    assert features.feature_item.notna().all()
    unknown_columns = [
        "execution_session", "outcome_available_session", "target_return", "target_label", "target_item"
    ]
    assert targets[unknown_columns].isna().all().all()
    assert last not in set(tables.samples.feature_session)


def test_training_filter_waits_for_the_target_session_to_finish(config, prices):
    labels = labels_module()
    samples = labels.build_label_tables(prices, config, theta=0.005).samples
    assert labels.eligible_training_samples(samples, "2024-01-05").empty
    known_monday = labels.eligible_training_samples(samples, "2024-01-08")
    assert len(known_monday) == 2
    assert (known_monday.feature_session == pd.Timestamp("2024-01-05")).all()
    assert (known_monday.outcome_available_session <= pd.Timestamp("2024-01-08")).all()


def test_training_filter_rejects_an_outcome_marked_known_before_execution(config, prices):
    labels = labels_module()
    samples = labels.build_label_tables(prices, config, theta=0.005).samples.copy()
    samples.loc[0, "outcome_available_session"] = samples.loc[0, "feature_session"]
    with pytest.raises(DataValidationError, match="availability"):
        labels.eligible_training_samples(samples, "2024-01-05")


@pytest.mark.parametrize("cutoff", ["2024-01-08 10:00", "2024-01-08T00:00+05:30"])
def test_training_cutoff_requires_explicit_session_close_date(config, prices, cutoff):
    labels = labels_module()
    samples = labels.build_label_tables(prices, config, theta=0.005).samples
    with pytest.raises(DataValidationError, match="session"):
        labels.eligible_training_samples(samples, cutoff)


def test_future_price_changes_cannot_change_features_available_today(config, prices):
    """Required leakage test, with a future-dependent target as a negative control."""
    labels = labels_module()
    cutoff = pd.Timestamp("2024-01-05")
    before = labels.build_label_tables(prices, config, theta=0.005)
    changed = prices.copy(deep=True)
    future = changed.date > cutoff
    changed.loc[future, "open"] *= 3.0
    changed.loc[future, "close"] *= 0.4
    changed.loc[future, "high"] = changed.loc[future, ["open", "close"]].max(axis=1) * 1.1
    changed.loc[future, "low"] = changed.loc[future, ["open", "close"]].min(axis=1) * 0.9
    after = labels.build_label_tables(changed, config, theta=0.005)
    pd.testing.assert_frame_equal(
        before.features.loc[before.features.feature_session <= cutoff].reset_index(drop=True),
        after.features.loc[after.features.feature_session <= cutoff].reset_index(drop=True),
        check_exact=True,
    )
    original_targets = before.targets.loc[before.targets.feature_session == cutoff, "target_return"]
    altered_targets = after.targets.loc[after.targets.feature_session == cutoff, "target_return"]
    assert not original_targets.equals(altered_targets)


def test_features_computed_on_a_prefix_match_full_history(config, prices):
    labels = labels_module()
    cutoff = pd.Timestamp("2024-01-05")
    full = labels.build_label_tables(prices, config, theta=0.005)
    prefix_prices = prices.loc[prices.date <= cutoff].copy()
    prefix = labels.build_label_tables(
        prefix_prices, replace(config, as_of="2024-01-05"), theta=0.005
    )
    pd.testing.assert_frame_equal(
        full.features.loc[full.features.feature_session <= cutoff].reset_index(drop=True),
        prefix.features, check_exact=True,
    )


def test_shuffled_input_does_not_mix_stock_lags_or_change_results(config, prices):
    labels = labels_module()
    before = labels.build_label_tables(prices, config, theta=0.005)
    after = labels.build_label_tables(prices.sample(frac=1.0, random_state=42), config, theta=0.005)
    for name in ("features", "targets", "samples"):
        pd.testing.assert_frame_equal(getattr(before, name), getattr(after, name), check_exact=True)


def test_building_labels_does_not_modify_the_price_input(config, prices):
    original = prices.copy(deep=True)
    labels_module().build_label_tables(prices, config, theta=0.005)
    pd.testing.assert_frame_equal(prices, original, check_exact=True)


def test_missing_prices_are_rejected_instead_of_becoming_neutral(config, prices):
    prices.loc[0, "close"] = np.nan
    with pytest.raises(DataValidationError, match="price"):
        labels_module().build_label_tables(prices, config, theta=0.005)


def test_cli_writes_derived_tables_with_source_lineage(config, prices, tmp_path, capsys):
    labels = labels_module()
    source, output = tmp_path / "source", tmp_path / "labels"
    source_manifest = write_snapshot(prices, config, source, min_sessions=2)
    before_hash = hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest()
    assert labels.main([
        "--snapshot", str(source), "--theta", "0.005", "--output", str(output)
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["supervised_sample_rows"] == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"] == source_manifest["sha256"] == before_hash
    assert manifest["theta"] == 0.005
    for filename in ("features.csv", "targets.csv", "samples.csv"):
        assert manifest["files"][filename] == hashlib.sha256((output / filename).read_bytes()).hexdigest()
    assert hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest() == before_hash
    assert len(pd.read_csv(output / "samples.csv")) == 4


def test_cli_refuses_to_overwrite_derived_results(config, prices, tmp_path, capsys):
    labels = labels_module()
    source, output = tmp_path / "source", tmp_path / "labels"
    write_snapshot(prices, config, source, min_sessions=2)
    args = ["--snapshot", str(source), "--output", str(output)]
    assert labels.main(args) == 0
    capsys.readouterr()
    assert labels.main(args) == 2
    assert "exists" in capsys.readouterr().err
