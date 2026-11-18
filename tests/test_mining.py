"""Guarded mining tests; all transactions and prices in this file are synthetic."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, SnapshotConfig, write_snapshot

STOCKS = ("AAA.NS", "BBB.NS")


def mining_module():
    try:
        from src import mining
    except ImportError:
        pytest.fail("src.mining must implement cutoff-aware guarded Apriori")
    return mining


@pytest.fixture
def samples():
    sessions = pd.bdate_range("2024-01-02", periods=9)
    rows = []
    for day in range(8):
        for ticker in STOCKS:
            feature = ("U" if day < 4 else "D") if ticker == "AAA.NS" else "N"
            target = ("U" if day < 4 else "D") if ticker == "AAA.NS" else (
                "U" if day in (0, 1, 2, 4) else "D"
            )
            symbol = ticker.removesuffix(".NS")
            rows.append({
                "feature_session": sessions[day],
                "feature_available_session": sessions[day],
                "execution_session": sessions[day + 1],
                "outcome_available_session": sessions[day + 1],
                "ticker": ticker,
                "feature_label": feature,
                "feature_item": f"{symbol}_{feature}",
                "target_label": target,
                "target_item": f"NEXT_{symbol}_{target}",
            })
    return pd.DataFrame(rows)


def test_encodes_one_boolean_transaction_per_day_not_per_stock(samples):
    encoded, sessions = mining_module().build_transactions(samples, STOCKS, "2024-01-12")
    assert len(encoded) == len(sessions) == 8
    assert encoded.dtypes.eq(bool).all()
    assert encoded.sum(axis=1).eq(4).all()
    assert encoded.index.equals(pd.DatetimeIndex(sessions.feature_session))


def test_window_counts_sessions_and_keeps_all_stock_items(samples):
    encoded, sessions = mining_module().build_transactions(samples, STOCKS, "2024-01-12", window=2)
    assert len(encoded) == 2
    assert sessions.execution_session.tolist() == list(pd.to_datetime(["2024-01-11", "2024-01-12"]))
    assert encoded.sum(axis=1).eq(4).all()


def test_encoding_excludes_outcomes_unknown_at_the_cutoff(samples):
    encoded, sessions = mining_module().build_transactions(samples, STOCKS, "2024-01-10")
    assert len(encoded) == 6
    assert sessions.execution_session.max() == pd.Timestamp("2024-01-10")
    assert sessions.feature_session.max() == pd.Timestamp("2024-01-09")


@pytest.mark.parametrize("window", [0, -1, 2.5])
def test_rejects_invalid_window_lengths(samples, window):
    with pytest.raises(DataValidationError, match="window"):
        mining_module().build_transactions(samples, STOCKS, "2024-01-12", window=window)


def test_refuses_to_call_short_history_a_full_window(samples):
    with pytest.raises(DataValidationError, match="history"):
        mining_module().build_transactions(samples, STOCKS, "2024-01-12", window=9)


def test_rejects_duplicate_stock_session_rows(samples):
    duplicated = pd.concat([samples, samples.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate"):
        mining_module().build_transactions(duplicated, STOCKS, "2024-01-12")


def test_rejects_an_incomplete_daily_stock_set(samples):
    with pytest.raises(DataValidationError, match="complete"):
        mining_module().build_transactions(samples.drop(index=0), STOCKS, "2024-01-12")


def test_rejects_a_future_target_supplied_as_a_feature(samples):
    samples.loc[0, "feature_item"] = "NEXT_AAA_U"
    with pytest.raises(DataValidationError, match="feature item"):
        mining_module().build_transactions(samples, STOCKS, "2024-01-12")


def test_rejects_a_target_without_the_next_prefix(samples):
    samples.loc[0, "target_item"] = "AAA_U"
    with pytest.raises(DataValidationError, match="target item"):
        mining_module().build_transactions(samples, STOCKS, "2024-01-12")


def test_rejects_disagreeing_execution_dates_within_one_transaction(samples):
    samples.loc[0, ["execution_session", "outcome_available_session"]] = pd.Timestamp("2024-01-04")
    with pytest.raises(DataValidationError, match="execution"):
        mining_module().build_transactions(samples, STOCKS, "2024-01-12")


def test_rule_metrics_use_daily_counts(samples):
    result = mining_module().mine_rules(
        samples, STOCKS, "2024-01-12", min_support=0.125, min_confidence=0.60
    )
    rule = result.rules.loc[
        result.rules.antecedents.map(lambda items: items == ("AAA_U",))
        & result.rules.consequent.eq("NEXT_BBB_U")
    ].iloc[0]
    assert rule.support == pytest.approx(3 / 8)
    assert rule.confidence == pytest.approx(3 / 4)
    assert rule.lift == pytest.approx(1.5)
    assert rule.antecedent_count == 4
    assert rule.consequent_count == 4
    assert rule.joint_count == 3
    assert rule.transaction_count == 8
    assert rule.target_ticker == "BBB.NS"
    assert rule.target_label == "U"


def test_all_mined_rules_have_only_past_antecedents_and_one_next_target(samples):
    result = mining_module().mine_rules(
        samples, STOCKS, "2024-01-12", min_support=0.125, min_confidence=0.60
    )
    assert not result.rules.empty
    assert result.rules.antecedents.map(len).eq(2).any()
    for rule in result.rules.itertuples():
        assert 1 <= len(rule.antecedents) <= 2
        assert all(not item.startswith("NEXT_") for item in rule.antecedents)
        assert isinstance(rule.consequent, str) and rule.consequent.startswith("NEXT_")
        assert rule.lift > 1


def test_independent_constant_features_produce_no_lift_above_one(samples):
    samples["feature_label"] = "N"
    samples["feature_item"] = samples.ticker.str.removesuffix(".NS") + "_N"
    result = mining_module().mine_rules(
        samples, STOCKS, "2024-01-12", min_support=0.125, min_confidence=0.50
    )
    assert result.rules.empty


def test_no_known_outcomes_is_a_valid_empty_result(samples):
    result = mining_module().mine_rules(samples, STOCKS, "2024-01-01")
    assert result.rules.empty and result.transactions.empty
    assert "antecedents" in result.rules and "confidence" in result.rules


@pytest.mark.parametrize("name,value", [
    ("min_support", 0.0), ("min_support", np.nan), ("min_support", 1.1),
    ("min_confidence", -0.1), ("min_confidence", np.inf), ("min_confidence", 1.1),
])
def test_rejects_invalid_mining_thresholds(samples, name, value):
    with pytest.raises(DataValidationError, match=name):
        mining_module().mine_rules(samples, STOCKS, "2024-01-12", **{name: value})


def test_stricter_filter_matches_direct_remining(samples):
    mining = mining_module()
    base = mining.mine_rules(samples, STOCKS, "2024-01-12", min_support=0.125, min_confidence=0.60)
    direct = mining.mine_rules(samples, STOCKS, "2024-01-12", min_support=0.25, min_confidence=0.80)
    selected = mining.select_rules(base, min_support=0.25, min_confidence=0.80)
    pd.testing.assert_frame_equal(selected, direct.rules)


@pytest.mark.parametrize("reuse", [False, True])
def test_exact_confidence_boundary_survives_direct_mining_and_reuse(samples, reuse):
    """15/25 is exactly 60%; mlxtend's intermediate arithmetic can drift below it."""
    mining = mining_module()
    sessions = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for day in range(39):
        for ticker in STOCKS:
            feature = ("U" if day < 25 else "D") if ticker == "AAA.NS" else "N"
            target = "N" if ticker == "AAA.NS" else ("U" if day < 15 else "D")
            symbol = ticker.removesuffix(".NS")
            rows.append({
                "feature_session": sessions[day], "feature_available_session": sessions[day],
                "execution_session": sessions[day + 1], "outcome_available_session": sessions[day + 1],
                "ticker": ticker, "feature_label": feature, "feature_item": f"{symbol}_{feature}",
                "target_label": target, "target_item": f"NEXT_{symbol}_{target}",
            })
    long_samples = pd.DataFrame(rows)
    result = mining.mine_rules(
        long_samples, STOCKS, sessions[-1], min_support=0.02,
        min_confidence=0.55 if reuse else 0.60,
    )
    rules = mining.select_rules(result, min_confidence=0.60) if reuse else result.rules
    selected = rules.loc[
        rules.antecedents.map(lambda items: items == ("AAA_U",))
        & rules.consequent.eq("NEXT_BBB_U")
    ]
    assert len(selected) == 1
    rule = selected.iloc[0]
    assert rule.antecedent_count == 25 and rule.joint_count == 15
    assert rule.confidence == 0.60
    assert rule.support == 15 / 39
    assert rule.lift == pytest.approx(1.56)


@pytest.mark.parametrize("thresholds", [{"min_support": 0.01}, {"min_confidence": 0.50}])
def test_cannot_relax_thresholds_without_remining(samples, thresholds):
    mining = mining_module()
    base = mining.mine_rules(samples, STOCKS, "2024-01-12", min_support=0.125, min_confidence=0.60)
    with pytest.raises(DataValidationError, match="relax"):
        mining.select_rules(base, **thresholds)


def test_shuffling_sample_rows_does_not_change_rule_order_or_values(samples):
    mining = mining_module()
    before = mining.mine_rules(samples, STOCKS, "2024-01-12", min_support=0.125)
    after = mining.mine_rules(samples.sample(frac=1.0, random_state=42), STOCKS, "2024-01-12", min_support=0.125)
    pd.testing.assert_frame_equal(before.rules, after.rules, check_exact=True)
    pd.testing.assert_frame_equal(before.transactions, after.transactions, check_exact=True)


def test_future_outcome_changes_cannot_change_fitted_rules(samples):
    mining = mining_module()
    cutoff = pd.Timestamp("2024-01-10")
    before = mining.mine_rules(samples, STOCKS, cutoff, min_support=0.125)
    assert not before.rules.empty
    changed = samples.copy(deep=True)
    future = changed.outcome_available_session > cutoff
    changed.loc[future, "feature_label"] = "N"
    changed.loc[future, "feature_item"] = changed.loc[future, "ticker"].str.removesuffix(".NS") + "_N"
    changed.loc[future, "target_label"] = "N"
    changed.loc[future, "target_item"] = "NEXT_" + changed.loc[future, "ticker"].str.removesuffix(".NS") + "_N"
    after = mining.mine_rules(changed, STOCKS, cutoff, min_support=0.125)
    pd.testing.assert_frame_equal(before.rules, after.rules, check_exact=True)
    pd.testing.assert_frame_equal(before.transactions, after.transactions, check_exact=True)


def test_cli_saves_training_provenance_and_refuses_overwrite(tmp_path, capsys):
    mining = mining_module()
    dates = pd.bdate_range("2024-01-02", periods=9)
    rows = []
    for ticker in (*STOCKS, "^NSEI"):
        for day, session in enumerate(dates):
            opening = 100.0 + day
            closing = opening * (1.01 if day % 2 else 0.99)
            rows.append({
                "date": session, "ticker": ticker, "open": opening,
                "high": max(opening, closing) + 1, "low": min(opening, closing) - 1,
                "close": closing, "volume": 0 if ticker == "^NSEI" else 1000,
                "dividends": 0.0, "stock_splits": 0.0,
            })
    source, output = tmp_path / "source", tmp_path / "mining"
    config = SnapshotConfig(start="2024-01-02", as_of="2024-01-12", stocks=STOCKS)
    prices = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    source_manifest = write_snapshot(prices, config, source, min_sessions=2)
    args = ["--snapshot", str(source), "--cutoff", "2024-01-12", "--window", "expanding", "--output", str(output)]
    assert mining.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["training_transactions"] == 7
    assert summary["metric_scope"] == "training_only"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"] == source_manifest["sha256"]
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert isinstance(json.loads((output / "rules.json").read_text()), list)
    assert hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest() == source_manifest["sha256"]
    assert mining.main(args) == 2
    assert "exists" in capsys.readouterr().err
