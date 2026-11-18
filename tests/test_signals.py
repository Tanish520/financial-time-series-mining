"""Frozen-rule matching checks using synthetic rules and observed-only features."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError
from src.mining import MiningResult, RULE_COLUMNS

STOCKS = ("AAA.NS", "BBB.NS")


def signals_module():
    try:
        from src import signals
    except ImportError:
        pytest.fail("src.signals must match frozen rules without future outcomes")
    return signals


def rule(antecedents=("AAA_U",), target="AAA.NS", label="U", confidence=0.8, count=100):
    joint = int(round(confidence * count))
    return {
        "antecedents": antecedents, "consequent": f"NEXT_{target.removesuffix('.NS')}_{label}",
        "target_ticker": target, "target_label": label, "confidence": confidence,
        "support": joint / 500, "lift": confidence / 0.2,
        "antecedent_support": count / 500, "consequent_support": 0.2,
        "joint_count": joint, "antecedent_count": count,
        "consequent_count": 100, "transaction_count": 500,
    }


def model(rows):
    return MiningResult(
        pd.DataFrame(rows, columns=RULE_COLUMNS), pd.DataFrame(), pd.DataFrame(),
        pd.Timestamp("2024-01-04"), 0.02, 0.60,
    )


@pytest.fixture
def features():
    rows = []
    for day, labels in (("2024-01-05", ("U", "D")), ("2024-01-08", ("N", "U"))):
        for ticker, label in zip(STOCKS, labels):
            rows.append({
                "feature_session": pd.Timestamp(day), "feature_available_session": pd.Timestamp(day),
                "ticker": ticker, "feature_return": {"U": 0.01, "D": -0.01, "N": 0.0}[label],
                "feature_label": label, "feature_item": ticker.removesuffix(".NS") + "_" + label,
            })
    return pd.DataFrame(rows)


def test_matching_up_rule_produces_a_long_with_provenance(features):
    result = signals_module().predict_signals(model([rule()]), features, STOCKS)
    row = result.loc[(result.feature_session == pd.Timestamp("2024-01-05")) & (result.ticker == "AAA.NS")].iloc[0]
    assert row.prediction_label == "U" and row.action == "LONG"
    assert row.consequent == "NEXT_AAA_U" and row.confidence == 0.8
    assert row.rule_fit_cutoff == pd.Timestamp("2024-01-04")


def test_highest_confidence_wins_even_when_that_means_cash(features):
    rules = [rule(), rule(("BBB_D",), label="D", confidence=0.9)]
    result = signals_module().predict_signals(model(rules), features, STOCKS)
    row = result.iloc[0]
    assert row.ticker == "AAA.NS" and row.prediction_label == "D"
    assert row.action == "CASH" and row.confidence == 0.9


def test_exact_conflicting_confidence_tie_means_cash_without_an_invented_class(features):
    rules = [rule(), rule(("BBB_D",), label="D")]
    row = signals_module().predict_signals(model(rules), features, STOCKS).iloc[0]
    assert row.action == "CASH" and row.reason == "confidence_tie"
    assert pd.isna(row.prediction_label) and pd.isna(row.consequent)
    assert row.matched_rule_count == 2


def test_same_class_tie_uses_stable_support_and_shorter_antecedent(features):
    rows = [rule(("AAA_U", "BBB_D")), rule()]
    first = signals_module().predict_signals(model(rows), features, STOCKS)
    second = signals_module().predict_signals(model(list(reversed(rows))), features, STOCKS)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first.iloc[0].matched_antecedents == ("AAA_U",)


@pytest.mark.parametrize("label", ["D", "N"])
def test_down_or_neutral_predictions_stay_in_cash_but_keep_their_class(features, label):
    result = signals_module().predict_signals(model([rule(label=label)]), features, STOCKS)
    assert result.iloc[0].prediction_label == label
    assert result.iloc[0].action == "CASH"


def test_no_match_is_unknown_and_cash_including_empty_rule_sets(features):
    result = signals_module().predict_signals(model([]), features, STOCKS)
    assert len(result) == 4
    assert result.action.eq("CASH").all() and result.reason.eq("no_match").all()
    assert result.prediction_label.isna().all()


def test_rejects_target_columns_in_the_feature_interface(features):
    features["target_return"] = 0.99
    with pytest.raises(DataValidationError, match="feature"):
        signals_module().predict_signals(model([rule()]), features, STOCKS)


def test_rejects_a_next_item_in_a_rule_antecedent(features):
    with pytest.raises(DataValidationError, match="antecedent"):
        signals_module().predict_signals(model([rule(("NEXT_BBB_D",))]), features, STOCKS)


def test_rejects_a_model_fitted_after_the_feature_session(features):
    future_model = replace(model([rule()]), fit_cutoff=pd.Timestamp("2024-01-08"))
    with pytest.raises(DataValidationError, match="cutoff"):
        signals_module().predict_signals(future_model, features, STOCKS)


def test_rejects_an_incomplete_predictor_cohort(features):
    with pytest.raises(DataValidationError, match="complete"):
        signals_module().predict_signals(model([rule()]), features.drop(index=1), STOCKS)


def test_later_feature_changes_cannot_change_an_earlier_prediction(features):
    predictor = model([rule()])
    before = signals_module().predict_signals(predictor, features, STOCKS)
    changed = features.copy()
    later = (changed.feature_session == pd.Timestamp("2024-01-08")) & changed.ticker.eq("AAA.NS")
    changed.loc[later, ["feature_label", "feature_item", "feature_return"]] = ["U", "AAA_U", 0.01]
    after = signals_module().predict_signals(predictor, changed, STOCKS)
    first = before.feature_session == pd.Timestamp("2024-01-05")
    pd.testing.assert_frame_equal(before.loc[first], after.loc[first], check_exact=True)
    assert before.loc[later, "action"].iloc[0] == "CASH"
    assert after.loc[later, "action"].iloc[0] == "LONG"


def test_shuffled_feature_rows_do_not_change_predictions(features):
    predictor = model([rule()])
    before = signals_module().predict_signals(predictor, features, STOCKS)
    after = signals_module().predict_signals(predictor, features.sample(frac=1, random_state=42), STOCKS)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
