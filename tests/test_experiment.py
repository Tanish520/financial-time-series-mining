"""Full rolling chronology, continuous capital and descriptive held-out scoring."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest import simulate_strategy
from src.data import DataValidationError, SnapshotConfig
from src.walkforward import RuleParameters, TuningGrid, run_fold


def experiment_module():
    try:
        from src import experiment
    except ImportError:
        pytest.fail("src.experiment must assemble complete rolling frozen predictions")
    return experiment


def asymmetry_module():
    try:
        from src import asymmetry
    except ImportError:
        pytest.fail("src.asymmetry must score held-out prediction-weighted base rates")
    return asymmetry


@pytest.fixture(scope="module")
def small_market():
    dates = pd.bdate_range("2024-01-02", periods=20)
    pattern = [0.012, 0.012, -0.012, -0.012, 0.003, -0.003]
    rows = []
    for ticker, offset in (("AAA.NS", 0), ("BBB.NS", 2), ("^NSEI", 0)):
        closing = 100.0
        for i, day in enumerate(dates):
            closing *= 1 + (pattern[(i + offset) % 6] if ticker != "^NSEI" else 0.001)
            intraday = (0.02 if pattern[(i - 1) % 6] > 0 else -0.02) * (-1 if ticker == "BBB.NS" else 1)
            opening = closing / (1 + intraday)
            rows.append({"date": day, "ticker": ticker, "open": opening, "close": closing,
                         "high": max(opening, closing) + 1, "low": min(opening, closing) - 1,
                         "volume": 0.0 if ticker == "^NSEI" else 1000.0,
                         "dividends": 0.0, "stock_splits": 0.0})
    config = SnapshotConfig(str(dates[0].date()), str(dates[-1].date()), stocks=("AAA.NS", "BBB.NS"))
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True), config, dates


def options():
    return {"grid": TuningGrid(windows=(4,), thetas=(0.005,), supports=(0.01,), confidences=(0.55,)),
            "fixed": RuleParameters(4, 0.005, 0.02, 0.60), "validation_sessions": 4,
            "test_sessions": 3, "min_validation_trades": 1, "min_validation_trade_sessions": 1}


@pytest.fixture(scope="module")
def complete_run(small_market):
    prices, config, _ = small_market
    return experiment_module().run_experiment(prices, config, **options())


def test_fold_can_freeze_decisions_without_redundant_outer_accounting(small_market):
    prices, config, _ = small_market
    normal = run_fold(prices, config, **options())
    frozen = run_fold(prices, config, score_outer=False, **options())
    pd.testing.assert_frame_equal(normal.predictions, frozen.predictions)
    pd.testing.assert_frame_equal(normal.tuning, frozen.tuning)
    assert frozen.metrics.empty and frozen.trades.empty and frozen.equity_curves.empty


def test_every_remaining_execution_has_exactly_one_frozen_decision(complete_run, small_market):
    _, _, dates = small_market
    run = complete_run
    assert run.folds.fold_id.tolist() == [0, 1, 2, 3]
    assert run.folds.test_sessions.tolist() == [3, 3, 3, 1]
    assert len(run.tuning) == 4
    assert run.predictions.groupby(["variant", "execution_session"]).size().eq(2).all()
    assert not run.predictions.duplicated(["variant", "ticker", "execution_session"]).any()
    for variant in ("TUNED", "FIXED"):
        observed = pd.DatetimeIndex(run.predictions.loc[run.predictions.variant.eq(variant), "execution_session"].unique())
        assert observed.equals(dates[10:])
    assert (run.predictions.rule_fit_cutoff <= run.predictions.feature_session).all()
    assert (run.training_sessions.execution_session <= run.training_sessions.fit_cutoff).all()
    assert (run.tuning.last_training_execution_session < run.tuning.validation_start).all()


def test_capital_compounds_through_fold_boundaries_and_each_cost_uses_same_signals(complete_run, small_market):
    prices, config, _ = small_market
    decisions = complete_run.predictions.loc[complete_run.predictions.variant.eq("TUNED")]
    assert decisions.action.eq("LONG").any()
    for cost in (0.0005, 0.001, 0.005):
        expected = simulate_strategy(prices, decisions, config, cost=cost)
        actual = complete_run.equity_curves.loc[
            complete_run.equity_curves.strategy.eq("TUNED_PORTFOLIO") & complete_run.equity_curves.cost.eq(cost)
        ]
        assert actual.equity.tolist() == expected.daily.equity.tolist()
        ledger = complete_run.trades.loc[
            complete_run.trades.strategy.eq("TUNED_PORTFOLIO") & complete_run.trades.cost.eq(cost)
        ]
        assert len(ledger) == decisions.action.eq("LONG").sum()
        assert actual.equity.iloc[-1] == pytest.approx(100000 + ledger.net_pnl.sum())
        second_fold = ledger.loc[ledger.fold_id.eq(1)]
        if not second_fold.empty:
            day = second_fold.entry_session.iloc[0]
            pos = expected.daily.index.get_loc(day)
            assert second_fold.entry_notional.iloc[0] == pytest.approx(expected.daily.equity.iloc[pos - 1] / 2)


def test_future_prices_cannot_revise_earlier_fold_selection_or_decisions(complete_run, small_market):
    prices, config, _ = small_market
    boundary = complete_run.folds.test_end.iloc[0]
    changed = prices.copy(deep=True)
    changed.loc[changed.date > boundary, ["open", "high", "low", "close"]] *= 1.4
    after = experiment_module().run_experiment(changed, config, **options())
    pd.testing.assert_frame_equal(complete_run.tuning.loc[complete_run.tuning.fold_id.eq(0)],
                                  after.tuning.loc[after.tuning.fold_id.eq(0)], check_exact=True)
    pd.testing.assert_frame_equal(complete_run.predictions.loc[complete_run.predictions.fold_id.eq(0)],
                                  after.predictions.loc[after.predictions.fold_id.eq(0)], check_exact=True)


def test_full_81_candidate_log_survives_final_partial_block(small_market):
    prices, config, dates = small_market
    short = prices.loc[prices.date <= dates[7]].copy()
    config = SnapshotConfig(config.start, str(dates[7].date()), stocks=config.stocks)
    run = experiment_module().run_experiment(
        short, config, grid=TuningGrid(windows=(1, 2, None)), fixed=RuleParameters(2, .005, .02, .60),
        validation_sessions=3, test_sessions=21,
    )
    assert len(run.folds) == 1 and run.folds.test_sessions.iloc[0] == 1
    assert len(run.tuning) == run.tuning.parameter_id.nunique() == 81


def test_outcomes_are_scored_after_predictions_and_latest_is_frozen(complete_run, small_market):
    _, config, dates = small_market
    run = complete_run
    assert (run.outcomes.outcome_available_session == run.outcomes.execution_session).all()
    assert len(run.outcomes) == len(run.predictions)
    assert run.latest_model["fit_cutoff"] == str(run.folds.fit_cutoff.iloc[-1].date())
    assert run.latest_model["stocks"] == list(config.stocks)
    assert run.latest_signals["observation_session"] == str(dates[-1].date())
    assert run.latest_signals["intended_execution"] == "next trading session"
    assert len(run.latest_signals["signals"]) == len(config.stocks)
    assert run.latest_prices.date.nunique() == 10


def test_experiment_save_is_complete_checksummed_and_immutable(complete_run, tmp_path):
    module = experiment_module()
    output = tmp_path / "experiment"
    module.write_experiment(complete_run, output, source_sha256="a" * 64)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 2 and manifest["evaluation_scope"] == "nested_walk_forward"
    assert manifest["fold_count"] == 4 and manifest["candidate_count_per_fold"] == 1
    assert len(manifest["files"]) == 15
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    def bad_constant(value):
        pytest.fail(value)
    json.loads((output / "latest_signals.json").read_text(), parse_constant=bad_constant)
    with pytest.raises(DataValidationError, match="exists"):
        module.write_experiment(complete_run, output, source_sha256="a" * 64)


def test_asymmetry_uses_prediction_weighted_group_base_rates():
    rows = []
    for ticker, actual, predicted in (
        ("AAA.NS", ["U", "U", "D", "N"], ["U", "U", "D", None]),
        ("BBB.NS", ["U", "D", "N", "N"], ["U", "D", "D", None]),
    ):
        for day, (truth, guess) in enumerate(zip(actual, predicted)):
            rows.append({"variant": "TUNED", "fold_id": 0, "ticker": ticker,
                         "execution_session": pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
                         "target_label": truth, "prediction_label": guess})
    summary, groups = asymmetry_module().score_asymmetry(pd.DataFrame(rows))
    up = summary.loc[summary.direction.eq("U")].iloc[0]
    assert up.prediction_count == up.correct_count == 3
    assert up.precision == 1 and up.expected_correct == 1.25
    assert up.base_rate == pytest.approx(1.25 / 3) and up.held_out_lift == pytest.approx(2.4)
    down = summary.loc[summary.direction.eq("D")].iloc[0]
    assert down.prediction_count == 3 and down.correct_count == 2
    assert down.base_rate == 0.25 and down.held_out_lift == pytest.approx(2 / 0.75)
    assert len(groups) == 4


@pytest.mark.parametrize("prediction", [None, "D"])
def test_asymmetry_undefined_cases_are_not_invented(prediction):
    data = pd.DataFrame({"variant": ["TUNED"], "fold_id": [0], "ticker": ["AAA.NS"],
                         "execution_session": [pd.Timestamp("2024-01-02")],
                         "target_label": ["U"], "prediction_label": [prediction]})
    summary, _ = asymmetry_module().score_asymmetry(data)
    down = summary.loc[summary.direction.eq("D")].iloc[0]
    assert pd.isna(down.held_out_lift)
    if prediction is None:
        assert down.prediction_count == 0 and pd.isna(down.precision) and pd.isna(down.base_rate)
    else:
        assert down.prediction_count == 1 and down.precision == 0 and down.base_rate == 0


def test_directional_forecast_without_mature_target_cannot_be_scored():
    data = pd.DataFrame({"variant": ["TUNED"], "fold_id": [0], "ticker": ["AAA.NS"],
                         "execution_session": [pd.Timestamp("2024-01-02")],
                         "target_label": [None], "prediction_label": ["U"]})
    with pytest.raises(DataValidationError, match="target"):
        asymmetry_module().score_asymmetry(data)
