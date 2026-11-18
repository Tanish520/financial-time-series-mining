"""Nested chronology checks on deterministic synthetic data, without providers."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, SnapshotConfig


def wf():
    try:
        from src import walkforward
    except ImportError:
        pytest.fail("src.walkforward must tune on past data and freeze an outer fold")
    return walkforward


@pytest.fixture(scope="module")
def market():
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows = []
    pattern = np.array([0.012, 0.012, -0.012, -0.012, 0.003, -0.003])
    for ticker, offset in (("AAA.NS", 0), ("BBB.NS", 2), ("^NSEI", 0)):
        close = 100.0
        for i, session in enumerate(dates):
            close *= 1 + (pattern[(i + offset) % 6] if ticker != "^NSEI" else 0.001)
            intraday = 0.02 if pattern[(i - 1) % 6] > 0 else -0.02
            if ticker == "BBB.NS":
                intraday = -intraday
            opening = close / (1 + intraday)
            rows.append({
                "date": session, "ticker": ticker, "open": opening,
                "high": max(opening, close) + 1, "low": min(opening, close) - 1,
                "close": close, "volume": 0.0 if ticker == "^NSEI" else 1000.0,
                "dividends": 0.0, "stock_splits": 0.0,
            })
    prices = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    config = SnapshotConfig(start="2024-01-02", as_of=str(dates[-1].date()), stocks=("AAA.NS", "BBB.NS"))
    return prices, config, dates


def small_options():
    module = wf()
    return {
        "grid": module.TuningGrid(windows=(4, 8, None), thetas=(0.005, 0.01),
                                  supports=(0.01, 0.20), confidences=(0.55, 0.70)),
        "fixed": module.RuleParameters(8, 0.005, 0.02, 0.60),
        "validation_sessions": 6, "test_sessions": 3,
        "min_validation_trades": 2, "min_validation_trade_sessions": 2,
    }


@pytest.fixture(scope="module")
def fold(market):
    prices, config, _ = market
    return wf().run_fold(prices, config, **small_options())


def test_default_grid_is_exactly_81_and_ids_are_canonical():
    module = wf()
    parameters = module.TuningGrid().parameters()
    assert len(parameters) == len({p.identifier for p in parameters}) == 81
    for parameter in parameters:
        assert json.loads(parameter.identifier) == parameter.as_dict()
    assert parameters == tuple(sorted(parameters, key=lambda p: p.identifier))


@pytest.mark.parametrize("kwargs", [
    {"windows": (0, 8)}, {"windows": (True,)}, {"windows": (8, 8)},
    {"thetas": (0.0,)}, {"supports": (0.0,)}, {"confidences": (np.nan,)},
    {"confidences": (0.6, 0.6)}, {"thetas": ()},
])
def test_invalid_grid_is_rejected(kwargs):
    with pytest.raises(DataValidationError):
        wf().TuningGrid(**kwargs).parameters()


def test_common_validation_and_exact_fit_lengths(fold, market):
    _, _, dates = market
    assert fold.cutoff == dates[15]
    assert fold.validation_cutoff == dates[9]
    assert fold.validation_sessions.equals(dates[10:16])
    assert fold.execution_sessions.equals(dates[16:19])
    scores = fold.tuning
    assert len(scores) == 24 and scores.parameter_id.nunique() == 24
    assert scores.validation_start.eq(dates[10]).all() and scores.validation_end.eq(dates[15]).all()
    assert scores.last_training_execution_session.eq(dates[9]).all()
    assert (scores.last_training_execution_session < scores.validation_start).all()
    assert scores.loc[scores.window.eq("4"), "training_transactions"].eq(4).all()
    assert scores.loc[scores.window.eq("8"), "training_transactions"].eq(8).all()
    assert scores.loc[scores.window.eq("expanding"), "training_transactions"].eq(8).all()
    assert scores.selection_cost.eq(0.001).all()


def test_selection_uses_validation_only_and_refit_includes_now_known_validation(fold):
    eligible = fold.tuning.loc[fold.tuning.eligible].sort_values(
        ["validation_net_return", "parameter_id"], ascending=[False, True], kind="stable"
    )
    assert not eligible.empty and fold.selected is not None
    assert fold.selected.identifier == eligible.iloc[0].parameter_id
    assert fold.tuning.selected.sum() == 1
    model = fold.models["TUNED"]
    assert model.fit_cutoff == fold.cutoff
    assert model.sessions.execution_session.max() == fold.cutoff
    assert len(model.sessions) == (fold.selected.window or 14)
    assert (model.sessions.outcome_available_session <= fold.cutoff).all()
    assert fold.models["FIXED"].fit_cutoff == fold.cutoff
    assert len(fold.models["FIXED"].sessions) == 8
    assert fold.predictions.groupby("variant").size().to_dict() == {"FIXED": 6, "TUNED": 6}
    assert fold.predictions.execution_session.min() > fold.cutoff
    assert (fold.predictions.rule_fit_cutoff <= fold.predictions.feature_session).all()


def test_equal_validation_returns_have_fixed_identifier_tie_break():
    scores = pd.DataFrame({"parameter_id": ["z", "a", "b"], "eligible": [True, True, False],
                           "validation_net_return": [0.10, 0.10, 0.99]})
    for seed in (0, 7):
        assert wf().choose_parameter_id(scores.sample(frac=1, random_state=seed)) == "a"


@pytest.mark.parametrize("guards", [
    {"min_validation_trades": 1000}, {"min_validation_trade_sessions": 7},
])
def test_failed_guard_preserves_outer_cash_days(market, guards):
    prices, config, _ = market
    result = wf().run_fold(prices, config, **(small_options() | guards))
    assert result.selected is None and not result.tuning.eligible.any()
    assert not result.tuning.selected.any()
    signals = result.predictions.loc[result.predictions.variant.eq("TUNED")]
    assert len(signals) == 6 and signals.action.eq("CASH").all()
    assert signals.prediction_label.isna().all()
    assert signals.reason.eq("no_eligible_configuration").all()
    assert result.models["TUNED"].rules.empty and result.models["TUNED"].sessions.empty
    metrics = result.metrics.loc[result.metrics.strategy.eq("TUNED_PORTFOLIO")]
    assert len(metrics) == 3 and metrics.net_return.eq(0).all() and metrics.trade_count.eq(0).all()


def test_future_bars_cannot_change_selection_models_or_first_outer_decision(market, fold):
    prices, config, _ = market
    changed = prices.copy(deep=True)
    future = changed.date > fold.cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 1.7
    after = wf().run_fold(changed, config, **small_options())
    pd.testing.assert_frame_equal(fold.tuning, after.tuning, check_exact=True)
    assert after.selected == fold.selected
    for variant in ("TUNED", "FIXED"):
        pd.testing.assert_frame_equal(fold.models[variant].rules, after.models[variant].rules, check_exact=True)
    before_first = fold.predictions.loc[fold.predictions.feature_session.eq(fold.cutoff)].reset_index(drop=True)
    after_first = after.predictions.loc[after.predictions.feature_session.eq(fold.cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_first, after_first, check_exact=True)
    assert not after.equity_curves.equals(fold.equity_curves)


def test_final_partial_fold_keeps_all_remaining_sessions(market):
    prices, config, dates = market
    result = wf().run_fold(prices, config, cutoff=str(dates[-3].date()), **small_options())
    assert result.execution_sessions.equals(dates[-2:])
    assert result.predictions.groupby("variant").size().eq(4).all()


@pytest.mark.parametrize("kwargs", [
    {"cutoff": "2024-01-03"}, {"cutoff": "2024-01-06"},
    {"cutoff": "2024-01-23T12:00:00"}, {"test_sessions": 0},
    {"validation_sessions": True}, {"min_validation_trades": -1},
])
def test_invalid_fold_requests_fail(market, kwargs):
    prices, config, _ = market
    with pytest.raises(DataValidationError):
        wf().run_fold(prices, config, **(small_options() | kwargs))


def test_same_frozen_decisions_and_cost_ledger_reconcile(fold):
    for variant in ("TUNED", "FIXED"):
        name = f"{variant}_PORTFOLIO"
        count = fold.predictions.loc[fold.predictions.variant.eq(variant), "action"].eq("LONG").sum()
        scenario_counts = []
        for cost in (0.0005, 0.001, 0.005):
            trades = fold.trades.loc[fold.trades.strategy.eq(name) & fold.trades.cost.eq(cost)]
            curve = fold.equity_curves.loc[fold.equity_curves.strategy.eq(name) & fold.equity_curves.cost.eq(cost)]
            assert len(trades) == count
            assert trades.transaction_cost.sum() == pytest.approx((trades.entry_notional * cost).sum())
            assert curve.equity.iloc[-1] == pytest.approx(100000 + trades.net_pnl.sum())
            scenario_counts.append(trades[["feature_session", "entry_session", "ticker"]].reset_index(drop=True))
        pd.testing.assert_frame_equal(scenario_counts[0], scenario_counts[1])
        pd.testing.assert_frame_equal(scenario_counts[1], scenario_counts[2])


def test_saved_fold_has_full_provenance_standard_json_and_no_overwrite(fold, tmp_path):
    output = tmp_path / "first_fold"
    wf().write_fold(fold, output, source_sha256="a" * 64)
    def invalid_constant(value):
        pytest.fail(f"Nonstandard JSON: {value}")
    manifest = json.loads((output / "manifest.json").read_text(), parse_constant=invalid_constant)
    assert manifest["source_sha256"] == "a" * 64
    assert manifest["evaluation_scope"] == "single_nested_fold_engineering_checkpoint"
    assert manifest["selected_parameters"] == fold.selected.as_dict()
    assert manifest["candidate_count"] == 24 and manifest["selection_cost"] == 0.001
    assert manifest["validation_sessions"] == 6 and manifest["min_validation_trades"] == 2
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    sessions = pd.read_csv(output / "training_sessions.csv")
    assert set(sessions.stage) == {"inner", "refit"}
    assert {"TUNED", "FIXED"}.issubset(set(sessions.variant))
    assert sessions.execution_session.le(sessions.fit_cutoff).all()
    assert len(json.loads((output / "metrics.json").read_text(), parse_constant=invalid_constant)) == 18
    with pytest.raises(DataValidationError, match="exists"):
        wf().write_fold(fold, output, source_sha256="a" * 64)
