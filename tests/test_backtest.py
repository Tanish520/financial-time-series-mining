"""Execution/metric checks use synthetic prices; no test fetches market data."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, SnapshotConfig, write_snapshot


def backtest_module():
    try:
        from src import backtest
    except ImportError:
        pytest.fail("src.backtest must implement dated, finite-capital cost accounting")
    return backtest


def pipeline_module():
    try:
        from src import pipeline
    except ImportError:
        pytest.fail("src.pipeline must run the offline fixed chronological holdout")
    return pipeline


@pytest.fixture
def config():
    return SnapshotConfig(start="2024-01-04", as_of="2024-01-10", stocks=("AAA.NS", "BBB.NS"))


@pytest.fixture
def prices():
    dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"])
    bars = {
        "AAA.NS": ([100, 100, 200, 200, 198], [100, 101, 202, 198, 220]),
        "BBB.NS": ([100] * 5, [100, 99, 102, 103, 100]),
        "^NSEI": ([1000] * 5, [1000, 1000, 1010, 1020, 1030]),
    }
    rows = []
    for ticker, (opens, closes) in bars.items():
        for session, opening, closing in zip(dates, opens, closes):
            rows.append({
                "date": session, "ticker": ticker, "open": float(opening),
                "high": float(max(opening, closing) + 1), "low": float(min(opening, closing) - 1),
                "close": float(closing), "volume": 0.0 if ticker == "^NSEI" else 1000.0,
                "dividends": 0.0, "stock_splits": 0.0,
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture
def signals(config):
    rows = []
    for session in pd.to_datetime(["2024-01-05", "2024-01-08"]):
        for ticker in config.stocks:
            long = ticker == "AAA.NS"
            rows.append({
                "feature_session": session, "ticker": ticker,
                "prediction_label": "U" if long else pd.NA,
                "action": "LONG" if long else "CASH", "reason": "matched_rule" if long else "no_match",
                "rule_fit_cutoff": pd.Timestamp("2024-01-04"),
                "consequent": "NEXT_AAA_U" if long else pd.NA,
                "confidence": 0.8 if long else np.nan,
            })
    return pd.DataFrame(rows)


def test_executes_at_next_open_across_weekend_and_charges_one_round_trip(config, prices, signals):
    bt = backtest_module()
    first = signals.loc[signals.feature_session == pd.Timestamp("2024-01-05")]
    result = bt.simulate_strategy(prices, first, config, cost=0.001, initial_capital=20000)
    trade = result.trades.iloc[0]
    assert trade.entry_session == trade.exit_session == pd.Timestamp("2024-01-08")
    assert trade.entry_price == 200 and trade.exit_price == 202
    assert trade.quantity == 50 and trade.entry_notional == 10000
    assert trade.gross_pnl == pytest.approx(100)
    assert trade.transaction_cost == pytest.approx(10)
    assert trade.net_pnl == pytest.approx(90)
    assert result.daily.equity.iloc[-1] == pytest.approx(20090)
    assert result.metrics["net_return"] == pytest.approx(0.0045)


def test_simultaneous_signals_share_one_finite_capital_budget(config, prices, signals):
    first = signals.loc[signals.feature_session == pd.Timestamp("2024-01-05")].copy()
    first["action"], first["prediction_label"] = "LONG", "U"
    result = backtest_module().simulate_strategy(prices, first, config, cost=0.001, initial_capital=20000)
    assert result.trades.entry_notional.sum() == pytest.approx(20000)
    assert result.daily.equity.iloc[-1] == pytest.approx(20280)


def test_later_positions_use_current_equity_instead_of_reusing_initial_capital(config, prices, signals):
    result = backtest_module().simulate_strategy(prices, signals, config, cost=0.001, initial_capital=20000)
    expected = 20000 * (1 + (0.01 - 0.001) / 2) * (1 + (-0.01 - 0.001) / 2)
    assert result.daily.equity.iloc[-1] == pytest.approx(expected)
    assert result.trades.entry_notional.iloc[1] == pytest.approx(20090 / 2)


@pytest.mark.parametrize("label", [pd.NA, "D", "N"])
def test_cash_days_stay_in_equity_curve_and_incur_no_costs(config, prices, signals, label):
    signals["action"], signals["prediction_label"] = "CASH", label
    result = backtest_module().simulate_strategy(prices, signals, config, cost=0.005, initial_capital=20000)
    assert len(result.daily) == 2 and result.daily.equity.eq(20000).all()
    assert result.trades.empty and result.daily.costs.eq(0).all()
    assert result.metrics["trade_count"] == 0 and result.metrics["net_return"] == 0
    assert result.metrics["sharpe"] is None and result.metrics["profit_factor"] is None
    assert result.metrics["win_rate"] is None


def test_first_day_loss_counts_in_maximum_drawdown(config, prices, signals):
    bad = prices.ticker.eq("AAA.NS") & prices.date.eq(pd.Timestamp("2024-01-08"))
    prices.loc[bad, "close"], prices.loc[bad, "low"] = 198.0, 197.0
    first = signals.loc[signals.feature_session == pd.Timestamp("2024-01-05")]
    result = backtest_module().simulate_strategy(prices, first, config, cost=0, initial_capital=20000)
    assert result.metrics["max_drawdown"] == pytest.approx(0.005)


@pytest.mark.parametrize("cost,net_pnl", [(0.0005, 95.0), (0.001, 90.0), (0.005, 50.0)])
def test_cost_scenarios_change_pnl_without_changing_quantity(config, prices, signals, cost, net_pnl):
    first = signals.loc[signals.feature_session == pd.Timestamp("2024-01-05")]
    result = backtest_module().simulate_strategy(prices, first, config, cost=cost, initial_capital=20000)
    assert result.trades.quantity.iloc[0] == 50
    assert result.trades.net_pnl.iloc[0] == pytest.approx(net_pnl)


def test_rejects_a_model_fit_after_its_feature_session(config, prices, signals):
    signals["rule_fit_cutoff"] = pd.Timestamp("2024-01-08")
    with pytest.raises(DataValidationError, match="cutoff"):
        backtest_module().simulate_strategy(prices, signals, config)


def test_rejects_a_missing_stock_decision(config, prices, signals):
    with pytest.raises(DataValidationError, match="complete"):
        backtest_module().simulate_strategy(prices, signals.drop(index=1), config)


def test_rejects_an_omitted_cash_or_trading_session(config, prices, signals):
    signals.loc[signals.feature_session == pd.Timestamp("2024-01-08"), "feature_session"] = pd.Timestamp("2024-01-09")
    with pytest.raises(DataValidationError, match="contiguous"):
        backtest_module().simulate_strategy(prices, signals, config)


def test_rejects_a_final_signal_without_a_known_next_session(config, prices, signals):
    last = signals.iloc[:2].copy()
    last["feature_session"] = pd.Timestamp("2024-01-10")
    with pytest.raises(DataValidationError, match="next session"):
        backtest_module().simulate_strategy(prices, last, config)


def test_rejects_long_actions_that_do_not_predict_up(config, prices, signals):
    signals.loc[0, "prediction_label"] = "D"
    with pytest.raises(DataValidationError, match="action"):
        backtest_module().simulate_strategy(prices, signals, config)


def test_zero_volume_long_bar_invalidates_assumed_execution_instead_of_skipping_it(config, prices, signals):
    prices.loc[prices.ticker.eq("AAA.NS") & prices.date.eq(pd.Timestamp("2024-01-08")), "volume"] = 0.0
    with pytest.raises(DataValidationError, match="volume"):
        backtest_module().simulate_strategy(prices, signals, config)


def test_buy_and_hold_uses_test_open_and_charges_cost_once(config, prices):
    days = pd.to_datetime(["2024-01-08", "2024-01-09"])
    results = backtest_module().simulate_buy_and_hold(prices, config, days, cost=0.001, initial_capital=20000)
    aaa = results["BUY_HOLD:AAA.NS"]
    assert aaa.daily.equity.iloc[0] == pytest.approx(20190)
    assert aaa.daily.equity.iloc[-1] == pytest.approx(19780)
    assert aaa.trades.transaction_cost.sum() == pytest.approx(20)
    assert results["BUY_HOLD_BASKET"].daily.equity.iloc[-1] == pytest.approx(20180)
    assert results["NIFTY50_PRICE_INDEX"].daily.equity.iloc[-1] == pytest.approx(20400)
    assert results["NIFTY50_PRICE_INDEX"].metrics["trade_count"] is None


def test_index_reference_is_not_charged_the_strategy_cost(config, prices):
    bt = backtest_module()
    days = pd.to_datetime(["2024-01-08", "2024-01-09"])
    low = bt.simulate_buy_and_hold(prices, config, days, cost=0.0005)["NIFTY50_PRICE_INDEX"]
    high = bt.simulate_buy_and_hold(prices, config, days, cost=0.005)["NIFTY50_PRICE_INDEX"]
    pd.testing.assert_frame_equal(low.daily, high.daily)


def test_metrics_use_net_trade_money_and_sample_daily_variance():
    bt = backtest_module()
    values = pd.Series([1090.0, 1080.0], index=pd.to_datetime(["2024-01-05", "2024-01-08"]))
    metrics = bt.performance_metrics(values, initial_capital=1000, net_pnls=pd.Series([90.0, -10.0]))
    daily = np.array([0.09, 1080 / 1090 - 1])
    assert metrics["net_return"] == pytest.approx(0.08)
    assert metrics["profit_factor"] == 9 and metrics["win_rate"] == 0.5
    assert metrics["trade_count"] == 2
    assert metrics["sharpe"] == pytest.approx(np.sqrt(252) * daily.mean() / daily.std(ddof=1))
    assert metrics["cagr"] == pytest.approx(1.08 ** (365.25 / 4) - 1)


def test_profitable_trades_without_losses_have_infinite_profit_factor():
    metrics = backtest_module().performance_metrics(
        pd.Series([1010.0], index=pd.to_datetime(["2024-01-08"])),
        initial_capital=1000, net_pnls=pd.Series([10.0]),
    )
    assert np.isinf(metrics["profit_factor"]) and metrics["win_rate"] == 1


@pytest.mark.parametrize("cost", [-0.1, np.nan, np.inf, 1.0])
def test_rejects_invalid_cost_assumptions(config, prices, signals, cost):
    with pytest.raises(DataValidationError, match="cost"):
        backtest_module().simulate_strategy(prices, signals, config, cost=cost)


def test_holdout_fits_once_before_test_and_keeps_all_cash_dates(config, prices):
    result = pipeline_module().run_holdout(
        prices, config, cutoff="2024-01-08", test_sessions=2, window=1,
    )
    assert result.model.rules.empty
    assert result.execution_sessions.tolist() == list(pd.to_datetime(["2024-01-09", "2024-01-10"]))
    assert len(result.predictions) == 4 and result.predictions.action.eq("CASH").all()
    assert result.model.sessions.outcome_available_session.max() <= pd.Timestamp("2024-01-08")
    rows = result.metrics.loc[result.metrics.strategy == "RULES_PORTFOLIO"]
    assert len(rows) == 3 and rows.net_return.eq(0).all() and rows.trade_count.eq(0).all()


def test_positive_trade_path_works_end_to_end_on_synthetic_data(config, prices):
    changed = prices.ticker.eq("BBB.NS") & prices.date.eq(pd.Timestamp("2024-01-09"))
    prices.loc[changed, "close"], prices.loc[changed, "low"] = 98.0, 97.0
    result = pipeline_module().run_holdout(
        prices, config, cutoff="2024-01-09", test_sessions=1, window=2,
    )
    assert not result.model.rules.empty
    aaa = result.predictions.loc[result.predictions.ticker.eq("AAA.NS")].iloc[0]
    assert aaa.prediction_label == "U" and aaa.action == "LONG"
    row = result.metrics.loc[result.metrics.strategy.eq("RULES_PORTFOLIO") & result.metrics.cost.eq(0.001)].iloc[0]
    assert row.net_return == pytest.approx(((220 / 198 - 1 - 0.001) + (0 - 0.001)) / 2)
    assert row.trade_count == 2


@pytest.mark.parametrize("cutoff,window,fit_feature,fit_execution", [
    ("2024-01-08", "1", "2024-01-05", "2024-01-08"),
    ("2024-01-05", "expanding", None, None),
])
def test_holdout_cli_saves_standard_json_and_preserves_source(
    config, prices, tmp_path, capsys, cutoff, window, fit_feature, fit_execution
):
    pipeline = pipeline_module()
    source, output = tmp_path / "source", tmp_path / "holdout"
    source_manifest = write_snapshot(prices, config, source, min_sessions=2)
    args = ["--snapshot", str(source), "--cutoff", cutoff, "--window", window,
            "--test-sessions", "2", "--output", str(output)]
    assert pipeline.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["test_sessions"] == 2 and summary["prediction_rows"] == 4
    def invalid_constant(value):
        pytest.fail(f"Nonstandard JSON constant: {value}")
    metrics = json.loads((output / "metrics.json").read_text(), parse_constant=invalid_constant)
    assert any(row["profit_factor"] == "Infinity" for row in metrics)
    manifest = json.loads((output / "manifest.json").read_text(), parse_constant=invalid_constant)
    assert manifest["source_sha256"] == source_manifest["sha256"]
    assert manifest["first_training_feature_session"] == fit_feature
    assert manifest["last_training_feature_session"] == fit_feature
    assert manifest["first_training_execution_session"] == fit_execution
    assert manifest["last_training_execution_session"] == fit_execution
    fitted_sessions = pd.read_csv(output / "training_sessions.csv")
    assert fitted_sessions.feature_session.tolist() == ([] if fit_feature is None else [fit_feature])
    assert fitted_sessions.execution_session.tolist() == ([] if fit_execution is None else [fit_execution])
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((source / "prices.csv").read_bytes()).hexdigest() == source_manifest["sha256"]
    assert pipeline.main(args) == 2
    assert "exists" in capsys.readouterr().err
