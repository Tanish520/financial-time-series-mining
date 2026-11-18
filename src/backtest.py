"""Daily-open/close simulation under explicit finite-capital and cost assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import DataValidationError, SnapshotConfig, validate_prices

TRADE_COLUMNS = [
    "feature_session", "entry_session", "exit_session", "ticker",
    "entry_price", "exit_price", "entry_notional", "quantity", "gross_return",
    "gross_pnl", "cost_rate", "transaction_cost", "net_pnl", "net_return",
    "prediction_label", "consequent", "confidence",
]


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _validate_assumptions(cost: float, initial_capital: float) -> None:
    if not np.isfinite(cost) or not 0 <= cost < 1:
        raise DataValidationError("cost must be a finite total round-trip fraction in [0, 1).")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise DataValidationError("initial_capital must be finite and positive.")


def performance_metrics(
    equity: pd.Series, *, initial_capital: float, net_pnls: pd.Series | None = None
) -> dict:
    """Use every daily close and initial capital; None trade PnLs means reference-only."""
    values = equity.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(equity.index)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise DataValidationError("Metrics require a nonempty positive finite equity path.")
    _validate_assumptions(0.0, initial_capital)
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise DataValidationError("Metrics require unique chronological execution dates.")
    path = np.r_[initial_capital, values]
    returns = np.diff(path) / path[:-1]
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = None if volatility <= 8 * np.finfo(float).eps else float(np.sqrt(252) * returns.mean() / volatility)
    days = (dates[-1].date() - dates[0].date()).days + 1
    annual_log_growth = np.log(values[-1] / initial_capital) * 365.25 / days
    cagr = None if annual_log_growth > np.log(np.finfo(float).max) else float(np.expm1(annual_log_growth))
    profit_factor = win_rate = trade_count = None
    if net_pnls is not None:
        pnls = net_pnls.to_numpy(dtype=float)
        if not np.isfinite(pnls).all():
            raise DataValidationError("Trade PnLs must be finite.")
        trade_count = len(pnls)
        if trade_count:
            gains, losses = float(pnls[pnls > 0].sum()), float(-pnls[pnls < 0].sum())
            profit_factor = gains / losses if losses else (float("inf") if gains else None)
            win_rate = float(np.count_nonzero(pnls > 0) / trade_count)
    return {
        "net_return": float(values[-1] / initial_capital - 1), "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(np.max(1 - path / np.maximum.accumulate(path))),
        "profit_factor": profit_factor, "win_rate": win_rate, "trade_count": trade_count,
    }


def _result(daily: pd.DataFrame, trades: pd.DataFrame, initial_capital: float, *, reference=False) -> BacktestResult:
    daily = daily.copy()
    prior = np.r_[initial_capital, daily["equity"].to_numpy()[:-1]]
    daily["equity_before"] = prior
    daily["daily_return"] = daily["equity"].to_numpy() / prior - 1
    daily["net_pnl"] = daily["equity"].to_numpy() - prior
    metrics = performance_metrics(
        daily["equity"], initial_capital=initial_capital,
        net_pnls=None if reference else trades["net_pnl"],
    )
    return BacktestResult(daily, trades, metrics)


def _calendar(prices: pd.DataFrame, config: SnapshotConfig) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(prices.loc[prices.ticker.eq(config.benchmark), "date"]).sort_values().as_unit("ns")


def _daily_dates(values) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(values)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise DataValidationError("Use nonmissing, timezone-naive daily session dates.")
    return dates.as_unit("ns")


def simulate_strategy(
    prices: pd.DataFrame, signals: pd.DataFrame, config: SnapshotConfig,
    *, cost: float = 0.001, initial_capital: float = 100000.0,
) -> BacktestResult:
    """Execute U decisions next session; each stock has 1/N of current capital."""
    _validate_assumptions(cost, initial_capital)
    validate_prices(prices, config, min_sessions=2)
    required = {"feature_session", "ticker", "prediction_label", "action", "rule_fit_cutoff"}
    if signals.empty or not required.issubset(signals.columns):
        raise DataValidationError("Backtesting requires explicit all-stock decisions, including cash days.")
    decisions = signals.copy()
    decisions["feature_session"] = _daily_dates(decisions["feature_session"])
    fits = _daily_dates(decisions["rule_fit_cutoff"])
    if (fits > pd.DatetimeIndex(decisions["feature_session"])).any():
        raise DataValidationError("A model fit cutoff is after its feature session.")
    if decisions.duplicated(["feature_session", "ticker"]).any():
        raise DataValidationError("Duplicate stock/session decisions.")
    complete = decisions.groupby("feature_session").ticker.agg(lambda values: set(values) == set(config.stocks))
    if not complete.all():
        raise DataValidationError("Every backtest session needs a complete stock decision cohort.")
    if not decisions.action.isin(["LONG", "CASH"]).all():
        raise DataValidationError("Invalid action; only LONG or CASH is allowed.")
    classes_valid = decisions.prediction_label.isna() | decisions.prediction_label.isin(["U", "D", "N"])
    is_up = decisions.prediction_label.eq("U").fillna(False)
    if not classes_valid.all() or not decisions.action.eq("LONG").equals(is_up):
        raise DataValidationError("A LONG action requires U; D/N/unknown predictions require CASH.")

    calendar = _calendar(prices, config)
    feature_days = _daily_dates(decisions.feature_session.unique()).sort_values()
    positions = calendar.get_indexer(feature_days)
    if (positions < 0).any():
        raise DataValidationError("A feature session is absent from the observed exchange calendar.")
    if positions[-1] + 1 >= len(calendar):
        raise DataValidationError("No known next session exists for the final signal.")
    if not feature_days.equals(calendar[positions[0]:positions[-1] + 1]):
        raise DataValidationError("The decision calendar must be contiguous, including cash sessions.")
    execution_days = calendar[positions + 1]
    mapping = dict(zip(feature_days, execution_days))
    if "execution_session" in decisions:
        supplied = _daily_dates(decisions.execution_session)
        expected = _daily_dates(decisions.feature_session.map(mapping))
        if not supplied.equals(expected):
            raise DataValidationError("A supplied execution date is not the observed next session.")
    quotes = prices.set_index(["date", "ticker"])
    capital = float(initial_capital)
    ledger, days = [], []
    for feature_day, execution_day in zip(feature_days, execution_days):
        allocation = capital / len(config.stocks)
        gross_pnl = fees = invested = 0.0
        cohort = decisions.loc[decisions.feature_session.eq(feature_day)].sort_values("ticker")
        for decision in cohort.to_dict("records"):
            if decision["action"] == "CASH":
                continue
            bar = quotes.loc[(execution_day, decision["ticker"])]
            if bar["volume"] <= 0:
                raise DataValidationError("A zero-volume LONG bar invalidates the assumed execution; audit the data.")
            opening, closing = float(bar["open"]), float(bar["close"])
            gross_return = closing / opening - 1
            gross, fee = allocation * gross_return, allocation * cost
            ledger.append({
                "feature_session": feature_day, "entry_session": execution_day,
                "exit_session": execution_day, "ticker": decision["ticker"],
                "entry_price": opening, "exit_price": closing, "entry_notional": allocation,
                "quantity": allocation / opening, "gross_return": gross_return,
                "gross_pnl": gross, "cost_rate": cost, "transaction_cost": fee,
                "net_pnl": gross - fee, "net_return": gross_return - cost,
                "prediction_label": "U", "consequent": decision.get("consequent", pd.NA),
                "confidence": decision.get("confidence", np.nan),
            })
            gross_pnl += gross
            fees += fee
            invested += allocation
        capital += gross_pnl - fees
        if not np.isfinite(capital) or capital <= 0:
            raise DataValidationError("Simulated capital is exhausted or nonfinite under the cost assumptions.")
        days.append({
            "execution_session": execution_day, "equity": capital,
            "gross_pnl": gross_pnl, "costs": fees, "invested_notional": invested,
        })
    daily = pd.DataFrame(days).set_index("execution_session")
    return _result(daily, pd.DataFrame(ledger, columns=TRADE_COLUMNS), initial_capital)


def simulate_buy_and_hold(
    prices: pd.DataFrame, config: SnapshotConfig, execution_sessions,
    *, cost: float = 0.001, initial_capital: float = 100000.0,
) -> dict[str, BacktestResult]:
    """Aligned holding controls with one flat round-trip cost and no rebalancing."""
    _validate_assumptions(cost, initial_capital)
    validate_prices(prices, config, min_sessions=2)
    days = _daily_dates(execution_sessions)
    calendar = _calendar(prices, config)
    if not len(days) or days.has_duplicates or not days.is_monotonic_increasing:
        raise DataValidationError("Holding controls require unique chronological execution sessions.")
    positions = calendar.get_indexer(days)
    if (positions < 0).any() or not days.equals(calendar[positions[0]:positions[-1] + 1]):
        raise DataValidationError("Holding controls require the same contiguous observed test sessions.")
    controls, basket_paths, basket_trades = {}, [], []
    for ticker in (*config.stocks, config.benchmark):
        reference = ticker == config.benchmark
        applied_cost = 0.0 if reference else cost
        bars = prices.loc[prices.ticker.eq(ticker)].set_index("date").loc[days]
        opening, closing = float(bars.open.iloc[0]), float(bars.close.iloc[-1])
        gross_values = initial_capital * bars.close.to_numpy(dtype=float) / opening
        charges = np.zeros(len(days), dtype=float)
        charges[0] += initial_capital * applied_cost / 2
        charges[-1] += initial_capital * applied_cost / 2
        daily = pd.DataFrame({
            "equity": gross_values - np.cumsum(charges), "costs": charges,
        }, index=pd.DatetimeIndex(days, name="execution_session"))
        gross_return = closing / opening - 1
        trade = {
            "feature_session": pd.NaT, "entry_session": days[0], "exit_session": days[-1],
            "ticker": ticker, "entry_price": opening, "exit_price": closing,
            "entry_notional": initial_capital, "quantity": initial_capital / opening,
            "gross_return": gross_return, "gross_pnl": initial_capital * gross_return,
            "cost_rate": applied_cost, "transaction_cost": initial_capital * applied_cost,
            "net_pnl": initial_capital * (gross_return - applied_cost),
            "net_return": gross_return - applied_cost,
            "prediction_label": pd.NA, "consequent": pd.NA, "confidence": np.nan,
        }
        trades = pd.DataFrame([] if reference else [trade], columns=TRADE_COLUMNS)
        key = "NIFTY50_PRICE_INDEX" if reference else f"BUY_HOLD:{ticker}"
        controls[key] = _result(daily, trades, initial_capital, reference=reference)
        if not reference:
            weight = 1 / len(config.stocks)
            basket_paths.append(daily["equity"].to_numpy() * weight)
            scaled = trade.copy()
            for name in ("entry_notional", "quantity", "gross_pnl", "transaction_cost", "net_pnl"):
                scaled[name] *= weight
            basket_trades.append(scaled)
    basket_daily = pd.DataFrame({"equity": np.sum(basket_paths, axis=0)}, index=days)
    basket_daily.index.name = "execution_session"
    controls["BUY_HOLD_BASKET"] = _result(
        basket_daily, pd.DataFrame(basket_trades, columns=TRADE_COLUMNS), initial_capital
    )
    return controls
