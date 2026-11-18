"""Small display-only transformations of saved rules, metrics and equity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.results import ResultBundle, ResultBundleError

METRIC_LABELS = {
    "net_return": "Net return", "cagr": "CAGR", "sharpe": "Sharpe",
    "max_drawdown": "Max drawdown", "profit_factor": "Profit factor",
    "win_rate": "Win rate", "trade_count": "Round trips",
}


def format_metric(metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    if np.isinf(number):
        return "∞" if number > 0 else "−∞"
    if metric in {"net_return", "cagr", "max_drawdown", "win_rate"}:
        return f"{number:.2%}"
    if metric == "trade_count":
        return str(int(number))
    return f"{number:.2f}"


def filter_rules(
    rules: pd.DataFrame, *, target: str = "All stocks", direction: str = "All",
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    mask = rules["confidence"] >= min_confidence
    if target != "All stocks":
        mask &= rules["target_ticker"].eq(target)
    if direction != "All":
        mask &= rules["target_label"].eq(direction)
    return rules.loc[mask].copy().reset_index(drop=True)


def backtest_tables(
    bundle: ResultBundle, *, scope: str = "Portfolio", cost: float = 0.001
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scope != "Portfolio" and scope not in bundle.tickers:
        raise ResultBundleError("Unknown saved stock scope.")
    names = {
        "RULES_PORTFOLIO" if scope == "Portfolio" else f"RULES:{scope}": "Rules strategy",
        "BUY_HOLD_BASKET" if scope == "Portfolio" else f"BUY_HOLD:{scope}": "Buy and hold",
        "NIFTY50_PRICE_INDEX": "NIFTY reference",
    }
    selected = bundle.metrics.loc[bundle.metrics.cost.eq(cost) & bundle.metrics.strategy.isin(names)]
    if len(selected) != 3 or set(selected.strategy) != set(names):
        raise ResultBundleError("The requested saved cost scenario is unavailable.")
    selected = selected.set_index("strategy").loc[list(names)]
    curve = bundle.equity.loc[bundle.equity.cost.eq(cost) & bundle.equity.strategy.isin(names)]
    chart = curve.pivot(index="execution_session", columns="strategy", values="equity")
    if set(chart.columns) != set(names) or chart.isna().any().any():
        raise ResultBundleError("The saved equity scenario has incomplete comparisons.")
    chart = chart[list(names)].rename(columns=names) / bundle.manifest["initial_capital"] * 100
    chart.index.name = "Session"
    metrics = pd.DataFrame({
        label: [format_metric(metric, selected.loc[name, metric]) for metric in METRIC_LABELS]
        for name, label in names.items()
    }, index=pd.Index(METRIC_LABELS.values(), name="Metric"))
    return chart, metrics, selected
