"""Display-only selections and formatting of verified saved experiment results."""

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
    min_confidence: float = 0.0, fold_id: int | None = None, variant: str | None = None,
) -> pd.DataFrame:
    mask = rules["confidence"] >= min_confidence
    if fold_id is not None:
        mask &= rules["fold_id"].eq(fold_id)
    if variant is not None:
        mask &= rules["variant"].eq(variant)
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
    names = {}
    if bundle.is_walk_forward:
        for variant, label in (("TUNED", "Tuned rules"), ("FIXED", "Fixed rules")):
            names[f"{variant}_PORTFOLIO" if scope == "Portfolio" else f"{variant}:{scope}"] = label
    else:
        names["RULES_PORTFOLIO" if scope == "Portfolio" else f"RULES:{scope}"] = "Rules strategy"
    names.update({
        "BUY_HOLD_BASKET" if scope == "Portfolio" else f"BUY_HOLD:{scope}": "Buy and hold",
        "NIFTY50_PRICE_INDEX": "NIFTY reference",
    })
    selected = bundle.metrics.loc[bundle.metrics.cost.eq(cost) & bundle.metrics.strategy.isin(names)]
    if len(selected) != len(names) or set(selected.strategy) != set(names):
        raise ResultBundleError("The requested saved cost scenario is unavailable.")
    selected = selected.set_index("strategy").loc[list(names)].copy()
    curve = bundle.equity.loc[bundle.equity.cost.eq(cost) & bundle.equity.strategy.isin(names)]
    if curve.duplicated(["strategy", "execution_session"]).any():
        raise ResultBundleError("The saved equity scenario contains duplicate observations.")
    chart = curve.pivot(index="execution_session", columns="strategy", values="equity").sort_index()
    if set(chart.columns) != set(names) or chart.isna().any().any():
        raise ResultBundleError("The saved equity scenario has incomplete comparisons.")
    chart = chart[list(names)].rename(columns=names) / bundle.manifest["initial_capital"] * 100
    chart.index.name = "Session"
    metrics = pd.DataFrame({
        label: [format_metric(metric, selected.loc[name, metric]) for metric in METRIC_LABELS]
        for name, label in names.items()
    }, index=pd.Index(METRIC_LABELS.values(), name="Metric"))
    return chart, metrics, selected


def tuning_tables(bundle: ResultBundle) -> tuple[pd.DataFrame, pd.Series, int]:
    """Count windows only where a fold has an actual eligible selection."""
    folds = bundle.folds.sort_values("fold_id").copy()
    winners = folds.loc[folds.fallback.isna() & folds.selected_window.notna()]
    frequencies = winners.selected_window.astype(str).value_counts().reindex(["250", "500", "expanding"], fill_value=0)
    frequencies.index.name = "Selected history"
    frequencies.name = "Winning folds"
    fallback_count = int(folds.fallback.notna().sum())
    display = pd.DataFrame({
        "Fold": folds.fold_id,
        "Test start": folds.test_start.dt.strftime("%d %b %Y"),
        "Test end": folds.test_end.dt.strftime("%d %b %Y"),
        "Window": folds.selected_window.fillna("N/A"),
        "Threshold θ": folds.theta.map(lambda v: format_metric("net_return", v)),
        "Min support": folds.min_support.map(lambda v: format_metric("net_return", v)),
        "Min confidence": folds.min_confidence.map(lambda v: format_metric("net_return", v)),
        "Validation net return": folds.validation_net_return.map(lambda v: format_metric("net_return", v)),
        "Validation trades": folds.validation_trade_count.map(lambda v: format_metric("trade_count", v)),
        "Validation trade sessions": folds.validation_trade_sessions.map(lambda v: format_metric("trade_count", v)),
        "Eligible candidates": folds.eligible_candidates,
        "Outer rules": folds.rule_count,
        "Status": folds.fallback.map(lambda v: "Eligible winner" if pd.isna(v) else "No eligible configuration · cash"),
    }).reset_index(drop=True)
    return display, frequencies, fallback_count


def asymmetry_tables(bundle: ResultBundle, *, variant: str = "TUNED") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = bundle.asymmetry.loc[bundle.asymmetry.variant.eq(variant)].set_index("direction").reindex(["U", "D"])
    if selected.prediction_count.isna().any():
        raise ResultBundleError("The requested saved asymmetry comparison is unavailable.")
    comparison = selected[["precision", "base_rate"]].rename(columns={"precision": "Precision", "base_rate": "Class base rate"})
    lift = selected[["held_out_lift"]].rename(columns={"held_out_lift": "Held-out lift"})
    counts = pd.DataFrame({
        "Predictions": selected.prediction_count.astype(int),
        "Correct": selected.correct_count.astype(int),
        "Precision": selected.precision.map(lambda v: format_metric("win_rate", v)),
        "Class base rate": selected.base_rate.map(lambda v: format_metric("win_rate", v)),
        "Expected correct": selected.expected_correct.map(lambda v: format_metric("expected_correct", v)),
        "Held-out lift": selected.held_out_lift.map(lambda v: format_metric("lift", v)),
    })
    counts.index.name = "Direction"
    return comparison, lift, counts
