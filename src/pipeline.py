"""Run a fixed chronological holdout entirely from the frozen local snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtest import TRADE_COLUMNS, simulate_buy_and_hold, simulate_strategy
from src.data import DataValidationError, SnapshotConfig, load_snapshot, validate_prices
from src.labels import build_label_tables
from src.mining import MiningResult, mine_rules
from src.signals import predict_signals

COST_SCENARIOS = (0.0005, 0.001, 0.005)


@dataclass(frozen=True)
class HoldoutResult:
    model: MiningResult
    predictions: pd.DataFrame
    execution_sessions: pd.DatetimeIndex
    metrics: pd.DataFrame
    equity_curves: pd.DataFrame
    trades: pd.DataFrame
    initial_capital: float


def run_holdout(
    prices: pd.DataFrame, config: SnapshotConfig, *, cutoff: str = "2022-12-30",
    test_sessions: int = 21, window: int | None = 500, theta: float = 0.005,
    min_support: float = 0.02, min_confidence: float = 0.60,
    costs: tuple[float, ...] = COST_SCENARIOS, initial_capital: float = 100000.0,
) -> HoldoutResult:
    """Fit once, freeze all dated predictions, then account for each cost scenario."""
    validate_prices(prices, config, min_sessions=2)
    calendar = pd.DatetimeIndex(
        prices.loc[prices.ticker.eq(config.benchmark), "date"]
    ).sort_values().as_unit("ns")
    try:
        cutoff = pd.Timestamp(cutoff)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("cutoff must identify an observed session-close date.") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize() or cutoff not in calendar:
        raise DataValidationError("cutoff must be an observed session-close date.")
    if isinstance(test_sessions, bool) or not isinstance(test_sessions, int) or test_sessions < 1:
        raise DataValidationError("test_sessions must be a positive integer.")
    if not costs or len(set(costs)) != len(costs):
        raise DataValidationError("Provide a nonempty set of distinct cost scenarios.")
    start = int(calendar.get_loc(cutoff))
    executions = calendar[start + 1:start + 1 + test_sessions]
    if len(executions) != test_sessions:
        raise DataValidationError("Insufficient observed sessions for the requested holdout.")
    feature_days = calendar[start:start + test_sessions]
    tables = build_label_tables(prices, config, theta=theta)
    model = mine_rules(
        tables.samples, config.stocks, cutoff, window=window,
        min_support=min_support, min_confidence=min_confidence,
    )
    observed_features = tables.features.loc[tables.features.feature_session.isin(feature_days)].copy()
    # This interface contains no realized outcome fields. Accounting starts only
    # after these predictions are fixed, and all costs use these same decisions.
    predictions = predict_signals(model, observed_features, config.stocks)
    predictions["execution_session"] = predictions.feature_session.map(dict(zip(feature_days, executions)))
    metric_rows, curves, ledgers = [], [], []
    for cost in costs:
        paths = {"RULES_PORTFOLIO": simulate_strategy(
            prices, predictions, config, cost=cost, initial_capital=initial_capital
        )}
        for ticker in config.stocks:
            stock_config = replace(config, stocks=(ticker,))
            stock_prices = prices.loc[prices.ticker.isin([ticker, config.benchmark])].copy()
            stock_decisions = predictions.loc[predictions.ticker.eq(ticker)].copy()
            paths[f"RULES:{ticker}"] = simulate_strategy(
                stock_prices, stock_decisions, stock_config, cost=cost, initial_capital=initial_capital
            )
        paths.update(simulate_buy_and_hold(
            prices, config, executions, cost=cost, initial_capital=initial_capital
        ))
        for name, result in paths.items():
            applied_cost = 0.0 if name == "NIFTY50_PRICE_INDEX" else cost
            metric_rows.append({"strategy": name, "cost": cost, "applied_cost": applied_cost, **result.metrics})
            curve = result.daily[["equity", "daily_return"]].reset_index()
            curves.append(curve.assign(strategy=name, cost=cost, applied_cost=applied_cost))
            if not result.trades.empty:
                ledgers.append(result.trades.assign(strategy=name, cost=cost))
    return HoldoutResult(
        model, predictions, executions, pd.DataFrame(metric_rows),
        pd.concat(curves, ignore_index=True),
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(columns=[*TRADE_COLUMNS, "strategy", "cost"]),
        initial_capital,
    )


def _json_safe(value):
    """Write undefined metrics as null and positive infinite profit factor as text."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _summary(result: HoldoutResult, source_sha256: str, theta: float, window: int | None) -> dict:
    portfolio = result.metrics.loc[result.metrics.strategy.eq("RULES_PORTFOLIO")]
    comparison = result.metrics.loc[
        result.metrics.strategy.isin(["RULES_PORTFOLIO", "BUY_HOLD_BASKET", "NIFTY50_PRICE_INDEX"])
        & result.metrics.cost.eq(0.001),
        ["strategy", "applied_cost", "net_return", "trade_count"],
    ]
    return {
        "status": "holdout_computed_offline", "evaluation_scope": "single_holdout_engineering_checkpoint",
        "source_sha256": source_sha256, "theta": theta,
        "window": "expanding" if window is None else window,
        "fit_cutoff": result.model.fit_cutoff.strftime("%Y-%m-%d"),
        "min_support": result.model.min_support, "min_confidence": result.model.min_confidence,
        "training_transactions": len(result.model.transactions), "model_rule_count": len(result.model.rules),
        "first_training_feature_session": None if result.model.sessions.empty else result.model.sessions.feature_session.min().strftime("%Y-%m-%d"),
        "last_training_feature_session": None if result.model.sessions.empty else result.model.sessions.feature_session.max().strftime("%Y-%m-%d"),
        "first_training_execution_session": None if result.model.sessions.empty else result.model.sessions.execution_session.min().strftime("%Y-%m-%d"),
        "last_training_execution_session": None if result.model.sessions.empty else result.model.sessions.execution_session.max().strftime("%Y-%m-%d"),
        "test_sessions": len(result.execution_sessions),
        "test_start": result.execution_sessions[0].strftime("%Y-%m-%d"),
        "test_end": result.execution_sessions[-1].strftime("%Y-%m-%d"),
        "prediction_rows": len(result.predictions), "initial_capital": result.initial_capital,
        "portfolio_metrics": portfolio.to_dict("records"),
        "comparison_at_0_1_percent_cost": comparison.to_dict("records"),
    }


def _write_result(result: HoldoutResult, output: Path, summary: dict) -> None:
    if output.exists():
        raise DataValidationError(f"Holdout output already exists: {output}. Preview without --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        staging = Path(staging)
        predictions = result.predictions.copy()
        predictions["matched_antecedents"] = predictions.matched_antecedents.map(lambda items: " & ".join(items))
        for filename, frame in (
            ("predictions.csv", predictions), ("equity.csv", result.equity_curves),
            ("metrics.csv", result.metrics), ("trades.csv", result.trades),
            ("training_sessions.csv", result.model.sessions),
        ):
            frame.to_csv(staging / filename, index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n")
        for filename, records in (
            ("metrics.json", result.metrics.to_dict("records")),
            ("rules.json", result.model.rules.to_dict("records")),
        ):
            (staging / filename).write_text(
                json.dumps(_json_safe(records), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
        files = ("predictions.csv", "equity.csv", "metrics.csv", "metrics.json", "trades.csv", "rules.json", "training_sessions.csv")
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
            "cost_convention": "Flat total round-trip fraction of entry notional; no charge for cash; buy-and-hold half at entry and half at exit; NIFTY reference uncosted.",
            "fill_convention": "Idealized next-session open and same-session close, fractional shares, fixed 1/N current-equity allocation, no cash interest.",
            "zero_volume_policy": "Abort an assumed LONG execution on a zero-volume bar; do not selectively drop the trade or session.",
            "annualization": "CAGR uses inclusive first-to-last execution calendar dates and 365.25 days/year; daily Sharpe uses sample std, sqrt(252), risk-free rate zero.",
            "metric_conventions": "Max drawdown is a positive loss magnitude including initial capital; undefined metrics null; positive profit factor with no losses is Infinity text in JSON.",
            "code_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("data.py", "labels.py", "mining.py", "signals.py", "backtest.py", "pipeline.py")
            },
            "software": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("pandas", "numpy", "mlxtend")},
            },
            "files": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in files},
        }
        (staging / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/day1_three_stocks"))
    parser.add_argument("--cutoff", default="2022-12-30")
    parser.add_argument("--window", default="500", help="Positive session count or expanding")
    parser.add_argument("--test-sessions", type=int, default=21)
    parser.add_argument("--theta", type=float, default=0.005)
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        try:
            window = None if args.window == "expanding" else int(args.window)
        except ValueError as exc:
            raise DataValidationError("window must be a positive integer or expanding.") from exc
        prices, source_manifest = load_snapshot(args.snapshot)
        result = run_holdout(
            prices, SnapshotConfig(**source_manifest["request"]), cutoff=args.cutoff,
            test_sessions=args.test_sessions, window=window, theta=args.theta,
            min_support=args.min_support, min_confidence=args.min_confidence,
            initial_capital=args.initial_capital,
        )
        summary = _summary(result, source_manifest["sha256"], args.theta, window)
        if args.output is not None:
            _write_result(result, args.output, summary)
            summary["output"] = str(args.output)
        print(json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Holdout error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
