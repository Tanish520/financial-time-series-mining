"""Run and save the complete offline, nested walk-forward course experiment."""

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
from typing import Callable

import numpy as np
import pandas as pd

from src.asymmetry import score_asymmetry
from src.backtest import TRADE_COLUMNS, simulate_buy_and_hold, simulate_strategy
from src.data import DataValidationError, SnapshotConfig, load_snapshot, validate_prices
from src.labels import build_label_tables, label_returns
from src.pipeline import COST_SCENARIOS, _json_safe
from src.signals import predict_signals
from src.walkforward import FIXED_PARAMETERS, RuleParameters, TuningGrid, run_fold


@dataclass(frozen=True)
class ExperimentResult:
    folds: pd.DataFrame
    tuning: pd.DataFrame
    predictions: pd.DataFrame
    training_sessions: pd.DataFrame
    rules: list[dict]
    metrics: pd.DataFrame
    equity_curves: pd.DataFrame
    trades: pd.DataFrame
    outcomes: pd.DataFrame
    asymmetry: pd.DataFrame
    asymmetry_groups: pd.DataFrame
    latest_model: dict
    latest_prices: pd.DataFrame
    latest_signals: dict
    protocol: dict


def _evaluate(prices, config, predictions, executions, initial_capital):
    """One continuous account per independent strategy/scenario, never per fold."""
    metric_rows, curves, ledgers = [], [], []
    for cost in COST_SCENARIOS:
        paths = {}
        for variant in ("TUNED", "FIXED"):
            decisions = predictions.loc[predictions.variant.eq(variant)].copy()
            paths[f"{variant}_PORTFOLIO"] = simulate_strategy(
                prices, decisions, config, cost=cost, initial_capital=initial_capital,
            )
            for ticker in config.stocks:
                stock_config = replace(config, stocks=(ticker,))
                paths[f"{variant}:{ticker}"] = simulate_strategy(
                    prices.loc[prices.ticker.isin([ticker, config.benchmark])].copy(),
                    decisions.loc[decisions.ticker.eq(ticker)].copy(), stock_config,
                    cost=cost, initial_capital=initial_capital,
                )
        paths.update(simulate_buy_and_hold(prices, config, executions, cost=cost, initial_capital=initial_capital))
        for name, path in paths.items():
            applied = 0.0 if name == "NIFTY50_PRICE_INDEX" else cost
            metric_rows.append({"strategy": name, "cost": cost, "applied_cost": applied, **path.metrics})
            curves.append(path.daily[["equity", "daily_return"]].reset_index().assign(
                strategy=name, cost=cost, applied_cost=applied,
            ))
            if not path.trades.empty:
                ledger = path.trades.copy()
                if name.startswith(("TUNED", "FIXED")):
                    variant = name.split(":" , 1)[0].removesuffix("_PORTFOLIO")
                    provenance = predictions.loc[predictions.variant.eq(variant),
                                                 ["feature_session", "ticker", "fold_id", "rule_fit_cutoff"]]
                    ledger = ledger.merge(provenance, on=["feature_session", "ticker"], validate="one_to_one")
                ledgers.append(ledger.assign(strategy=name, cost=cost))
    return (pd.DataFrame(metric_rows), pd.concat(curves, ignore_index=True),
            pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(columns=[*TRADE_COLUMNS, "strategy", "cost", "fold_id", "rule_fit_cutoff"]))


def _outcomes(prices: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    realized = prices[["date", "ticker", "open", "close"]].copy()
    realized["target_return"] = realized.close / realized.open - 1
    result = predictions.merge(
        realized[["date", "ticker", "target_return"]].rename(columns={"date": "execution_session"}),
        on=["execution_session", "ticker"], validate="many_to_one",
    )
    result["outcome_available_session"] = result.execution_session
    result["target_label"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for theta, indices in result.loc[result.theta.notna()].groupby("theta").groups.items():
        result.loc[indices, "target_label"] = label_returns(result.loc[indices, "target_return"], theta)
    return result[["variant", "fold_id", "feature_session", "execution_session", "outcome_available_session",
                   "ticker", "theta", "prediction_label", "target_return", "target_label"]]


def run_experiment(
    prices: pd.DataFrame, config: SnapshotConfig, *, grid: TuningGrid = TuningGrid(),
    fixed: RuleParameters = FIXED_PARAMETERS, validation_sessions: int = 63,
    test_sessions: int = 21, min_validation_trades: int = 20,
    min_validation_trade_sessions: int = 5, initial_capital: float = 100000.0,
    progress: Callable[[dict], None] | None = None,
) -> ExperimentResult:
    """Score every remaining observed session using the unchanged nested protocol."""
    validate_prices(prices, config, min_sessions=2)
    calendar = pd.DatetimeIndex(prices.loc[prices.ticker.eq(config.benchmark), "date"]).sort_values().as_unit("ns")
    cutoff, fold_id = None, 0
    folds, tuning, predictions, sessions, rules = [], [], [], [], []
    while True:
        fold = run_fold(
            prices, config, cutoff=cutoff, grid=grid, fixed=fixed,
            validation_sessions=validation_sessions, test_sessions=test_sessions,
            min_validation_trades=min_validation_trades,
            min_validation_trade_sessions=min_validation_trade_sessions,
            initial_capital=initial_capital, score_outer=False,
        )
        selected = fold.selected
        winner = None if selected is None else fold.tuning.loc[fold.tuning.selected].iloc[0]
        row = {
            "fold_id": fold_id, "fit_cutoff": fold.cutoff, "inner_fit_cutoff": fold.validation_cutoff,
            "validation_start": fold.validation_sessions[0], "validation_end": fold.validation_sessions[-1],
            "test_start": fold.execution_sessions[0], "test_end": fold.execution_sessions[-1],
            "test_sessions": len(fold.execution_sessions), "eligible_candidates": int(fold.tuning.eligible.sum()),
            "selected_window": None if selected is None else str(selected.as_dict()["window"]),
            "theta": None if selected is None else selected.theta,
            "min_support": None if selected is None else selected.min_support,
            "min_confidence": None if selected is None else selected.min_confidence,
            "validation_net_return": None if winner is None else winner.validation_net_return,
            "validation_trade_count": None if winner is None else int(winner.validation_trade_count),
            "validation_trade_sessions": None if winner is None else int(winner.validation_trade_sessions),
            "rule_count": len(fold.models["TUNED"].rules), "fixed_rule_count": len(fold.models["FIXED"].rules),
            "fallback": "no_eligible_configuration" if selected is None else None,
        }
        folds.append(row)
        tuning.append(fold.tuning.assign(fold_id=fold_id))
        predictions.append(fold.predictions.assign(fold_id=fold_id))
        sessions.append(fold.training_sessions.assign(fold_id=fold_id))
        for variant, model in fold.models.items():
            theta = fixed.theta if variant == "FIXED" else (None if selected is None else selected.theta)
            rules.extend(dict(variant=variant, fold_id=fold_id, fit_cutoff=fold.cutoff.strftime("%Y-%m-%d"),
                              theta=theta, **rule) for rule in model.rules.to_dict("records"))
        if progress is not None:
            progress(row)
        if fold.execution_sessions[-1] == calendar[-1]:
            break
        cutoff = fold.execution_sessions[-1].strftime("%Y-%m-%d")
        fold_id += 1

    decisions = pd.concat(predictions, ignore_index=True)
    execution_days = pd.DatetimeIndex(decisions.execution_session.unique()).sort_values().as_unit("ns")
    metric_table, curves, ledger = _evaluate(prices, config, decisions, execution_days, initial_capital)
    outcomes = _outcomes(prices, decisions)
    asymmetry, groups = score_asymmetry(outcomes)
    final_model = fold.models["TUNED"]
    final_parameters = None if fold.selected is None else fold.selected.as_dict()
    latest_model = {
        "stocks": list(config.stocks), "benchmark": config.benchmark,
        "fit_cutoff": fold.cutoff.strftime("%Y-%m-%d"),
        "theta": None if selected is None else selected.theta,
        "parameters": final_parameters, "fallback": row["fallback"],
        "rules": final_model.rules.to_dict("records"),
    }
    latest_prices = prices.loc[prices.date.isin(calendar[-10:])].copy()
    latest_features = build_label_tables(prices, config, theta=fixed.theta if selected is None else selected.theta).features
    cached = predict_signals(final_model, latest_features.loc[latest_features.feature_session.eq(calendar[-1])].copy(), config.stocks)
    if selected is None:
        cached["reason"] = "no_eligible_configuration"
    elif final_model.rules.empty:
        cached["reason"] = "no_rules_final_fold"
    latest_signals = {
        "signals": cached.to_dict("records"), "observation_session": calendar[-1].strftime("%Y-%m-%d"),
        "fit_cutoff": latest_model["fit_cutoff"], "status": "cached", "fetched_at_utc": "",
        "message": "Frozen completed daily cache; use refresh to request recent bars.",
        "intended_execution": "next trading session",
    }
    protocol = {
        **fold.protocol, "fold_count": len(folds), "candidate_count_per_fold": len(grid.parameters()),
        "test_start": execution_days[0].strftime("%Y-%m-%d"), "test_end": execution_days[-1].strftime("%Y-%m-%d"),
        "test_sessions": len(execution_days), "fallback_folds": sum(item["fallback"] is not None for item in folds),
        "window_selection_counts": pd.Series([item["selected_window"] for item in folds if item["selected_window"] is not None], dtype="string").value_counts().sort_index().to_dict(),
        "source_start": calendar[0].strftime("%Y-%m-%d"), "source_end": calendar[-1].strftime("%Y-%m-%d"),
        "source_sessions": len(calendar),
    }
    return ExperimentResult(pd.DataFrame(folds), pd.concat(tuning, ignore_index=True), decisions,
                            pd.concat(sessions, ignore_index=True), rules, metric_table, curves, ledger,
                            outcomes, asymmetry, groups, latest_model, latest_prices, latest_signals, protocol)


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.bool_):
        return bool(value)
    return _json_safe(value)


def summarize_experiment(result: ExperimentResult) -> dict:
    return {
        "status": "complete_walk_forward_computed_offline", "evaluation_scope": "nested_walk_forward",
        **result.protocol, "prediction_rows": len(result.predictions), "tuning_rows": len(result.tuning),
        "comparison_at_0_1_percent_cost": result.metrics.loc[
            result.metrics.cost.eq(0.001) & result.metrics.strategy.isin(
                ["TUNED_PORTFOLIO", "FIXED_PORTFOLIO", "BUY_HOLD_BASKET", "NIFTY50_PRICE_INDEX"]
            )
        ].to_dict("records"),
        "asymmetry_summary": result.asymmetry.to_dict("records"),
    }


def write_experiment(result: ExperimentResult, output: Path, *, source_sha256: str,
                     source_manifest: dict | None = None) -> None:
    output = Path(output)
    if output.exists():
        raise DataValidationError(f"Experiment output already exists: {output}. Choose a new run directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        staging = Path(staging)
        predictions = result.predictions.copy()
        predictions["matched_antecedents"] = predictions.matched_antecedents.map(lambda items: " & ".join(items))
        for name, frame in (
            ("folds", result.folds), ("tuning", result.tuning), ("predictions", predictions),
            ("training_sessions", result.training_sessions), ("equity", result.equity_curves),
            ("metrics", result.metrics), ("trades", result.trades), ("outcomes", result.outcomes),
            ("asymmetry", result.asymmetry), ("asymmetry_groups", result.asymmetry_groups),
            ("latest_prices", result.latest_prices),
        ):
            frame.to_csv(staging / f"{name}.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n")
        latest_model = {**result.latest_model, "source_sha256": source_sha256}
        for name, records in (("metrics", result.metrics.to_dict("records")), ("rules", result.rules),
                              ("latest_model", latest_model), ("latest_signals", result.latest_signals)):
            (staging / f"{name}.json").write_text(json.dumps(_safe(records), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_sha256, **summarize_experiment(result),
            "source_correction": None if source_manifest is None else source_manifest.get("correction"),
            "source_audit_files": {} if source_manifest is None else source_manifest.get("auxiliary_files", {}),
            "cost_convention": "Flat total round-trip entry-notional fraction. Same frozen decisions at every cost. Holding controls pay once, half entry/half exit; index reference uncosted.",
            "fill_convention": "Idealized next open to same-session close, fractional shares, 1/N current equity, cash earns zero; capital compounds continuously across all folds.",
            "availability_convention": "Dates mean information after that exchange session closes; fitting outcomes mature at execution close.",
            "annualization": "Inclusive first-to-last execution calendar days; 365.25 days/year CAGR; sample daily std and sqrt(252) Sharpe at zero risk-free rate.",
            "metric_conventions": "Drawdown includes initial capital; profit factor and win rate use net currency PnL; undefined is null and positive infinity is Infinity text.",
            "asymmetry_convention": "Held-out precision and prediction-count-weighted fold/stock class base rates; lift=total correct/total expected correct. Unknown predictions stay unknown.",
            "limitations": ["Current convenience sample; historical index membership is not reconstructed.",
                            "Historical adjusted prices are not a point-in-time archive.",
                            "Idealized fills and flat friction scenarios, without broker-specific charges or liquidity modeling.",
                            "81 candidates may overfit validation; no formal significance claim.",
                            "Adaptive window selection alone does not establish concept drift.",
                            "Tuned-vs-fixed comparison combines window and threshold tuning; it is not a G1/G2 ablation."],
            "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py")) if path.name != "latest.py"},
            "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("pandas", "numpy", "mlxtend")}},
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/full_ten_stocks_corrected"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise DataValidationError(f"Experiment output already exists: {args.output}.")
        prices, manifest = load_snapshot(args.snapshot)
        def progress(row):
            print(f"Fold {row['fold_id'] + 1}: cutoff={row['fit_cutoff'].date()}, test={row['test_start'].date()}..{row['test_end'].date()}, eligible={row['eligible_candidates']}, window={row['selected_window'] or 'cash fallback'}", file=sys.stderr, flush=True)
        result = run_experiment(prices, SnapshotConfig(**manifest["request"]), progress=progress)
        write_experiment(result, args.output, source_sha256=manifest["sha256"], source_manifest=manifest)
        summary = summarize_experiment(result)
        summary.pop("grid")
        print(json.dumps(_safe({**summary, "source_sha256": manifest["sha256"], "output": str(args.output)}), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Experiment error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
