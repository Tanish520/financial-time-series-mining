"""Inspect one nested chronological fold entirely from the frozen snapshot.

This checkpoint tunes on a common past-only block, refits through the outer
cutoff, and freezes the next block. The complete rolling experiment follows
after inspecting this fold; this module does not claim a completed study.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtest import TRADE_COLUMNS, simulate_buy_and_hold, simulate_strategy
from src.data import DataValidationError, SnapshotConfig, load_snapshot, validate_prices
from src.labels import LabelTables, build_label_tables
from src.mining import MiningResult, mine_rules, select_rules
from src.pipeline import COST_SCENARIOS, _json_safe
from src.signals import predict_signals

SELECTION_COST = 0.001


@dataclass(frozen=True)
class RuleParameters:
    window: int | None
    theta: float
    min_support: float
    min_confidence: float

    def __post_init__(self) -> None:
        if self.window is not None and (
            isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1
        ):
            raise DataValidationError("window must be a positive integer or None for expanding.")
        for name in ("theta", "min_support", "min_confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise DataValidationError(f"{name} must be a finite probability.")
            lower = value >= 0 if name == "min_confidence" else value > 0
            upper = value < 1 if name == "theta" else value <= 1
            if not lower or not upper:
                raise DataValidationError(f"{name} is outside its allowed probability range.")

    def as_dict(self) -> dict:
        return {"window": "expanding" if self.window is None else self.window,
                "theta": self.theta, "min_support": self.min_support,
                "min_confidence": self.min_confidence}

    @property
    def identifier(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


FIXED_PARAMETERS = RuleParameters(500, 0.005, 0.02, 0.60)


@dataclass(frozen=True)
class TuningGrid:
    windows: tuple[int | None, ...] = (250, 500, None)
    thetas: tuple[float, ...] = (0.0025, 0.005, 0.01)
    supports: tuple[float, ...] = (0.01, 0.02, 0.05)
    confidences: tuple[float, ...] = (0.55, 0.60, 0.70)

    def parameters(self) -> tuple[RuleParameters, ...]:
        axes = (self.windows, self.thetas, self.supports, self.confidences)
        if any(not axis or len(set(axis)) != len(axis) for axis in axes):
            raise DataValidationError("Grid axes must be nonempty and contain distinct values.")
        parameters = [RuleParameters(*values) for values in product(*axes)]
        return tuple(sorted(parameters, key=lambda value: value.identifier))


@dataclass(frozen=True)
class FoldResult:
    cutoff: pd.Timestamp
    validation_cutoff: pd.Timestamp
    validation_sessions: pd.DatetimeIndex
    execution_sessions: pd.DatetimeIndex
    selected: RuleParameters | None
    models: dict[str, MiningResult]
    tuning: pd.DataFrame
    predictions: pd.DataFrame
    training_sessions: pd.DataFrame
    metrics: pd.DataFrame
    equity_curves: pd.DataFrame
    trades: pd.DataFrame
    protocol: dict


def choose_parameter_id(scores: pd.DataFrame) -> str | None:
    """Exact objective ties use a predeclared canonical order, never test results."""
    eligible = scores.loc[scores.eligible].sort_values(
        ["validation_net_return", "parameter_id"], ascending=[False, True], kind="stable"
    )
    return None if eligible.empty else str(eligible.iloc[0].parameter_id)


def _fit_bounds(model: MiningResult) -> dict:
    return {
        f"{edge}_training_{kind}_session": (
            pd.NaT if model.sessions.empty else getattr(model.sessions[f"{kind}_session"], operation)()
        )
        for edge, operation in (("first", "min"), ("last", "max"))
        for kind in ("feature", "execution")
    }


def _fit(tables: LabelTables, stocks: tuple[str, ...], cutoff: pd.Timestamp, p: RuleParameters) -> MiningResult:
    return mine_rules(tables.samples, stocks, cutoff, window=p.window,
                      min_support=p.min_support, min_confidence=p.min_confidence)


def _predict(model: MiningResult, tables: LabelTables, stocks: tuple[str, ...],
             calendar: pd.DatetimeIndex, executions: pd.DatetimeIndex) -> pd.DataFrame:
    positions = calendar.get_indexer(executions)
    feature_days = calendar[positions - 1]
    observed = tables.features.loc[tables.features.feature_session.isin(feature_days)].copy()
    signals = predict_signals(model, observed, stocks)
    signals["execution_session"] = signals.feature_session.map(dict(zip(feature_days, executions)))
    return signals


def run_fold(
    prices: pd.DataFrame, config: SnapshotConfig, *, cutoff: str | None = None,
    grid: TuningGrid = TuningGrid(), fixed: RuleParameters = FIXED_PARAMETERS,
    validation_sessions: int = 63, test_sessions: int = 21,
    min_validation_trades: int = 20, min_validation_trade_sessions: int = 5,
    initial_capital: float = 100000.0, score_outer: bool = True,
) -> FoldResult:
    """Tune only before the outer cutoff, then freeze both tuned and fixed rules.

    Nondefault grids/window counts exist for deterministic small tests and are
    recorded verbatim. The CLI uses the approved, frozen 81-configuration grid.
    """
    validate_prices(prices, config, min_sessions=2)
    parameters = grid.parameters()
    for name, value in (("validation_sessions", validation_sessions), ("test_sessions", test_sessions),
                        ("min_validation_trades", min_validation_trades),
                        ("min_validation_trade_sessions", min_validation_trade_sessions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DataValidationError(f"{name} must be a positive integer.")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise DataValidationError("initial_capital must be finite and positive.")
    calendar = pd.DatetimeIndex(prices.loc[prices.ticker.eq(config.benchmark), "date"]).sort_values().as_unit("ns")
    labels = {theta: build_label_tables(prices, config, theta=theta)
              for theta in sorted({p.theta for p in (*parameters, fixed)})}
    known_sessions = pd.DatetimeIndex(next(iter(labels.values())).samples.execution_session.unique()).sort_values().as_unit("ns")
    minimum_fit = max((p.window or 1) for p in (*parameters, fixed))
    required = minimum_fit + validation_sessions
    if len(known_sessions) < required:
        raise DataValidationError(f"Insufficient eligible history: need {required} fitting plus validation outcomes.")
    try:
        outer_cutoff = known_sessions[required - 1] if cutoff is None else pd.Timestamp(cutoff)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("cutoff must be an observed session-close date.") from exc
    if (pd.isna(outer_cutoff) or outer_cutoff.tzinfo is not None
            or outer_cutoff != outer_cutoff.normalize() or outer_cutoff not in calendar):
        raise DataValidationError("cutoff must be an observed session-close date.")
    eligible = known_sessions[known_sessions <= outer_cutoff]
    if len(eligible) < required:
        raise DataValidationError(f"Insufficient eligible history at cutoff: need {required} outcomes.")
    validation_days = eligible[-validation_sessions:]
    inner_cutoff = calendar[calendar.get_loc(validation_days[0]) - 1]
    start = int(calendar.get_loc(outer_cutoff)) + 1
    executions = calendar[start:start + test_sessions]
    if not len(executions):
        raise DataValidationError("At least one next observed execution session is required.")

    cache: dict[tuple[int | None, float], MiningResult] = {}
    session_frames, scores = [], []
    for p in parameters:
        key = (p.window, p.theta)
        if key not in cache:
            base_parameters = replace(p, min_support=min(grid.supports), min_confidence=min(grid.confidences))
            cache[key] = _fit(labels[p.theta], config.stocks, inner_cutoff, base_parameters)
            session_frames.append(cache[key].sessions.assign(
                stage="inner", variant="VALIDATION", fit_cutoff=inner_cutoff,
                fit_id=f"{p.window if p.window is not None else 'expanding'}:{p.theta}",
            ))
        base = cache[key]
        model = replace(base, rules=select_rules(base, min_support=p.min_support, min_confidence=p.min_confidence),
                        min_support=p.min_support, min_confidence=p.min_confidence)
        decisions = _predict(model, labels[p.theta], config.stocks, calendar, validation_days)
        validation = simulate_strategy(prices, decisions, config, cost=SELECTION_COST, initial_capital=initial_capital)
        count = validation.metrics["trade_count"]
        trade_days = validation.trades.entry_session.nunique()
        scores.append({
            "parameter_id": p.identifier, **p.as_dict(),
            "window": "expanding" if p.window is None else str(p.window),
            "selection_cost": SELECTION_COST, "fit_cutoff": inner_cutoff,
            **_fit_bounds(model), "training_transactions": len(model.sessions),
            "validation_start": validation_days[0], "validation_end": validation_days[-1],
            "rule_count": len(model.rules), "validation_net_return": validation.metrics["net_return"],
            "validation_trade_count": count, "validation_trade_sessions": trade_days,
            "eligible": count >= min_validation_trades and trade_days >= min_validation_trade_sessions,
        })
    tuning = pd.DataFrame(scores)
    selected_id = choose_parameter_id(tuning)
    selected = next((p for p in parameters if p.identifier == selected_id), None)
    tuning["selected"] = tuning.parameter_id.eq(selected_id)

    # Outer outcomes have not been used above. Refit only after fixing selection.
    fixed_model = _fit(labels[fixed.theta], config.stocks, outer_cutoff, fixed)
    if selected is None:
        tuned_model = replace(fixed_model, rules=fixed_model.rules.iloc[:0].copy(),
                              transactions=fixed_model.transactions.iloc[:0].copy(),
                              sessions=fixed_model.sessions.iloc[:0].copy())
    else:
        tuned_model = _fit(labels[selected.theta], config.stocks, outer_cutoff, selected)
    models = {"TUNED": tuned_model, "FIXED": fixed_model}
    predictions = []
    for variant, model in models.items():
        p = selected if variant == "TUNED" else fixed
        signals = _predict(model, labels[fixed.theta if p is None else p.theta], config.stocks, calendar, executions)
        if p is None:
            signals["reason"] = "no_eligible_configuration"
        predictions.append(signals.assign(variant=variant, theta=np.nan if p is None else p.theta,
                                          parameter_id=None if p is None else p.identifier))
        if not model.sessions.empty:
            session_frames.append(model.sessions.assign(
                stage="refit", variant=variant, fit_cutoff=outer_cutoff, fit_id=p.identifier,
            ))
    predictions = pd.concat(predictions, ignore_index=True)

    # All decisions are frozen before costs or outer realized returns are scored.
    metric_rows, curves, ledgers = [], [], []
    for cost in COST_SCENARIOS if score_outer else ():
        paths = {
            f"{variant}_PORTFOLIO": simulate_strategy(
                prices, predictions.loc[predictions.variant.eq(variant)].copy(), config,
                cost=cost, initial_capital=initial_capital,
            ) for variant in models
        }
        paths.update(simulate_buy_and_hold(prices, config, executions, cost=cost, initial_capital=initial_capital))
        for name, result in paths.items():
            applied = 0.0 if name == "NIFTY50_PRICE_INDEX" else cost
            metric_rows.append({"strategy": name, "cost": cost, "applied_cost": applied, **result.metrics})
            curves.append(result.daily[["equity", "daily_return"]].reset_index().assign(
                strategy=name, cost=cost, applied_cost=applied,
            ))
            if not result.trades.empty:
                ledgers.append(result.trades.assign(strategy=name, cost=cost))
    protocol = {
        "grid": [p.as_dict() for p in parameters], "fixed_parameters": fixed.as_dict(),
        "validation_sessions": validation_sessions, "requested_test_sessions": test_sessions,
        "min_validation_trades": min_validation_trades,
        "min_validation_trade_sessions": min_validation_trade_sessions,
        "selection_cost": SELECTION_COST, "cost_scenarios": list(COST_SCENARIOS),
        "initial_capital": initial_capital, "stocks": list(config.stocks), "benchmark": config.benchmark,
        "tie_break": "maximum eligible validation net return, then ascending canonical parameter JSON",
    }
    return FoldResult(
        outer_cutoff, inner_cutoff, validation_days, executions, selected, models,
        tuning, predictions, pd.concat(session_frames, ignore_index=True),
        pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(columns=["strategy", "cost", "applied_cost"]),
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["execution_session", "equity", "daily_return", "strategy", "cost", "applied_cost"]),
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame(columns=[*TRADE_COLUMNS, "strategy", "cost"]),
        protocol,
    )


def summarize_fold(result: FoldResult) -> dict:
    def date(value):
        return None if pd.isna(value) else value.strftime("%Y-%m-%d")
    return {
        "status": "nested_fold_computed_offline",
        "evaluation_scope": "single_nested_fold_engineering_checkpoint",
        **result.protocol, "fit_cutoff": date(result.cutoff),
        "inner_fit_cutoff": date(result.validation_cutoff),
        "validation_start": date(result.validation_sessions[0]), "validation_end": date(result.validation_sessions[-1]),
        "test_start": date(result.execution_sessions[0]), "test_end": date(result.execution_sessions[-1]),
        "test_sessions": len(result.execution_sessions), "candidate_count": len(result.tuning),
        "eligible_candidates": int(result.tuning.eligible.sum()),
        "selected_parameters": None if result.selected is None else result.selected.as_dict(),
        "fallback": "no_eligible_configuration" if result.selected is None else None,
        "models": {variant: {"rule_count": len(model.rules), "training_transactions": len(model.sessions),
                             **{key: date(value) for key, value in _fit_bounds(model).items()}}
                   for variant, model in result.models.items()},
        "comparison_at_0_1_percent_cost": result.metrics.loc[
            result.metrics.cost.eq(SELECTION_COST)
            & result.metrics.strategy.isin(["TUNED_PORTFOLIO", "FIXED_PORTFOLIO", "BUY_HOLD_BASKET", "NIFTY50_PRICE_INDEX"]),
            ["strategy", "applied_cost", "net_return", "trade_count"],
        ].to_dict("records"),
    }


def write_fold(result: FoldResult, output: Path, *, source_sha256: str) -> None:
    """Publish a new checksummed local result directory; never overwrite a run."""
    output = Path(output)
    if output.exists():
        raise DataValidationError(f"Fold output already exists: {output}. Preview without --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        staging = Path(staging)
        predictions = result.predictions.copy()
        predictions["matched_antecedents"] = predictions.matched_antecedents.map(lambda items: " & ".join(items))
        for name, frame in (("tuning", result.tuning), ("predictions", predictions),
                            ("training_sessions", result.training_sessions), ("equity", result.equity_curves),
                            ("metrics", result.metrics), ("trades", result.trades)):
            frame.to_csv(staging / f"{name}.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n")
        rules = [dict(variant=variant, **row) for variant, model in result.models.items() for row in model.rules.to_dict("records")]
        for name, records in (("metrics", result.metrics.to_dict("records")), ("rules", rules)):
            (staging / f"{name}.json").write_text(
                json.dumps(_json_safe(records), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_sha256, **summarize_fold(result),
            "availability_convention": "All dates denote information after the named session close; targets mature at execution close.",
            "cost_convention": "Flat total round-trip entry-notional fraction; holding controls pay once, half entry/half exit; index reference uncosted.",
            "fill_convention": "Idealized next open to same-session close; fractional shares; 1/N current equity; uninvested cash earns zero.",
            "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("data.py", "labels.py", "mining.py", "signals.py", "backtest.py", "pipeline.py", "walkforward.py")},
            "software": {"python": platform.python_version(),
                         **{name: importlib.metadata.version(name) for name in ("pandas", "numpy", "mlxtend")}},
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/day1_three_stocks"))
    parser.add_argument("--cutoff", help="Observed session close; default is the earliest eligible cutoff")
    parser.add_argument("--output", type=Path, help="New result directory; omit for a read-only preview")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise DataValidationError(f"Fold output already exists: {args.output}. Preview without --output.")
        prices, manifest = load_snapshot(args.snapshot)
        result = run_fold(prices, SnapshotConfig(**manifest["request"]), cutoff=args.cutoff)
        summary = summarize_fold(result)
        # The saved manifest retains the full grid; keep the console concise.
        summary.pop("grid")
        summary["source_sha256"] = manifest["sha256"]
        if args.output is not None:
            write_fold(result, args.output, source_sha256=manifest["sha256"])
            summary["output"] = str(args.output)
        print(json.dumps(_json_safe(summary), sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Walk-forward error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
