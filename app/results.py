"""Verify and read precomputed display bundles without research/provider imports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BUNDLE = ARTIFACTS / "full_walkforward"
LEGACY_BUNDLE = ARTIFACTS / "day1_holdout"
BUNDLE_FILES = {
    "predictions.csv", "equity.csv", "metrics.csv", "metrics.json",
    "trades.csv", "rules.json", "training_sessions.csv",
}
FULL_BUNDLE_FILES = BUNDLE_FILES | {
    "folds.csv", "tuning.csv", "outcomes.csv", "asymmetry.csv",
    "asymmetry_groups.csv", "latest_model.json", "latest_prices.csv", "latest_signals.json",
}
RULE_COLUMNS = [
    "antecedents", "consequent", "target_ticker", "target_label", "support",
    "confidence", "lift", "antecedent_support", "consequent_support",
    "joint_count", "antecedent_count", "consequent_count", "transaction_count",
]
METRICS = {"net_return", "cagr", "sharpe", "max_drawdown", "profit_factor", "win_rate", "trade_count"}


class ResultBundleError(ValueError):
    """The saved experiment is missing, inconsistent or no longer verifies."""


def _validate_manifest(meta: dict, version: int) -> None:
    required = {"schema_version", "evaluation_scope", "files", "initial_capital", "test_start",
                "test_end", "test_sessions", "source_sha256"}
    required |= ({"stocks", "benchmark", "fold_count", "cost_scenarios", "grid", "selection_cost",
                  "min_validation_trades", "min_validation_trade_sessions"} if version == 2 else
                 {"fit_cutoff", "window", "theta", "min_support", "min_confidence",
                  "training_transactions", "model_rule_count"})
    if not required.issubset(meta):
        raise ResultBundleError("Result bundle manifest fields are incomplete.")
    values = [("initial_capital", 0, None)]
    if version == 2:
        values.append(("selection_cost", -1, 1))
        if (not isinstance(meta["stocks"], list) or not meta["stocks"]
                or not all(isinstance(ticker, str) and ticker for ticker in meta["stocks"])
                or len(set(meta["stocks"])) != len(meta["stocks"])
                or not isinstance(meta["benchmark"], str) or meta["benchmark"] in meta["stocks"]):
            raise ResultBundleError("Result bundle manifest stock universe is invalid.")
    for name, lower, upper in values:
        value = meta[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or value <= lower or (name == "selection_cost" and value < 0)
                or (upper is not None and value >= upper)):
            raise ResultBundleError(f"Result bundle manifest {name} is invalid.")
    counts = ["test_sessions"] + (["fold_count", "min_validation_trades", "min_validation_trade_sessions"] if version == 2 else [])
    for name in counts:
        if isinstance(meta[name], bool) or not isinstance(meta[name], int) or meta[name] < 1:
            raise ResultBundleError(f"Result bundle manifest {name} must be a positive integer.")


@dataclass(frozen=True)
class ResultBundle:
    manifest: dict
    rules: pd.DataFrame
    predictions: pd.DataFrame
    equity: pd.DataFrame
    metrics: pd.DataFrame
    trades: pd.DataFrame
    training_sessions: pd.DataFrame
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuning: pd.DataFrame = field(default_factory=pd.DataFrame)
    outcomes: pd.DataFrame = field(default_factory=pd.DataFrame)
    asymmetry: pd.DataFrame = field(default_factory=pd.DataFrame)
    asymmetry_groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_model: dict = field(default_factory=dict)
    latest_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    latest_signals: dict = field(default_factory=dict)

    @property
    def is_walk_forward(self) -> bool:
        return self.manifest["schema_version"] == 2

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest.get("stocks", self.predictions["ticker"].unique())))


def default_bundle() -> Path:
    """The completed study is the default; legacy previews require an explicit path."""
    return DEFAULT_BUNDLE


def _required(frame: pd.DataFrame, names: set[str], label: str) -> None:
    if not names.issubset(frame.columns):
        raise ResultBundleError(f"Result bundle {label} columns are incomplete.")


def _csv(payloads: dict[str, bytes], name: str, dates: tuple[str, ...] = (), **kwargs) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(payloads[name]), **kwargs)
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def _validate_scenarios(bundle: ResultBundle) -> None:
    metrics, equity = bundle.metrics, bundle.equity
    _required(metrics, {"strategy", "cost", "applied_cost", *METRICS}, "metric")
    _required(equity, {"strategy", "cost", "applied_cost", "execution_session", "equity"}, "equity")
    if metrics.duplicated(["strategy", "cost"]).any() or equity.duplicated(["strategy", "cost", "execution_session"]).any():
        raise ResultBundleError("Result bundle contains duplicate scenario rows.")
    for name, frame in (("metric", metrics), ("equity", equity)):
        costs = frame["cost"].to_numpy(dtype=float)
        applied = frame["applied_cost"].to_numpy(dtype=float)
        expected = np.where(frame.strategy.eq("NIFTY50_PRICE_INDEX"), 0.0, costs)
        if not np.isfinite(costs).all() or not np.array_equal(applied, expected):
            raise ResultBundleError(f"Result bundle {name} applied costs are inconsistent.")
    values = equity.equity.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ResultBundleError("Result bundle equity must be positive and finite.")
    dates = pd.DatetimeIndex(sorted(bundle.predictions.execution_session.unique()))
    strategies = {"BUY_HOLD_BASKET", "NIFTY50_PRICE_INDEX", *[f"BUY_HOLD:{t}" for t in bundle.tickers]}
    variants = ("TUNED", "FIXED") if bundle.is_walk_forward else ("RULES",)
    for variant in variants:
        strategies |= {f"{variant}_PORTFOLIO", *[f"{variant}:{t}" for t in bundle.tickers]}
    expected_costs = set(bundle.manifest.get("cost_scenarios", metrics.cost.unique()))
    if not expected_costs or set(metrics.cost) != expected_costs or set(equity.cost) != expected_costs:
        raise ResultBundleError("Result bundle cost scenarios are inconsistent.")
    for cost in expected_costs:
        selected = metrics.loc[metrics.cost.eq(cost)]
        curve = equity.loc[equity.cost.eq(cost)]
        if set(selected.strategy) != strategies or set(curve.strategy) != strategies:
            raise ResultBundleError("Result bundle scenario comparisons are incomplete.")
        for _, group in curve.groupby("strategy", sort=False):
            if not pd.DatetimeIndex(group.execution_session.sort_values()).equals(dates):
                raise ResultBundleError("Result bundle equity scenarios do not share every test session.")


def _validate_full(bundle: ResultBundle) -> None:
    meta, folds, predictions = bundle.manifest, bundle.folds, bundle.predictions
    _required(folds, {"fold_id", "fit_cutoff", "inner_fit_cutoff", "validation_start", "validation_end",
                      "test_start", "test_end", "test_sessions", "eligible_candidates", "selected_window",
                      "theta", "min_support", "min_confidence", "validation_net_return", "validation_trade_count",
                      "validation_trade_sessions", "rule_count", "fallback"}, "fold")
    _required(predictions, {"fold_id", "variant"}, "prediction")
    _required(bundle.training_sessions, {"fold_id", "variant", "stage", "fit_id", "fit_cutoff",
                                        "feature_session", "execution_session", "outcome_available_session"}, "training-session")
    _required(bundle.tuning, {"fold_id", "window", "theta", "min_support", "min_confidence",
                             "validation_net_return", "validation_trade_count", "validation_trade_sessions",
                             "eligible", "selected", "fit_cutoff"}, "tuning")
    _required(bundle.outcomes, {"fold_id", "variant", "ticker", "feature_session", "execution_session",
                               "outcome_available_session", "theta", "prediction_label", "target_return", "target_label"}, "outcome")
    _required(bundle.asymmetry_groups, {"fold_id", "variant", "ticker", "direction", "observations",
                                       "prediction_count", "correct_count", "class_base_rate", "expected_correct"}, "asymmetry-group")
    _required(bundle.trades, {"strategy", "cost", "ticker", "entry_session", "exit_session", "net_pnl"}, "trade")
    if (len(folds) != meta["fold_count"] or not len(folds) or folds.fold_id.isna().any()
            or folds.fold_id.duplicated().any() or not folds.fold_id.mod(1).eq(0).all()):
        raise ResultBundleError("Result bundle fold count or identifiers are inconsistent.")
    if set(predictions.fold_id) != set(folds.fold_id) or set(predictions.variant) != {"TUNED", "FIXED"}:
        raise ResultBundleError("Result bundle prediction folds or variants are inconsistent.")
    if predictions.duplicated(["fold_id", "variant", "execution_session", "ticker"]).any():
        raise ResultBundleError("Result bundle contains duplicate predictions.")
    if set(predictions.ticker) != set(meta["stocks"]):
        raise ResultBundleError("Result bundle stock universe is inconsistent.")
    for _, cohort in predictions.groupby(["variant", "execution_session"], sort=False):
        if len(cohort) != len(bundle.tickers) or set(cohort.ticker) != set(bundle.tickers):
            raise ResultBundleError("Result bundle contains an incomplete prediction cohort.")
    for fold in folds.itertuples():
        if not (fold.inner_fit_cutoff < fold.validation_start <= fold.validation_end <= fold.fit_cutoff < fold.test_start <= fold.test_end):
            raise ResultBundleError("Result bundle fold availability dates are inconsistent.")
        fallback = pd.notna(fold.fallback)
        if fallback and fold.fallback != "no_eligible_configuration":
            raise ResultBundleError("Result bundle has an unknown fold fallback.")
        if fallback != pd.isna(fold.selected_window):
            raise ResultBundleError("Result bundle fallback and selected parameters are inconsistent.")
        for variant in ("TUNED", "FIXED"):
            selected = predictions.loc[predictions.fold_id.eq(fold.fold_id) & predictions.variant.eq(variant)]
            sessions = selected.execution_session.drop_duplicates().sort_values()
            if len(sessions) != fold.test_sessions or sessions.min() != fold.test_start or sessions.max() != fold.test_end:
                raise ResultBundleError("Result bundle fold test dates or counts are inconsistent.")
            if not selected.rule_fit_cutoff.eq(fold.fit_cutoff).all():
                raise ResultBundleError("Result bundle prediction fitting provenance is inconsistent.")
    _required(bundle.asymmetry, {"variant", "direction", "prediction_count", "correct_count", "precision",
                                  "base_rate", "expected_correct", "held_out_lift"}, "asymmetry")
    if bundle.asymmetry.duplicated(["variant", "direction"]).any():
        raise ResultBundleError("Result bundle has duplicate asymmetry summaries.")
    if set(zip(bundle.asymmetry.variant, bundle.asymmetry.direction)) != {(v, d) for v in ("TUNED", "FIXED") for d in ("U", "D")}:
        raise ResultBundleError("Result bundle asymmetry comparisons are incomplete.")
    model, latest = bundle.latest_model, bundle.latest_signals
    if (set(model["stocks"]) != set(bundle.tickers) or model["source_sha256"] != meta["source_sha256"]
            or model["benchmark"] != meta["benchmark"]):
        raise ResultBundleError("Result bundle latest-model provenance is inconsistent.")
    if pd.Timestamp(model["fit_cutoff"]) != folds.fit_cutoff.max():
        raise ResultBundleError("Result bundle latest model does not use the final fold.")
    final = folds.sort_values("fit_cutoff").iloc[-1]
    final_fallback = None if pd.isna(final.fallback) else final.fallback
    if model["fallback"] != final_fallback:
        raise ResultBundleError("Result bundle latest model disagrees with the final tuned fallback.")
    if final_fallback is not None:
        if model["theta"] is not None or model["parameters"] is not None or model["rules"]:
            raise ResultBundleError("Result bundle final fallback cannot have a selected model or rules.")
    else:
        expected = {"window": "expanding" if final.selected_window == "expanding" else int(final.selected_window),
                    "theta": float(final.theta), "min_support": float(final.min_support),
                    "min_confidence": float(final.min_confidence)}
        parameters = model["parameters"]
        if (not isinstance(parameters, dict) or set(parameters) != set(expected)
                or parameters["window"] != expected["window"]
                or any(not np.isclose(parameters[name], expected[name], rtol=0, atol=8*np.finfo(float).eps)
                       for name in ("theta", "min_support", "min_confidence"))
                or not np.isclose(model["theta"], expected["theta"], rtol=0, atol=8*np.finfo(float).eps)):
            raise ResultBundleError("Result bundle latest parameters differ from the final tuned fold.")
    final_rules = bundle.rules.loc[bundle.rules.fold_id.eq(final.fold_id) & bundle.rules.variant.eq("TUNED"), RULE_COLUMNS]
    def canonical_rule(row):
        if not isinstance(row, dict) or set(row) != set(RULE_COLUMNS):
            raise ResultBundleError("Result bundle latest rules have incomplete columns.")
        normalized = {**row, "antecedents": sorted(row["antecedents"])}
        return json.dumps(normalized, sort_keys=True, allow_nan=False)
    if sorted(map(canonical_rule, model["rules"])) != sorted(map(canonical_rule, final_rules.to_dict("records"))):
        raise ResultBundleError("Result bundle latest rules differ from the final frozen tuned rules.")
    if not {"signals", "observation_session", "fit_cutoff", "status", "message", "fetched_at_utc"}.issubset(latest):
        raise ResultBundleError("Result bundle cached latest-signal fields are incomplete.")
    if latest["fit_cutoff"] != model["fit_cutoff"] or latest["status"] not in {"cached", "fallback"}:
        raise ResultBundleError("Result bundle cached latest-signal provenance is inconsistent.")
    signal_rows = pd.DataFrame(latest["signals"])
    _required(signal_rows, {"ticker", "prediction_label", "action", "rule_fit_cutoff", "feature_session"}, "latest-signal")
    if len(signal_rows) != len(bundle.tickers) or set(signal_rows.ticker) != set(bundle.tickers):
        raise ResultBundleError("Result bundle cached latest signals are incomplete.")
    valid_classes = signal_rows.prediction_label.isna() | signal_rows.prediction_label.isin(["U", "D", "N"])
    if (not valid_classes.all() or not signal_rows.action.isin(["LONG", "CASH"]).all()
            or not np.array_equal(signal_rows.action.eq("LONG").to_numpy(),
                                  signal_rows.prediction_label.eq("U").fillna(False).to_numpy(dtype=bool))):
        raise ResultBundleError("Result bundle cached signal actions conflict with their predicted classes.")
    if (final_fallback is not None or not model["rules"]) and (
        not signal_rows.action.eq("CASH").all() or not signal_rows.prediction_label.isna().all()
    ):
        raise ResultBundleError("Result bundle final fallback or no-rule model must retain CASH/unknown signals.")
    observation = pd.Timestamp(latest["observation_session"])
    if pd.isna(observation) or observation < pd.Timestamp(model["fit_cutoff"]):
        raise ResultBundleError("Result bundle latest observation precedes the frozen model.")
    if not signal_rows.rule_fit_cutoff.eq(model["fit_cutoff"]).all() or not pd.to_datetime(signal_rows.feature_session).eq(observation).all():
        raise ResultBundleError("Result bundle cached latest-signal timestamps are inconsistent.")
    _required(bundle.latest_prices, {"date", "ticker", "open", "high", "low", "close", "volume", "dividends", "stock_splits"}, "latest-price")
    if bundle.latest_prices.duplicated(["date", "ticker"]).any():
        raise ResultBundleError("Result bundle latest prices contain duplicates.")
    expected_stocks = {*bundle.tickers, meta["benchmark"]}
    cached = bundle.latest_prices.loc[bundle.latest_prices.date.le(observation)]
    shared = cached.groupby("date").ticker.agg(lambda tickers: set(tickers) == expected_stocks)
    if len(shared) < 2 or not shared.all() or cached.date.max() != observation:
        raise ResultBundleError("Result bundle cached latest prices do not form completed common sessions.")


@lru_cache(maxsize=3)
def _parse_verified(manifest_bytes: bytes, payload_items: tuple[tuple[str, bytes], ...]) -> ResultBundle:
    """Cache parsing by verified content, never by path or modification timestamp."""
    meta = json.loads(manifest_bytes)
    payloads = dict(payload_items)
    full = meta["schema_version"] == 2
    predictions = _csv(payloads, "predictions.csv", ("feature_session", "execution_session", "rule_fit_cutoff"))
    _required(predictions, {"ticker", "feature_session", "execution_session", "rule_fit_cutoff", "prediction_label"}, "prediction")
    equity = _csv(payloads, "equity.csv", ("execution_session",))
    metrics = pd.DataFrame(json.loads(payloads["metrics.json"]))
    rule_columns = [*RULE_COLUMNS, *(["fold_id", "variant", "fit_cutoff", "theta"] if full else [])]
    rule_records = json.loads(payloads["rules.json"])
    if not isinstance(rule_records, list) or any(not isinstance(row, dict) or not set(rule_columns).issubset(row) for row in rule_records):
        raise ResultBundleError("Result bundle rule columns are incomplete.")
    rules = pd.DataFrame(rule_records, columns=rule_columns)
    for antecedents in rules.antecedents:
        if not isinstance(antecedents, list) or not antecedents or not all(isinstance(item, str) for item in antecedents):
            raise ResultBundleError("Result bundle rule antecedents must be observed item strings.")
    rules["antecedents"] = rules.antecedents.map(tuple)
    if full:
        rules["fit_cutoff"] = pd.to_datetime(rules.fit_cutoff)
    sessions = _csv(payloads, "training_sessions.csv", ("feature_session", "execution_session", "outcome_available_session", "fit_cutoff"))
    extra = {}
    if full:
        extra = {
            "folds": _csv(payloads, "folds.csv", ("fit_cutoff", "inner_fit_cutoff", "validation_start", "validation_end", "test_start", "test_end"), dtype={"selected_window": "string"}),
            "tuning": _csv(payloads, "tuning.csv", ("fit_cutoff", "validation_start", "validation_end"), dtype={"window": "string"}),
            "outcomes": _csv(payloads, "outcomes.csv", ("feature_session", "execution_session", "outcome_available_session")),
            "asymmetry": _csv(payloads, "asymmetry.csv"),
            "asymmetry_groups": _csv(payloads, "asymmetry_groups.csv"),
            "latest_model": json.loads(payloads["latest_model.json"]),
            "latest_prices": _csv(payloads, "latest_prices.csv", ("date",)),
            "latest_signals": json.loads(payloads["latest_signals.json"]),
        }
    bundle = ResultBundle(meta, rules, predictions, equity, metrics, _csv(payloads, "trades.csv"), sessions, **extra)
    capital = float(meta["initial_capital"])
    if not np.isfinite(capital) or capital <= 0:
        raise ResultBundleError("Result bundle initial capital is invalid.")
    if predictions.empty or predictions.execution_session.isna().any():
        raise ResultBundleError("Result bundle prediction dates are missing.")
    if (predictions.execution_session.min().strftime("%Y-%m-%d") != meta["test_start"]
            or predictions.execution_session.max().strftime("%Y-%m-%d") != meta["test_end"]):
        raise ResultBundleError("Result bundle test dates do not match its manifest.")
    if "prediction_rows" in meta and len(predictions) != meta["prediction_rows"]:
        raise ResultBundleError("Result bundle prediction counts do not match its manifest.")
    if "test_sessions" in meta and predictions.execution_session.nunique() != meta["test_sessions"]:
        raise ResultBundleError("Result bundle test-session count is inconsistent.")
    if full:
        _validate_full(bundle)
    elif len(rules) != meta["model_rule_count"] or len(sessions) != meta["training_transactions"]:
        raise ResultBundleError("Result bundle fitting counts do not match its manifest.")
    _validate_scenarios(bundle)
    return bundle


def load_bundle(directory: str | Path | None = None) -> ResultBundle:
    """Recheck every file's bytes before parsing or reusing a content-keyed parse."""
    directory = Path(directory) if directory is not None else default_bundle()
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        version = manifest["schema_version"]
        expected = BUNDLE_FILES if version == 1 else FULL_BUNDLE_FILES if version == 2 else None
        if expected is None or not isinstance(manifest["files"], dict) or set(manifest["files"]) != expected:
            raise ResultBundleError("Unsupported result bundle schema or file set.")
        scopes = {1: "single_holdout_engineering_checkpoint", 2: "nested_walk_forward"}
        if manifest["evaluation_scope"] != scopes[version]:
            raise ResultBundleError("Unsupported result bundle evaluation scope.")
        _validate_manifest(manifest, version)
        payloads = []
        for name in sorted(expected):
            payload = (directory / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != manifest["files"][name]:
                raise ResultBundleError(f"Result bundle checksum mismatch: {name}.")
            payloads.append((name, payload))
        bundle = _parse_verified(manifest_bytes, tuple(payloads))
        # Callers cannot poison the content cache through a mutable DataFrame/dict.
        values = {item.name: getattr(bundle, item.name) for item in fields(bundle)}
        return ResultBundle(**{key: value.copy(deep=True) if isinstance(value, pd.DataFrame)
                               else json.loads(json.dumps(value)) for key, value in values.items()})
    except ResultBundleError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ResultBundleError(f"Cannot load a complete saved result bundle: {exc}") from exc


def package_bundle(source: str | Path, destination: str | Path) -> None:
    """Explicitly curate verified derived artifacts; never download or overwrite."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ResultBundleError(f"Display bundle already exists: {destination}.")
    bundle = load_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as staging:
        staging = Path(staging)
        for name in (*sorted(bundle.manifest["files"]), "manifest.json"):
            shutil.copyfile(source / name, staging / name)
        load_bundle(staging)
        staging.rename(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a verified offline display bundle.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    try:
        package_bundle(args.source, args.output)
        print(f"Packaged verified display bundle: {args.output}")
        return 0
    except ResultBundleError as exc:
        print(f"Bundle error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
