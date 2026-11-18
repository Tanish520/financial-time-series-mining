"""Verify and read a precomputed display bundle; no research/provider imports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import pandas as pd

DEFAULT_BUNDLE = Path(__file__).resolve().parent / "artifacts/day1_holdout"
BUNDLE_FILES = {
    "predictions.csv", "equity.csv", "metrics.csv", "metrics.json",
    "trades.csv", "rules.json", "training_sessions.csv",
}
RULE_COLUMNS = [
    "antecedents", "consequent", "target_ticker", "target_label", "support",
    "confidence", "lift", "antecedent_support", "consequent_support",
    "joint_count", "antecedent_count", "consequent_count", "transaction_count",
]


class ResultBundleError(ValueError):
    """The saved experiment is missing, inconsistent or no longer verifies."""


@dataclass(frozen=True)
class ResultBundle:
    manifest: dict
    rules: pd.DataFrame
    predictions: pd.DataFrame
    equity: pd.DataFrame
    metrics: pd.DataFrame
    trades: pd.DataFrame
    training_sessions: pd.DataFrame

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.predictions["ticker"].unique()))


def load_bundle(directory: str | Path = DEFAULT_BUNDLE) -> ResultBundle:
    """Parse the exact bytes checked against the producer's file checksums."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        if manifest["schema_version"] != 1 or set(manifest["files"]) != BUNDLE_FILES:
            raise ResultBundleError("Unsupported result bundle schema or file set.")
        if manifest["evaluation_scope"] != "single_holdout_engineering_checkpoint":
            raise ResultBundleError("This preview expects a fixed-holdout result bundle.")
        payloads = {name: (directory / name).read_bytes() for name in BUNDLE_FILES}
        for name, payload in payloads.items():
            if hashlib.sha256(payload).hexdigest() != manifest["files"][name]:
                raise ResultBundleError(f"Result bundle checksum mismatch: {name}.")
        predictions = pd.read_csv(BytesIO(payloads["predictions.csv"]), parse_dates=[
            "feature_session", "execution_session", "rule_fit_cutoff",
        ])
        equity = pd.read_csv(BytesIO(payloads["equity.csv"]), parse_dates=["execution_session"])
        metrics = pd.DataFrame(json.loads(payloads["metrics.json"]))
        rules = pd.DataFrame(json.loads(payloads["rules.json"]), columns=RULE_COLUMNS)
        trades = pd.read_csv(BytesIO(payloads["trades.csv"]))
        sessions = pd.read_csv(BytesIO(payloads["training_sessions.csv"]), parse_dates=[
            "feature_session", "execution_session", "outcome_available_session",
        ])
        if len(predictions) != manifest["prediction_rows"] or len(rules) != manifest["model_rule_count"]:
            raise ResultBundleError("Result bundle counts do not match its manifest.")
        if len(sessions) != manifest["training_transactions"]:
            raise ResultBundleError("Result bundle fitting-session count is inconsistent.")
        if predictions.execution_session.nunique() != manifest["test_sessions"]:
            raise ResultBundleError("Result bundle test-session count is inconsistent.")
        if (predictions.execution_session.min().strftime("%Y-%m-%d") != manifest["test_start"]
                or predictions.execution_session.max().strftime("%Y-%m-%d") != manifest["test_end"]):
            raise ResultBundleError("Result bundle test dates do not match its manifest.")
        if metrics.duplicated(["strategy", "cost"]).any() or equity.duplicated(["strategy", "cost", "execution_session"]).any():
            raise ResultBundleError("Result bundle contains duplicate scenario rows.")
        return ResultBundle(manifest, rules, predictions, equity, metrics, trades, sessions)
    except ResultBundleError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ResultBundleError(f"Cannot load a complete saved result bundle: {exc}") from exc


def package_bundle(source: str | Path, destination: str | Path) -> None:
    """Explicitly curate verified derived artifacts; never download or overwrite."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ResultBundleError(f"Display bundle already exists: {destination}.")
    load_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as staging:
        staging = Path(staging)
        for name in (*sorted(BUNDLE_FILES), "manifest.json"):
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
