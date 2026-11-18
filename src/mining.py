"""Mine next-session association rules using only cutoff-eligible daily data.

All reported rule metrics are training statistics. Mining a strong association
does not establish held-out predictive accuracy or a profitable trading rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.data import DataValidationError, SnapshotConfig, load_snapshot
from src.labels import build_label_tables, eligible_training_samples

MAX_ITEMSET_LENGTH = 3
RULE_COLUMNS = [
    "antecedents", "consequent", "target_ticker", "target_label", "support",
    "confidence", "lift", "antecedent_support", "consequent_support",
    "joint_count", "antecedent_count", "consequent_count", "transaction_count",
]
SESSION_COLUMNS = ["feature_session", "execution_session", "outcome_available_session"]


@dataclass(frozen=True)
class MiningResult:
    rules: pd.DataFrame
    transactions: pd.DataFrame
    sessions: pd.DataFrame
    fit_cutoff: pd.Timestamp
    min_support: float
    min_confidence: float


def build_transactions(
    samples: pd.DataFrame,
    stocks: tuple[str, ...],
    cutoff: str | pd.Timestamp,
    window: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode complete stock cohorts after filtering by known outcome dates."""
    symbols = [ticker.removesuffix(".NS") for ticker in stocks]
    if not stocks or len(set(symbols)) != len(symbols):
        raise DataValidationError("Provide a nonempty stock set with distinct item symbols.")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1
    ):
        raise DataValidationError("window must be a positive session count or None for expanding.")
    required = {"ticker", "feature_item", "target_item"}
    if not required.issubset(samples.columns):
        raise DataValidationError("Mining requires stock, feature item and target item columns.")
    known = eligible_training_samples(samples, cutoff)
    if known.empty:
        if window is not None:
            raise DataValidationError(f"Insufficient eligible history for a {window}-session window.")
        return (
            pd.DataFrame(index=pd.DatetimeIndex([], name="feature_session"), dtype=bool),
            pd.DataFrame({name: pd.Series(dtype="datetime64[ns]") for name in SESSION_COLUMNS}),
        )
    if known.duplicated(["feature_session", "ticker"]).any():
        raise DataValidationError("Duplicate stock/session rows cannot form daily transactions.")
    if known.groupby("feature_session")["execution_session"].nunique().gt(1).any():
        raise DataValidationError("Stocks disagree on the execution session within a transaction.")
    sessions = known[SESSION_COLUMNS].drop_duplicates().sort_values(
        "execution_session", kind="stable"
    ).reset_index(drop=True)
    if sessions["execution_session"].duplicated().any():
        raise DataValidationError("Each execution session must correspond to one feature session.")
    if window is not None:
        if len(sessions) < window:
            raise DataValidationError(f"Insufficient eligible history: {len(sessions)} sessions for window={window}.")
        sessions = sessions.tail(window).reset_index(drop=True)
    known = known.loc[known["feature_session"].isin(sessions["feature_session"])].sort_values(
        ["feature_session", "ticker"], kind="stable"
    )
    complete = known.groupby("feature_session")["ticker"].agg(lambda values: set(values) == set(stocks))
    if not complete.all():
        raise DataValidationError("Each daily transaction requires the complete requested stock set.")
    if not known["feature_label"].isin(["U", "D", "N"]).all():
        raise DataValidationError("Invalid feature item label; expected U/D/N.")
    if not known["target_label"].isin(["U", "D", "N"]).all():
        raise DataValidationError("Invalid target item label; expected U/D/N.")
    item_symbol = known["ticker"].str.removesuffix(".NS")
    expected_features = item_symbol + "_" + known["feature_label"]
    expected_targets = "NEXT_" + item_symbol + "_" + known["target_label"]
    if not known["feature_item"].eq(expected_features).fillna(False).all():
        raise DataValidationError("Invalid feature item: only that stock's observed label is allowed.")
    if not known["target_item"].eq(expected_targets).fillna(False).all():
        raise DataValidationError("Invalid target item: expected exactly that stock's NEXT_ label.")

    by_day = {
        session: sorted(group["feature_item"].tolist() + group["target_item"].tolist())
        for session, group in known.groupby("feature_session", sort=True)
    }
    transactions = [by_day[session] for session in sessions["feature_session"]]
    encoder = TransactionEncoder()
    values = encoder.fit_transform(transactions)
    encoded = pd.DataFrame(
        values, columns=encoder.columns_,
        index=pd.DatetimeIndex(sessions["feature_session"], name="feature_session"), dtype=bool,
    )
    return encoded, sessions


def _validate_thresholds(min_support: float, min_confidence: float) -> tuple[float, float]:
    for name, value, lower in (
        ("min_support", min_support, 0), ("min_confidence", min_confidence, -1),
    ):
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            raise DataValidationError(f"{name} must be a finite probability.")
        if not 0 <= value <= 1 or (lower == 0 and value == 0):
            raise DataValidationError(f"{name} is outside its allowed probability range.")
    return float(min_support), float(min_confidence)


def mine_rules(
    samples: pd.DataFrame,
    stocks: tuple[str, ...],
    cutoff: str | pd.Timestamp,
    *,
    window: int | None = None,
    min_support: float = 0.02,
    min_confidence: float = 0.60,
) -> MiningResult:
    """Fit cutoff-eligible rules with one/two feature items and one NEXT_ item."""
    min_support, min_confidence = _validate_thresholds(min_support, min_confidence)
    encoded, sessions = build_transactions(samples, stocks, cutoff, window=window)
    rules = pd.DataFrame(columns=RULE_COLUMNS)
    count = len(encoded)
    if count:
        frequent = apriori(
            encoded, min_support=min_support, use_colnames=True, max_len=MAX_ITEMSET_LENGTH
        )
        if not frequent.empty and frequent["itemsets"].map(len).ge(2).any():
            raw_rules = association_rules(
                # Do not let intermediate library rounding drop exact-boundary
                # rules before recovering their integer transaction counts.
                frequent, num_itemsets=count, metric="confidence", min_threshold=0.0,
                return_metrics=["antecedent support", "consequent support", "support", "confidence", "lift"],
            )
            feature_items = {f"{ticker.removesuffix('.NS')}_{label}" for ticker in stocks for label in "UDN"}
            target_items = {
                f"NEXT_{ticker.removesuffix('.NS')}_{label}": (ticker, label)
                for ticker in stocks for label in "UDN"
            }
            kept = []
            for row in raw_rules.to_dict("records"):
                antecedents, consequents = row["antecedents"], row["consequents"]
                if not 1 <= len(antecedents) <= 2 or not antecedents.issubset(feature_items):
                    continue
                if len(consequents) != 1:
                    continue
                consequent = next(iter(consequents))
                if consequent not in target_items:
                    continue
                joint_count = int(round(row["support"] * count))
                antecedent_count = int(round(row["antecedent support"] * count))
                consequent_count = int(round(row["consequent support"] * count))
                support = joint_count / count
                confidence = joint_count / antecedent_count
                if support < min_support or confidence < min_confidence:
                    continue
                # Exact integer comparison avoids retaining numerical lift=1+epsilon.
                if joint_count * count <= antecedent_count * consequent_count:
                    continue
                target_ticker, target_label = target_items[consequent]
                kept.append({
                    "antecedents": tuple(sorted(antecedents)), "consequent": consequent,
                    "target_ticker": target_ticker, "target_label": target_label,
                    "support": support, "confidence": confidence,
                    "lift": joint_count * count / (antecedent_count * consequent_count),
                    "antecedent_support": antecedent_count / count,
                    "consequent_support": consequent_count / count,
                    "joint_count": joint_count, "antecedent_count": antecedent_count,
                    "consequent_count": consequent_count, "transaction_count": count,
                })
            if kept:
                rules = pd.DataFrame(kept, columns=RULE_COLUMNS).sort_values(
                    ["confidence", "lift", "support", "antecedents", "consequent"],
                    ascending=[False, False, False, True, True], kind="stable",
                ).reset_index(drop=True)
    return MiningResult(rules, encoded, sessions, pd.Timestamp(cutoff), min_support, min_confidence)


def select_rules(
    result: MiningResult, *, min_support: float | None = None, min_confidence: float | None = None
) -> pd.DataFrame:
    """Reuse an already-mined result only at equal or stricter thresholds."""
    support = result.min_support if min_support is None else min_support
    confidence = result.min_confidence if min_confidence is None else min_confidence
    support, confidence = _validate_thresholds(support, confidence)
    if support < result.min_support or confidence < result.min_confidence:
        raise DataValidationError("Cannot relax thresholds on filtered rules; re-mine at the lower thresholds.")
    return result.rules.loc[
        (result.rules["support"] >= support) & (result.rules["confidence"] >= confidence)
    ].reset_index(drop=True).copy()


def _summary(result: MiningResult, theta: float, window: int | None, source_sha256: str) -> dict:
    sessions = result.sessions
    return {
        "status": "rules_mined_offline", "metric_scope": "training_only",
        "source_sha256": source_sha256, "theta": theta,
        "fit_cutoff": result.fit_cutoff.strftime("%Y-%m-%d"),
        "window": "expanding" if window is None else window,
        "min_support": result.min_support, "min_confidence": result.min_confidence,
        "max_itemset_length": MAX_ITEMSET_LENGTH,
        "training_transactions": len(result.transactions), "rule_count": len(result.rules),
        "first_training_feature_session": None if sessions.empty else sessions.feature_session.min().strftime("%Y-%m-%d"),
        "last_training_feature_session": None if sessions.empty else sessions.feature_session.max().strftime("%Y-%m-%d"),
        "first_training_execution_session": None if sessions.empty else sessions.execution_session.min().strftime("%Y-%m-%d"),
        "last_training_execution_session": None if sessions.empty else sessions.execution_session.max().strftime("%Y-%m-%d"),
        "example_rule": None if result.rules.empty else result.rules.to_dict("records")[0],
    }


def _write_result(result: MiningResult, output: Path, summary: dict) -> None:
    if output.exists():
        raise DataValidationError(f"Mining output already exists: {output}. Preview without --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        staging = Path(staging)
        (staging / "rules.json").write_text(
            json.dumps(result.rules.to_dict("records"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        readable = result.rules.copy()
        readable["antecedents"] = readable["antecedents"].map(lambda items: " & ".join(items))
        readable.to_csv(staging / "rules.csv", index=False, float_format="%.17g", lineterminator="\n")
        result.transactions.to_csv(staging / "transactions.csv", date_format="%Y-%m-%d", lineterminator="\n")
        result.sessions.to_csv(staging / "sessions.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
        filenames = ("rules.json", "rules.csv", "transactions.csv", "sessions.csv")
        manifest = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
            "code_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("data.py", "labels.py", "mining.py")
            },
            "software": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("pandas", "numpy", "mlxtend")},
            },
            "files": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in filenames},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/day1_three_stocks"))
    parser.add_argument("--cutoff", default="2022-12-30", help="Session-close fitting cutoff: YYYY-MM-DD")
    parser.add_argument("--window", default="500", help="Positive session count or expanding")
    parser.add_argument("--theta", type=float, default=0.005, help="Decimal return: 0.005 means 0.5%%")
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--output", type=Path, help="Optional new directory for training artifacts")
    args = parser.parse_args(argv)
    try:
        try:
            window = None if args.window == "expanding" else int(args.window)
        except ValueError as exc:
            raise DataValidationError("window must be a positive integer or expanding.") from exc
        prices, source_manifest = load_snapshot(args.snapshot)
        config = SnapshotConfig(**source_manifest["request"])
        tables = build_label_tables(prices, config, theta=args.theta)
        result = mine_rules(
            tables.samples, config.stocks, args.cutoff, window=window,
            min_support=args.min_support, min_confidence=args.min_confidence,
        )
        summary = _summary(result, args.theta, window, source_manifest["sha256"])
        if args.output is not None:
            _write_result(result, args.output, summary)
            summary["output"] = str(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Mining error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
