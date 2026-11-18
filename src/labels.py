"""Causal stock features and next-session intraday targets, built offline.

Session dates are availability labels: information is known after that session's
close. They are not claims that closing information is available at midnight.
Features and targets deliberately live in separate tables.
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

from src.data import DataValidationError, SnapshotConfig, load_snapshot, validate_prices

# Only numerical boundary equality, not a tunable financial threshold.
BOUNDARY_ATOL = float(8 * np.finfo(np.float64).eps)


@dataclass(frozen=True)
class LabelTables:
    features: pd.DataFrame
    targets: pd.DataFrame
    samples: pd.DataFrame


def label_returns(returns: pd.Series, theta: float) -> pd.Series:
    """Label decimal returns U/D/N; preserve unavailable returns as pd.NA."""
    try:
        theta = float(theta)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("theta must be a finite decimal return between 0 and 1.") from exc
    if not np.isfinite(theta) or not 0 < theta < 1:
        raise DataValidationError("theta must be a finite decimal return between 0 and 1.")
    try:
        values = returns.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Returns must be numeric, finite values or missing.") from exc
    if np.isinf(values).any():
        raise DataValidationError("Returns must be finite or missing, never infinite.")
    result = pd.Series(pd.NA, index=returns.index, name=returns.name, dtype="string")
    result.loc[~np.isnan(values)] = "N"
    result.loc[values > theta + BOUNDARY_ATOL] = "U"
    result.loc[values < -theta - BOUNDARY_ATOL] = "D"
    return result


def build_label_tables(
    prices: pd.DataFrame, config: SnapshotConfig, theta: float = 0.005
) -> LabelTables:
    """Build stock-only tables without imputing or inventing a future session."""
    # The source loader enforces research coverage; two sessions also permit a
    # valid prefix/latest-signal calculation and small deterministic test inputs.
    validate_prices(prices, config, min_sessions=2)
    work = prices.loc[prices["ticker"].isin(config.stocks)].sort_values(
        ["ticker", "date"], kind="stable"
    ).reset_index(drop=True).copy()
    grouped = work.groupby("ticker", sort=False)
    feature_return = work["close"] / grouped["close"].shift(1) - 1
    target_return = grouped["close"].shift(-1) / grouped["open"].shift(-1) - 1
    next_session = grouped["date"].shift(-1)
    feature_label = label_returns(feature_return, theta)
    target_label = label_returns(target_return, theta)
    item_symbol = work["ticker"].str.removesuffix(".NS")

    features = pd.DataFrame({
        "feature_session": work["date"],
        "feature_available_session": work["date"],
        "ticker": work["ticker"],
        "feature_return": feature_return,
        "feature_label": feature_label,
        "feature_item": item_symbol + "_" + feature_label,
    })
    targets = pd.DataFrame({
        "feature_session": work["date"],
        "ticker": work["ticker"],
        "execution_session": next_session,
        "outcome_available_session": next_session,
        "target_return": target_return,
        "target_label": target_label,
        "target_item": "NEXT_" + item_symbol + "_" + target_label,
    })
    order = ["feature_session", "ticker"]
    features = features.sort_values(order, kind="stable").reset_index(drop=True)
    targets = targets.sort_values(order, kind="stable").reset_index(drop=True)
    samples = features.merge(targets, on=order, validate="one_to_one", sort=False)
    samples = samples.dropna(subset=["feature_label", "target_label"]).reset_index(drop=True)
    return LabelTables(features=features, targets=targets, samples=samples)


def eligible_training_samples(samples: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Return complete samples known AFTER the specified session has closed."""
    try:
        cutoff = pd.Timestamp(cutoff)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("cutoff must be a session-close date: YYYY-MM-DD.") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise DataValidationError("Use a date-only session-close cutoff, not an intraday timestamp.")
    required = {
        "feature_session", "feature_available_session", "execution_session",
        "outcome_available_session", "feature_label", "target_label",
    }
    if not required.issubset(samples.columns) or samples[list(required)].isna().any().any():
        raise DataValidationError("Training samples require complete labels and availability dates.")
    invalid = (
        (samples["feature_available_session"] != samples["feature_session"])
        | (samples["feature_session"] >= samples["execution_session"])
        | (samples["outcome_available_session"] != samples["execution_session"])
    )
    if invalid.any():
        raise DataValidationError("Training sample availability does not match its execution session.")
    return samples.loc[samples["outcome_available_session"] <= cutoff].reset_index(drop=True).copy()


def _summary(tables: LabelTables, theta: float, source_sha256: str) -> dict:
    latest_date = tables.features["feature_session"].max()
    latest = tables.features.loc[tables.features["feature_session"] == latest_date]
    return {
        "status": "labels_computed_offline",
        "theta": float(theta),
        "source_sha256": source_sha256,
        "feature_rows": len(tables.features),
        "known_feature_rows": int(tables.features["feature_label"].notna().sum()),
        "target_rows": len(tables.targets),
        "known_target_rows": int(tables.targets["target_label"].notna().sum()),
        "supervised_sample_rows": len(tables.samples),
        "supervised_feature_sessions": int(tables.samples["feature_session"].nunique()),
        "latest_feature_session": latest_date.strftime("%Y-%m-%d"),
        "latest_features": latest[["ticker", "feature_item", "feature_return"]].to_dict("records"),
    }


def _write_tables(tables: LabelTables, output: Path, summary: dict) -> None:
    if output.exists():
        raise DataValidationError(f"Derived output already exists: {output}. Preview without --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        staging = Path(staging)
        checksums = {}
        for name in ("features", "targets", "samples"):
            filename = f"{name}.csv"
            path = staging / filename
            getattr(tables, name).to_csv(
                path, index=False, date_format="%Y-%m-%d",
                float_format="%.17g", lineterminator="\n",
            )
            checksums[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "availability_convention": "after the named Asia/Kolkata exchange session has closed",
            "boundary_atol": BOUNDARY_ATOL,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "software": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("pandas", "numpy")},
            },
            **summary,
            "files": checksums,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/day1_three_stocks"))
    parser.add_argument("--theta", type=float, default=0.005, help="Decimal return: 0.005 means 0.5%%")
    parser.add_argument("--output", type=Path, help="Optional new directory for derived CSVs")
    args = parser.parse_args(argv)
    try:
        prices, source_manifest = load_snapshot(args.snapshot)
        config = SnapshotConfig(**source_manifest["request"])
        tables = build_label_tables(prices, config, theta=args.theta)
        summary = _summary(tables, args.theta, source_manifest["sha256"])
        if args.output is not None:
            _write_tables(tables, args.output, summary)
            summary["output"] = str(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Label error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
