"""Offline, bounded correction of verified 18 March 2025 Yahoo placeholders.

Primary NSE OHLCV replaces only flat zero-volume stock bars on that date.
An adjustment scale inferred from the preceding adjusted close and NSE's
previous close is independently checked against the following session. This
retains the existing historical adjusted-price convention; it is not a
point-in-time corporate-action archive. Original inputs remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data import DataValidationError, SnapshotConfig, load_snapshot, validate_prices, write_snapshot

CORRECTION_DATE = pd.Timestamp("2025-03-18")
PRIMARY_FILES = ("sec_bhavdata_full_18032025.csv", "sec_bhavdata_full_19032025.csv")
SCALE_TOLERANCE = 1e-5


def _primary(frame: pd.DataFrame, expected_date: str, columns: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if not {"SYMBOL", "SERIES", "DATE1", *columns}.issubset(frame.columns):
        raise DataValidationError("Primary NSE file lacks required OHLCV/date fields.")
    try:
        dates = pd.to_datetime(frame.DATE1.str.strip(), format="%d-%b-%Y")
    except (ValueError, TypeError, AttributeError) as exc:
        raise DataValidationError("Primary NSE file has invalid session dates.") from exc
    if not dates.eq(pd.Timestamp(expected_date)).all():
        raise DataValidationError("Primary NSE file's actual date differs from its requested session.")
    frame = frame.loc[frame.SERIES.str.strip().eq("EQ")].copy()
    frame["SYMBOL"] = frame.SYMBOL.str.strip()
    if frame.SYMBOL.duplicated().any():
        raise DataValidationError("Primary NSE EQ rows must be unique per symbol.")
    return frame.set_index("SYMBOL")


def correct_placeholder_bars(prices, config, primary, following) -> tuple[pd.DataFrame, list[dict]]:
    validate_prices(prices, config, min_sessions=2)
    current = _primary(primary, "2025-03-18", {"PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"})
    next_day = _primary(following, "2025-03-19", {"CLOSE_PRICE"})
    result = prices.copy(deep=True)
    bad = result.date.eq(CORRECTION_DATE) & result.ticker.isin(config.stocks) & result.volume.eq(0)
    if not bad.any():
        raise DataValidationError("No documented 18 March 2025 stock placeholders need correction.")
    audit = []
    for index, row in result.loc[bad].iterrows():
        if not row[["open", "high", "low", "close"]].eq(row.close).all():
            raise DataValidationError("Only the documented flat zero-volume placeholder pattern may be corrected.")
        if row.dividends != 0 or row.stock_splits != 0:
            raise DataValidationError("A corporate action on the correction date requires a separate adjustment audit.")
        symbol = row.ticker.removesuffix(".NS")
        if symbol not in current.index or symbol not in next_day.index:
            raise DataValidationError("A requested placeholder has no unique primary NSE EQ observation.")
        source = current.loc[symbol]
        if not np.isfinite(source[["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]].to_numpy(dtype=float)).all():
            raise DataValidationError("Primary OHLCV must be finite.")
        volume = float(source.TTL_TRD_QNTY)
        if volume <= 0 or not volume.is_integer() or source.PREV_CLOSE <= 0:
            raise DataValidationError("Primary replacement requires positive traded volume and previous close.")
        before = result.loc[result.ticker.eq(row.ticker) & result.date.eq("2025-03-17"), "close"]
        after = result.loc[result.ticker.eq(row.ticker) & result.date.eq("2025-03-19"), "close"]
        if len(before) != 1 or len(after) != 1 or next_day.loc[symbol, "CLOSE_PRICE"] <= 0:
            raise DataValidationError("Both adjacent adjusted-price observations must exist.")
        scale = float(before.iloc[0] / source.PREV_CLOSE)
        following_scale = float(after.iloc[0] / next_day.loc[symbol, "CLOSE_PRICE"])
        discrepancy = abs(scale / following_scale - 1)
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(discrepancy) or discrepancy > SCALE_TOLERANCE:
            raise DataValidationError("Adjacent primary observations do not verify a constant adjustment scale.")
        fields = {"open": "OPEN_PRICE", "high": "HIGH_PRICE", "low": "LOW_PRICE", "close": "CLOSE_PRICE"}
        replacement = {name: float(source[original] * scale) for name, original in fields.items()}
        replacement["volume"] = int(volume)
        original = {name: float(row[name]) for name in (*fields, "volume")}
        for name, value in replacement.items():
            result.at[index, name] = value
        audit.append({"date": "2025-03-18", "ticker": row.ticker,
                      "original": original, "corrected": replacement,
                      "adjustment_scale": scale, "following_session_scale": following_scale,
                      "scale_relative_discrepancy": discrepancy})
    validate_prices(result, config, min_sessions=2)
    return result, audit


def create_corrected_snapshot(source: Path, primary_directory: Path, output: Path) -> dict:
    source, primary_directory, output = Path(source), Path(primary_directory), Path(output)
    if output.exists():
        raise DataValidationError(f"Corrected snapshot already exists: {output}.")
    prices, original = load_snapshot(source)
    inputs = [pd.read_csv(primary_directory / name, skipinitialspace=True) for name in PRIMARY_FILES]
    config = SnapshotConfig(**original["request"])
    corrected, audit = correct_placeholder_bars(prices, config, *inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as staging:
        prepared = Path(staging) / "snapshot"
        excluded = (pd.read_csv(source / "excluded_non_sessions.csv", parse_dates=["date"])
                    if original.get("excluded_non_sessions") else None)
        manifest = write_snapshot(corrected, config, prepared, min_sessions=original["minimum_sessions"], excluded_rows=excluded)
        for name in PRIMARY_FILES:
            shutil.copyfile(primary_directory / name, prepared / name)
        shutil.copyfile(source / "manifest.json", prepared / "source_manifest.json")
        (prepared / "corrections.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
        auxiliary = (*PRIMARY_FILES, "source_manifest.json", "corrections.json")
        manifest["auxiliary_files"] = {name: hashlib.sha256((prepared / name).read_bytes()).hexdigest() for name in auxiliary}
        manifest["correction"] = {
            "source_sha256": original["sha256"], "date": "2025-03-18", "row_count": len(audit),
            "method": "Replace only verified flat zero-volume placeholders with NSE EQ OHLCV; apply previous-session adjusted/NSE-close scale; verify following-session scale within 1e-5.",
            "sources": [f"https://archives.nseindia.com/products/content/{name}" for name in PRIMARY_FILES],
            "volume_convention": "Corrected volume is the primary exchange's traded share count; volume is only an execution-validity check, never a model feature.",
            "repair_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        prepared.rename(output)
    load_snapshot(output)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/raw/full_ten_stocks"))
    parser.add_argument("--primary-directory", type=Path, default=Path("data/raw/primary_audit_2025_03_18"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/full_ten_stocks_corrected"))
    args = parser.parse_args(argv)
    try:
        manifest = create_corrected_snapshot(args.snapshot, args.primary_directory, args.output)
        print(json.dumps({"status": "primary_source_correction_verified", "sha256": manifest["sha256"],
                          "output": str(args.output), "correction": manifest["correction"]}, indent=2, sort_keys=True))
        return 0
    except (OSError, DataValidationError) as exc:
        print(f"Correction error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
