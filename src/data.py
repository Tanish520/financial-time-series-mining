"""Create a frozen daily-price snapshot and validate it entirely offline.

Only ``download_snapshot`` imports yfinance. All other functions work against
in-memory tables or the saved CSV and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

INITIAL_STOCKS = ("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS")
BENCHMARK = "^NSEI"
MIN_SESSIONS = 1200
PRICE_COLUMNS = ("open", "high", "low", "close")
CSV_COLUMNS = ("date", "ticker", *PRICE_COLUMNS, "volume", "dividends", "stock_splits")
# A verified exchange closure, not a license to drop arbitrary invalid bars.
NSE_NON_SESSIONS = {
    "2026-01-15": {
        "reason": "NSE capital-market trading holiday for Maharashtra municipal elections",
        "source_url": "https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf",
    },
}
DOWNLOAD_OPTIONS = {
    "interval": "1d",
    "auto_adjust": True,
    "back_adjust": False,
    "actions": True,
    "repair": False,
    "keepna": True,
    "ignore_tz": True,
    "threads": False,
    "progress": False,
    "group_by": "ticker",
    "multi_level_index": True,
    "timeout": 30,
}


class DataValidationError(ValueError):
    """The requested or stored observations violate the research data contract."""


@dataclass(frozen=True)
class SnapshotConfig:
    start: str
    as_of: str
    stocks: tuple[str, ...] = INITIAL_STOCKS
    benchmark: str = BENCHMARK

    def __post_init__(self) -> None:
        object.__setattr__(self, "stocks", tuple(self.stocks))
        try:
            start, end = date.fromisoformat(self.start), date.fromisoformat(self.as_of)
        except (TypeError, ValueError) as exc:
            raise DataValidationError("Use ISO dates: YYYY-MM-DD.") from exc
        if self.start != start.isoformat() or self.as_of != end.isoformat():
            raise DataValidationError("Use ISO dates: YYYY-MM-DD.")
        if start > end:
            raise DataValidationError("start must not be after as_of.")
        if end >= datetime.now(ZoneInfo("Asia/Kolkata")).date():
            raise DataValidationError(
                "Use completed historical sessions: as_of must be before today's India date."
            )
        if not self.stocks or len(set(self.stocks)) != len(self.stocks):
            raise DataValidationError("Provide at least one stock and distinct stock tickers.")
        if any(not isinstance(ticker, str) or not ticker.strip() for ticker in self.tickers):
            raise DataValidationError("Every ticker must be a nonempty string.")
        if self.benchmark in self.stocks:
            raise DataValidationError("Stock tickers and the benchmark must be distinct.")

    @property
    def tickers(self) -> tuple[str, ...]:
        return (*self.stocks, self.benchmark)


def normalize_download(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Convert ticker-first Yahoo columns to dated rows without filling NaNs."""
    if raw is None or raw.empty:
        raise DataValidationError("The provider download returned no daily prices.")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise DataValidationError("Expected ticker-first MultiIndex download columns.")
    available = set(raw.columns.get_level_values(0))
    missing = set(tickers) - available
    if missing:
        raise DataValidationError(f"The download is missing requested tickers: {sorted(missing)}")

    dates = pd.DatetimeIndex(raw.index)
    if dates.tz is not None:
        dates = dates.tz_convert("Asia/Kolkata").tz_localize(None)
    if not dates.equals(dates.normalize()):
        raise DataValidationError("Expected daily session dates, not intraday timestamps.")
    names = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits",
    }
    frames = []
    for ticker in tickers:
        frame = raw[ticker].rename(columns=names).copy()
        missing_fields = set((*PRICE_COLUMNS, "volume")) - set(frame.columns)
        if missing_fields:
            raise DataValidationError(f"Missing price fields for {ticker}: {sorted(missing_fields)}")
        for column in ("dividends", "stock_splits"):
            if column not in frame:
                frame[column] = np.nan
        frame.index = dates
        frame.index.name = "date"
        frame["ticker"] = ticker
        frames.append(frame.reset_index()[list(CSV_COLUMNS)])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "ticker"], kind="stable"
    ).reset_index(drop=True)


def _examples(prices: pd.DataFrame, mask) -> str:
    return prices.loc[mask, ["date", "ticker"]].head(3).to_string(index=False)


def validate_prices(
    prices: pd.DataFrame,
    config: SnapshotConfig,
    *,
    min_sessions: int = MIN_SESSIONS,
) -> dict:
    """Reject incomplete or inconsistent bars; return the observed coverage."""
    if min_sessions < 1:
        raise DataValidationError("min_sessions must be positive.")
    if prices.empty or set(CSV_COLUMNS) != set(prices.columns):
        raise DataValidationError(f"Expected nonempty price columns: {CSV_COLUMNS}")
    if prices["ticker"].isna().any() or set(prices["ticker"]) != set(config.tickers):
        raise DataValidationError("The price ticker set does not match the requested tickers.")
    dates = pd.DatetimeIndex(prices["date"])
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise DataValidationError("Expected timezone-naive, nonmissing daily session dates.")
    if (dates < pd.Timestamp(config.start)).any() or (dates > pd.Timestamp(config.as_of)).any():
        raise DataValidationError("Price dates fall outside the requested start/as_of bounds.")
    if dates.strftime("%Y-%m-%d").isin(NSE_NON_SESSIONS).any():
        raise DataValidationError("An official NSE non-trading day is present in accepted prices.")
    duplicate = prices.duplicated(["date", "ticker"], keep=False)
    if duplicate.any():
        raise DataValidationError(f"Duplicate stock/session observations:\n{_examples(prices, duplicate)}")

    try:
        values = prices[list(PRICE_COLUMNS)].to_numpy(dtype=float)
        volume = prices["volume"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Non-numeric price or volume fields.") from exc
    bad_price = (~np.isfinite(values) | (values <= 0)).any(axis=1)
    if bad_price.any():
        raise DataValidationError(f"Missing, infinite or nonpositive price:\n{_examples(prices, bad_price)}")
    bad_volume = ~np.isfinite(volume) | (volume < 0)
    if bad_volume.any():
        raise DataValidationError(f"Missing, infinite or negative volume:\n{_examples(prices, bad_volume)}")
    opening, high, low, closing = values.T
    tolerance = np.maximum(np.abs(high), 1.0) * 1e-8
    bad_range = (
        (low > high + tolerance)
        | (np.minimum(opening, closing) < low - tolerance)
        | (np.maximum(opening, closing) > high + tolerance)
    )
    if bad_range.any():
        raise DataValidationError(f"Inconsistent OHLC range:\n{_examples(prices, bad_range)}")

    reference = pd.DatetimeIndex(
        prices.loc[prices["ticker"] == config.benchmark, "date"]
    ).sort_values()
    for ticker in config.tickers:
        sessions = pd.DatetimeIndex(prices.loc[prices["ticker"] == ticker, "date"]).sort_values()
        if not sessions.equals(reference):
            missing = reference.difference(sessions).strftime("%Y-%m-%d").tolist()[:3]
            extra = sessions.difference(reference).strftime("%Y-%m-%d").tolist()[:3]
            raise DataValidationError(
                f"Session calendar mismatch for {ticker}: missing={missing}, extra={extra}."
            )
    if len(reference) < min_sessions:
        raise DataValidationError(f"Only {len(reference)} sessions; need at least {min_sessions}.")
    staleness = (date.fromisoformat(config.as_of) - reference[-1].date()).days
    if staleness > 7:
        raise DataValidationError(f"Latest provider session is {staleness} calendar days before as_of.")

    return {
        "rows": len(prices),
        "first_session": reference[0].strftime("%Y-%m-%d"),
        "last_session": reference[-1].strftime("%Y-%m-%d"),
        "sessions_per_ticker": {ticker: len(reference) for ticker in config.tickers},
    }


def write_snapshot(
    prices: pd.DataFrame,
    config: SnapshotConfig,
    destination: str | Path,
    *,
    min_sessions: int = MIN_SESSIONS,
    excluded_rows: pd.DataFrame | None = None,
) -> dict:
    """Validate, stage and publish a new snapshot; never overwrite an old one."""
    destination = Path(destination)
    if destination.exists():
        raise DataValidationError(f"Frozen snapshot already exists: {destination}. Use check instead.")
    coverage = validate_prices(prices, config, min_sessions=min_sessions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as staging:
        staging = Path(staging)
        canonical = prices[list(CSV_COLUMNS)].sort_values(
            ["date", "ticker"], kind="stable"
        ).reset_index(drop=True)
        csv_path = staging / "prices.csv"
        canonical.to_csv(
            csv_path, index=False, date_format="%Y-%m-%d",
            float_format="%.17g", lineterminator="\n",
        )
        exclusions = []
        audit_checksum = None
        if excluded_rows is not None and not excluded_rows.empty:
            for session, group in excluded_rows.groupby("date", sort=True):
                day = pd.Timestamp(session).strftime("%Y-%m-%d")
                if day not in NSE_NON_SESSIONS:
                    raise DataValidationError(f"No verified NSE closure authorizes excluding {day}.")
                exclusions.append({"date": day, "rows": len(group), **NSE_NON_SESSIONS[day]})
            audit_path = staging / "excluded_non_sessions.csv"
            excluded_rows[list(CSV_COLUMNS)].sort_values(["date", "ticker"]).to_csv(
                audit_path, index=False, date_format="%Y-%m-%d",
                float_format="%.17g", lineterminator="\n",
            )
            audit_checksum = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "provider": "Yahoo Finance via yfinance",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "session_timezone": "Asia/Kolkata",
            "request": asdict(config),
            "end_exclusive": (date.fromisoformat(config.as_of) + timedelta(days=1)).isoformat(),
            "download_options": DOWNLOAD_OPTIONS,
            "minimum_sessions": min_sessions,
            "excluded_non_sessions": exclusions,
            "excluded_rows_sha256": audit_checksum,
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "software": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("yfinance", "pandas", "numpy")},
            },
            **coverage,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    return manifest


def load_snapshot(destination: str | Path) -> tuple[pd.DataFrame, dict]:
    """Check bytes and coverage before returning local prices; no network code."""
    destination = Path(destination)
    try:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise DataValidationError("Unsupported snapshot manifest schema.")
        config = SnapshotConfig(**manifest["request"])
        min_sessions = manifest["minimum_sessions"]
        csv_bytes = (destination / "prices.csv").read_bytes()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Cannot read a complete snapshot at {destination}: {exc}") from exc
    if hashlib.sha256(csv_bytes).hexdigest() != manifest.get("sha256"):
        raise DataValidationError("Snapshot checksum mismatch: the saved price bytes changed.")
    for name, expected in manifest.get("auxiliary_files", {}).items():
        if not isinstance(name, str) or Path(name).name != name:
            raise DataValidationError("Invalid snapshot audit filename.")
        try:
            actual = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        except OSError as exc:
            raise DataValidationError("Missing snapshot audit checksum file.") from exc
        if actual != expected:
            raise DataValidationError("Snapshot audit checksum mismatch: primary correction evidence changed.")
    if manifest.get("excluded_non_sessions"):
        try:
            audit_bytes = (destination / "excluded_non_sessions.csv").read_bytes()
        except OSError as exc:
            raise DataValidationError("Missing checksum-protected non-session audit file.") from exc
        if hashlib.sha256(audit_bytes).hexdigest() != manifest.get("excluded_rows_sha256"):
            raise DataValidationError("Non-session audit checksum mismatch: saved bytes changed.")
    prices = pd.read_csv(destination / "prices.csv", parse_dates=["date"])
    coverage = validate_prices(prices, config, min_sessions=min_sessions)
    if any(manifest.get(key) != value for key, value in coverage.items()):
        raise DataValidationError("Saved coverage metadata does not match the price file.")
    return prices, manifest


def download_snapshot(
    config: SnapshotConfig,
    destination: str | Path,
    *,
    min_sessions: int = MIN_SESSIONS,
) -> dict:
    """The only network boundary; downloading is an explicit user command."""
    if Path(destination).exists():
        raise DataValidationError(f"Frozen snapshot already exists: {destination}. Use check instead.")
    import yfinance as yf

    raw = yf.download(
        tickers=list(config.tickers), start=config.start,
        end=(date.fromisoformat(config.as_of) + timedelta(days=1)).isoformat(),
        **DOWNLOAD_OPTIONS,
    )
    prices = normalize_download(raw, config.tickers)
    closed = pd.DatetimeIndex(prices["date"]).strftime("%Y-%m-%d").isin(NSE_NON_SESSIONS)
    return write_snapshot(
        prices.loc[~closed].reset_index(drop=True), config, destination,
        min_sessions=min_sessions, excluded_rows=prices.loc[closed].copy(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download a new immutable historical snapshot")
    download.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD")
    download.add_argument("--as-of", required=True, help="Inclusive completed historical date")
    download.add_argument("--stocks", nargs="+", default=INITIAL_STOCKS)
    download.add_argument("--output", type=Path, default=Path("data/raw/day1_three_stocks"))
    check = commands.add_parser("check", help="Verify the saved snapshot without network access")
    check.add_argument("--snapshot", type=Path, default=Path("data/raw/day1_three_stocks"))
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            config = SnapshotConfig(args.start, args.as_of, tuple(args.stocks))
            manifest = download_snapshot(config, args.output)
            destination = args.output
        else:
            _, manifest = load_snapshot(args.snapshot)
            destination = args.snapshot
        summary = {
            "status": "downloaded" if args.command == "download" else "verified_offline",
            "snapshot": str(destination),
            **{key: manifest[key] for key in (
                "first_session", "last_session", "rows", "sessions_per_ticker", "sha256",
                "excluded_non_sessions",
            )},
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (DataValidationError, OSError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
