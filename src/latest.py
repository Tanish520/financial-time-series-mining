"""Apply the final historical model to completed bars, with an explicit live boundary.

Importing this module and the default cache-only path perform no network access.
Nothing here mines rules, writes files, or changes the historical result bundle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data import CSV_COLUMNS, DOWNLOAD_OPTIONS, DataValidationError, normalize_download, validate_prices
from src.labels import label_returns
from src.mining import MiningResult, RULE_COLUMNS, SESSION_COLUMNS
from src.signals import SIGNAL_COLUMNS, predict_signals

INDIA = ZoneInfo("Asia/Kolkata")
MAX_LIVE_AGE_DAYS = 7


@dataclass(frozen=True)
class LatestResult:
    signals: pd.DataFrame
    observation_session: pd.Timestamp
    fit_cutoff: pd.Timestamp
    status: str
    message: str
    fetched_at_utc: str
    source_sha256: str
    intended_execution: str = "next trading session"
    educational_disclaimer: str = "Educational project; not investment advice."


@dataclass(frozen=True)
class _FrozenModel:
    model: MiningResult
    stocks: tuple[str, ...]
    benchmark: str
    theta: float | None
    fallback: str | None
    source_sha256: str


@dataclass(frozen=True)
class _RecentWindow:
    """The shared validator's bounds, independent of the machine's wall clock."""

    start: str
    as_of: str
    tickers: tuple[str, ...]
    benchmark: str


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        raise DataValidationError("now must be a datetime; a naive datetime is interpreted as UTC.")
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _load_model(spec: dict) -> _FrozenModel:
    required = {
        "stocks", "benchmark", "fit_cutoff", "theta", "parameters", "fallback",
        "rules", "source_sha256",
    }
    if not isinstance(spec, dict) or not required.issubset(spec):
        raise DataValidationError("Latest signals require the complete frozen final-fold model specification.")
    stocks, benchmark = spec["stocks"], spec["benchmark"]
    if not isinstance(stocks, (list, tuple)) or not stocks or any(
        not isinstance(ticker, str) or not ticker.strip() for ticker in stocks
    ):
        raise DataValidationError("The frozen model needs nonempty stock tickers.")
    stocks = tuple(stocks)
    if len({ticker.removesuffix(".NS") for ticker in stocks}) != len(stocks):
        raise DataValidationError("The frozen model needs distinct stock item symbols.")
    if not isinstance(benchmark, str) or not benchmark.strip() or benchmark in stocks:
        raise DataValidationError("The benchmark must be a distinct, nonempty ticker.")
    try:
        cutoff = pd.Timestamp(spec["fit_cutoff"])
    except (TypeError, ValueError) as exc:
        raise DataValidationError("The frozen model fit cutoff must be YYYY-MM-DD.") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is not None or spec["fit_cutoff"] != cutoff.strftime("%Y-%m-%d"):
        raise DataValidationError("The frozen model fit cutoff must be YYYY-MM-DD.")
    checksum = spec["source_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in checksum
    ):
        raise DataValidationError("The frozen model requires its historical source_sha256 provenance.")
    fallback, rows = spec["fallback"], spec["rules"]
    if fallback not in (None, "no_eligible_configuration"):
        raise DataValidationError("Unknown final-fold fallback reason.")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != set(RULE_COLUMNS) for row in rows
    ):
        raise DataValidationError("Frozen rules must contain the existing mining rule columns.")
    parameters, theta = spec["parameters"], spec["theta"]
    if fallback is not None:
        if rows or theta is not None or parameters is not None:
            raise DataValidationError("A no-eligible-configuration fallback cannot contain rules or selected parameters.")
        parameters = {}
    else:
        if not isinstance(parameters, dict):
            raise DataValidationError("A selected frozen model requires its recorded parameters.")
        label_returns(pd.Series([0.0]), theta)
        theta = float(theta)
    model = MiningResult(
        rules=pd.DataFrame(rows, columns=RULE_COLUMNS), transactions=pd.DataFrame(),
        sessions=pd.DataFrame(columns=SESSION_COLUMNS), fit_cutoff=cutoff,
        min_support=parameters.get("min_support", 0.02),
        min_confidence=parameters.get("min_confidence", 0.60),
    )
    return _FrozenModel(model, stocks, benchmark, theta, fallback, checksum)


def fetch_recent_prices(stocks: tuple[str, ...], benchmark: str, now: datetime) -> pd.DataFrame:
    """Network-only helper; Yahoo's end date excludes the current India session."""
    import yfinance as yf

    today = _utc_now(now).astimezone(INDIA).date()
    tickers = (*stocks, benchmark)
    raw = yf.download(
        tickers=list(tickers), start=(today - timedelta(days=45)).isoformat(),
        end=today.isoformat(), **DOWNLOAD_OPTIONS,
    )
    return normalize_download(raw, tickers)


def _completed_prices(prices: pd.DataFrame, frozen: _FrozenModel, now: datetime) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty or set(prices.columns) != set(CSV_COLUMNS):
        raise DataValidationError("Expected nonempty daily prices with the original OHLCV/actions schema.")
    try:
        dates = pd.DatetimeIndex(prices["date"])
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Recent bars require valid daily session dates.") from exc
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise DataValidationError("Recent bars require nonmissing, timezone-naive daily session dates.")
    completed = dates < pd.Timestamp(now.astimezone(INDIA).date())
    accepted = prices.loc[completed].copy()
    accepted["date"] = dates[completed]
    if accepted.empty:
        raise DataValidationError("No completed India session is available.")
    window = _RecentWindow(
        accepted.date.min().strftime("%Y-%m-%d"), accepted.date.max().strftime("%Y-%m-%d"),
        (*frozen.stocks, frozen.benchmark), frozen.benchmark,
    )
    # Reuse all original bar, ticker and exact-calendar checks. These bounds use
    # the supplied clock; SnapshotConfig itself consults the machine's date.
    validate_prices(accepted, window, min_sessions=2)
    observation = accepted.loc[accepted.ticker == frozen.benchmark, "date"].max()
    if frozen.model.fit_cutoff > observation:
        raise DataValidationError("The frozen model fit cutoff is after the latest observed feature session.")
    return accepted.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _cash_signals(frozen: _FrozenModel, observation: pd.Timestamp, reason: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "feature_session": observation, "ticker": ticker, "prediction_label": pd.NA,
            "action": "CASH", "reason": reason, "matched_antecedents": (),
            "consequent": pd.NA, "confidence": np.nan, "lift": np.nan, "support": np.nan,
            "matched_rule_count": 0, "rule_fit_cutoff": frozen.model.fit_cutoff,
        }
        for ticker in sorted(frozen.stocks)
    ], columns=SIGNAL_COLUMNS)


def _result(
    accepted: pd.DataFrame, frozen: _FrozenModel, status: str, fetched_at: str, detail: str,
) -> LatestResult:
    sessions = accepted.loc[accepted.ticker == frozen.benchmark, "date"].sort_values()
    observation, previous = sessions.iloc[-1], sessions.iloc[-2]
    reason = frozen.fallback or ("no_rules_final_fold" if frozen.model.rules.empty else None)
    if frozen.fallback:
        signals = _cash_signals(frozen, observation, frozen.fallback)
    else:
        stocks = accepted.loc[accepted.ticker.isin(frozen.stocks)]
        current = stocks.loc[stocks.date == observation].set_index("ticker").reindex(frozen.stocks)
        prior = stocks.loc[stocks.date == previous].set_index("ticker").reindex(frozen.stocks)
        returns = (current.close / prior.close - 1).reset_index(drop=True)
        labels = label_returns(returns, frozen.theta)
        features = pd.DataFrame({
            "feature_session": observation, "feature_available_session": observation,
            "ticker": list(frozen.stocks), "feature_return": returns,
            "feature_label": labels,
            "feature_item": pd.Series(frozen.stocks).str.removesuffix(".NS") + "_" + labels,
        })
        signals = predict_signals(frozen.model, features, frozen.stocks)
        if reason:
            signals["reason"] = reason
    message = (
        f"{detail} Observation session {observation:%Y-%m-%d}; "
        f"frozen final-fold rule fit cutoff {frozen.model.fit_cutoff:%Y-%m-%d}. "
        "Intended execution: next trading session."
    )
    if reason:
        message += f" CASH/unknown: {reason}; no earlier model is substituted."
    return LatestResult(
        signals, observation, frozen.model.fit_cutoff, status, message, fetched_at,
        frozen.source_sha256,
    )


def get_latest_signals(
    model_spec: dict,
    cached_prices: pd.DataFrame,
    *,
    fetch_live: bool = False,
    now: datetime | None = None,
    fetcher: Callable[[tuple[str, ...], str, datetime], pd.DataFrame] | None = None,
) -> LatestResult:
    """Match the final historical model; live failures use only validated cache.

    ``now`` defaults to current UTC; naive values are interpreted as UTC. Current
    and future India dates are always excluded, even after market close. Live
    data must be at most seven calendar days old and cannot precede a valid
    cached observation. Cached data has no age cap. For a failed refresh,
    ``fetched_at_utc`` records the attempt, not a successful fetch; cache-only
    calls leave it empty. Invalid cache raises DataValidationError.
    """
    frozen, clock = _load_model(model_spec), _utc_now(now)
    fetched_at, live_error = "", None
    validated_cache = None
    if fetch_live:
        fetched_at = clock.isoformat()
        try:
            recent = (fetcher or fetch_recent_prices)(frozen.stocks, frozen.benchmark, clock)
            accepted = _completed_prices(recent, frozen, clock)
            observation = accepted.loc[accepted.ticker == frozen.benchmark, "date"].max()
            age = (clock.astimezone(INDIA).date() - observation.date()).days
            if age > MAX_LIVE_AGE_DAYS:
                raise DataValidationError(
                    f"Stale provider observation {observation:%Y-%m-%d}: {age} calendar days "
                    f"before India today; live limit is {MAX_LIVE_AGE_DAYS} calendar days."
                )
            try:
                validated_cache = _completed_prices(cached_prices, frozen, clock)
            except DataValidationError:
                # A bad cache cannot prevent a valid fresh provider cohort.
                pass
            if validated_cache is not None:
                cached_observation = validated_cache.loc[
                    validated_cache.ticker == frozen.benchmark, "date"
                ].max()
                if observation < cached_observation:
                    raise DataValidationError(
                        f"Provider observation {observation:%Y-%m-%d} is older than "
                        f"validated cached observation {cached_observation:%Y-%m-%d}."
                    )
        except Exception as exc:
            # Provider failures and invalid provider cohorts share the same
            # conservative fallback. Never combine new bars with cached bars.
            live_error = str(exc) or type(exc).__name__
        else:
            return _result(accepted, frozen, "live", fetched_at, "Live provider fetch validated.")
    try:
        accepted = validated_cache if validated_cache is not None else _completed_prices(cached_prices, frozen, clock)
    except DataValidationError as exc:
        detail = f" Live refresh failed: {live_error}." if live_error else ""
        raise DataValidationError(f"Latest signal unavailable: cached data invalid: {exc}.{detail}") from exc
    if live_error:
        detail = f"Live refresh failed ({live_error}); fallback to cached snapshot (stale data)."
        return _result(accepted, frozen, "fallback", fetched_at, detail)
    return _result(accepted, frozen, "cached", "", "Cached frozen snapshot (stale data); no live fetch attempted.")
