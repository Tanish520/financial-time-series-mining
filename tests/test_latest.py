"""Latest signals keep the live boundary explicit and the historical model frozen."""

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError, DOWNLOAD_OPTIONS
from src.mining import RULE_COLUMNS

STOCKS = ("AAA.NS", "BBB.NS")
BENCHMARK = "^NSEI"
NOW = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)


def latest_module():
    try:
        from src import latest
    except ImportError:
        pytest.fail("src.latest must expose an explicit, frozen-rule latest-signal boundary")
    return latest


def prices_for(days=("2024-01-02", "2024-01-03")):
    rows = []
    for index, session in enumerate(pd.to_datetime(days)):
        for ticker in (*STOCKS, BENCHMARK):
            close = 100.0 + (2.0 * index if ticker != "BBB.NS" else -2.0 * index)
            rows.append({
                "date": session, "ticker": ticker, "open": 100.0,
                "high": max(100.0, close) + 1.0, "low": min(100.0, close) - 1.0,
                "close": close, "volume": 0.0 if ticker == BENCHMARK else 1000.0,
                "dividends": 0.0, "stock_splits": 0.0,
            })
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def frozen_rule(antecedents=("AAA_U",), target="AAA.NS", label="U"):
    return {
        "antecedents": list(antecedents), "consequent": f"NEXT_{target.removesuffix('.NS')}_{label}",
        "target_ticker": target, "target_label": label, "support": 0.16,
        "confidence": 0.8, "lift": 4.0, "antecedent_support": 0.2,
        "consequent_support": 0.2, "joint_count": 80, "antecedent_count": 100,
        "consequent_count": 100, "transaction_count": 500,
    }


@pytest.fixture
def model_spec():
    return {
        "stocks": list(STOCKS), "benchmark": BENCHMARK, "fit_cutoff": "2024-01-02",
        "theta": 0.005, "parameters": {"min_support": 0.02, "min_confidence": 0.60},
        "fallback": None, "rules": [frozen_rule()], "source_sha256": "a" * 64,
    }


def test_cached_default_never_imports_yfinance_or_calls_a_fetcher(model_spec, monkeypatch):
    latest = latest_module()
    monkeypatch.setitem(sys.modules, "yfinance", None)

    def forbidden_fetch(*args, **kwargs):
        pytest.fail("cache-only use must never cross the network boundary")

    cached = prices_for()
    untouched = cached.copy(deep=True)
    original_model = deepcopy(model_spec)
    result = latest.get_latest_signals(model_spec, cached, now=NOW, fetcher=forbidden_fetch)

    assert result.status == "cached" and result.fetched_at_utc == ""
    assert result.observation_session == pd.Timestamp("2024-01-03")
    assert result.fit_cutoff == pd.Timestamp("2024-01-02")
    assert result.source_sha256 == model_spec["source_sha256"]
    assert result.intended_execution == "next trading session"
    assert result.educational_disclaimer == "Educational project; not investment advice."
    assert "cached" in result.message.lower()
    assert result.signals.iloc[0].action == "LONG"
    assert result.signals.iloc[0].rule_fit_cutoff == result.fit_cutoff
    pd.testing.assert_frame_equal(cached, untouched, check_exact=True)
    assert model_spec == original_model


def test_current_and_future_india_dates_are_excluded_before_bar_validation(model_spec):
    latest = latest_module()
    completed = prices_for(("2024-01-04", "2024-01-05"))
    incomplete = prices_for(("2024-01-08", "2024-01-09")).iloc[[0, 5]].copy()
    incomplete.loc[:, "close"] = np.nan
    all_bars = pd.concat([completed, incomplete], ignore_index=True)
    result = latest.get_latest_signals(model_spec, all_bars, now=NOW)
    assert result.observation_session == pd.Timestamp("2024-01-05")
    assert set(result.signals.feature_session) == {pd.Timestamp("2024-01-05")}


def test_india_calendar_boundary_excludes_session_even_before_utc_midnight(model_spec):
    latest = latest_module()
    now = datetime(2024, 1, 4, 20, 0, tzinfo=timezone.utc)  # Jan 5 in India.
    result = latest.get_latest_signals(
        model_spec, prices_for(("2024-01-03", "2024-01-04", "2024-01-05")), now=now,
    )
    assert result.observation_session == pd.Timestamp("2024-01-04")


@pytest.mark.parametrize("broken", ["latest_stock", "prior_stock", "benchmark", "ticker", "duplicate"])
def test_cache_rejects_incomplete_or_mismatched_cohorts_without_backfilling(model_spec, broken):
    cached = prices_for()
    if broken == "latest_stock":
        cached = cached.loc[~((cached.ticker == "BBB.NS") & (cached.date == cached.date.max()))]
    elif broken == "prior_stock":
        cached = cached.loc[~((cached.ticker == "AAA.NS") & (cached.date == cached.date.min()))]
    elif broken == "benchmark":
        cached = cached.loc[~((cached.ticker == BENCHMARK) & (cached.date == cached.date.max()))]
    elif broken == "ticker":
        cached.loc[cached.ticker == "BBB.NS", "ticker"] = "OTHER.NS"
    else:
        cached = pd.concat([cached, cached.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="(?i)(unavailable|calendar|cohort|ticker|duplicate)"):
        latest_module().get_latest_signals(model_spec, cached, now=NOW)


def test_a_single_completed_session_cannot_supply_a_preceding_close(model_spec):
    with pytest.raises(DataValidationError, match="(?i)(preceding|sessions|unavailable)"):
        latest_module().get_latest_signals(model_spec, prices_for(("2024-01-03",)), now=NOW)


@pytest.mark.parametrize("bad_close", [np.nan, np.inf, 0.0])
def test_invalid_preceding_close_is_never_imputed(model_spec, bad_close):
    cached = prices_for()
    cached.loc[(cached.ticker == "AAA.NS") & (cached.date == cached.date.min()), "close"] = bad_close
    with pytest.raises(DataValidationError, match="(?i)(price|unavailable)"):
        latest_module().get_latest_signals(model_spec, cached, now=NOW)


def test_explicit_live_refresh_uses_only_the_frozen_model_and_latest_completed_features(model_spec, monkeypatch):
    from src import mining, signals

    latest = latest_module()
    calls = []

    def fetcher(stocks, benchmark, now):
        calls.append((stocks, benchmark, now))
        return prices_for(("2024-01-04", "2024-01-05", "2024-01-08"))

    def forbidden_mine(*args, **kwargs):
        pytest.fail("refresh must not retrain the final historical model")

    real_predict = signals.predict_signals

    def check_observed_interface(model, features, stocks):
        assert set(features.columns) == signals.FEATURE_COLUMNS
        assert model.transactions.empty and model.sessions.empty
        assert list(model.rules.columns) == RULE_COLUMNS
        assert model.fit_cutoff == pd.Timestamp(model_spec["fit_cutoff"])
        return real_predict(model, features, stocks)

    monkeypatch.setattr(mining, "mine_rules", forbidden_mine)
    monkeypatch.setattr(latest, "predict_signals", check_observed_interface)
    result = latest.get_latest_signals(model_spec, prices_for(), fetch_live=True, now=NOW, fetcher=fetcher)
    assert calls == [(STOCKS, BENCHMARK, NOW)]
    assert result.status == "live"
    assert result.fetched_at_utc == NOW.isoformat()
    assert result.observation_session == pd.Timestamp("2024-01-05")
    assert result.signals.iloc[0].action == "LONG"
    assert "2024-01-02" in result.message and "2024-01-05" in result.message


@pytest.mark.parametrize("failure", ["exception", "empty", "partial_latest", "partial_prior", "invalid_bar", "wrong_ticker"])
def test_failed_or_invalid_live_refresh_uses_truthfully_dated_cache(model_spec, failure):
    live = prices_for(("2024-01-04", "2024-01-05"))
    if failure == "empty":
        live = pd.DataFrame()
    elif failure in {"partial_latest", "partial_prior"}:
        day = live.date.max() if failure == "partial_latest" else live.date.min()
        live = live.loc[~(live.ticker.eq("BBB.NS") & live.date.eq(day))]
    elif failure == "invalid_bar":
        live.loc[0, "close"] = np.nan
    elif failure == "wrong_ticker":
        live.loc[live.ticker == "BBB.NS", "ticker"] = "OTHER.NS"

    def fetcher(*args):
        if failure == "exception":
            raise RuntimeError("provider unavailable")
        return live

    result = latest_module().get_latest_signals(
        model_spec, prices_for(), fetch_live=True, now=NOW, fetcher=fetcher,
    )
    assert result.status == "fallback" and result.fetched_at_utc == NOW.isoformat()
    assert result.observation_session == pd.Timestamp("2024-01-03")
    assert "fallback" in result.message.lower() and "cache" in result.message.lower()
    assert "2024-01-03" in result.message
    assert "stale" in result.message.lower()


def test_fallback_preserves_the_live_error_reason(model_spec):
    def fetcher(*args):
        raise RuntimeError("provider unavailable")

    result = latest_module().get_latest_signals(
        model_spec, prices_for(), fetch_live=True, now=NOW, fetcher=fetcher,
    )
    assert "provider unavailable" in result.message


def test_live_data_must_not_precede_the_frozen_fit_cutoff(model_spec):
    model_spec["fit_cutoff"] = "2024-01-05"
    with pytest.raises(DataValidationError, match="cutoff"):
        latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)


def test_frozen_down_rule_stays_in_cash_and_keeps_its_class(model_spec):
    model_spec["rules"] = [frozen_rule(("BBB_D",), target="BBB.NS", label="D")]
    result = latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)
    row = result.signals.set_index("ticker").loc["BBB.NS"]
    assert row.prediction_label == "D" and row.action == "CASH"
    assert row.consequent == "NEXT_BBB_D"


def test_a_next_item_cannot_enter_a_frozen_rule_antecedent(model_spec):
    model_spec["rules"][0]["antecedents"] = ["NEXT_AAA_U"]
    with pytest.raises(DataValidationError, match="antecedent"):
        latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)


@pytest.mark.parametrize("fallback", [False, True])
def test_final_fold_no_rule_states_keep_dated_cash_unknown_without_substituting_a_model(model_spec, fallback):
    model_spec["rules"] = []
    if fallback:
        model_spec.update(fallback="no_eligible_configuration", theta=None, parameters=None)
    result = latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)
    assert result.observation_session == pd.Timestamp("2024-01-03")
    assert result.fit_cutoff == pd.Timestamp(model_spec["fit_cutoff"])
    assert len(result.signals) == len(STOCKS)
    assert result.signals.action.eq("CASH").all()
    assert result.signals.prediction_label.isna().all()
    reason = "no_eligible_configuration" if fallback else "no_rules_final_fold"
    assert result.signals.reason.eq(reason).all()
    assert reason in result.message


def test_fallback_model_cannot_smuggle_in_rules(model_spec):
    model_spec.update(fallback="no_eligible_configuration", theta=None, parameters=None)
    with pytest.raises(DataValidationError, match="(?i)(fallback|rules)"):
        latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)


@pytest.mark.parametrize("field,value", [
    ("stocks", ["AAA.NS", "AAA.NS"]), ("benchmark", "AAA.NS"),
    ("fit_cutoff", "2024-01-02T12:00:00"), ("theta", None),
    ("theta", np.nan), ("parameters", None), ("source_sha256", ""),
])
def test_invalid_frozen_model_provenance_or_configuration_is_rejected(model_spec, field, value):
    model_spec[field] = value
    with pytest.raises(DataValidationError):
        latest_module().get_latest_signals(model_spec, prices_for(), now=NOW)


def test_no_completed_cohort_in_either_source_is_unavailable(model_spec):
    today_only = prices_for(("2024-01-08",))
    with pytest.raises(DataValidationError, match="(?i)unavailable"):
        latest_module().get_latest_signals(
            model_spec, today_only, fetch_live=True, now=NOW, fetcher=lambda *args: today_only,
        )


def test_valid_live_response_does_not_require_a_valid_cache(model_spec):
    result = latest_module().get_latest_signals(
        model_spec, pd.DataFrame(), fetch_live=True, now=NOW,
        fetcher=lambda *args: prices_for(("2024-01-04", "2024-01-05")),
    )
    assert result.status == "live" and result.observation_session == pd.Timestamp("2024-01-05")


def test_older_live_response_never_replaces_a_newer_validated_cache(model_spec):
    result = latest_module().get_latest_signals(
        model_spec, prices_for(("2024-01-04", "2024-01-05")), fetch_live=True, now=NOW,
        fetcher=lambda *args: prices_for(("2024-01-02", "2024-01-03")),
    )
    assert result.status == "fallback" and result.observation_session == pd.Timestamp("2024-01-05")
    assert result.fetched_at_utc == NOW.isoformat()
    assert "older" in result.message.lower()
    assert "2024-01-03" in result.message and "2024-01-05" in result.message


def test_year_old_live_response_cannot_replace_valid_frozen_cache(model_spec):
    model_spec["fit_cutoff"] = "2023-01-02"
    result = latest_module().get_latest_signals(
        model_spec, prices_for(), fetch_live=True, now=NOW,
        fetcher=lambda *args: prices_for(("2023-01-02", "2023-01-03")),
    )
    assert result.status == "fallback" and result.observation_session == pd.Timestamp("2024-01-03")
    assert result.fetched_at_utc == NOW.isoformat()
    assert "stale" in result.message.lower() and "2023-01-03" in result.message


def test_live_freshness_limit_uses_injected_india_today_and_allows_stale_cache(model_spec):
    model_spec["fit_cutoff"] = "2023-12-29"
    now = datetime(2024, 1, 8, 20, 0, tzinfo=timezone.utc)  # Jan 9 in India: provider is 8 days old.
    result = latest_module().get_latest_signals(
        model_spec, prices_for(("2023-12-28", "2023-12-29")), fetch_live=True, now=now,
        fetcher=lambda *args: prices_for(("2023-12-29", "2024-01-01")),
    )
    assert result.status == "fallback" and result.observation_session == pd.Timestamp("2023-12-29")
    assert result.fetched_at_utc == now.isoformat()
    assert "stale" in result.message.lower()
    assert "2024-01-01" in result.message and "8 calendar days" in result.message
    assert "7" in result.message


def test_live_observation_exactly_seven_calendar_days_old_is_accepted(model_spec):
    model_spec["fit_cutoff"] = "2023-12-29"
    result = latest_module().get_latest_signals(
        model_spec, prices_for(("2023-12-28", "2023-12-29")), fetch_live=True, now=NOW,
        fetcher=lambda *args: prices_for(("2023-12-29", "2024-01-01")),
    )
    assert result.status == "live" and result.observation_session == pd.Timestamp("2024-01-01")


def test_cache_only_observation_has_no_freshness_age_cap(model_spec):
    now = datetime(2025, 1, 8, 12, 0, tzinfo=timezone.utc)
    result = latest_module().get_latest_signals(model_spec, prices_for(), now=now)
    assert result.status == "cached" and result.fetched_at_utc == ""
    assert result.observation_session == pd.Timestamp("2024-01-03")
    assert "stale" in result.message.lower()


def test_network_helper_requests_original_adjustment_policy_and_excludes_today(model_spec, monkeypatch):
    latest = latest_module()
    recent = prices_for(("2024-01-04", "2024-01-05"))
    fields = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends", "stock_splits": "Stock Splits",
    }
    raw = pd.concat({
        ticker: group.set_index("date")[list(fields)].rename(columns=fields)
        for ticker, group in recent.groupby("ticker")
    }, axis=1)
    observed = {}

    def download(**kwargs):
        observed.update(kwargs)
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = latest.fetch_recent_prices(STOCKS, BENCHMARK, NOW)
    assert observed["tickers"] == [*STOCKS, BENCHMARK]
    assert observed["end"] == "2024-01-08"
    assert pd.Timestamp(observed["start"]) <= pd.Timestamp("2023-12-18")
    for key, value in DOWNLOAD_OPTIONS.items():
        assert observed[key] == value
    pd.testing.assert_frame_equal(result, recent, check_dtype=False)
