"""Version 2 display contract checks using synthetic data confined to tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import results, view_data

ENTRY = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
COSTS = [0.0005, 0.001, 0.005]
STOCKS = ["AAA.NS", "BBB.NS"]


def _write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _rehash(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in manifest["files"]}
    _write_json(path, manifest)


@pytest.fixture
def full_bundle(tmp_path):
    """Two synthetic folds: one winner, one explicit no-eligible fallback."""
    directory = tmp_path / "full"
    directory.mkdir()
    days = pd.bdate_range("2024-01-08", periods=4)
    cutoffs = ["2024-01-05", "2024-01-09"]
    folds = pd.DataFrame([
        {"fold_id": 1, "fit_cutoff": cutoffs[0], "inner_fit_cutoff": "2023-10-05",
         "validation_start": "2023-10-06", "validation_end": cutoffs[0],
         "test_start": str(days[0].date()), "test_end": str(days[1].date()), "test_sessions": 2,
         "eligible_candidates": 1, "selected_window": "250", "theta": 0.005,
         "min_support": 0.02, "min_confidence": 0.6, "validation_net_return": 0.01,
         "validation_trade_count": 20, "validation_trade_sessions": 5,
         "rule_count": 1, "fallback": None},
        {"fold_id": 2, "fit_cutoff": cutoffs[1], "inner_fit_cutoff": "2023-10-09",
         "validation_start": "2023-10-10", "validation_end": cutoffs[1],
         "test_start": str(days[2].date()), "test_end": str(days[3].date()), "test_sessions": 2,
         "eligible_candidates": 0, "selected_window": None, "theta": None,
         "min_support": None, "min_confidence": None, "validation_net_return": None,
         "validation_trade_count": 0, "validation_trade_sessions": 0,
         "rule_count": 0, "fallback": "no_eligible_configuration"},
    ])
    folds.to_csv(directory / "folds.csv", index=False)
    tuning = pd.DataFrame([
        {"fold_id": 1, "window": "250", "theta": 0.005, "min_support": 0.02,
         "min_confidence": 0.6, "validation_net_return": 0.01,
         "validation_trade_count": 20, "validation_trade_sessions": 5,
         "eligible": True, "selected": True, "fit_cutoff": "2023-10-05"},
        {"fold_id": 2, "window": "500", "theta": 0.005, "min_support": 0.02,
         "min_confidence": 0.6, "validation_net_return": 0.03,
         "validation_trade_count": 2, "validation_trade_sessions": 1,
         "eligible": False, "selected": False, "fit_cutoff": "2023-10-09"},
    ])
    tuning.to_csv(directory / "tuning.csv", index=False)
    predictions = []
    for i, day in enumerate(days):
        fold = i // 2 + 1
        for variant in ("TUNED", "FIXED"):
            for ticker in STOCKS:
                label = "U" if ticker == STOCKS[0] else "D"
                if fold == 2 and variant == "TUNED":
                    label = None
                predictions.append({
                    "fold_id": fold, "variant": variant, "feature_session": str((day - pd.offsets.BDay()).date()),
                    "execution_session": str(day.date()), "rule_fit_cutoff": cutoffs[fold - 1],
                    "ticker": ticker, "prediction_label": label, "action": "LONG" if label == "U" else "CASH",
                    "reason": "matched_rule" if label else "no_match", "matched_antecedents": "AAA_U" if label else "",
                    "consequent": f"NEXT_{ticker.removesuffix('.NS')}_{label}" if label else None,
                    "confidence": 0.7 if label else None, "lift": 1.4 if label else None,
                    "support": 0.1 if label else None, "matched_rule_count": 1 if label else 0,
                })
    pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
    sessions = pd.DataFrame([
        {"fold_id": fold, "variant": variant, "stage": "refit", "fit_id": "fixture", "fit_cutoff": cutoff,
         "feature_session": str((pd.Timestamp(cutoff) - pd.offsets.BDay()).date()),
         "execution_session": cutoff, "outcome_available_session": cutoff}
        for fold, cutoff in enumerate(cutoffs, 1) for variant in ("TUNED", "FIXED")
        if not (fold == 2 and variant == "TUNED")
    ])
    sessions.to_csv(directory / "training_sessions.csv", index=False)
    rule = {"antecedents": ["AAA_U"], "consequent": "NEXT_AAA_U", "target_ticker": "AAA.NS",
            "target_label": "U", "support": 0.1, "confidence": 0.7, "lift": 1.4,
            "antecedent_support": 0.2, "consequent_support": 0.5, "joint_count": 10,
            "antecedent_count": 20, "consequent_count": 50, "transaction_count": 100}
    rules = [{**rule, "fold_id": fold, "variant": variant, "fit_cutoff": cutoff, "theta": 0.005}
             for fold, cutoff in enumerate(cutoffs, 1) for variant in ("TUNED", "FIXED")
             if not (fold == 2 and variant == "TUNED")]
    _write_json(directory / "rules.json", rules)
    names = ["TUNED_PORTFOLIO", "FIXED_PORTFOLIO", "BUY_HOLD_BASKET", "NIFTY50_PRICE_INDEX"]
    names += [f"{variant}:{ticker}" for ticker in STOCKS for variant in ("TUNED", "FIXED", "BUY_HOLD")]
    metrics, equity = [], []
    for cost in COSTS:
        for n, name in enumerate(names):
            applied = 0 if name == "NIFTY50_PRICE_INDEX" else cost
            gain = 0.01 * (n + 1) - applied
            metrics.append({"strategy": name, "cost": cost, "applied_cost": applied,
                            "net_return": gain, "cagr": gain, "sharpe": None,
                            "max_drawdown": 0, "profit_factor": "Infinity" if n else None,
                            "win_rate": 1 if n else None, "trade_count": None if name == "NIFTY50_PRICE_INDEX" else n})
            for i, day in enumerate(days):
                equity.append({"strategy": name, "cost": cost, "applied_cost": applied,
                               "execution_session": str(day.date()), "equity": 100000 * (1 + gain * (i + 1) / 4),
                               "daily_return": gain / 4})
    pd.DataFrame(metrics).to_csv(directory / "metrics.csv", index=False)
    _write_json(directory / "metrics.json", metrics)
    pd.DataFrame(equity).to_csv(directory / "equity.csv", index=False)
    pd.DataFrame(columns=["strategy", "cost", "ticker", "entry_session", "exit_session", "net_pnl"]).to_csv(directory / "trades.csv", index=False)
    pd.DataFrame(columns=["fold_id", "variant", "ticker", "feature_session", "execution_session", "outcome_available_session", "theta", "prediction_label", "target_return", "target_label"]).to_csv(directory / "outcomes.csv", index=False)
    asymmetry = [
        {"variant": variant, "direction": direction, "prediction_count": count, "correct_count": correct,
         "precision": precision, "base_rate": base, "expected_correct": expected, "held_out_lift": lift}
        for variant in ("TUNED", "FIXED") for direction, count, correct, precision, base, expected, lift in (
            ("U", 2, 1, 0.5, 0.25, 0.5, 2.0), ("D", 0, 0, None, None, 0.0, None))
    ]
    pd.DataFrame(asymmetry).to_csv(directory / "asymmetry.csv", index=False)
    pd.DataFrame(columns=["fold_id", "variant", "ticker", "direction", "prediction_count", "correct_count", "observations", "class_base_rate", "expected_correct"]).to_csv(directory / "asymmetry_groups.csv", index=False)
    model = {"stocks": STOCKS, "benchmark": "^NSEI", "fit_cutoff": cutoffs[-1],
             "theta": None, "parameters": None, "fallback": "no_eligible_configuration",
             "rules": [], "source_sha256": "a" * 64}
    _write_json(directory / "latest_model.json", model)
    _write_json(directory / "latest_signals.json", {
        "signals": [{"ticker": ticker, "prediction_label": None, "action": "CASH",
                     "reason": "no_eligible_configuration", "matched_antecedents": [],
                     "consequent": None, "confidence": None, "lift": None, "support": None,
                     "matched_rule_count": 0, "rule_fit_cutoff": cutoffs[-1],
                     "feature_session": str(days[-1].date())} for ticker in STOCKS],
        "observation_session": str(days[-1].date()), "fit_cutoff": cutoffs[-1], "status": "cached",
        "message": "Frozen completed daily cache; use refresh to request recent bars.",
        "fetched_at_utc": "", "intended_execution": "next trading session",
    })
    latest = [{"date": str(day.date()), "ticker": ticker, "open": 100, "high": 102,
               "low": 99, "close": 101 + i, "volume": 1000, "dividends": 0, "stock_splits": 0}
              for i, day in enumerate(days[-2:]) for ticker in [*STOCKS, "^NSEI"]]
    pd.DataFrame(latest).to_csv(directory / "latest_prices.csv", index=False)
    manifest = {"schema_version": 2, "evaluation_scope": "nested_walk_forward", "stocks": STOCKS,
                "benchmark": "^NSEI", "initial_capital": 100000, "cost_scenarios": COSTS,
                "source_sha256": "a" * 64, "test_start": str(days[0].date()), "test_end": str(days[-1].date()),
                "fold_count": 2, "test_sessions": 4, "prediction_rows": len(predictions),
                "selection_cost": 0.001, "min_validation_trades": 20, "min_validation_trade_sessions": 5,
                "grid": {"windows": [250, 500, "expanding"], "thetas": [0.0025, 0.005, 0.01],
                         "supports": [0.01, 0.02, 0.05], "confidences": [0.55, 0.60, 0.70]},
                "files": {path.name: "" for path in directory.iterdir()}}
    _write_json(directory / "manifest.json", manifest)
    _rehash(directory)
    return directory


def _app(monkeypatch, directory):
    monkeypatch.setenv("ADM_RESULTS_DIR", str(directory))
    return AppTest.from_file(str(ENTRY), default_timeout=20).run()


def test_v2_verified_loader_reads_all_saved_display_tables(full_bundle):
    bundle = results.load_bundle(full_bundle)
    assert bundle.is_walk_forward
    assert bundle.tickers == tuple(STOCKS)
    assert len(bundle.folds) == 2 and len(bundle.tuning) == 2
    assert len(bundle.rules) == 3 and len(bundle.asymmetry) == 4
    assert bundle.latest_model["theta"] is None
    assert len(bundle.latest_prices) == 6


@pytest.mark.parametrize("scope", ["Portfolio", *STOCKS])
@pytest.mark.parametrize("cost", COSTS)
def test_v2_backtest_scope_and_cost_keep_four_aligned_saved_strategies(full_bundle, scope, cost):
    bundle = results.load_bundle(full_bundle)
    chart, formatted, raw = view_data.backtest_tables(bundle, scope=scope, cost=cost)
    assert chart.shape == (4, 4)
    assert list(chart.columns) == ["Tuned rules", "Fixed rules", "Buy and hold", "NIFTY reference"]
    assert raw.cost.eq(cost).all()
    assert raw.loc["NIFTY50_PRICE_INDEX", "applied_cost"] == 0
    assert len(formatted) == 7
    tuned = "TUNED_PORTFOLIO" if scope == "Portfolio" else f"TUNED:{scope}"
    assert chart["Tuned rules"].iloc[-1] == pytest.approx(100 * (1 + raw.loc[tuned, "net_return"]))


def test_window_frequencies_exclude_fallback_instead_of_assigning_a_window(full_bundle):
    bundle = results.load_bundle(full_bundle)
    winners, frequencies, fallbacks = view_data.tuning_tables(bundle)
    assert len(winners) == 2 and fallbacks == 1
    assert frequencies.to_dict() == {"250": 1, "500": 0, "expanding": 0}
    fallback = winners.iloc[1]
    assert fallback["Window"] == "N/A" and fallback["Validation net return"] == "N/A"


def test_rules_filter_preserves_fold_and_variant_provenance(full_bundle):
    bundle = results.load_bundle(full_bundle)
    selected = view_data.filter_rules(bundle.rules, fold_id=1, variant="FIXED", target="AAA.NS", direction="U", min_confidence=0.6)
    assert len(selected) == 1
    assert selected.iloc[0].fit_cutoff == pd.Timestamp("2024-01-05")
    assert view_data.filter_rules(bundle.rules, fold_id=2, variant="TUNED").empty


def test_asymmetry_preserves_null_denominators_and_counts(full_bundle):
    bundle = results.load_bundle(full_bundle)
    comparison, lift, counts = view_data.asymmetry_tables(bundle, variant="TUNED")
    assert comparison.loc["U", "Precision"] == 0.5
    assert comparison.loc["U", "Class base rate"] == 0.25
    assert lift.loc["U", "Held-out lift"] == 2
    assert np.isnan(lift.loc["D", "Held-out lift"])
    assert counts.loc["D", "Predictions"] == 0
    assert counts.loc["D", "Precision"] == "N/A"


def test_cached_loader_reverifies_changed_bytes_before_using_parsed_cache(full_bundle):
    results.load_bundle(full_bundle)
    path = full_bundle / "asymmetry.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(results.ResultBundleError, match="checksum"):
        results.load_bundle(full_bundle)


@pytest.mark.parametrize("filename,column", [
    ("training_sessions.csv", "fold_id"), ("tuning.csv", "min_confidence"),
    ("latest_prices.csv", "dividends"), ("outcomes.csv", "target_label"),
    ("asymmetry_groups.csv", "expected_correct"),
])
def test_rehashed_incomplete_payload_schema_is_rejected(full_bundle, filename, column):
    path = full_bundle / filename
    pd.read_csv(path).drop(columns=column).to_csv(path, index=False)
    _rehash(full_bundle)
    with pytest.raises(results.ResultBundleError, match="columns"):
        results.load_bundle(full_bundle)


def test_latest_model_cannot_substitute_an_earlier_winner_for_final_fallback(full_bundle):
    path = full_bundle / "latest_model.json"
    model = json.loads(path.read_text())
    previous = json.loads((full_bundle / "rules.json").read_text())[0]
    model.update(fallback=None, theta=.005,
                 parameters={"window": 250, "theta": .005, "min_support": .02, "min_confidence": .6},
                 rules=[{name: previous[name] for name in results.RULE_COLUMNS}])
    _write_json(path, model)
    _rehash(full_bundle)
    with pytest.raises(results.ResultBundleError, match="final|fallback"):
        results.load_bundle(full_bundle)


def test_final_fallback_cache_cannot_claim_a_long_signal(full_bundle):
    path = full_bundle / "latest_signals.json"
    cached = json.loads(path.read_text())
    cached["signals"][0].update(prediction_label="U", action="LONG")
    _write_json(path, cached)
    _rehash(full_bundle)
    with pytest.raises(results.ResultBundleError, match="fallback|cash|CASH"):
        results.load_bundle(full_bundle)


def test_rule_antecedent_mutation_cannot_poison_verified_parse_cache(full_bundle):
    first = results.load_bundle(full_bundle)
    original = tuple(first.rules.at[0, "antecedents"])
    try:
        first.rules.at[0, "antecedents"].append("UNVERIFIED_U")
    except AttributeError:
        pass
    again = results.load_bundle(full_bundle)
    assert tuple(again.rules.at[0, "antecedents"]) == original


@pytest.mark.parametrize("selection_cost", ["missing", "0.001", None, -0.1, 1.0, True])
def test_invalid_selection_cost_metadata_is_rejected_before_page_load(full_bundle, selection_cost):
    path = full_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    if selection_cost == "missing":
        manifest.pop("selection_cost")
    else:
        manifest["selection_cost"] = selection_cost
    _write_json(path, manifest)
    with pytest.raises(results.ResultBundleError, match="manifest|selection"):
        results.load_bundle(full_bundle)


@pytest.mark.parametrize("mutation", ["missing_comparison", "cost_mismatch", "duplicate_prediction", "wrong_fold_count"])
def test_verified_but_inconsistent_v2_payloads_are_rejected(full_bundle, mutation):
    if mutation == "missing_comparison":
        path = full_bundle / "equity.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[~((frame.strategy == "FIXED_PORTFOLIO") & (frame.cost == 0.001))]
        frame.to_csv(path, index=False)
    elif mutation == "cost_mismatch":
        path = full_bundle / "metrics.json"
        records = json.loads(path.read_text())
        records[0]["applied_cost"] = 0
        _write_json(path, records)
    elif mutation == "duplicate_prediction":
        path = full_bundle / "predictions.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.head(1)], ignore_index=True).to_csv(path, index=False)
    else:
        path = full_bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["fold_count"] = 3
        _write_json(path, manifest)
    _rehash(full_bundle)
    with pytest.raises(results.ResultBundleError):
        bundle = results.load_bundle(full_bundle)
        view_data.backtest_tables(bundle)


def test_v2_packaging_copies_only_the_verified_contract(full_bundle, tmp_path):
    target = tmp_path / "packaged"
    (full_bundle / "untracked-secret.txt").write_text("not part of display contract")
    results.package_bundle(full_bundle, target)
    manifest = json.loads((target / "manifest.json").read_text())
    assert {path.name for path in target.iterdir()} == {*manifest["files"], "manifest.json"}
    assert results.load_bundle(target).is_walk_forward


@pytest.mark.parametrize("page", ["Rules Explorer", "Tuning & Drift", "Backtest", "Asymmetry", "Latest Signals"])
def test_all_v2_views_render_offline_without_research_imports(full_bundle, monkeypatch, page):
    for name in ("src", "src.latest", "src.mining", "src.pipeline", "src.walkforward", "yfinance", "mlxtend"):
        monkeypatch.setitem(sys.modules, name, None)
    app = _app(monkeypatch, full_bundle)
    if page != "Rules Explorer":
        app.selectbox(key="page").set_value(page).run()
    assert not app.exception and not app.error
    assert any("Educational project; not investment advice." in item.value for item in app.caption)


def test_v2_rules_fallback_and_tuning_nulls_are_honest(full_bundle, monkeypatch):
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="rule_fold").set_value(2).run()
    assert not app.exception
    assert any("No rules met" in item.value for item in app.info)
    app.selectbox(key="page").set_value("Tuning & Drift").run()
    assert not app.exception
    assert any("1" == item.value and "Fallback" in item.label for item in app.metric)
    table = app.dataframe[0].value
    assert table.loc[1, "Window"] == "N/A"
    assert table.loc[1, "Validation net return"] == "N/A"


def test_rules_explorer_counts_refit_training_sessions(full_bundle, monkeypatch):
    path = full_bundle / "training_sessions.csv"
    sessions = pd.read_csv(path)
    sessions["stage"] = "refit"
    sessions.to_csv(path, index=False)
    _rehash(full_bundle)
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="rule_variant").set_value("FIXED").run()
    assert not app.exception
    assert any("Training only · 1 transactions" in item.value for item in app.caption)


def test_v2_backtest_widgets_follow_saved_scope_and_cost(full_bundle, monkeypatch):
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="page").set_value("Backtest").run()
    app.selectbox(key="scope").set_value("BBB.NS").run()
    app.selectbox(key="cost").set_value(0.005).run()
    assert not app.exception
    table = app.dataframe[0].value.set_index("Metric")
    assert table.loc["Net return", "Tuned rules"] == "7.50%"
    assert table.loc["Net return", "Fixed rules"] == "8.50%"


def test_backtest_chart_wraps_legend_and_keeps_years_visible_on_phones(full_bundle, monkeypatch):
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="page").set_value("Backtest").run()
    assert not app.exception
    spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    encoding = spec.get("encoding", spec.get("layer", [{}])[0].get("encoding", {}))
    color = encoding["color"]
    assert color.get("legend", {}).get("columns", 99) <= 2
    assert "%Y" in encoding["x"]["axis"].get("format", "")


def test_v2_latest_initial_page_has_model_and_observation_age_and_explicit_refresh(full_bundle, monkeypatch):
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="page").set_value("Latest Signals").run()
    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    assert "09 Jan 2024" in captions and "11 Jan 2024" in captions
    assert any("Refresh" in item.label for item in app.button)
    assert any("cached" in item.value.lower() or "fallback" in item.value.lower() for item in app.info)
    assert any("next trading session" in item.value.lower() for item in app.markdown)


def test_latest_refresh_is_explicit_and_success_is_cached_for_repeated_clicks(full_bundle, monkeypatch):
    from types import ModuleType, SimpleNamespace

    calls = []
    module = ModuleType("src.latest")

    def fetch(model_spec, cached_prices, *, fetch_live=False):
        calls.append(fetch_live)
        saved = json.loads((full_bundle / "latest_signals.json").read_text())
        return SimpleNamespace(signals=pd.DataFrame(saved["signals"]),
                               observation_session=pd.Timestamp(saved["observation_session"]),
                               fit_cutoff=pd.Timestamp(saved["fit_cutoff"]), status="live",
                               message="Completed provider bars accepted.", fetched_at_utc="2024-01-11T18:00:00+00:00")

    module.get_latest_signals = fetch
    monkeypatch.setitem(sys.modules, "src.latest", module)
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="page").set_value("Latest Signals").run()
    assert calls == []
    app.button(key="refresh_latest").click().run()
    assert not app.exception and calls == [True]
    app.button(key="refresh_latest").click().run()
    assert not app.exception and calls == [True]
    assert any("live" in item.value.lower() for item in app.success)


def test_latest_refresh_failure_keeps_verified_cache_visible(full_bundle, monkeypatch):
    from types import ModuleType

    module = ModuleType("src.latest")

    def unavailable(*args, **kwargs):
        raise ValueError("Provider unavailable")

    module.get_latest_signals = unavailable
    monkeypatch.setitem(sys.modules, "src.latest", module)
    # Unique model cache key prevents reuse of the successful request from another test.
    model_path = full_bundle / "latest_model.json"
    model = json.loads(model_path.read_text())
    model["test_request_nonce"] = "failure-case"
    _write_json(model_path, model)
    _rehash(full_bundle)
    app = _app(monkeypatch, full_bundle)
    app.selectbox(key="page").set_value("Latest Signals").run()
    app.button(key="refresh_latest").click().run()
    assert not app.exception and not app.error
    assert any("fallback" in item.value.lower() for item in app.warning)
    assert len(app.dataframe[0].value) == len(STOCKS)


def test_corrupt_v2_bundle_renders_a_clear_error(full_bundle, monkeypatch):
    path = full_bundle / "latest_signals.json"
    path.write_bytes(path.read_bytes() + b"\n")
    app = _app(monkeypatch, full_bundle)
    assert not app.exception
    assert any("Saved results are unavailable" in item.value for item in app.error)
    assert not app.metric
