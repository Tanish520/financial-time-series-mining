"""Read-only artifact/UI tests; the bundled holdout contains actual saved results."""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "app/artifacts/day1_holdout"
ENTRY = ROOT / "app/streamlit_app.py"


@pytest.fixture(autouse=True)
def legacy_runtime_bundle(monkeypatch):
    """Keep the original engineering-checkpoint regressions on their own bundle."""
    monkeypatch.setenv("ADM_RESULTS_DIR", str(BUNDLE))


def results_module():
    try:
        from app import results
    except ImportError:
        pytest.fail("app.results must verify and load the saved display bundle")
    return results


def view_module():
    try:
        from app import view_data
    except ImportError:
        pytest.fail("app.view_data must select and format saved experiment results")
    return view_data


def run_app():
    if not ENTRY.is_file():
        pytest.fail("app/streamlit_app.py must render the approved five-view preview")
    return AppTest.from_file(str(ENTRY), default_timeout=20).run()


def test_loads_curated_results_without_research_or_provider_modules(monkeypatch):
    for name in ("yfinance", "mlxtend", "src.mining", "src.pipeline"):
        monkeypatch.setitem(sys.modules, name, None)
    bundle = results_module().load_bundle(BUNDLE)
    assert bundle.manifest["evaluation_scope"] == "single_holdout_engineering_checkpoint"
    assert len(bundle.predictions) == 63 and len(bundle.equity) == 567
    assert len(bundle.training_sessions) == 500 and bundle.rules.empty
    assert bundle.tickers == ("HDFCBANK.NS", "RELIANCE.NS", "TCS.NS")


def test_rejects_modified_bundle_instead_of_showing_unverified_results(tmp_path):
    results = results_module()
    target = tmp_path / "bundle"
    shutil.copytree(BUNDLE, target)
    path = target / "metrics.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(results.ResultBundleError, match="checksum"):
        results.load_bundle(target)


def test_missing_bundle_has_a_clear_loader_error(tmp_path):
    results = results_module()
    with pytest.raises(results.ResultBundleError, match="bundle"):
        results.load_bundle(tmp_path / "absent")


def test_explicit_packaging_preserves_bytes_and_refuses_overwrite(tmp_path):
    results = results_module()
    target = tmp_path / "packaged"
    results.package_bundle(BUNDLE, target)
    for source in BUNDLE.iterdir():
        assert (target / source.name).read_bytes() == source.read_bytes()
    assert results.load_bundle(target).manifest == results.load_bundle(BUNDLE).manifest
    with pytest.raises(results.ResultBundleError, match="exists"):
        results.package_bundle(BUNDLE, target)


@pytest.mark.parametrize("metric,value,expected", [
    ("net_return", 0.0, "0.00%"), ("net_return", -0.0136645897, "-1.37%"),
    ("win_rate", 1 / 3, "33.33%"), ("sharpe", None, "N/A"),
    ("profit_factor", np.nan, "N/A"), ("profit_factor", "Infinity", "∞"),
    ("profit_factor", np.inf, "∞"), ("trade_count", 0.0, "0"),
])
def test_metric_formatting_preserves_units_and_undefined_values(metric, value, expected):
    assert view_module().format_metric(metric, value) == expected


def test_display_filters_select_existing_rules_without_recomputing_them():
    rules = pd.DataFrame([
        {"target_ticker": "AAA.NS", "target_label": "U", "confidence": 0.7},
        {"target_ticker": "BBB.NS", "target_label": "D", "confidence": 0.65},
        {"target_ticker": "AAA.NS", "target_label": "N", "confidence": 0.8},
    ])
    before = rules.copy(deep=True)
    selected = view_module().filter_rules(rules, target="AAA.NS", direction="All", min_confidence=0.75)
    assert selected.target_label.tolist() == ["N"]
    pd.testing.assert_frame_equal(rules, before, check_exact=True)


def test_cost_selector_reads_the_matching_saved_controls_and_curve():
    bundle = results_module().load_bundle(BUNDLE)
    view = view_module()
    chart, metrics, raw = view.backtest_tables(bundle, scope="Portfolio", cost=0.001)
    assert len(chart) == 21 and chart["Rules strategy"].eq(100).all()
    assert metrics.loc["Net return", "Rules strategy"] == "0.00%"
    assert metrics.loc["Net return", "Buy and hold"] == "-1.37%"
    assert metrics.loc["Sharpe", "Rules strategy"] == "N/A"
    _, higher_cost, _ = view.backtest_tables(bundle, scope="Portfolio", cost=0.005)
    assert higher_cost.loc["Net return", "Buy and hold"] == "-1.77%"
    assert raw.loc["NIFTY50_PRICE_INDEX", "applied_cost"] == 0


def test_unknown_cost_cannot_silently_fall_back_to_another_scenario():
    bundle = results_module().load_bundle(BUNDLE)
    with pytest.raises(results_module().ResultBundleError, match="scenario"):
        view_module().backtest_tables(bundle, scope="Portfolio", cost=0.25)


def test_rules_view_has_the_actual_empty_state_and_project_summary():
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Financial Pattern Mining"
    assert app.selectbox(key="page").value == "Rules Explorer"
    assert any("investigates" in element.value for element in app.markdown)
    assert any("No rules met" in element.value for element in app.info)


def test_backtest_widgets_show_saved_values_for_cost_and_stock_scope():
    app = run_app()
    app.selectbox(key="page").set_value("Backtest").run()
    assert not app.exception
    assert app.metric[0].value == "0.00%"
    app.selectbox(key="cost").set_value(0.005).run()
    table = app.dataframe[0].value.set_index("Metric")
    assert table.loc["Net return", "Buy and hold"] == "-1.77%"
    app.selectbox(key="scope").set_value("RELIANCE.NS").run()
    assert not app.exception
    assert app.metric[0].value == "0.00%"
    assert app.dataframe[0].value.set_index("Metric").loc["Round trips", "Buy and hold"] == "1"


@pytest.mark.parametrize("page,message", [
    ("Tuning & Drift", "has not run"),
    ("Asymmetry", "No U/D forecasts"),
    ("Latest Signals", "not enabled"),
])
def test_pending_views_do_not_invent_results(page, message):
    app = run_app()
    app.selectbox(key="page").set_value(page).run()
    assert not app.exception
    assert any(message in element.value for element in app.info)
    assert not app.metric


def test_app_pages_run_with_research_and_provider_imports_blocked(monkeypatch):
    for name in ("yfinance", "mlxtend", "src.mining", "src.pipeline", "src.backtest"):
        monkeypatch.setitem(sys.modules, name, None)
    app = run_app()
    app.selectbox(key="page").set_value("Backtest").run()
    assert not app.exception and not app.error


def test_missing_results_produces_an_honest_ui_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ADM_RESULTS_DIR", str(tmp_path / "missing"))
    app = run_app()
    assert not app.exception
    assert any("Saved results are unavailable" in element.value for element in app.error)
    assert not app.metric
