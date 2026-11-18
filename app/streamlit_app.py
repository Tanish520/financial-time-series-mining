"""Single-column Streamlit preview backed only by a verified result bundle."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app.results import DEFAULT_BUNDLE, ResultBundleError, load_bundle
from app.view_data import backtest_tables, filter_rules, format_metric

PAGES = ["Rules Explorer", "Tuning & Drift", "Backtest", "Asymmetry", "Latest Signals"]


def pretty_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def rules_page(bundle) -> None:
    st.markdown(
        "This project **investigates** interpretable stock patterns, adaptive training windows "
        f"and validation after trading costs. This preview uses {len(bundle.tickers)} stocks and a fixed holdout; "
        "walk-forward tuning is next."
    )
    st.header("Rules Explorer")
    meta = bundle.manifest
    if bundle.rules.empty:
        st.info("No rules met the fixed cutoffs in this training window. The holdout strategy stayed in cash.")
    st.caption(
        f"Training only · {meta['training_transactions']} sessions · "
        f"Support ≥ {meta['min_support']:.0%} · Confidence ≥ {meta['min_confidence']:.0%} · Lift > 1"
    )
    target = st.selectbox("Target stock", ["All stocks", *bundle.tickers], key="rule_target")
    direction = st.selectbox(
        "Direction", ["All", "U", "D", "N"], key="rule_direction",
        format_func=lambda value: {"All": "All directions", "U": "Up (U)", "D": "Down (D)", "N": "Neutral (N)"}[value],
    )
    confidence = st.slider("Display confidence at least (%)", 0, 100, int(round(meta["min_confidence"] * 100)), 5)
    selected = filter_rules(bundle.rules, target=target, direction=direction, min_confidence=confidence / 100)
    display = selected[["antecedents", "consequent", "support", "confidence", "lift"]].copy()
    display["antecedents"] = display.antecedents.map(lambda items: " & ".join(items))
    for name in ("support", "confidence"):
        display[name] = display[name].map(lambda value: f"{value:.2%}")
    display = display.rename(columns={
        "antecedents": "Observed pattern", "consequent": "Next-session target",
        "support": "Support", "confidence": "Confidence", "lift": "Lift",
    })
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("These filters only narrow saved rules. They do not re-mine data or change the model.")
    st.markdown("**Key Insight:** Training confidence describes historical matches. Backtest evaluates the later sessions.")


def tuning_page(bundle) -> None:
    st.header("Tuning & Drift")
    st.info("Walk-forward tuning has not run yet. There are no selected-window frequencies to report.")
    st.subheader("Fixed settings in this preview")
    meta = bundle.manifest
    st.table(pd.DataFrame({
        "Setting": ["Training window", "Movement threshold", "Minimum support", "Minimum confidence", "Fit cutoff"],
        "Value": ["Expanding history" if meta["window"] == "expanding" else f"{meta['window']} sessions", f"{meta['theta']:.2%}", f"{meta['min_support']:.2%}",
                  f"{meta['min_confidence']:.2%}", pretty_date(meta["fit_cutoff"])],
    }))
    st.write("Next, the same validation dates will compare 250-session, 500-session and expanding histories. The test period will stay outside parameter selection.")


def backtest_page(bundle) -> None:
    st.header("Backtest")
    scope = st.selectbox("View", ["Portfolio", *bundle.tickers], key="scope")
    costs = sorted(bundle.metrics.cost.unique().tolist())
    default = costs.index(0.001) if 0.001 in costs else 0
    cost = st.selectbox("Total round-trip cost", costs, index=default, format_func=lambda value: f"{value:.2%}", key="cost")
    chart, metrics, raw = backtest_tables(bundle, scope=scope, cost=cost)
    strategy = "RULES_PORTFOLIO" if scope == "Portfolio" else f"RULES:{scope}"
    st.metric("Strategy net return", format_metric("net_return", raw.loc[strategy, "net_return"]))
    st.caption(f"{format_metric('trade_count', raw.loc[strategy, 'trade_count'])} round trips · Max drawdown {format_metric('max_drawdown', raw.loc[strategy, 'max_drawdown'])}")
    st.subheader("Equity through the holdout")
    st.line_chart(
        chart, height=300, width="stretch", x_label="Execution session",
        y_label="Value of initial 100", color=["#0F766E", "#4F46E5", "#94A3B8"],
    )
    st.caption("Initial allocation = 100. Plotted points are each test session's closing value. NIFTY is an uncosted price-index reference.")
    st.subheader("Comparison metrics")
    st.dataframe(metrics.reset_index(), hide_index=True, width="stretch", height="content")
    st.info(f"This is one fixed {bundle.manifest['test_sessions']}-session engineering holdout. Holding cash during a falling month does not demonstrate predictive skill.")
    with st.expander("How to read these results"):
        st.write("Each stock gets 1/N of current capital. U forecasts enter at the next open and exit that close; D, N and unmatched cases stay in cash.")
        st.write("Costs are flat round-trip assumptions on entry notional. Holding controls pay once across the holding period; the index reference has no modeled implementation cost.")
        st.write("N/A means a statistic is undefined. Annualized values over a short interval are descriptive; they are not reliable long-run forecasts.")
    st.download_button("Download selected metrics", raw.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="holdout_metrics.csv", mime="text/csv")


def asymmetry_page(bundle) -> None:
    st.header("Asymmetry")
    counts = [int(bundle.predictions.prediction_label.eq(label).sum()) for label in ("U", "D")]
    if not sum(counts):
        st.info("No U/D forecasts were produced in this holdout. Precision and lift are undefined.")
    else:
        st.info("Forecast counts are available; held-out precision and lift have not been computed in this preview.")
    st.table(pd.DataFrame({"Direction": ["Up (U)", "Down (D)"], "Forecasts": counts}))
    st.write("The bearish-versus-bullish observation will be investigated using held-out forecasts and class base rates after walk-forward evaluation.")


def latest_page(bundle) -> None:
    st.header("Latest Signals")
    st.info("Latest signals are not enabled in this preview. Daily refresh and cached fallback are planned for Day 4.")
    st.write(f"The available experiment is historical: {pretty_date(bundle.manifest['test_start'])}–{pretty_date(bundle.manifest['test_end'])}. It is not a current market signal.")


def main() -> None:
    st.set_page_config(page_title="Financial Pattern Mining", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")
    st.title("Financial Pattern Mining")
    st.caption("CS G520 · BITS Pilani Hyderabad · Topic 21")
    try:
        bundle = load_bundle(Path(os.environ.get("ADM_RESULTS_DIR", str(DEFAULT_BUNDLE))))
    except ResultBundleError as exc:
        logging.warning("Saved experiment unavailable: %s", exc)
        st.error("Saved results are unavailable or failed verification. Ask the project team to refresh the saved experiment.")
        st.stop()
    meta = bundle.manifest
    st.caption(f"Fixed holdout · {len(bundle.tickers)} stocks · Fit: {pretty_date(meta['fit_cutoff'])} · Test: {pretty_date(meta['test_start'])}–{pretty_date(meta['test_end'])}")
    page = st.selectbox("Page", PAGES, key="page")
    try:
        {"Rules Explorer": rules_page, "Tuning & Drift": tuning_page,
         "Backtest": backtest_page, "Asymmetry": asymmetry_page, "Latest Signals": latest_page}[page](bundle)
    except ResultBundleError as exc:
        logging.warning("Saved scenario unavailable: %s", exc)
        st.error("This saved comparison is unavailable. Ask the project team to refresh the experiment.")
    st.divider()
    st.caption("Tanish Kumar Gupta · Aayush Gupt")
    st.caption("Educational project; not investment advice.")


if __name__ == "__main__":
    main()
