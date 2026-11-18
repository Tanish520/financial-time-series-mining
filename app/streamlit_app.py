"""Five single-column views backed by verified, precomputed experiment results."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.results import ResultBundleError, default_bundle, load_bundle
from app.view_data import asymmetry_tables, backtest_tables, filter_rules, format_metric, tuning_tables

PAGES = ["Rules Explorer", "Tuning & Drift", "Backtest", "Asymmetry", "Latest Signals"]
VARIANT_LABELS = {"TUNED": "Tuned rules", "FIXED": "Fixed rules"}


def pretty_date(value) -> str:
    return "N/A" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%d %b %Y")


def pattern_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return " & ".join(value) or "N/A"
    return "N/A" if value is None or pd.isna(value) or value == "" else str(value)


def rules_page(bundle) -> None:
    full, meta = bundle.is_walk_forward, bundle.manifest
    st.markdown(
        "This project **investigates** interpretable stock patterns, adaptive training windows "
        "and validation after trading costs. "
        + (f"Saved walk-forward results compare tuned and fixed rules across {len(bundle.tickers)} stocks."
           if full else f"This preview uses {len(bundle.tickers)} stocks and a fixed holdout; walk-forward tuning is next.")
    )
    st.header("Rules Explorer")
    fold_id, variant = None, None
    if full:
        ids = bundle.folds.fold_id.astype(int).sort_values().tolist()
        by_id = bundle.folds.set_index("fold_id")
        fold_id = st.selectbox("Outer fold", ids, index=len(ids) - 1, key="rule_fold",
                               format_func=lambda value: f"Fold {value} · {pretty_date(by_id.loc[value, 'test_start'])}–{pretty_date(by_id.loc[value, 'test_end'])}")
        variant = st.selectbox("Rule model", list(VARIANT_LABELS), format_func=VARIANT_LABELS.get, key="rule_variant")
        fold = by_id.loc[fold_id]
        cutoff = fold.fit_cutoff
        support = fold.min_support if variant == "TUNED" else 0.02
        confidence = fold.min_confidence if variant == "TUNED" else 0.60
        theta = fold.theta if variant == "TUNED" else 0.005
        sessions = bundle.training_sessions
        sessions = sessions.loc[sessions.fold_id.eq(fold_id) & sessions.variant.eq(variant)]
        if "stage" in sessions:
            sessions = sessions.loc[sessions.stage.isin(["refit", "outer"])]
        st.caption(f"Frozen fit cutoff · {pretty_date(cutoff)} · Later test block · {pretty_date(fold.test_start)}–{pretty_date(fold.test_end)}")
        if pd.notna(fold.fallback) and variant == "TUNED":
            st.info("No eligible configuration passed the validation trade guards. This tuned fold stayed in cash.")
        st.caption(f"Training only · {len(sessions)} transactions · Movement threshold θ {format_metric('net_return', theta)} · Support ≥ {format_metric('net_return', support)} · Confidence ≥ {format_metric('win_rate', confidence)} · Lift > 1")
        if not sessions.empty:
            st.caption(f"Training features · {pretty_date(sessions.feature_session.min())}–{pretty_date(sessions.feature_session.max())} · Outcomes available through {pretty_date(sessions.outcome_available_session.max())}")
    else:
        support, confidence = meta["min_support"], meta["min_confidence"]
        st.caption(f"Training only · {meta['training_transactions']} sessions · Support ≥ {support:.0%} · Confidence ≥ {confidence:.0%} · Lift > 1 · Fit cutoff {pretty_date(meta['fit_cutoff'])}")
    model_rules = filter_rules(bundle.rules, fold_id=fold_id, variant=variant)
    if model_rules.empty:
        st.info("No rules met the saved cutoffs for this model. Its strategy stayed in cash during this test block.")
    target = st.selectbox("Target stock", ["All stocks", *bundle.tickers], key="rule_target")
    direction = st.selectbox("Direction", ["All", "U", "D", "N"], key="rule_direction",
                             format_func=lambda value: {"All": "All directions", "U": "Up (U)", "D": "Down (D)", "N": "Neutral (N)"}[value])
    initial = 0 if pd.isna(confidence) else int(round(confidence * 100))
    confidence_filter = st.slider("Display confidence at least (%)", 0, 100, initial, 5, key="rule_confidence")
    selected = filter_rules(model_rules, target=target, direction=direction, min_confidence=confidence_filter / 100)
    display = selected[["antecedents", "consequent", "support", "confidence", "lift"]].copy()
    display["antecedents"] = display.antecedents.map(pattern_text)
    for name in ("support", "confidence"):
        display[name] = display[name].map(lambda value: format_metric("win_rate", value))
    display = display.rename(columns={"antecedents": "Observed pattern", "consequent": "Next-session target",
                                      "support": "Training support", "confidence": "Training confidence", "lift": "Training lift"})
    st.dataframe(display, hide_index=True, width="stretch")
    if not model_rules.empty and selected.empty:
        st.info("No saved rules match the display filters.")
    st.caption("Filters narrow saved rules. Support, confidence and lift describe the fitting transactions; they are not held-out accuracy.")
    st.markdown("**Key Insight:** A rule is frozen before its test block. Later predictions and trading results appear in Backtest and Asymmetry.")
    with st.expander("Rule provenance"):
        st.write(f"{len(selected)} displayed rules from {len(model_rules)} saved rules for this model.")
        st.caption(f"Frozen data SHA-256 · {meta.get('source_sha256', 'Unavailable')}")
        if not selected.empty:
            st.dataframe(selected.drop(columns=["antecedents"]).reset_index(drop=True), hide_index=True, width="stretch")


def tuning_page(bundle) -> None:
    st.header("Tuning & Drift")
    meta = bundle.manifest
    if not bundle.is_walk_forward:
        st.info("Walk-forward tuning has not run yet. There are no selected-window frequencies to report.")
        st.subheader("Fixed settings in this preview")
        st.table(pd.DataFrame({"Setting": ["Training window", "Movement threshold", "Minimum support", "Minimum confidence", "Fit cutoff"],
                               "Value": ["Expanding history" if meta["window"] == "expanding" else f"{meta['window']} sessions",
                                         f"{meta['theta']:.2%}", f"{meta['min_support']:.2%}", f"{meta['min_confidence']:.2%}", pretty_date(meta["fit_cutoff"])]}))
        return
    chosen, frequencies, fallbacks = tuning_tables(bundle)
    st.markdown("Adaptive forgetting compares 250-session, 500-session and expanding histories on common past validation dates. Each winner is refit, then frozen for the next test block.")
    st.metric("Folds with an eligible winner", str(len(bundle.folds) - fallbacks))
    st.metric("Fallback folds · no eligible configuration", str(fallbacks))
    st.subheader("Chosen settings by outer fold")
    st.dataframe(chosen, hide_index=True, width="stretch")
    st.caption(f"Selection cost · {meta['selection_cost']:.2%} total round trip · Guards · at least {meta.get('min_validation_trades', 'N/A')} validation trades across {meta.get('min_validation_trade_sessions', 'N/A')} sessions. Validation returns select settings; test returns do not.")
    st.subheader("Selected training histories")
    if frequencies.sum():
        st.bar_chart(frequencies, color="#0F766E", height=250, width="stretch", y_label="Winning folds")
    else:
        st.info("No fold had an eligible winner; window frequencies are all zero.")
    st.table(frequencies.to_frame())
    st.caption("Window counts include only folds with an actual eligible winner. Fallback folds stay in cash and have no selected window.")
    st.markdown("**Key Insight:** Changing selected windows investigates adaptive forgetting. It does not establish that concept drift caused the selections or that tuning improves future returns.")
    with st.expander("Inspect saved validation candidates"):
        ids = bundle.folds.fold_id.astype(int).sort_values().tolist()
        fold_id = st.selectbox("Candidate scores for fold", ids, key="tuning_fold")
        candidates = bundle.tuning.loc[bundle.tuning.fold_id.eq(fold_id)].copy()
        st.dataframe(candidates, hide_index=True, width="stretch")
        st.caption("Every candidate score comes from its validation block. Searching a grid can overfit this block; the later outer test assesses the combined tuning procedure.")


def backtest_page(bundle) -> None:
    st.header("Backtest")
    scope = st.selectbox("View", ["Portfolio", *bundle.tickers], key="scope")
    costs = sorted(bundle.metrics.cost.unique().tolist())
    default = costs.index(0.001) if 0.001 in costs else 0
    cost = st.selectbox("Total round-trip cost", costs, index=default, format_func=lambda value: f"{value:.2%}", key="cost")
    chart, metrics, raw = backtest_tables(bundle, scope=scope, cost=cost)
    prefix = "TUNED" if bundle.is_walk_forward else "RULES"
    strategy = f"{prefix}_PORTFOLIO" if scope == "Portfolio" else f"{prefix}:{scope}"
    st.metric("Tuned strategy net return" if bundle.is_walk_forward else "Strategy net return", format_metric("net_return", raw.loc[strategy, "net_return"]))
    st.caption(f"{format_metric('trade_count', raw.loc[strategy, 'trade_count'])} round trips · Max drawdown {format_metric('max_drawdown', raw.loc[strategy, 'max_drawdown'])}")
    st.subheader("Equity across held-out sessions" if bundle.is_walk_forward else "Equity through the holdout")
    colors = ["#0F766E", "#F59E0B", "#4F46E5", "#94A3B8"] if bundle.is_walk_forward else ["#0F766E", "#4F46E5", "#94A3B8"]
    plotted = chart.rename_axis("execution_session").reset_index().melt(
        id_vars="execution_session", var_name="Strategy", value_name="Indexed equity"
    )
    equity_chart = alt.Chart(plotted).mark_line().encode(
        x=alt.X("execution_session:T", title="Execution session",
                axis=alt.Axis(format="%Y-%m", tickCount=4, labelAngle=-30)),
        y=alt.Y("Indexed equity:Q", title="Value of initial 100"),
        color=alt.Color("Strategy:N", scale=alt.Scale(domain=chart.columns.tolist(), range=colors),
                        legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=120)),
        tooltip=[alt.Tooltip("execution_session:T", title="Date", format="%Y-%m-%d"),
                 alt.Tooltip("Strategy:N"), alt.Tooltip("Indexed equity:Q", format=".2f")],
    ).properties(height=300)
    st.altair_chart(equity_chart, width="stretch")
    st.caption("Initial allocation = 100. Points show closing equity on every test session, including cash days. The NIFTY reference is an uncosted price index.")
    st.subheader("Comparison metrics")
    st.dataframe(metrics.reset_index(), hide_index=True, width="stretch", height="content")
    if bundle.is_walk_forward:
        st.info("The cost selector reads saved accounting for the same frozen decisions; it does not retune them. Tuned versus fixed evaluates the full tuning procedure and cannot isolate the contribution of training-window choice.")
    else:
        st.info(f"This is one fixed {bundle.manifest['test_sessions']}-session engineering holdout. Holding cash during a falling month does not demonstrate predictive skill.")
    with st.expander("How to read these results"):
        st.write("Each stock receives 1/N of current equity. U forecasts enter at the next open and exit that close; D, N and unmatched cases stay in cash. Cash earns zero. Fills are idealized and fractional shares are allowed.")
        st.write("Costs are total round-trip fractions of entry notional. Holding controls pay once, half at entry and half at final exit. The equal-initial-weight holding basket is not rebalanced daily. The NIFTY 50 price-index reference omits dividends and implementation costs; adjusted-stock prices are a holding proxy.")
        st.write("CAGR uses elapsed calendar time. Daily Sharpe uses 252 sessions per year, sample standard deviation and a zero risk-free rate. Drawdown includes initial capital. Net win rate and profit factor use the net trade ledger. N/A means undefined; ∞ profit factor means gains with no losses.")
        st.write("Selected present-day companies create survivorship and selection bias. Provider-adjusted history is not a point-in-time archive. These results describe this saved sample and do not establish future performance.")
    st.download_button("Download selected metrics", raw.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="walkforward_metrics.csv" if bundle.is_walk_forward else "holdout_metrics.csv", mime="text/csv")


def asymmetry_page(bundle) -> None:
    st.header("Asymmetry")
    if not bundle.is_walk_forward:
        counts = [int(bundle.predictions.prediction_label.eq(label).sum()) for label in ("U", "D")]
        st.info("No U/D forecasts were produced in this holdout. Precision and lift are undefined." if not sum(counts)
                else "Forecast counts are available; held-out precision and lift have not been computed in this preview.")
        st.table(pd.DataFrame({"Direction": ["Up (U)", "Down (D)"], "Forecasts": counts}))
        return
    variant = st.selectbox("Prediction model", list(VARIANT_LABELS), format_func=VARIANT_LABELS.get, key="asymmetry_variant")
    comparison, lift, counts = asymmetry_tables(bundle, variant=variant)
    st.markdown("One frozen prediction per stock and test session enters this comparison. Down predictions are scored even though the trading policy keeps them in cash.")
    st.subheader("Precision and class base rate")
    st.bar_chart(comparison, stack=False, color=["#0F766E", "#94A3B8"], height=260, width="stretch", y_label="Fraction")
    st.subheader("Held-out lift")
    st.bar_chart(lift, color="#0F766E", height=220, width="stretch", y_label="Correct / expected correct")
    st.dataframe(counts.reset_index(), hide_index=True, width="stretch")
    st.caption("U = up · D = down. N/A means no predictions or a zero denominator; missing bars are not replaced by zero.")
    valid = lift["Held-out lift"].dropna()
    if counts.Predictions.sum() == 0:
        st.info("No U/D forecasts were produced for this saved model. Precision and held-out lift are undefined.")
    elif len(valid) < 2:
        st.info("At least one direction has no defined lift. These saved results do not support a complete U/D comparison.")
    else:
        st.info(f"Saved descriptive lift · U {format_metric('lift', lift.loc['U', 'Held-out lift'])} · D {format_metric('lift', lift.loc['D', 'Held-out lift'])}. This comparison does not establish a universal directional advantage or statistical significance.")
    with st.expander("How the base rate is weighted"):
        st.write("Each fold and stock uses its own test class rate at that fold's frozen threshold. Expected correct counts sum prediction count × class rate. The displayed base rate is weighted by prediction count, and held-out lift is total correct / total expected correct.")
        st.write("Test outcomes are used only after predictions are frozen. Training confidence is a separate statistic and is not held-out precision.")


class LatestRefreshFallback(Exception):
    """A fallback result is displayed but is not saved as a successful fetch."""

    def __init__(self, payload):
        self.payload = payload
        super().__init__(payload["message"])


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def refresh_latest(model_spec: dict, cached_prices: pd.DataFrame) -> dict:
    """Only an explicit button click reaches the provider-facing module."""
    from src.latest import get_latest_signals

    result = get_latest_signals(model_spec, cached_prices, fetch_live=True)
    payload = {"signals": result.signals.copy(), "observation_session": result.observation_session,
               "fit_cutoff": result.fit_cutoff, "status": result.status,
               "message": result.message, "fetched_at_utc": result.fetched_at_utc}
    if result.status != "live":
        raise LatestRefreshFallback(payload)
    return payload


def latest_page(bundle) -> None:
    st.header("Latest Signals")
    if not bundle.is_walk_forward:
        st.info("Latest signals are not enabled in this preview. Daily refresh and cached fallback require the full saved experiment.")
        st.write(f"The available experiment is historical: {pretty_date(bundle.manifest['test_start'])}–{pretty_date(bundle.manifest['test_end'])}. It is not a current market signal.")
        return
    fingerprint = (bundle.manifest["files"]["latest_model.json"], bundle.manifest["files"]["latest_prices.csv"], bundle.manifest["files"]["latest_signals.json"])
    if st.session_state.get("latest_bundle") != fingerprint:
        st.session_state["latest_bundle"] = fingerprint
        st.session_state["latest_payload"] = bundle.latest_signals
    st.markdown("This demonstration applies the final historical fold's **frozen rules** to completed daily observations. Intended execution: **next trading session**.")
    st.caption(f"Frozen model fit cutoff · {pretty_date(bundle.latest_model['fit_cutoff'])} · Movement threshold θ {format_metric('net_return', bundle.latest_model.get('theta'))}")
    if st.button("Refresh completed daily bars", key="refresh_latest", type="primary"):
        try:
            with st.spinner("Requesting recent completed daily bars…"):
                st.session_state["latest_payload"] = refresh_latest(bundle.latest_model, bundle.latest_prices)
        except LatestRefreshFallback as exc:
            st.session_state["latest_payload"] = exc.payload
        except Exception as exc:
            logging.warning("Latest-signal refresh unavailable: %s", exc)
            st.session_state["latest_payload"] = {**bundle.latest_signals, "status": "fallback",
                                                  "message": "Refresh unavailable. Showing the verified frozen cache; observations may be stale."}
    st.caption("Successful refreshes are cached for 15 minutes. A refresh updates observations and signals; the fitted rules stay frozen.")
    payload = st.session_state["latest_payload"]
    status = payload["status"]
    if status == "live":
        st.success(f"Live refresh · {payload['message']}")
    elif status == "fallback":
        st.warning(f"Cached fallback · {payload['message']}")
    else:
        st.info(f"Cached completed bars · {payload['message']} This frozen snapshot may be stale.")
    st.caption(f"Observation session · {pretty_date(payload['observation_session'])} · Rule fitting cutoff · {pretty_date(payload['fit_cutoff'])}")
    fetched = payload.get("fetched_at_utc")
    st.caption(f"Fetched at (UTC) · {fetched or 'Frozen saved cache; no live fetch in this session'}")
    if bundle.latest_model.get("fallback"):
        st.info("The final fold had no eligible tuned configuration. Its frozen model has no rules, so the demonstration remains in cash.")
    signals = pd.DataFrame(payload["signals"])
    display = pd.DataFrame({
        "Stock": signals.ticker,
        "Prediction": signals.prediction_label.map(pattern_text),
        "Action": signals.action,
        "Observed pattern": signals.get("matched_antecedents", pd.Series(index=signals.index, dtype=object)).map(pattern_text),
        "Next-session target": signals.get("consequent", pd.Series(index=signals.index, dtype=object)).map(pattern_text),
        "Training confidence": signals.get("confidence", pd.Series(index=signals.index, dtype=float)).map(lambda v: format_metric("win_rate", v)),
        "Training lift": signals.get("lift", pd.Series(index=signals.index, dtype=float)).map(lambda v: format_metric("lift", v)),
        "Reason": signals.get("reason", pd.Series(index=signals.index, dtype=object)).map(pattern_text),
    })
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption("An unavailable prediction appears as N/A and stays in cash. A newer observation does not make the historical model newer.")
    with st.expander("Completed bars in the frozen cache"):
        st.dataframe(bundle.latest_prices.sort_values(["date", "ticker"]), hide_index=True, width="stretch")
        st.caption("These are the packaged cached bars. Refreshed signals above may use later completed observations.")


def main() -> None:
    st.set_page_config(page_title="Financial Pattern Mining", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")
    st.title("Financial Pattern Mining")
    st.caption("CS G520 · BITS Pilani Hyderabad · Topic 21")
    try:
        bundle = load_bundle(Path(os.environ.get("ADM_RESULTS_DIR", str(default_bundle()))))
    except ResultBundleError as exc:
        logging.warning("Saved experiment unavailable: %s", exc)
        st.error("Saved results are unavailable or failed verification. Ask the project team to refresh the saved experiment.")
        st.stop()
    meta = bundle.manifest
    if bundle.is_walk_forward:
        st.caption(f"Nested walk-forward · {len(bundle.tickers)} stocks · {meta['fold_count']} folds · Test: {pretty_date(meta['test_start'])}–{pretty_date(meta['test_end'])}")
    else:
        st.caption(f"Fixed holdout · {len(bundle.tickers)} stocks · Fit: {pretty_date(meta['fit_cutoff'])} · Test: {pretty_date(meta['test_start'])}–{pretty_date(meta['test_end'])}")
    page = st.selectbox("Page", PAGES, key="page")
    try:
        {"Rules Explorer": rules_page, "Tuning & Drift": tuning_page, "Backtest": backtest_page,
         "Asymmetry": asymmetry_page, "Latest Signals": latest_page}[page](bundle)
    except ResultBundleError as exc:
        logging.warning("Saved scenario unavailable: %s", exc)
        st.error("This saved comparison is unavailable. Ask the project team to refresh the experiment.")
    st.divider()
    st.caption("Tanish Kumar Gupta · Aayush Gupt")
    st.caption("Educational project; not investment advice.")


if __name__ == "__main__":
    main()
