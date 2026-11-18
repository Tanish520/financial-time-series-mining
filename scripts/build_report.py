"""Build the report and export figures from a verified saved experiment only."""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.results import load_bundle
from app.view_data import METRIC_LABELS, format_metric

NAMES = {"TUNED_PORTFOLIO": "Tuned rules", "FIXED_PORTFOLIO": "Fixed rules",
         "BUY_HOLD_BASKET": "Holding basket", "NIFTY50_PRICE_INDEX": "NIFTY reference"}
COLORS = ["#0F766E", "#D99320", "#5046AE", "#7C8CA0"]
REFERENCES = [
    ("Srivastava, T., Mullick, I., and Bedi, J. (2024). Association mining based deep learning approach for financial time-series forecasting. Applied Soft Computing 155, 111469.", "https://doi.org/10.1016/j.asoc.2024.111469"),
    ("Nair, B. B., et al. (2015). A stock trading recommender system based on temporal association rule mining. SAGE Open 5(2).", "https://doi.org/10.1177/2158244015579941"),
    ("Nair, B. B., Kumar, P. K. S., Sakthivel, N. R., and Vipin, U. (2017). Clustering stock price time series data to generate stock trading recommendations: An empirical study. Expert Systems with Applications 70, 20-36.", "https://doi.org/10.1016/j.eswa.2016.11.002"),
    ("Tusien, E., Kwasniewska, A., and Weichbroth, P. (2025). Mining frequent sequences with time constraints from high-frequency data. International Journal of Financial Studies 13(2), 55.", "https://doi.org/10.3390/ijfs13020055"),
    ("Tsinaslanidis, P. E. (2018). Subsequence dynamic time warping for charting: Bullish and bearish class predictions for NYSE stocks. Expert Systems with Applications 94, 193-204.", "https://doi.org/10.1016/j.eswa.2017.10.055"),
]


def fmt(value, decimals=2):
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.{decimals}f}"


def figures(bundle, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#172B4D", "text.color": "#172B4D"})
    selected = bundle.equity.loc[bundle.equity.cost.eq(.001)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (name, label), color in zip(NAMES.items(), COLORS):
        path = selected.loc[selected.strategy.eq(name)].sort_values("execution_session")
        ax.plot(path.execution_session, path.equity / bundle.manifest["initial_capital"] * 100,
                label=label, color=color, linewidth=1.8)
    ax.set_ylabel("Value of initial 100")
    ax.set_xlabel("Held-out execution session")
    ax.set_title("Continuous held-out equity at 0.1% total round-trip cost", loc="left", fontsize=11)
    ax.grid(alpha=.16)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "equity.png", dpi=240)
    plt.close(fig)

    asym = bundle.asymmetry.loc[bundle.asymmetry.variant.eq("TUNED")].set_index("direction").reindex(["U", "D"])
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
    axes[0].bar(x-.16, asym.precision.to_numpy(float), width=.32, color=COLORS[0], label="Precision")
    axes[0].bar(x+.16, asym.base_rate.to_numpy(float), width=.32, color=COLORS[3], label="Weighted base rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Held-out directional precision", loc="left", fontsize=11)
    axes[1].bar(x, asym.held_out_lift.to_numpy(float), width=.48, color=COLORS[0])
    axes[1].axhline(1, color=COLORS[3], linestyle="--", linewidth=1)
    axes[1].set_ylabel("Correct / expected correct")
    axes[1].set_title("Held-out lift", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xticks(x, ["Up (U)", "Down (D)"])
        ax.grid(axis="y", alpha=.16)
    for i, row in enumerate(asym.itertuples()):
        if pd.isna(row.precision): axes[0].text(i, .04, "N/A", ha="center")
        if pd.isna(row.held_out_lift): axes[1].text(i, .04, "N/A", ha="center")
    fig.tight_layout()
    fig.savefig(directory / "asymmetry.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    order = {"250": 0, "500": 1, "expanding": 2}
    for window, color in zip(order, COLORS):
        subset = bundle.folds.loc[bundle.folds.selected_window.eq(window).fillna(False)]
        ax.scatter(subset.fit_cutoff, [order[window]]*len(subset), color=color, s=38)
    missing = bundle.folds.loc[bundle.folds.selected_window.isna()]
    if len(missing): ax.scatter(missing.fit_cutoff, [-1]*len(missing), color="#B45353", marker="x", s=40)
    ax.set_yticks([-1, 0, 1, 2], ["Cash fallback", "250 sessions", "500 sessions", "Expanding"])
    ax.set_ylim(-1.4, 2.4)
    ax.set_xlabel("Outer fitting cutoff")
    ax.set_title("History selected by past validation return", loc="left", fontsize=11)
    ax.grid(alpha=.16)
    fig.tight_layout()
    fig.savefig(directory / "window_selection.png", dpi=240)
    plt.close(fig)


def content(bundle):
    m = bundle.manifest
    metrics = bundle.metrics.loc[bundle.metrics.cost.eq(.001)].set_index("strategy")
    tuned, fixed, holding = (metrics.loc[name] for name in ("TUNED_PORTFOLIO", "FIXED_PORTFOLIO", "BUY_HOLD_BASKET"))
    difference = float(tuned.net_return - fixed.net_return)
    eligible = int(bundle.folds.selected_window.notna().sum())
    frequencies = bundle.folds.selected_window.dropna().value_counts().to_dict()
    asym = bundle.asymmetry.loc[bundle.asymmetry.variant.eq("TUNED")].set_index("direction")
    up, down = asym.loc["U"], asym.loc["D"]
    if pd.isna(up.held_out_lift) or pd.isna(down.held_out_lift):
        direction_text = "At least one direction has undefined lift, so a complete bearish/bullish comparison is unavailable."
    else:
        comparison = "higher" if down.held_out_lift > up.held_out_lift else "lower" if down.held_out_lift < up.held_out_lift else "equal"
        direction_text = f"Down-prediction lift is {comparison} than up-prediction lift in this sample ({fmt(down.held_out_lift)} versus {fmt(up.held_out_lift)}). This is descriptive evidence, without a significance test or a claim of a universal bearish advantage."
    trades = bundle.trades.loc[bundle.trades.strategy.eq("TUNED_PORTFOLIO") & bundle.trades.cost.eq(.001)].sort_values(["entry_session", "ticker"])
    if trades.empty:
        example = "No tuned trade was executed. A trade-profit example is therefore unavailable; zero-trade metrics remain explicitly undefined where appropriate."
    else:
        t = trades.iloc[0]
        example = (f"First chronological tuned trade: {t.ticker}, {pd.Timestamp(t.entry_session):%d %B %Y}. "
                   f"Entry notional INR {t.entry_notional:,.2f}; open {t.entry_price:,.2f}; close {t.exit_price:,.2f}. "
                   f"Gross PnL INR {t.gross_pnl:,.2f}; cost {0.001:.1%} x notional = INR {t.transaction_cost:,.2f}; "
                   f"net PnL INR {t.net_pnl:,.2f}. These are saved ledger values, not illustrative invented returns.")
    return {
        "abstract": (f"We investigate adaptive history length, automated association-rule thresholds and trading-cost-aware evaluation "
                     f"for {len(bundle.tickers)} selected Indian equities. The frozen snapshot has {m['source_sessions']:,} sessions per instrument. "
                     f"Nested validation scores {len(bundle.tuning):,} configurations across {m['fold_count']} outer folds, covering {m['test_sessions']} held-out sessions. "
                     f"At 0.1% total round-trip cost, tuned rules return {format_metric('net_return', tuned.net_return)}, fixed rules {format_metric('net_return', fixed.net_return)}, "
                     f"and the equal-initial-weight holding basket {format_metric('net_return', holding.net_return)}. "
                     "The application presents frozen evidence and preserves negative results and cash fallbacks."),
        "comparison": (f"Tuned-minus-fixed return is {difference*100:+.2f} percentage points at 0.1% cost. "
                       "This compares the complete tuning procedure; it cannot attribute a difference specifically to window choice without an ablation. "
                       "Holding controls retain overnight exposure while the rule strategy is intraday, so exposure and opportunity differ."),
        "windows": (f"{eligible} of {m['fold_count']} folds selected an eligible configuration; {m['fold_count']-eligible} used the explicit cash fallback. "
                    f"Winning histories: 250={int(frequencies.get('250',0))}, 500={int(frequencies.get('500',0))}, expanding={int(frequencies.get('expanding',0))}. "
                    "Window changes investigate adaptive forgetting; they do not identify a causal regime change or prove concept drift was detected."),
        "asymmetry": direction_text,
        "example": example,
        "data": (f"Source coverage: {m['source_start']} to {m['source_end']}; 10 selected stocks plus the NIFTY 50 price index. "
                 "The universe is a present-day convenience sample rather than reconstructed historical index membership. "
                 "Prices are provider-adjusted OHLC; cash dividends and split information are retained. No prices are forward-filled."),
        "correction": ("Data audit found nine flat, zero-volume stock placeholders on 18 March 2025. Primary NSE EQ OHLCV replaced only those rows in a new immutable snapshot. "
                       "The previous adjusted-close / NSE previous-close ratio sets the price scale; the following session independently verifies it within 1e-5. "
                       "All original and replacement values and primary-file hashes are retained. The separate verified 15 January 2026 exchange closure is excluded for all instruments. "
                       "This uses a static adjusted-price archive and does not establish point-in-time corporate-action accuracy."),
    }


def tables(bundle):
    m = bundle.manifest
    metrics = bundle.metrics.loc[bundle.metrics.cost.eq(.001)].set_index("strategy")
    metric_rows = [["Metric", *NAMES.values()]]
    for key, label in METRIC_LABELS.items():
        metric_rows.append([label, *[format_metric(key, metrics.loc[name, key]) for name in NAMES]])
    cost_rows = [["Round-trip cost", "Tuned return", "Fixed return", "Holding basket"]]
    for cost in sorted(bundle.metrics.cost.unique()):
        paths = bundle.metrics.loc[bundle.metrics.cost.eq(cost)].set_index("strategy")
        cost_rows.append([f"{cost:.2%}", *[format_metric("net_return", paths.loc[name, "net_return"]) for name in ("TUNED_PORTFOLIO", "FIXED_PORTFOLIO", "BUY_HOLD_BASKET")]])
    asym = bundle.asymmetry.loc[bundle.asymmetry.variant.eq("TUNED")].set_index("direction")
    asym_rows = [["Direction", "Forecasts", "Correct", "Precision", "Base rate", "Lift"]]
    for direction in ("U", "D"):
        row = asym.loc[direction]
        asym_rows.append([direction, str(int(row.prediction_count)), str(int(row.correct_count)),
                          format_metric("win_rate", row.precision), format_metric("win_rate", row.base_rate), fmt(row.held_out_lift)])
    return metric_rows, cost_rows, asym_rows


def markdown_report(bundle, output, text, data_tables):
    def md_table(rows):
        return "\n".join(["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"]*len(rows[0])) + " |",
                          *["| " + " | ".join(row) + " |" for row in rows[1:]]])
    metrics, costs, asym = data_tables
    m = bundle.manifest
    lines = ["# Mining Stock Price Data to Discover Temporal Trends and Market Patterns", "",
             "CS G520 Advanced Data Mining - BITS Pilani Hyderabad", "",
             "Tanish Kumar Gupta (team leader) and Aayush Gupt", "", "## Abstract", "", text["abstract"], "",
             "## Motivation and contribution", "",
             "The Phase 1 survey motivates G1 (adaptive forgetting), G2 (manual thresholds) and G3 (economic validation). This implementation addresses or investigates these gaps; it does not claim to solve market prediction. Srivastava et al. (2024) motivate inter-stock association mining, Nair et al. (2015) temporal ARM and Nair et al. (2017) alternative recommendation configurations. Tusien et al. (2025) and Tsinaslanidis (2018) motivate a descriptive U/D comparison. This is a methodological comparison, not a direct numerical replication.", "",
             "## Data and quality audit", "", text["data"], "", ", ".join(bundle.tickers), "", text["correction"], "",
             "## Method", "",
             "Feature at close t: close[t]/close[t-1]-1. Target: close[t+1]/open[t+1]-1. U is above theta, D below -theta, and N otherwise; threshold equality is N. Targets become training information only after execution-session close. Each transaction has one feature and one NEXT_ outcome per stock. Apriori itemsets have length at most three; eligible rules have one/two observed antecedents, exactly one NEXT_ consequent and training lift > 1.", "",
             "Grid: windows {250,500,expanding}; theta {0.25%,0.5%,1%}; support {1%,2%,5%}; confidence {55%,60%,70%}. All 81 configurations face the same preceding 63 validation executions. Selection maximizes net validation portfolio return at 0.1% cost, requiring 20 trades across five sessions. Exact score ties use canonical parameter ordering. Winners refit through the outer cutoff and freeze for the next 21 executions; the final partial block is retained. No eligible configuration means cash. Fixed control: 500 / theta 0.5% / support 2% / confidence 60%.", "",
             "U signals buy at the next open and sell that close; D/N/unknown stay in cash. Highest training confidence resolves matches; class conflicts at identical top confidence mean cash. Each stock has 1/N of current pre-session equity; unused allocations remain cash. Full-run capital compounds across fold boundaries. Every cost scenario uses the same frozen decisions. Costs are flat total round-trip fractions of entry notional, not broker-specific tax/fee schedules. The 0.5% case is a stress scenario; exact replication of Nair's fee convention is not claimed.", "",
             "## Results", "", f"Held-out interval: {m['test_start']} to {m['test_end']}; {m['test_sessions']} sessions, {m['fold_count']} folds. NIFTY is an uncosted price-index reference.", "", md_table(costs), "", md_table(metrics), "", text["comparison"], "",
             "![Continuous held-out equity](report_figures/equity.png)", "", "## Tuning and drift", "", text["windows"], "", "![Selected fitting history](report_figures/window_selection.png)", "",
             "## Held-out asymmetry", "", md_table(asym), "", text["asymmetry"], "",
             "Within each fold/stock, estimate class base rate using its selected theta. Sum forecast_count x group_base_rate to obtain expected correct counts. Aggregated precision = correct/forecasts; weighted base rate = expected/forecasts; lift = correct/expected. All realized outcomes are read after predictions are frozen. These base rates never tune the model.", "", "![Held-out directional comparison](report_figures/asymmetry.png)", "",
             "## Worked ledger example", "", text["example"], "", "## Validation and limitations", "",
             "Tests cover future-data perturbation, mature outcome cutoffs, NEXT_ direction guards, simultaneous capital budgets, overnight-gap exclusion, round-trip costs, first-loss drawdown, continuous multi-fold equity, exact objective ties, final partial blocks, immutable artifacts, weighted asymmetry, completed latest cohorts and offline app behavior. The production experiment runs with networking denied. App pages read checksummed precomputed artifacts; only an explicit Latest Signals refresh contacts Yahoo.", "",
             "Current-stock selection causes survivorship/selection bias. Provider-adjusted prices are not point-in-time. Idealized fills omit detailed liquidity, market impact, taxes and execution queues. Searching 81 settings can overfit validation. Stocks and overlapping historical windows are dependent; no significance or universal directional edge is claimed. Tuning-vs-fixed combines G1/G2 and is not an ablation. A negative return or failure to beat holding is a valid finding.", "",
             "Metrics: net return, calendar CAGR (inclusive dates, 365.25 days/year), sample daily Sharpe (252, risk-free zero), drawdown including initial capital, profit factor and win rate from net currency PnL, and trade count. Undefined metrics are N/A; positive profit factor without losses is infinite. Holding controls pay cost once, half entry/half final exit on initial notional, with no basket rebalancing.", "",
             "## Reproduction and application", "", "```bash\nmake test\nmake reproduce\nmake report\nmake app\n```", "",
             "Research reproduction requires the frozen corrected snapshot under data/raw/full_ten_stocks_corrected. The self-contained local archive includes it. A fresh GitHub checkout can display the packaged results without raw data. See docs/DEPLOYMENT.md and docs/DEMO.md. Live deployment status and exact verified test count are recorded in README.md; account acceptance is not inferred.", "",
             f"Frozen source SHA-256: `{m['source_sha256']}`", "", "## References", ""]
    for i, (citation, url) in enumerate(REFERENCES, 1): lines.append(f"{i}. {citation} [{url}]({url})")
    lines += ["", "Primary data audit: [NSE 18 March 2025](https://archives.nseindia.com/products/content/sec_bhavdata_full_18032025.csv), [NSE 19 March 2025](https://archives.nseindia.com/products/content/sec_bhavdata_full_19032025.csv), [NSE closure circular](https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf).", "",
              "Software: [mlxtend Apriori](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/), [yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), [Streamlit](https://docs.streamlit.io/).", "",
              "> **Key Insight:** Parameters are chosen on past validation data; later test outcomes judge the frozen procedure. Attractive training rules and a changing window alone do not establish an economic advantage."]
    (output / "project_report.md").write_text("\n".join(lines) + "\n")


def pdf_report(bundle, output, text, data_tables):
    pdfmetrics.registerFont(TTFont("DejaVu", font_manager.findfont("DejaVu Sans")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("ReportBody", fontName="DejaVu", fontSize=9, leading=12.8, spaceAfter=8, textColor=colors.HexColor("#172B4D"))
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=7.7, leading=10.5, spaceAfter=5)
    heading = ParagraphStyle("ReportHeading", parent=body, fontName="DejaVuBold", fontSize=18, leading=23, spaceBefore=2, spaceAfter=12)
    sub = ParagraphStyle("ReportSub", parent=body, fontName="DejaVuBold", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=7)
    title = ParagraphStyle("ReportTitle", parent=heading, fontSize=25, leading=31, spaceAfter=13)
    header_cell = ParagraphStyle("HeaderCell", parent=small, fontName="DejaVuBold", textColor=colors.white, spaceAfter=0)
    cell = ParagraphStyle("Cell", parent=small, spaceAfter=0)
    story = []
    def p(value, style=body): return Paragraph(escape(str(value)), style)
    def add(value, style=body): story.append(p(value, style))
    def table(rows, widths):
        rendered = [[p(value, header_cell if i == 0 else cell) for value in row] for i, row in enumerate(rows)]
        item = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#0F766E")),
                                  ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#F2F6F8"),colors.white]),
                                  ("VALIGN", (0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),6),
                                  ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),6),
                                  ("BOTTOMPADDING",(0,0),(-1,-1),6), ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#0F766E"))]))
        story.extend([item, Spacer(1,10)])
    def figure(name, height): story.extend([Image(str(output / "report_figures" / name), width=477, height=height), Spacer(1,8)])
    def page(): story.append(PageBreak())
    m = bundle.manifest
    add("PHASE 2 / TOPIC 21", sub)
    add("Mining stock price data", title)
    add("Temporal trends and market patterns", heading)
    add("CS G520 Advanced Data Mining | BITS Pilani Hyderabad", sub)
    add("Tanish Kumar Gupta (team leader) and Aayush Gupt")
    add(datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y"), small)
    add("Abstract", sub); add(text["abstract"])
    table([["Gap investigated", "Implementation"],
           ["G1: changing relationships", "Validation chooses 250, 500 or expanding fitting history."],
           ["G2: manual cutoffs", "Support, confidence and movement threshold tuned within past data."],
           ["G3: economic validation", "Three modeled costs, continuous equity and aligned holding controls."]], [149,328])
    add("Data and scope", sub); add(text["data"])
    add(", ".join(bundle.tickers), small)
    add("Key Insight: This application investigates the gaps. It does not claim to solve market prediction or concept drift.", sub)
    page()
    add("Method and temporal controls", heading)
    for value in [
        "1. At close t, label close[t]/close[t-1]-1 as U, D or N. The next target is close[t+1]/open[t+1]-1 and matures only after that execution closes. Equality at the threshold is neutral.",
        "2. Apriori uses daily all-stock transactions. Keep one/two observed antecedents and one NEXT_ consequent, at most three total items and training lift > 1. NEXT_ items never enter predictors.",
        "3. Every candidate fits before the same 63 validation executions. Optimize validation net return at 0.1% cost, requiring 20 trades on five sessions. Exact ties use canonical parameter order. No eligible candidate means cash.",
        "4. Refit the winner through the outer cutoff, freeze the next 21 executions, then repeat. Prior test outcomes may enter later history once observed. Keep the final partial block; never revise earlier decisions.",
        "5. Highest training confidence selects a matched class. Conflicting equal-confidence classes mean cash. U buys at next open and exits that close; D/N/unknown hold cash. Each stock receives 1/N of current equity; capital persists across folds.",
    ]: add(value)
    table([["Parameter", "Frozen candidate values"], ["Fitting history", "250 / 500 / expanding"],
           ["Movement threshold", "0.25% / 0.5% / 1%"], ["Support / confidence", "1%, 2%, 5% / 55%, 60%, 70%"],
           ["Reporting costs", "0.05%, 0.1%, 0.5% total round trip; same signals"],
           ["Fixed control", "500 sessions; theta 0.5%; support 2%; confidence 60%"]], [149,328])
    add("A saved ledger example", sub); add(text["example"])
    page()
    add("Held-out economic results", heading)
    add(f"{m['test_start']} to {m['test_end']} | {m['test_sessions']} sessions | {m['fold_count']} outer folds")
    table(data_tables[1], [119,119,119,120])
    figure("equity.png", 250)
    table(data_tables[0], [117,90,90,90,90])
    add(text["comparison"])
    add("NIFTY is an uncosted price-index reference. Stock holding paths are adjusted-price proxies. Costs are flat scenario assumptions; the 0.5% case is not claimed to replicate Nair's exact fee convention.", small)
    page()
    add("Tuning, forgetting and data quality", heading)
    add(text["windows"])
    figure("window_selection.png", 210)
    add("Why validation still needs a later test", sub)
    add("Searching 81 configurations can overfit the 63-session validation block. Outer performance evaluates the entire selection process. An eligible negative-return winner is retained if it is the best eligible candidate; settings are not changed after inspecting test losses.")
    add("Primary-source price audit", sub); add(text["correction"])
    add("Data checksum", sub); add(m["source_sha256"], small)
    add("The manifests retain the uncorrected snapshot checksum, all nine original/replacement rows, primary-file hashes, creator hashes and package versions. Source caches and prior checkpoint results are preserved.")
    add("Key Insight: A data correction needs independent evidence and a new snapshot. It must not become an outcome-dependent trade filter.", sub)
    page()
    add("Directional asymmetry", heading)
    add("U and D predictions are compared against the base rate for their own fold and stock. D predictions remain recorded even though the trading policy does not short.")
    figure("asymmetry.png", 221)
    table(data_tables[2], [62,75,70,90,90,90])
    add(text["asymmetry"])
    add("Weighted base-rate calculation", sub)
    add("For group g = fold/stock, expected correct = forecast_count[g] x test_class_rate[g]. Sum these expectations across groups. Precision = total correct / total forecasts; weighted base rate = expected / forecasts; held-out lift = total correct / expected. A missing or zero denominator stays N/A.")
    add("Test base rates are descriptive after-the-fact quantities. They never select a model, confidence cutoff or trading action. They also prevent different fold thresholds from being compared as though their class prevalences were identical.")
    add("Literature context", sub)
    add("Tusien et al. (2025) and Tsinaslanidis (2018) motivated this observation to test. Their markets, sampling and methods differ, so the chart is not a direct replication. Srivastava et al. (2024) and Nair et al. (2015, 2017) provide association/recommendation context rather than comparable return benchmarks.")
    page()
    add("Verification, limits and reproduction", heading)
    add("Checks cover future-price perturbation, mature outcomes, NEXT_ guards, exact ties, simultaneous capital budgets, round-trip costs, overnight-gap exclusion, first-loss drawdown, continuous capital, final partial folds, immutable hashes, weighted base rates, completed latest cohorts and offline app operation.")
    add("Limits: selected current stocks create survivorship/selection bias; adjusted history is not point-in-time. Idealized fractional fills omit detailed liquidity, taxes and market impact. Cross-stock and temporal dependence preclude a significance claim here. Tuned-versus-fixed is a combined G1/G2 comparison, not an ablation. Weak or negative returns remain valid findings.")
    add("The app reads precomputed results. Only an explicit Latest Signals refresh contacts Yahoo; successful refreshes cache for 15 minutes and failures show dated cached evidence. A fresh observation does not retrain the final historical model.")
    add("Reproduce: make test; make reproduce; make report; make app. Research requires the corrected frozen snapshot included in the local project archive. GitHub display works without raw data. README records exact verification and deployment status; docs/DEMO.md gives the demonstration sequence.")
    add("References", sub)
    for index, (citation, url) in enumerate(REFERENCES, 1):
        add(f"[{index}] {citation} {url}", small)
    add("Data: Yahoo via yfinance; NSE security bhavcopies 18/19 March 2025 and closure circular CMTR72260. Software references: rasbt.github.io/mlxtend; ranaroussi.github.io/yfinance; docs.streamlit.io. Detailed links appear in the Markdown report.", small)
    add("Educational project; not investment advice.", sub)
    directory = output / "pdf"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "financial_time_series_report.pdf"
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D5E1E7")); canvas.line(59, 39, A4[0]-59, 39)
        canvas.setFont("DejaVu", 7); canvas.setFillColor(colors.HexColor("#607384"))
        canvas.drawString(59, 26, "CS G520 | Topic 21 | Frozen walk-forward experiment")
        canvas.drawRightString(A4[0]-59, 26, str(doc.page))
    SimpleDocTemplate(str(target), pagesize=A4, rightMargin=59, leftMargin=59,
                      topMargin=47, bottomMargin=50, title="Mining Stock Price Data - Phase 2",
                      author="Tanish Kumar Gupta and Aayush Gupt").build(story, onFirstPage=footer, onLaterPages=footer)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "app/artifacts/full_walkforward")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    bundle = load_bundle(args.results)
    if not bundle.is_walk_forward: raise SystemExit("Report requires the completed walk-forward bundle.")
    args.output.mkdir(parents=True, exist_ok=True)
    figures(bundle, args.output / "report_figures")
    text, data_tables = content(bundle), tables(bundle)
    markdown_report(bundle, args.output, text, data_tables)
    target = pdf_report(bundle, args.output, text, data_tables)
    print(json.dumps({"report": str(target), "markdown": str(args.output / 'project_report.md'),
                      "source_sha256": bundle.manifest['source_sha256']}))


if __name__ == "__main__":
    main()
