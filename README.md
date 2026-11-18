# Financial Time-Series Pattern Mining

CS G520 Advanced Data Mining, BITS Pilani Hyderabad. Topic 21, Phase 2: **Mining Stock Price Data to Discover Temporal Trends and Market Patterns**.

**Team:** Tanish Kumar Gupta (leader) and Aayush Gupt.

A responsive Streamlit application investigating adaptive training-history selection, automatically tuned association-rule thresholds and trading-cost-aware evaluation. The completed offline study uses ten selected Indian stocks plus NIFTY, a frozen six-year snapshot, nested chronological validation and an untuned control. It does not claim to solve market prediction or concept drift.

## Results

The saved study covers **24 February 2023–17 November 2026: 917 held-out sessions, 44 outer folds and 3,564 candidate scores**. Every fold evaluates the same 81-configuration grid. Portfolio capital compounds across fold boundaries.

| Total round-trip cost | Tuned rules | Fixed rules | Holding basket |
| --- | ---: | ---: | ---: |
| 0.05% | -8.48% | +0.05% | +45.62% |
| 0.10% | -13.30% | +0.04% | +45.57% |
| 0.50% | -43.78% | -0.08% | +45.17% |

The uncosted NIFTY 50 price-index reference returned **+44.32%** over the same interval. At 0.1% cost, tuned rules executed 1,082 round trips; the fixed control executed only three and was largely cash. Holding includes overnight exposure; rule trades are intraday. The tuned procedure did not outperform these controls in this sample. Preserve that negative finding rather than retuning against the test results.

Training histories won 27 folds (250 sessions), 10 folds (500) and seven folds (expanding); all 44 folds had an eligible configuration. This investigates adaptive forgetting, without establishing causal concept drift. Tuned U precision was 38.08% against a weighted base rate of 36.20%; D precision was 42.85% against 41.12%. The corresponding lifts were 1.052 and 1.042: higher raw D precision does not establish a stronger adjusted directional edge.

> **Key Insight:** Strong training rules and more active tuning did not produce an economic advantage here. A properly separated, reproducible negative result is a valid project outcome.

## Application and report

The source and saved display results are published in the [GitHub repository](https://github.com/Tanish520/financial-time-series-mining). The local app is available at **http://127.0.0.1:8501** while its server is running. Hosted deployment status is recorded in [deployment notes](docs/DEPLOYMENT.md).

- [Six-page report](output/pdf/financial_time_series_report.pdf), [Markdown report](output/project_report.md), and exported [figures](output/report_figures).
- [Demo script](docs/DEMO.md), [contributor notes](docs/CONTRIBUTING.md), and [approved design](docs/superpowers/specs/2026-11-18-financial-time-series-mining-design.md).
- [Implementation and completion checklist](docs/superpowers/plans/2026-11-18-complete-project.md); [archived initial checkpoints](docs/ENGINEERING_LOG.md).

| Page | Implemented behavior |
| --- | --- |
| Rules Explorer | Short project summary; final fitted rules by fold/model/stock/direction/confidence; training statistics and fitting dates |
| Tuning & Drift | Actual chosen parameters and validation scores, all candidate logs, winner frequencies and separate fallback counts |
| Backtest | Portfolio/stock and cost selectors; continuous tuned/fixed/holding/index curves; seven metrics and execution assumptions |
| Asymmetry | Held-out U/D precision, prediction-weighted base rates, lift and counts; descriptive interpretation |
| Latest Signals | Precomputed cached signals, final frozen model provenance, explicit daily refresh, 15-minute successful-fetch cache and dated fallback |

Core pages read the verified 15-payload result bundle under app/artifacts/full_walkforward. They do not import research/provider code or contact a market API. Every payload is checked against its manifest before parsing; cached parsing is keyed by verified contents. Missing, inconsistent or corrupted results produce an explicit error. The older Day 1 holdout remains available only through an explicit ADM_RESULTS_DIR override.

## Run locally

Use Python **3.13**, matching the pinned environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
make app
```

Or run Streamlit directly:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

On Windows, replace .venv/bin/python with .venv\Scripts\python.exe and use the direct commands if Make is unavailable. The UI is centered and single-column, with horizontally scrollable data tables for narrow screens. No database, login or API key is required by the app.

## Reproduce and verify

```bash
make test
make reproduce
make report
```

The last full verification passed **273 tests** with warnings treated as errors; dependency checks passed. Tests include leakage perturbation, next-session maturity, exact-threshold rules, finite capital/costs, continuous folds, weighted asymmetry, all app pages, immutable checksums, latest-data boundaries and a hosted entry-point launch from an unrelated working directory. Independent research and app reviews found no remaining material issues after regression fixes.

The research command requires the frozen corrected snapshot under data/raw/full_ten_stocks_corrected. It is included in the self-contained local archive; raw caches stay excluded from Git. A fresh GitHub checkout can display all saved results immediately without raw data. To repeat the study using an existing snapshot:

```bash
.venv/bin/python -m src.experiment --snapshot data/raw/full_ten_stocks_corrected --output data/results/reproduction
```

Output directories are immutable. If reproduction already exists, choose another directory or use `make reproduce RUN_DIR=data/results/another_run`. The experiment never silently downloads or refreshes data. Both complete runs finished with OS networking denied. A second full run reproduced all 15 payload files byte for byte; its manifest differed only in the creation timestamp. File hashes, creator hashes, candidate selections, timing boundaries, ledgers and weighted asymmetry were independently reconciled.

Report generation reads the verified display bundle only and exports a six-page PDF, Markdown and three figures. The PDF was rendered and all six pages inspected. Historical checkpoint instructions remain in the engineering log; they are not the current research evaluation.

Packaging a new verified result bundle is explicit:

```bash
.venv/bin/python -m app.results --source data/results/another_run --output app/artifacts/another_run
```

To inspect that bundle, set ADM_RESULTS_DIR when launching Streamlit. There is no page-load mining or tuning.

## Frozen data and audit

Ten stocks: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, ITC.NS, LT.NS, SUNPHARMA.NS and MARUTI.NS; benchmark: ^NSEI. Provider metadata verified NSE, INR, Asia/Kolkata and the expected instrument identities. This is a convenience sample, not reconstructed historical NIFTY membership.

Accepted coverage: **17 November 2020–17 November 2026**, 1,482 sessions per instrument and 16,302 rows including the index. Adjusted OHLC are used consistently. Missing prices are not filled. Current-day bars are excluded from research downloads; all research uses the frozen snapshot.

Two documented quality decisions:

1. Exclude 15 January 2026 for all instruments because NSE's capital-market [closure circular](https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf) confirms a non-session. Retain original excluded provider rows and checksums.
2. Replace exactly nine flat, zero-volume stock placeholders on 18 March 2025 using primary NSE EQ bars from [18 March](https://archives.nseindia.com/products/content/sec_bhavdata_full_18032025.csv). Scale raw OHLC by the preceding adjusted-close / NSE previous-close ratio; independently check that scale against [19 March](https://archives.nseindia.com/products/content/sec_bhavdata_full_19032025.csv). No corporate actions occurred on the repaired date. Every other price row remains unchanged. The original snapshot, primary CSVs and original/replacement audit rows remain available.

```text
Original full snapshot SHA-256:
4db6f1637529f374f1576a1b46d99d3e3c77e47c099c3f239ae50c04c1de0802
Corrected study snapshot SHA-256:
5286eeaa261ed3abdae66fc105971a2dc78c0e5d134dd761f2bd6fbbe70f0c34
```

The bounded correction is reproducible offline with `python -m src.repair` against the preserved original snapshot and primary audit files. The output must be a new directory. Every primary audit file is checksum-protected on load. A zero-volume LONG execution still aborts a backtest; the accounting engine never selectively drops a trade after observing its outcome.

Historical adjusted prices are not a point-in-time corporate-action archive. Corrected volume is the primary exchange's share count and is used solely as an execution-validity check, not as a mining feature.

## Evaluation protocol

1. **Observed feature:** close[t]/close[t-1]-1. **Target:** next-session close/open-1. U is above theta, D below -theta, N otherwise; equality is neutral. Target information matures at the following close.
2. **Rules:** Apriori and association_rules; one/two observed antecedents, one NEXT_ consequent, total length at most three and training lift > 1. Support/confidence are rebuilt from integer counts to preserve exact threshold boundaries.
3. **Grid:** support {0.01,0.02,0.05}, confidence {0.55,0.60,0.70}, theta {0.0025,0.005,0.01}, windows {250,500,expanding}. Nine low-threshold fits support 81 scored configurations per fold.
4. **Validation:** common preceding 63 executions; fitting history ends before validation. Maximize net portfolio return at fixed 0.1% cost with at least 20 trades across five sessions. Exact score ties use canonical parameter order. No eligible candidate means dated cash decisions.
5. **Outer test:** refit through its cutoff, freeze up to 21 next executions, then advance. Earlier observed outcomes may become later history; current/future outcomes never tune the fold. The last partial block has 14 sessions.
6. **Execution:** highest training confidence resolves matches; conflicting top-confidence classes mean cash. U enters next open and exits that close; D/N/unknown remain cash. Each stock gets 1/N of current capital; unused cash is not redistributed and earns zero.
7. **Controls:** fixed 500/theta0.005/support0.02/confidence0.60 at the same outer cutoffs; per-stock holding; equal-initial-weight holding basket; uncosted NIFTY price index. Standalone stock paths are independent comparisons, not extra portfolio positions.
8. **Costs/metrics:** flat total round-trip fractions of entry notional. Holding pays once, split half entry/half final exit on initial notional. Report net return, inclusive-calendar CAGR, sample daily Sharpe at zero risk-free rate, drawdown including initial capital, net-money profit factor/win rate and trade count. N/A remains undefined; profit factor is infinite only for positive gains without losses.
9. **Asymmetry:** score frozen U/D forecasts against each fold/stock's test class rates at its selected theta. Aggregate expected correct as sum(forecasts × class rate), then lift = total correct/expected. Test base rates are descriptive and never affect selection.

Searching 81 settings can overfit validation. Adaptive windows alone do not demonstrate drift, and tuned-vs-fixed combines window/threshold choices. There is no formal significance claim. Idealized fills omit detailed liquidity, taxes and market impact. The 0.5% stress scenario is not claimed to reproduce Nair's exact transaction-cost convention.

## Latest Signals boundary

Refresh is explicit and applies the final historical TUNED model without retraining or substituting an earlier successful fold. It accepts only aligned completed daily cohorts with a preceding close. Current/future India dates are excluded. Responses older than a validated cache or more than seven calendar days old trigger a dated cached fallback. Stale cache is allowed with visible labeling; invalid cache produces an unavailable state. Successful refreshes cache for 15 minutes.

## Repository map

```text
src/          data, labels, mining, signals, backtest, walkforward, experiment, asymmetry, latest, repair
app/          Streamlit views, read-only adapters and verified saved artifacts
tests/        temporal, accounting, schema, UI, freshness and reproducibility checks
scripts/      report generation and local project archive
output/       report PDF/Markdown and exported figures
docs/         design, implementation logs, deployment and demo instructions
data/         ignored immutable raw/processed/research caches
```

Responsive web delivery is the team's chosen implementation. Instructor acceptance of this format and the exact submission date were not supplied; no institutional approval is asserted.

Educational project; not investment advice.
