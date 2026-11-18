# Financial Time-Series Pattern Mining

CS G520 Advanced Data Mining, BITS Pilani Hyderabad. Team: Tanish Kumar Gupta and Aayush Gupt.

The project investigates adaptive fitting-window selection, automatic association-rule thresholds and trading-cost-aware validation. It uses daily stock movements to mine interpretable rules for the following session. It does not claim to solve market prediction or concept drift.

## Current checkpoint: Day 2, Step 1

Implemented: the frozen data snapshot, causal labels, guarded Apriori, features-only matching, finite-capital execution, cost-aware holdout, five-view Streamlit preview and the first nested walk-forward fold. The new fold scores all 81 configurations on common past-only validation dates and explicitly stays in cash when none passes the predeclared trade-count guard. The complete rolling experiment is next; one inspected fold is an engineering checkpoint, not the final research evaluation.

The approved [design](docs/superpowers/specs/2026-11-18-financial-time-series-mining-design.md) explains the research protocol. The [current checklist](docs/superpowers/plans/2026-11-18-day2-first-fold.md) covers the inspected fold. Earlier checklists record the [app](docs/superpowers/plans/2026-11-18-day1-streamlit.md), [backtesting](docs/superpowers/plans/2026-11-18-day1-backtest.md), [mining](docs/superpowers/plans/2026-11-18-day1-mining.md), [labels](docs/superpowers/plans/2026-11-18-day1-labels.md) and the [data snapshot](docs/superpowers/plans/2026-11-18-day1-data-snapshot.md).

## Setup

Use Python 3.13, matching the environment used for this checkpoint. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On the project owner's Mac, the Python executable is /opt/homebrew/bin/python3.13. On Windows, use .venv\Scripts\python.exe in place of .venv/bin/python.

requirements.txt pins the installed data, mining, Streamlit and testing packages, including transitive dependencies. mlxtend 0.25.0 and Streamlit 1.67.0 were installed using the preceding pins as constraints; previous package versions were preserved.

## Open the local app

The preview is running on this workstation at **http://127.0.0.1:8501**. To start it after it stops, run from the repository root:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

This is a local preview, not a published app. Binding to loopback keeps it on this workstation; testing a physical phone will use the later hosted deployment. The current layout uses centered, single-column native Streamlit controls.

| View | Current behavior |
| --- | --- |
| Rules Explorer | Project summary, fixed-fit dates/thresholds, target/direction/confidence display filters, and the genuine zero-rule state |
| Tuning & Drift | Actual fixed settings with an explicit notice that walk-forward selection has not run; no invented winner frequencies |
| Backtest | Portfolio or stock view, all three cost scenarios, saved equity curves, seven metrics and cost/fill assumptions |
| Asymmetry | Actual zero U/D forecast counts; precision/lift remain undefined and no performance chart is invented |
| Latest Signals | Explicitly not enabled; the displayed experiment is historical, with daily refresh/cached fallback deferred |

The Backtest cost selector changes the selected saved accounting scenario, not the model. For example, at 0.10% the holding basket shows -1.37%, while at 0.50% it shows -1.77%; the cash strategy stays at 0.00%. NIFTY remains an uncosted price-index reference. Undefined values display N/A; positive infinite profit factor displays the infinity symbol.

> **Key Insight:** A display filter or cost selector must not secretly retune a model. The app presents the verified saved experiment and clearly identifies work that has not run yet.

### App data and fresh checkouts

The app reads the versioned eight-file bundle in app/artifacts/day1_holdout, approximately 90 KB. It contains only the derived holdout files and their original manifest, copied byte for byte. A fresh checkout can display these results without downloading raw prices or importing the research/provider modules. The app verifies file checksums on each rerun and parses the verified bytes; missing or corrupt results produce an explicit error instead of fallback data.

The research caches under data/ stay local and ignored by Git. Updating the display bundle is an explicit operation, separate from running the app:

```bash
.venv/bin/python -m app.results --source data/results/day1_holdout_fixed_2022_12_30 --output app/artifacts/another_holdout
```

The packaging command verifies the source and refuses to overwrite an existing bundle. To preview a different compatible fixed-holdout bundle, set ADM_RESULTS_DIR to its directory when launching Streamlit. Walk-forward bundle support will be added with that evaluation stage.

### App checks and verification limit

```bash
.venv/bin/python -m pytest tests/test_app.py -q -W error
.venv/bin/python -m pytest -q -W error
.venv/bin/python -m pip check
```

Observed at the app checkpoint: 22 app tests passed and dependency checks passed. The full suite now has 166 passing tests including the first-fold checks. AppTest exercises navigation, filters, scope/cost selection, formatting, empty/pending states and missing-result handling. The app tests also passed with OS networking denied; research and provider imports are blocked in a separate check. Both the local root and health endpoint returned HTTP 200. A read-only app review found no material issues and confirmed all 12 scope/cost combinations select matching saved curves and metrics.

Visual desktop/phone QA remains **unverified** in this session: no browser provider was available, and native Chrome/Brave access returned cgWindowNotFound. No screenshot or viewport success is claimed. The local URL is available for manual review; confirm narrow-screen layout before final deployment. The preview server remains available during the ongoing project and can be stopped from its terminal with Ctrl+C.

Streamlit references: [AppTest](https://docs.streamlit.io/develop/concepts/app-testing/get-started), [dataframe API](https://docs.streamlit.io/develop/api-reference/data/st.dataframe).

## Check the existing snapshot without downloading

The working directory already contains the initial snapshot. This command does not import yfinance or contact Yahoo:

```bash
.venv/bin/python -m src.data check --snapshot data/raw/day1_three_stocks
```

Observed output from the accepted snapshot:

```text
status: verified_offline
first_session: 2020-11-17
last_session: 2026-11-17
rows: 5928
sessions per ticker: 1482
tickers: HDFCBANK.NS, RELIANCE.NS, TCS.NS, ^NSEI
sha256: 668376329abdc570b4724454c921bd0e712b359af18692ec36cb0e30303ae601
```

The real CLI prints JSON and also lists the documented non-session exclusion. It verifies both accepted prices and the non-session audit file, then reruns price/calendar checks. The command was also executed successfully inside a macOS process sandbox with all networking denied, and separately with yfinance imports blocked.

## Initial download on a fresh checkout

The generated data are intentionally excluded from Git. On a fresh checkout, run this explicit network command once:

```bash
.venv/bin/python -m src.data download --start 2020-11-17 --as-of 2026-11-17 --output data/raw/day1_three_stocks
```

The command refuses to overwrite an existing snapshot. Use check on the current machine. A deliberate later download must use a different output directory and creates a separate manifest; never refresh the experiment's inputs silently.

The default stock list is RELIANCE.NS, TCS.NS and HDFCBANK.NS; ^NSEI is always included as the benchmark. A separate one-time Yahoo history-metadata check returned the following identities, all on NSE, denominated in INR, with Asia/Kolkata as the exchange timezone:

| Symbol | Provider identity | Type |
| --- | --- | --- |
| RELIANCE.NS | Reliance Industries Limited | EQUITY |
| TCS.NS | Tata Consultancy Services Limited | EQUITY |
| HDFCBANK.NS | HDFC Bank Limited | EQUITY |
| ^NSEI | NIFTY 50 | INDEX |

The observed metadata and verification timestamp are saved in data/raw/instrument-verification.json, separately from the immutable price snapshot. This verifies current instrument identity, not point-in-time NIFTY membership. No metadata request is needed for offline analysis.

The requested as-of date must be before today's date in Asia/Kolkata, conservatively excluding the entire current session even after its close. Yahoo's end bound is exclusive, so the request uses 2026-11-18 to include 2026-11-17.

Yahoo can revise historical adjusted prices. A later fresh download is therefore not guaranteed to reproduce the original checksum. Exact experiment reproducibility requires the original frozen snapshot together with its manifest and the pinned environment. The local cached files, not a future provider response, are the research inputs.

## Saved files and data policy

```text
data/raw/day1_three_stocks/
    prices.csv                   # accepted trading sessions
    manifest.json                # request, settings, versions, coverage, checksums
    excluded_non_sessions.csv    # original observations for the verified holiday
```

CSV schema:

```text
date,ticker,open,high,low,close,volume,dividends,stock_splits
```

- Daily OHLC prices are consistently adjusted: auto_adjust=True, back_adjust=False. Corporate actions are requested; unavailable optional action fields remain missing.
- Automatic price repair is disabled. Missing, infinite, nonpositive or impossible OHLC observations fail validation; prices are never forward-filled.
- Stock and benchmark session dates must agree. Missing trading observations are not replaced with zero returns or N labels.
- Volume must be finite and nonnegative. Zero volume is permitted, including index volume; it is not independently proof of an executable market fill.
- At least 1,200 aligned sessions are required for an accepted research download. A last observation more than seven calendar days before as-of is rejected as stale.
- The manifest records price SHA-256, actual coverage, normalized request, explicit provider settings, creator-code checksum and Python/package versions.

### Diagnosed holiday placeholder

The provider response included 15 January 2026 with three flat, zero-volume stock bars and an empty NIFTY bar. NSE officially declared that date a capital-market trading holiday in [circular NSE/CMTR/72260](https://nsearchives.nseindia.com/content/circulars/CMTR72260.pdf), dated 12 January 2026.

The downloader removes that explicitly verified non-session from the analytical calendar for every ticker. It retains the four original rows in excluded_non_sessions.csv, records the source and reason in manifest.json, and checksums the audit file. No arbitrary invalid trading session receives this treatment. Keeping the placeholder would invent a market session and could distort later returns, labels and fold lengths.

## Run the checks

```bash
.venv/bin/python -m pytest -q -W error
.venv/bin/python -m pip check
```

The suite covers data validity, matching calendars, completed-session cutoffs, immutable writes, offline loading, checksum tampering, explicit provider settings and audited holiday exclusion. Label tests also verify the exact return formulas, threshold boundaries, unavailable outcomes, per-stock shifts, next-session dates, outcome availability and future-data independence. It uses clearly synthetic in-memory tables; the tests do not download market prices.

The required leakage test changes every price bar after a chosen session, while keeping OHLC internally valid. Features through the cutoff must remain exactly equal. A negative control confirms that targets depending on the changed future bars do change, so an ineffective mutation cannot make the test pass. A separate prefix test compares features computed from the observed prefix against those from the full dataset.

## Build or inspect the labels offline

This repeatable command reads the existing cache and prints counts and the latest observed feature items:

```bash
.venv/bin/python -m src.labels --theta 0.005
```

Theta is a decimal return: 0.005 means 0.5%. U means return above theta, D means below -theta, and N is the interval between them including equality. A documented eight-float64-epsilon tolerance handles division round-off at exact boundaries; it is not a tuned market parameter. Missing returns remain missing, and the benchmark is excluded from stock items.

Observed output at theta=0.005:

| Quantity | Actual count |
| --- | ---: |
| Feature rows, including unavailable initial features | 4,446 |
| Known features | 4,443 |
| Target rows, including unknown final targets | 4,446 |
| Known targets | 4,443 |
| Complete supervised sample rows | 4,440 |
| Feature sessions represented in complete samples | 1,480 |

Each of the three stocks lacks a previous close on the first cached session, so the first feature is unavailable. Each also lacks the next session on the last cached session, so its final target and execution date are unknown. The final feature remains usable. Neither boundary is filled with N.

To write the derived files once to a new directory:

```bash
.venv/bin/python -m src.labels --theta 0.005 --output data/processed/day1_labels_theta_0p005
```

That directory contains features.csv, targets.csv, samples.csv and manifest.json. If it already exists, use the preview command without --output. The manifest records source checksum, theta, numerical tolerance, creator-code checksum, software versions, counts and derived-file checksums. The source snapshot remains unchanged. The label preview also passed with all networking denied.

features.csv contains no NEXT_ items. targets.csv holds NEXT_ outcomes separately, with execution_session and outcome_available_session. samples.csv is the complete join, but it is not automatically valid training input at every historical cutoff. Later training must first call eligible_training_samples(samples, cutoff), which admits only outcomes known after the specified session has closed.

Availability columns contain exchange session dates, interpreted as after-close availability. A printed midnight is a date representation, not a claim that close prices or targets were known at midnight. The helper rejects intraday and timezone-aware cutoffs so they cannot be mistaken for session-close dates.

## Worked example from the actual cache

The last two adjusted RELIANCE closes in this snapshot are:

| Session | Adjusted close |
| --- | ---: |
| 2026-11-16 | 1344.80004883 |
| 2026-11-17 | 1367.00000000 |

Calculated directly from the cache, close-to-close return is (1367 / 1344.80004883 - 1) * 100 = approximately 1.650799%, producing RELIANCE_U for 17 November at theta=0.005. That session's next-session target is unknown in this snapshot, so the row cannot enter supervised training.

> **Key Insight:** A return needs two genuine consecutive trading sessions. A provider's holiday placeholder must not become an extra neutral day, and reproducing an experiment starts with preserving the exact input data.

For a complete observed feature/target pair, consider RELIANCE's feature on 16 November:

| Component | Observed inputs | Return | Label at 0.5% |
| --- | --- | ---: | --- |
| Feature for 16 November | Close 13 Nov: 1339.69995117; close 16 Nov: 1344.80004883 | +0.380690% | RELIANCE_N |
| Target executed 17 November | Open 17 Nov: 1339.00; close 17 Nov: 1367.00 | +2.091113% | NEXT_RELIANCE_U |

The implementation computes these using the full cached precision, not the rounded display values. The target uses that next session's open, not the previous close. The row is ineligible for training at the 16 November close and eligible only after the 17 November close. It is an observed label example, not a prediction or a strategy profit.

> **Key Insight:** Computing a future target for historical evaluation is legitimate. Letting a training fit see that target before its execution session has finished is look-ahead bias.

## Mine a declared training window offline

The predeclared fixed mining check uses 500 eligible sessions through the 30 December 2022 close, theta=0.005, support=0.02 and confidence=0.60. These are fixed defaults, not selected by comparing subsequent outcomes.

```bash
.venv/bin/python -m src.mining --cutoff 2022-12-30 --window 500 --theta 0.005 --min-support 0.02 --min-confidence 0.60
```

Observed output:

```text
status: rules_mined_offline
metric_scope: training_only
training_transactions: 500
first_training_feature_session: 2020-12-24
last_training_feature_session: 2022-12-29
first_training_execution_session: 2020-12-28
last_training_execution_session: 2022-12-30
rule_count: 0
example_rule: null
```

This is a valid empty result. Do not lower thresholds after inspecting the result to force a demonstration rule. The independent count audit found 174 candidates satisfying support and lift, but none satisfied the fixed 60% confidence threshold. The thresholds in the approved Day 2 tuning grid will be compared only within chronological validation.

One transaction represents one complete day of all three stocks' feature items and next-session target items. Consequently the support denominator is 500 days, not 1,500 stock-day rows. The fitting window is selected by outcome availability before encoding. The encoder sees only this training subset. Apriori's maximum itemset length is three, including the consequent. Final rules must contain one or two observed feature antecedents, exactly one NEXT_ consequent, and lift strictly above one. Integer counts enforce the lift comparison without mistaking floating-point round-off for positive lift.

Final metrics are reconstructed from integer transaction counts before confidence filtering and stricter reuse. A regression covers 15 matching outcomes in 25 antecedent occurrences: exactly 60% must pass an inclusive 60% cutoff, even if intermediate library arithmetic reports 59.99999999999999%. Review identified this boundary issue; both the direct-mining and stricter-reuse paths are checked with warnings treated as errors.

To save a new training-artifact directory once:

```bash
.venv/bin/python -m src.mining --cutoff 2022-12-30 --window 500 --theta 0.005 --min-support 0.02 --min-confidence 0.60 --output data/results/day1_mining_fixed_2022_12_30
```

The directory contains rules.json (an empty list for this fixed run), rules.csv (a header-only table), transactions.csv, sessions.csv and manifest.json. The manifest records training dates, thresholds, training-only metric scope, source/code/software lineage and file checksums. Existing output directories are never overwritten. Preview without --output to repeat the calculation.

The public select_rules helper reuses an already-mined result only at equal or stricter support/confidence thresholds. Relaxing either cutoff raises an error because filtered-out itemsets cannot be recovered without re-mining. This will allow Day 2 to score the grid without repeating every Apriori fit.

### Actual rejected-candidate arithmetic

The highest-confidence candidate among those meeting support and lift was:

```text
HDFCBANK_U & TCS_D -> NEXT_RELIANCE_U
```

| Count/statistic | Observed training value |
| --- | ---: |
| Training days | 500 |
| Days with the antecedent | 37 |
| Days with the target, irrespective of antecedent | 173 |
| Days with both antecedent and target | 21 |
| Support: 21 / 500 | 4.2% |
| Confidence: 21 / 37 | 56.7568% |
| Target base rate: 173 / 500 | 34.6% |
| Lift: (21 / 37) / (173 / 500) | 1.64037 |

This candidate is rejected because confidence is below 60%. The target means next-session open-to-close return above 0.5%, not merely a positive return. These are training co-occurrence statistics, not out-of-sample accuracy, statistical significance or trading performance. Support, confidence and lift follow the [official mlxtend definitions](https://rasbt.github.io/mlxtend/api_modules/mlxtend.frequent_patterns/association_rules/).

After saving the artifacts, independently check those counts directly:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
x = pd.read_csv('data/results/day1_mining_fixed_2022_12_30/transactions.csv')
antecedent = x[['HDFCBANK_U', 'TCS_D']].all(axis=1)
target = x['NEXT_RELIANCE_U']
print('days:', len(x), 'antecedent:', int(antecedent.sum()),
      'target:', int(target.sum()), 'joint:', int((antecedent & target).sum()))
PY
```

Expected counts: days 500, antecedent 37, target 173, joint 21.

> **Key Insight:** A pattern can occur more often than its base rate and still fail the declared evidence threshold. Empty rule sets are legitimate outcomes; training confidence is not proof of future performance.

## First chronological holdout

Run the complete fixed holdout offline:

```bash
.venv/bin/python -m src.pipeline --cutoff 2022-12-30 --test-sessions 21
```

The fixed 500-session model is fitted once through the declared close. src/signals.py accepts only the six observed feature columns, rejects target-bearing inputs and models fitted after a feature session, and generates decisions without realized outcomes. Accounting then uses the next observed session's open and close. The same decisions are evaluated at all three cost assumptions; no tuning or threshold changes occur in this command.

Observed test interval: 2-31 January 2023, 21 execution sessions and 63 stock-day decisions. All decisions are cash because the retained rule set is empty; their forecast classes remain unknown, not N. Strategy net return, CAGR and maximum drawdown are zero at all three cost levels. Trade count is zero, while Sharpe, profit factor and win rate are undefined and saved as null.

At the 0.1% total round-trip cost scenario:

| Comparison | Observed holding-period return |
| --- | ---: |
| Rules portfolio, cash | 0.0000% |
| Equal-initial-weight holding basket, after modeled costs | -1.3665% |
| NIFTY 50 price-index reference, uncosted | -2.5897% |

Individual stock holding controls returned approximately -7.7921% for RELIANCE, +5.2371% for TCS and -1.5444% for HDFCBANK after the same once-per-holding round-trip charge. Standalone per-stock rule paths remain cash. These paths are independent strategy comparisons, not additional positions added to one capital budget.

The cash outcome over this month does not establish predictive skill or resolve the research gaps. Preserve this result and proceed to the predeclared walk-forward evaluation rather than changing parameters in response to this test period.

### Matching and accounting conventions

- Highest training confidence wins. Exact top-confidence conflicts between classes produce cash and an unknown forecast. Same-class ties use lift, support, shorter antecedent and canonical item ordering. D and N forecasts remain recorded but do not open short positions.
- Every session has an explicit decision for every stock, including cash. Missing or duplicate decisions, omitted sessions and unknown execution dates fail validation rather than disappear from performance calculations.
- Each long receives 1/N of current pre-session capital; unused allocations remain cash. Positions use fractional shares, enter at the next open and exit that close. No overnight exposure or cash interest is modeled for the rule strategy.
- The total round-trip cost c is deducted once against entry notional. This is a flat friction scenario, not a broker-specific Indian charge schedule. The 0.5% scenario is not claimed as a direct replication of Nair's cost convention.
- A zero-volume LONG bar invalidates the assumed execution and aborts the run. It is not used as an after-the-fact filter that quietly removes a trade or date.
- Holding controls enter at the first test open and exit at the last close. They retain overnight moves, charge c/2 at entry and c/2 at exit against initial notional, and do not rebalance the initial basket weights. The NIFTY price-index reference is uncosted and has no hypothetical trade ledger.
- Metrics use all daily observations, including cash days. Maximum drawdown includes starting capital and is reported as a positive loss magnitude. Profit factor and win rate use net currency PnL, not directional-label correctness.
- CAGR uses inclusive calendar dates and 365.25 days/year. Daily Sharpe uses sample standard deviation, a zero risk-free rate and sqrt(252) scaling. Short-period annualization is descriptive and depends on assumptions; it is not a dependable long-run forecast. See [Sharpe's primary discussion of the ratio and time dependence](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).

The synthetic accounting test allocates 10,000 INR from a two-stock 20,000 INR portfolio to a position opening at 200 and closing at 202. The executed calculation produces 100 INR gross PnL, 10 INR cost at 0.1%, and 90 INR net PnL; total portfolio equity becomes 20,090 INR. The synthetic inputs deliberately include an overnight gap so using the previous close instead of the execution open would fail the test. This is a test example, not an observed strategy profit.

> **Key Insight:** A valid backtest must account for when a signal was knowable, where entry capital came from, and every modeled cost. Holding cash in a falling month is an observed outcome, not evidence that a model predicted the decline.

### Saved holdout bundle

Create a new result directory once:

```bash
.venv/bin/python -m src.pipeline --cutoff 2022-12-30 --test-sessions 21 --output data/results/day1_holdout_fixed_2022_12_30
```

That directory already exists in this workspace; preview without --output to repeat the calculation. It contains:

- predictions.csv: all 63 dated, cost-independent stock decisions with rule provenance.
- equity.csv: 567 rows, covering all 21 sessions for nine comparisons at three cost scenarios.
- metrics.csv and metrics.json: 27 comparison/scenario rows with the seven requested metrics. Undefined metrics are null in JSON; positive profit factor without losses is the text Infinity, avoiding nonstandard bare JSON constants.
- trades.csv: labelled strategy/control ledgers. The actual run contains 18 control-ledger rows and zero rule-strategy trades; do not interpret control trades as mined signals.
- rules.json: the empty frozen rule list.
- training_sessions.csv: the exact 500 fitting session pairs, with all outcomes known by the cutoff.
- manifest.json: exact training/test boundaries, assumptions, source/creator/software lineage and checksums for the saved files.

The final command passed with networking denied. All saved file and creator checksums were verified, the source/label/mining artifacts were unchanged, and each non-index equity path reconciled with its net ledger. A positive-trade synthetic integration case checks the active strategy path separately from the real empty-rule case. Review found no outstanding timing, accounting, benchmark or metric issues after adding the exact fitting-date provenance.

## First nested walk-forward fold

Run the production grid on the frozen three-stock snapshot:

```bash
.venv/bin/python -m src.walkforward
```

This read-only preview recomputes the earliest eligible fold. To inspect a different eligible outer cutoff, use --cutoff YYYY-MM-DD. To save a run, add --output with a new directory name. The accepted first run is already saved in data/results/day2_first_fold; existing outputs are never overwritten.

src/walkforward.py implements immutable grid parameters, canonical tie ordering, nine cached inner fits and 81 validation scores. All candidates face the same 63 execution sessions at a fixed 0.1% round-trip cost. Eligibility requires at least 20 stock trades on at least five sessions. Candidate fitting lengths are 250, 500 or expanding, ending before validation. An eligible winner is refitted through the outer cutoff and frozen for the next up-to-21 executions. If none qualifies, there is no selected configuration and every outer decision is cash with an unknown forecast. The fixed 500/0.005/0.02/0.60 control is fitted at the same outer cutoff without inner selection.

Observed split:

| Stage | Dates / count |
| --- | --- |
| Inner fitting, 250 sessions | Execution outcomes: 25 November 2021–25 November 2022 |
| Inner fitting, 500 / expanding | Execution outcomes: 19 November 2020–25 November 2022; 500 sessions for both at this first cutoff |
| Common validation | 28 November 2022–23 February 2023; 63 executions |
| Outer cutoff close | 23 February 2023 |
| Frozen outer test | 24 February–27 March 2023; 21 executions |

Actual outcome: all 81 configurations were scored; zero were eligible. The largest validation trade count was 14, below the required 20. Preserve the guard instead of relaxing it after seeing the result. The tuned portfolio therefore has zero trades, zero return and a flat 100,000 INR equity path. The fixed control also retains zero rules at this cutoff. No adaptive window won this fold, and the identical 500/expanding histories at the first cutoff provide no window-length contrast yet.

At the 0.1% cost scenario, the holding basket returned -5.9526% and the uncosted NIFTY price-index reference returned -3.4429%. Holding cash during this interval does not establish predictive skill. These figures describe this one checkpoint, not the aggregate walk-forward experiment.

The saved nine-file directory contains:

- tuning.csv: all 81 scores, exact candidate parameters, validation trades/days, eligibility, selection flag and fitting boundaries.
- predictions.csv: 126 cost-independent decisions, covering both variants, three stocks and 21 executions. No fallback forecast is converted into N.
- training_sessions.csv: exact inner/refit sample pairs, with stage, variant, fit identifier and availability cutoff. No fictitious tuned fit is recorded for the fallback.
- equity.csv: 441 rows, covering seven comparisons at three costs. Both portfolio variants use the same frozen decisions at every cost.
- metrics.csv/json: 21 comparison/scenario rows. Undefined metrics remain null and positive infinite profit factor is text in JSON.
- trades.csv: 18 holding-control ledger rows; zero model trades. rules.json contains the genuine empty final rule lists.
- manifest.json: source/code/software hashes, full grid, guards, objective/tie rule, split boundaries, selected parameters or explicit fallback, and eight file checksums.

Verify the checkpoint:

```bash
.venv/bin/python -m pytest tests/test_walkforward.py -q -W error
.venv/bin/python -m pytest -q -W error
```

Observed: 24 focused tests and 166 total tests pass. Checks include positive synthetic trading, both eligibility guards, canonical objective ties, exact common validation/fit boundaries, permitted historical refitting, final partial blocks, future-price perturbation, all-cost ledger reconciliation and saved provenance. The actual 81-configuration run passed with OS networking denied. A second offline run reproduced all eight result files byte for byte and every manifest field except creation time. Independent read-only review found no material issues, confirmed exactly nine inner fits and checked cached rules against direct mining for every configuration. The source snapshot, prior research code and curated Day 1 display bundle were unchanged.

The running Streamlit preview still displays its verified Day 1 holdout bundle. Full rolling results and their app integration are the next checkpoint; the single-fold files are not silently substituted into the old display contract.

> **Key Insight:** A tuning procedure may correctly select nothing. Here, 14 observed validation trades do not meet the predeclared 20-trade minimum, so the auditable result is a dated cash fallback rather than a forced winner.

## Research limits and next checkpoint

The chosen stocks are today's convenience sample, creating selection/survivorship bias. Historical adjusted data are not a point-in-time archive, and ^NSEI is a price index rather than an investable total-return instrument. Later validation will maintain chronological information cutoffs and explicitly modeled costs; no current data checks establish predictive value.

Next, after team confirmation: roll the inspected procedure across the remaining 21-session blocks, preserving continuous portfolio capital and earlier frozen predictions. Record all 81 scores per fold and compare the combined tuned procedure against the fixed control. The app will then receive the computed tuning results and out-of-sample curves. Keep the research pipeline offline; the optional Latest Signals page remains separate.

Primary software reference: [yfinance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
