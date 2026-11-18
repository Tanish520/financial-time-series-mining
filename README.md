# Financial Time-Series Pattern Mining

CS G520 Advanced Data Mining, BITS Pilani Hyderabad. Team: Tanish Kumar Gupta and Aayush Gupt.

The project investigates adaptive fitting-window selection, automatic association-rule thresholds and trading-cost-aware validation. It uses daily stock movements to mine interpretable rules for the following session. It does not claim to solve market prediction or concept drift.

## Current checkpoint: Day 1, Step 3

Implemented: a validated, immutable three-stock-plus-NIFTY data snapshot, offline cache reader, causal U/D/N features, next-session intraday targets, outcome-availability filtering and guarded Apriori mining. The fixed training-only mining checkpoint retains zero rules at its declared cutoffs. Prediction matching, walk-forward results, backtests and Streamlit pages are subsequent checkpoints; there are no empirical strategy results yet.

The approved [design](docs/superpowers/specs/2026-11-18-financial-time-series-mining-design.md) explains the research protocol. The [current implementation checklist](docs/superpowers/plans/2026-11-18-day1-mining.md) covers mining. Earlier checklists record [labels](docs/superpowers/plans/2026-11-18-day1-labels.md) and the [data snapshot](docs/superpowers/plans/2026-11-18-day1-data-snapshot.md).

## Setup

Use Python 3.13, matching the environment used for this checkpoint. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On the project owner's Mac, the Python executable is /opt/homebrew/bin/python3.13. On Windows, use .venv\Scripts\python.exe in place of .venv/bin/python.

requirements.txt pins the currently installed data, mining and testing packages, including transitive dependencies. mlxtend 0.25.0 was installed with the preceding pins as constraints; previous package versions were preserved. App dependencies will be added at that checkpoint.

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

Final metrics are reconstructed from integer transaction counts before confidence filtering and stricter reuse. A regression covers 15 matching outcomes in 25 antecedent occurrences: exactly 60% must pass an inclusive 60% cutoff, even if intermediate library arithmetic reports 59.99999999999999%. Review identified this boundary issue; both the direct-mining and stricter-reuse paths are now checked. The current suite has 79 passing tests with warnings treated as errors.

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

## Research limits and next checkpoint

The chosen stocks are today's convenience sample, creating selection/survivorship bias. Historical adjusted data are not a point-in-time archive, and ^NSEI is a price index rather than an investable total-return instrument. Later validation will maintain chronological information cutoffs and explicitly modeled costs; no current data checks establish predictive value.

Next, after team confirmation: add deterministic rule matching and the first chronological, cost-aware holdout backtest. Preserve no-rule cash behavior and report it honestly. Keep the research pipeline offline; the optional Latest Signals page remains separate.

Primary software reference: [yfinance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
