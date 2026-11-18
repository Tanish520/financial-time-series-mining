# Financial Time-Series Pattern Mining

CS G520 Advanced Data Mining, BITS Pilani Hyderabad. Team: Tanish Kumar Gupta and Aayush Gupt.

The project investigates adaptive fitting-window selection, automatic association-rule thresholds and trading-cost-aware validation. It uses daily stock movements to mine interpretable rules for the following session. It does not claim to solve market prediction or concept drift.

## Current checkpoint: Day 1, Step 1

Implemented: a validated, immutable three-stock-plus-NIFTY data snapshot and offline cache reader. Labels, mining, walk-forward results, backtests and Streamlit pages are subsequent checkpoints; there are no empirical strategy results yet.

The approved [design](docs/superpowers/specs/2026-11-18-financial-time-series-mining-design.md) explains the research protocol. The [current implementation checklist](docs/superpowers/plans/2026-11-18-day1-data-snapshot.md) covers this data step.

## Setup

Use Python 3.13, matching the environment used for this checkpoint. From the repository root:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On the project owner's Mac, the Python executable is /opt/homebrew/bin/python3.13. On Windows, use .venv\Scripts\python.exe in place of .venv/bin/python.

requirements.txt pins the currently installed data and testing packages, including transitive dependencies. App/mining dependencies will be added when those checkpoints are implemented.

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

The suite covers data validity, matching calendars, completed-session cutoffs, immutable writes, offline loading, checksum tampering, explicit provider settings and audited holiday exclusion. It uses clearly synthetic in-memory tables; the tests do not download market prices. The future-data perturbation test belongs to the next labels checkpoint, where the feature calculation will exist.

## Worked example from the actual cache

The last two adjusted RELIANCE closes in this snapshot are:

| Session | Adjusted close |
| --- | ---: |
| 2026-11-16 | 1344.80004883 |
| 2026-11-17 | 1367.00000000 |

Calculated directly from the cache, close-to-close return is (1367 / 1344.80004883 - 1) * 100 = approximately 1.650799%. The next checkpoint will turn returns into U/D/N features; this observation is not a prediction or a trading return.

> **Key Insight:** A return needs two genuine consecutive trading sessions. A provider's holiday placeholder must not become an extra neutral day, and reproducing an experiment starts with preserving the exact input data.

## Research limits and next checkpoint

The chosen stocks are today's convenience sample, creating selection/survivorship bias. Historical adjusted data are not a point-in-time archive, and ^NSEI is a price index rather than an investable total-return instrument. Later validation will maintain chronological information cutoffs and explicitly modeled costs; no current data checks establish predictive value.

Next, after team confirmation: implement dated close-to-close features, next-session open-to-close targets and a test proving that changes after session t cannot alter features available at t. Keep the research pipeline offline; the optional Latest Signals page remains separate.

Primary software reference: [yfinance download options](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
