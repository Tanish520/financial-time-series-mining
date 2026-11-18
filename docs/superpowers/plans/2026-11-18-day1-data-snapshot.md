# Day 1 Data Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Create and verify an immutable three-stock-plus-NIFTY snapshot that subsequent experiments can read without a network connection.

**Architecture:** A single src/data.py module provides validation, normalized Yahoo downloads, snapshot writing and offline loading. A CSV contains dated bars; a JSON manifest records their checksum, request, coverage and software versions. The command line separates explicit downloading from offline checking.

**Tech Stack:** Python 3.13, pandas, numpy, yfinance, pytest; exact installed versions recorded in requirements.txt. Execute inline at the user's requested checkpoints; stop after this step before labels or mining.

---

## Files and contracts

- src/data.py: SnapshotConfig, DataValidationError, normalize_download, validate_prices, write_snapshot, load_snapshot, download_snapshot and command-line entry point.
- tests/test_data.py: real in-memory price tables and temporary on-disk snapshots; no live downloads in unit tests.
- requirements.txt: exact package versions after installation.
- .gitignore: environments, credentials and generated data excluded from source control; original project PDFs stay local.
- pytest.ini: discover tests/.
- README.md: setup, one explicit download command, one offline check command, test command, actual coverage and limitations.

The canonical long-form CSV columns are date, ticker, open, high, low, close, volume, dividends and stock_splits. Rows are ordered by date then ticker. Every accepted ticker must have the benchmark's session dates; no missing-price filling or unexplained session deletion.

## Task 1: Prepare the environment and observe failing contract tests

- [x] Read the approved design and inspect the empty repository.
- [x] Find Python 3.13 and create .venv.
- [x] Install yfinance, pandas, numpy and pytest, then freeze their exact resolved versions.
- [x] Add tests before implementation for missing prices, duplicate observations, impossible OHLC, zero stock volume allowance, negative volume rejection, index zero volume, unequal calendars, incomplete-session rejection, and offline checksum enforcement.
- [x] Run the tests and inspect the expected missing-function/module failure:

```bash
.venv/bin/python -m pytest tests/test_data.py -q
```

The central cache test uses the following API:

```python
write_snapshot(prices, config, destination, min_sessions=2)
loaded, manifest = load_snapshot(destination)
assert manifest["sha256"] == hashlib.sha256(
    (destination / "prices.csv").read_bytes()
).hexdigest()
pd.testing.assert_frame_equal(loaded, prices, check_dtype=False)
```

These functions are implemented by Task 2. DataValidationError is a ValueError subclass, and SnapshotConfig holds start/as_of ISO dates, stock symbols and the benchmark symbol.

## Task 2: Implement the minimal data boundary

- [x] Implement SnapshotConfig with explicit dates and default three-stock universe; prohibit as_of on or after the current India date for this conservative research downloader.
- [x] Implement normalize_download for Yahoo's ticker-first MultiIndex response, retaining missing prices for validation rather than filling them. Preserve optional corporate-action columns as unavailable when absent.
- [x] Implement validate_prices: exact ticker set; ISO midnight dates within the request; no duplicate ticker/date; finite positive OHLC; nonnegative finite volume; high/low consistency; matching benchmark calendar; minimum coverage. Produce errors with affected ticker/date examples.
- [x] Implement write_snapshot: validate before writing, refuse overwrite, stage the CSV and manifest, store SHA-256 and actual coverage, then publish the complete directory.
- [x] Implement load_snapshot: verify schema/checksum/recorded coverage and rerun validation. Do not import or call yfinance on this path.
- [x] Implement download_snapshot with explicit yfinance settings: interval=1d, auto_adjust=True, back_adjust=False, actions=True, repair=False, keepna=True, ignore_tz=True, threads=False and ticker-first columns. End is as_of plus one day because Yahoo's end bound is exclusive.
- [x] Add download/check CLI commands, with clear errors and a nonzero exit code on invalid data.
- [x] Rerun the tests; fix implementation defects without weakening the data contract.

## Task 3: Produce the real snapshot and inspect it

- [x] Run the explicit initial download:

```bash
.venv/bin/python -m src.data download --start 2020-11-17 --as-of 2026-11-17 --output data/raw/day1_three_stocks
```

Expected structural output: a JSON summary naming RELIANCE.NS, TCS.NS, HDFCBANK.NS and ^NSEI, observed first/last sessions, per-ticker counts and the snapshot checksum. Do not predict actual price values or claim a successful download in advance.

- [x] If provider data violate the contract, inspect the offending raw observations and identify the source before changing coverage or adjustment policy. Never substitute simulated prices for market data.
- [x] Run the offline check:

```bash
.venv/bin/python -m src.data check --snapshot data/raw/day1_three_stocks
```

Expected: the checksum and observed coverage match the saved manifest; no download occurs. Repeat with yfinance imports disabled to demonstrate that cache reads do not depend on the provider package.

- [x] Print a small actual price example and calculate its close-to-close return directly from the snapshot for the mentoring explanation. This is arithmetic inspection, not the next labels/mining implementation step.

## Task 4: Document, verify and pause

- [x] Write README.md with the exact commands, observed snapshot coverage, data adjustment policy, conservative cutoff and survivorship/point-in-time limitations.
- [x] Run the full current test suite and dependency checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

- [x] Review the changed files and source-control exclusions. Create a local feature-branch checkpoint if git author identity is available; never push or publish as part of this data step.
- [x] Update this checklist with actual outcomes and any diagnosed data issues.
- [x] Give Tanish runnable commands, the observed output, one numerical explanation and a Key Insight box. Stop and wait for confirmation before implementing labels, mining or the app.

## Subsequent checkpoints, already covered by the approved design

The next checkpoint adds causal return labels and their future-data perturbation test. Later Day 1 checkpoints add guarded Apriori mining, a simple chronological backtest and the app skeleton. Day 2 adds common-date walk-forward tuning; Day 3 adds cost-aware results, benchmarks and asymmetry; Day 4 adds deployment, documentation and Latest Signals if time permits. Each receives its own concrete implementation checklist when reached, so unapproved later steps are not executed now.

## Observed results and resolved data issue

- Accepted snapshot: 5,928 rows, 1,482 sessions per ticker, 2020-11-17 through 2026-11-17. SHA-256: 668376329abdc570b4724454c921bd0e712b359af18692ec36cb0e30303ae601.
- The first strict validation rejected Yahoo's 2026-01-15 placeholder. Primary-source NSE circular CMTR72260 confirms a capital-market holiday. The implementation now excludes only this verified non-session and preserves all four original rows plus their checksum and source. Ordinary missing prices still fail validation. This corrects the session calendar rather than fills or deletes an unexplained trading observation.
- Offline checking succeeded with all network access denied by a macOS process sandbox, and separately with yfinance imports disabled.
- An independent read-only code review found no material issues. A follow-up metadata check confirmed all four requested identities, NSE exchange, INR currency and Asia/Kolkata timezone; the evidence is stored separately in data/raw/instrument-verification.json and summarized in README.md.
- The installed pandas 3.1 parser deprecates float_precision because all modes use the same converter. A warnings-as-errors reproduction confirmed the warning; removing the redundant argument left the snapshot round-trip tests passing without warnings.
- No labels, mined rules, strategy performance, application pages or deployment are claimed at this checkpoint.
