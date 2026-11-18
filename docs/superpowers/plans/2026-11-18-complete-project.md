# Complete Financial Time-Series Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, plus superpowers:dispatching-parallel-agents for independent components and superpowers:requesting-code-review before completion. Track steps below.

**Goal:** Finish the approved ten-stock course project, including reproducible full walk-forward results, all five app views, report, demo and the deployment possible with available account access.

**Architecture:** Retain the audited single-fold engine. src/experiment.py rolls it across the observed calendar and simulates concatenated frozen decisions once per cost, preserving capital. src/asymmetry.py scores held-out U/D predictions; src/latest.py is the separate optional network boundary. A version-2 saved bundle feeds the Streamlit app without retraining. Reports and demo artifacts derive only from saved results.

**Tech Stack:** Existing pinned Python 3.13, pandas, numpy, mlxtend, Streamlit, pytest and matplotlib. Report PDF uses an isolated document runtime if necessary. Deployment targets Streamlit Community Cloud; do not substitute a static host for the Python app.

---

The user explicitly requested completion in one go on 18 November 2026, superseding the earlier instruction to wait after each checkpoint. The approved research design remains unchanged. No new approval gate is required for routine implementation, tests, repository creation or preparing deployment.

## Task 1: Full data and rolling evaluation

- [x] Verify all ten proposed Yahoo instrument identities and freeze a new 2020-11-17..2026-11-17 snapshot. Keep the original three-stock snapshot unchanged and preserve the documented NSE non-session exclusion.
- [x] Write failing tests for a complete rolling calendar, final partial fold, exactly 81 candidate logs per production fold, selection/refit timing, no retrospective predictions and continuous equity across folds.
- [x] Add score_outer=False to run_fold so full orchestration can retain frozen decisions without resetting capital or calculating redundant fold-local performance. Existing behavior remains the default.
- [x] Implement src/experiment.py with run_experiment(prices, config, ...), assemble fold logs/rules/provenance, and evaluate tuned/fixed portfolios plus standalone stock paths and holding controls over identical sessions. Use the existing accounting implementation.
- [x] Write failing tests for prediction-weighted held-out base rates, U/D precision/lift, unknown forecasts and zero denominators; implement src/asymmetry.py.
- [x] Run the full ten-stock production grid offline and store an immutable checksummed bundle. Progress goes to stderr and numerical results to JSON/files. Verify hashes, chronology, counts, eligible objective selection and ledger reconciliation.

Production command:

```bash
.venv/bin/python -m src.experiment --snapshot data/raw/full_ten_stocks_corrected --output data/results/full_walkforward
```

## Task 2: Shared saved-result contract

- [x] Write manifest schema_version=2 and evaluation_scope=nested_walk_forward with exact grid, guards, source/code/software hashes and all payload hashes. Use the fixed names below.
- [x] Keep folds.csv, tuning.csv, predictions.csv, training_sessions.csv, equity.csv, metrics.csv/json, trades.csv, rules.json, outcomes.csv, asymmetry.csv, asymmetry_groups.csv, latest_model.json, latest_prices.csv and latest_signals.json. Existing files are never overwritten.
- [x] folds.csv contains fold_id, fit_cutoff, inner_fit_cutoff, validation_start/end, test_start/end, test_sessions, eligible_candidates, selected_window, theta, min_support/confidence, validation_net_return, validation_trade_count/sessions, rule_count and fallback. Fallback rows have missing selected parameters, not an invented winner.
- [x] predictions.csv retains the single-fold columns plus fold_id and variant=TUNED/FIXED. All variants use identical contiguous execution dates. All rule metrics are training statistics.
- [x] rules.json is a list of final fitted rules tagged fold_id, variant, fit_cutoff and theta. Existing antecedents/consequent/support/confidence/lift fields remain unchanged.
- [x] equity/metrics/trades strategy identifiers are TUNED_PORTFOLIO, FIXED_PORTFOLIO, TUNED:TICKER, FIXED:TICKER, BUY_HOLD:TICKER, BUY_HOLD_BASKET and NIFTY50_PRICE_INDEX. Cost is the scenario; applied_cost is zero for the index.
- [x] asymmetry.csv has variant, direction, prediction_count, correct_count, precision, base_rate, expected_correct and held_out_lift. Base rate is prediction-weighted across fold/stock groups; undefined values remain missing.
- [x] latest_model.json has stocks, benchmark, fit_cutoff, theta, parameters, fallback, rules and source_sha256. It represents the final historical TUNED fold, including a real no-eligible fallback. latest_prices.csv holds the last ten cached completed sessions in the original long-price schema.

## Task 3: Latest signals and complete app

- [x] Independently implement/test src/latest.py and tests/test_latest.py: completed aligned cohorts only, explicit live fetch, 15-minute UI cache, cache fallback with timestamp/status, frozen rule provenance and unknown-class preservation. No future/current-session bar is admitted.
- [x] Upgrade the read-only app loader/adapters and tests for the version-2 contract while retaining supported Day 1 data. Package verified display artifacts explicitly under app/artifacts/full_walkforward.
- [x] Wire Rules Explorer with fold/variant/target/direction/confidence filters; Tuning & Drift with all real winners, frequency counts and fallback count; Backtest with tuned/fixed/holding curves and seven metrics; Asymmetry with precision/base-rate and lift charts plus counts; Latest Signals with explicit refresh and educational disclaimer.
- [x] Test every page, scenario selection, corrupt/missing artifacts, no-rule/no-trade states and offline core operation. Attempt supported desktop/narrow browser QA and record limitations accurately.

## Task 4: Report, deployment and handoff

- [x] Generate a concise report in Markdown and PDF using real saved outputs, with method, research gaps addressed/investigated, net costs, fixed comparison, asymmetry, limitations and bibliography. Verify primary sources before asserting exact paper-specific costs. Do not claim concept drift is solved or detected just because a window changes.
- [x] Create exported equity/asymmetry/tuning figures, a short live-demo script, a contributor guide, a one-command research entry point and deployment instructions. Keep optional stretch work out of the critical path.
- [x] Add a lightweight CI workflow and a container deployment option. Verify a fresh local start using the packaged results; run the full test suite with warnings as errors and complete independent review.
- [x] Publish a private GitHub repository using authenticated account access after all source is reviewable. Connector lacks repository creation; authenticated gh is the fallback for this operation. Keep secrets and original PDFs excluded. The user explicitly approved making only this course repository public on 19 November 2026 for Streamlit access; that visibility change is now verified.
- [ ] Attempt Streamlit Community Cloud publication through supported available UI/account access. Verify a deployed URL before claiming success. If hosting authentication or unavailable UI blocks publication, finish all local/deployment artifacts and report that exact remaining external action.
- [ ] Commit final verified work, generate a self-contained local source/results archive, record actual completion/verification limits and give launch/report/repository links. Do not ask to continue another implementation checkpoint.

## Verified research and app outcome

- Full corrected ten-stock study: 44 folds, 917 test sessions, 3,564 scores, 18,340 decisions and a final 14-session partial fold. All 44 folds have eligible winners; history frequencies 27/10/7.
- At 0.1% cost: tuned -13.30%, fixed +0.04%, holding basket +45.57%, uncosted NIFTY +44.32%. Negative tuned results are retained.
- Nine flat zero-volume Yahoo placeholders on 18 March 2025 were corrected from primary NSE observations in a new immutable snapshot; every other row is unchanged. Original and correction source hashes and all evidence are retained.
- Independent research review reconciled chronology, selected objectives, continuous capital, cost ledgers, controls, asymmetry and all source/artifact hashes with no material findings.
- Independent app review prompted regressions for missing schemas/metadata, final-model consistency, immutable nested cache values and stale-live responses. All material findings were fixed and re-reviewed.
- Full test suite: 273 passing with warnings as errors, including an isolated hosted entry-point launch. The core display/latest test set also passed with OS networking denied before the small legend-layout refinement.
- A second complete run with OS networking denied reproduced all 15 payloads byte for byte; only the manifest creation timestamp differs. The full release commit passed remote GitHub Actions on Python 3.13.
- Six-page report generated from verified outputs, rendered and all six pages visually inspected. Desktop Rules/Backtest and a 390-pixel phone-emulated Backtest layout inspected; clipped legend found and refined to two columns with year-bearing date labels. Physical phone testing is not claimed.
- GitHub repository: https://github.com/Tanish520/financial-time-series-mining, branch main. Initially private; now public with explicit user authorization. CI passed for release commit 10c2866.
- Initial cloud launch exposed a console-script import-path assumption. Reproduced the identical failure in an isolated subprocess, added a failing regression, made the entry point resolve its own repository path, and verified all 273 tests. Cloud selected Python 3.14; CI now checks both 3.13 and 3.14 before final hosted verification.
- Live deployment and final archive regeneration are the remaining release steps.
