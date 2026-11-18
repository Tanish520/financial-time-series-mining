# Day 2 First Walk-Forward Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce one auditable outer fold with 81 past-only validation scores, a frozen selected model, an untuned control and the next up-to-21 sessions of decisions.

**Architecture:** Add src/walkforward.py around the existing labels, mining, signals and accounting interfaces. Cache three label tables and nine minimum-threshold inner fits; filter those fits for all 81 configurations. Keep the existing Day 1 app and artifacts unchanged until the full walk-forward result contract is ready.

**Tech Stack:** Existing pinned Python 3.13, pandas, numpy, mlxtend and pytest; no new dependency or network call.

---

This continues the approved design on the existing feature branch. The user has authorized the next checkpoint; no further design approval is needed. This checkpoint is a single inspected fold, followed by user confirmation before the full rolling experiment and app integration.

## Task 1: Temporal and selection tests

Files: create tests/test_walkforward.py; create src/walkforward.py only after observing missing-feature failures.

- [x] Add a deterministic synthetic market with alternating U/D features and causal intraday outcomes. Use small explicit windows for fast temporal tests; these are unit-test inputs, not a changed research protocol.
- [x] Assert the default grid contains 81 unique configurations and stable serialized identifiers. Reject duplicate/invalid probabilities, windows, dates and inadequate history.
- [x] Assert all candidates share validation dates, finite fitting lengths count execution sessions, and fitting outcomes end before validation starts.
- [x] Assert selection maximizes eligible validation return with a fixed identifier tie order. Enforce both minimum trades and distinct trade-session guards; no eligible candidate produces dated cash decisions.
- [x] Assert refitting can include the now-historical validation block, while outer execution dates are strictly after the outer cutoff. Perturb future prices and verify unchanged tuning, fitted rules and first outer decision.
- [x] Assert final partial blocks retain the same protocol, costs use identical decisions, and saved files have checksums and full fitting-date provenance.
- [x] Run the tests and observe expected failures before implementation:

```bash
.venv/bin/python -m pytest tests/test_walkforward.py -q --tb=short
```

## Task 2: One-fold engine

Files: create src/walkforward.py.

- [x] Define immutable RuleParameters and TuningGrid, a FoldResult, and run_fold(prices, config, cutoff=None, ...). RuleParameters.identifier is sorted-key compact JSON, with expanding serialized as text. Resolve exact return ties by ascending identifier; do not use test scores.
- [x] Determine the earliest cutoff with 500 fitting outcomes plus 63 validation outcomes. Complete labeled outcomes begin at the third raw session because the first feature needs a preceding close. Explicit cutoffs must also have enough history and at least one future execution.
- [x] Take the final 63 known execution dates as validation. Inner fitting cutoff is the observed session immediately before its first execution. Cache each theta/window fit at support=0.01, confidence=0.55, then select stricter rules using select_rules.
- [x] Freeze each candidate across validation; use simulate_strategy at cost=0.001 and record every score, eligibility, count, dates and rule count. Require 20 stock trades on at least five sessions by default.
- [x] Refit the eligible winner through the outer cutoff. If none qualifies, use an empty dated model, unknown predictions and cash. Refit the fixed 500/0.005/0.02/0.60 control at the same cutoff without tuning.
- [x] Freeze predictions before evaluating the next up-to-21 executions at all three approved costs. Compare tuned/fixed portfolios and aligned holding controls. Record this as an engineering checkpoint, not the completed research experiment.
- [x] Save tuning.csv, predictions.csv, training_sessions.csv, equity.csv, metrics.csv/json, trades.csv and rules.json in a new directory with a manifest. Record source/code/software hashes, exact grid, guards, dates, costs and winner/fallback. Refuse overwrites and preserve the source snapshot.
- [x] Provide the offline CLI:

```bash
.venv/bin/python -m src.walkforward --output data/results/day2_first_fold
```

Expected structure: 81 tuning rows, 21 outer executions when available, one selected configuration or an explicit no-eligible fallback, and both TUNED and FIXED dated decisions. Numerical values must come from the actual run.

## Task 3: Audit, review and teaching checkpoint

Files: update README.md and this checklist after actual verification.

- [x] Run focused tests with warnings as errors, then the full suite after implementation is stable.
- [x] Execute the production 81-configuration first fold with external networking denied. Independently inspect split boundaries, objective selection, fit counts, saved hashes, ledger reconciliation and unchanged original artifacts.
- [x] Follow requesting-code-review: dispatch a fresh read-only reviewer with the approved design and focused diff; address material findings with regression checks.
- [x] Record actual results and single-command reproduction in README.md. State that the app still displays the Day 1 bundle and the full rolling experiment is the next checkpoint.
- [x] Commit the verified checkpoint on the current branch. Explain one actual fold with dates/numbers, provide the runnable check, end with a Key Insight and wait for confirmation.

## Observed results

- First production cutoff: 23 February 2023. Inner fitting ends 25 November 2022; common validation runs 28 November 2022–23 February 2023 (63 executions). Frozen test: 24 February–27 March 2023 (21 executions).
- All 81 configurations scored. Maximum validation trade count: 14; no candidate meets the unchanged 20-trade guard. No selected parameters or window winner; the outer tuned portfolio explicitly stays in cash. Fixed control also has zero final rules.
- At 0.1% total round-trip cost: tuned and fixed return 0%, holding basket -5.9526%, uncosted NIFTY reference -3.4429%. These are one-fold engineering observations, not the completed research evaluation.
- Saved 126 decisions, 441 equity rows and 18 holding-control ledger rows in a nine-file local result directory. All eight payload hashes and code/source lineage verified; non-index paths reconcile with their ledgers.
- 24 focused and 166 total tests pass with warnings as errors. Both production runs passed with networking denied; all eight result files reproduced byte for byte and the manifests match except creation time.
- Independent read-only review found no material issues. It confirmed nine cached inner fits, compared all 81 cached rule sets with direct mining, checked fitting/validation/fallback provenance and independently replayed numeric CSVs.
- Original snapshot, existing research source and curated app bundle remain unchanged. The app continues to display its Day 1 holdout pending the full rolling experiment and new display contract.
