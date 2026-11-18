# Day 1 Matching and Holdout Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce deterministic, features-only forecasts and an auditable 21-session chronological holdout with finite-capital accounting, three cost assumptions and aligned controls.

**Architecture:** src/signals.py matches frozen MiningResult rules against observed feature tables without receiving outcomes. src/backtest.py executes dated decisions at the next observed session's open/close and computes equity, ledgers, controls and metrics. src/pipeline.py orchestrates the fixed holdout offline and optionally saves a checksummed result bundle; later walk-forward work can reuse these functions.

**Tech Stack:** Existing pinned Python 3.13, pandas, numpy, mlxtend and pytest. No additional dependency, data download, UI or automatic tuning in this checkpoint.

---

## Files and interfaces

- Create src/signals.py: predict_signals(model, features, stocks).
- Create src/backtest.py: BacktestResult, simulate_strategy, simulate_buy_and_hold, performance_metrics.
- Create src/pipeline.py: HoldoutResult, run_holdout, offline CLI and result serialization.
- Create tests/test_signals.py and tests/test_backtest.py: deterministic matching, temporal boundaries, accounting, metrics, controls and integration checks.
- Modify README.md with actual holdout dates, counts and computed results. Keep existing source, price, label and mining artifacts unchanged.

## Task 1: Frozen-rule matching

- [x] Write failing tests for highest confidence, exact conflicting ties -> cash, stable same-class tie resolution, D/N -> cash, no match -> unknown prediction/cash, and shuffled inputs.
- [x] Reject NEXT_ antecedents, target-bearing feature tables, incomplete predictor cohorts, and a model fitted after a requested feature session.
- [x] Implement features-only matching. Equal-confidence rules for one class sort by lift, support, shorter antecedent, then canonical items. Never consult realized outcomes.
- [x] Verify later feature changes cannot alter earlier predictions.

```bash
.venv/bin/python -m pytest tests/test_signals.py -q -W error
```

## Task 2: Execution, metrics and controls

- [x] Write failing tests using clearly synthetic prices: next-session open, weekend mapping, simultaneous allocations, daily compounding, cost-once accounting, cash days, missing decisions and first-day drawdown.
- [x] Implement simulate_strategy(prices, signals, config, cost=0.001, initial_capital=100000). Require a complete contiguous sequence of all-stock decisions. Execute only U/LONG, with each position receiving 1/N of pre-session equity. Retain cash days and charge c times entry notional once per round trip.
- [x] Save execution/feature dates, prices, fractional quantities, notional, gross/net PnL and fees. Reject ambiguous dates or inconsistent action/prediction pairs rather than inventing fills.
- [x] Implement net return, calendar CAGR, daily Sharpe (252, ddof=1, zero cash/risk-free rate), positive-magnitude maximum drawdown including initial capital, net profit factor, win rate and trade count. No trades/zero variance get N/A for undefined metrics. Positive gains without losses get infinite profit factor.
- [x] Define calendar CAGR over inclusive first-to-last execution calendar dates; explain that short-period annualization is descriptive, not a reliable long-run estimate.
- [x] Implement stock buy-and-hold and an equal-initial-weight basket over identical execution dates, with c/2 at entry and c/2 at exit against initial notional. Implement an uncosted NIFTY price-index reference separately. No daily basket rebalancing.
- [x] Test exact fee treatment and date alignment, cash metrics, profit-factor edge cases and Sharpe inputs.

```bash
.venv/bin/python -m pytest tests/test_backtest.py -q -W error
```

Worked synthetic accounting contract: capital 20,000 INR, two stocks, only one long allocation of 10,000 INR, next-session open 200 and close 202. At c=0.001, gross PnL=100 INR, fee=10 INR, net PnL=90 INR, portfolio closes at 20,090 INR. These are test inputs, not observed market returns.

## Task 3: Offline fixed holdout

- [x] Implement run_holdout: validate a real session-close cutoff, fit the unchanged fixed configuration once, select the next 21 observed execution sessions, and freeze predictions before any return accounting. Feature dates must be the immediately preceding observed sessions.
- [x] Simulate the same predictions at c in {0.0005, 0.001, 0.005}. Report portfolio, standalone per-stock and corresponding buy-and-hold controls plus the NIFTY reference. Costs do not alter rule fitting or predictions.
- [x] Verify the pipeline on a temporary synthetic snapshot, including standard JSON serialization of undefined/infinite metrics and source checksums.
- [x] Provide a repeatable preview:

```bash
.venv/bin/python -m src.pipeline --cutoff 2022-12-30 --test-sessions 21
```

- [x] Add optional --output for predictions.csv, equity.csv, metrics.csv/metrics.json, trades.csv, rules.json and manifest.json. Save source/code/software/file checksums, cost and fill conventions, and exact fitting/test dates. Refuse overwrite.
- [x] Expect cash for the fixed real model's zero rules, but report only observed outputs. Do not lower thresholds or skip the cash result.

## Task 4: Verification, review and handoff

- [x] Run the full suite and source-control checks. Finish requesting-code-review and fix material findings with regressions before final artifacts.
- [x] Execute the actual holdout with networking denied, inspect source lineage, ledger/equity consistency and controls, then save the final result bundle.
- [x] Record the computed results and limitations in README.md. Cite Sharpe's primary metric discussion for the metric context; clearly label the short holdout as an engineering checkpoint.
- [x] Commit on the existing feature branch. Give runnable commands, actual outputs and a Key Insight, then pause before app/walk-forward work.

## Observed results and review

- New tests: 13 matching and 28 execution/metric/holdout checks; full suite 120 passing with warnings treated as errors. A synthetic end-to-end case exercises nonempty learned rules and positive trade execution; the actual fixed model exercises cash behavior.
- The actual offline holdout used 21 execution sessions from 2023-01-02 through 2023-01-31, following the 2022-12-30 fit cutoff. All 500 fitting outcomes were known by that cutoff. All 63 stock-day decisions remained cash because no rules met the unchanged fixed thresholds.
- At every cost tier, strategy net return/CAGR/max drawdown were zero, trade count zero, and undefined trade/volatility metrics null. At c=0.001, the holding basket returned -1.3664589707% and the uncosted NIFTY reference -2.5896570556%. These are a single engineering holdout, not a claim of predictive skill.
- Final artifacts are in data/results/day1_holdout_fixed_2022_12_30. Counts: 500 training-session rows, 63 decisions, 567 equity rows, 27 metric rows and 18 labelled control-ledger rows. Zero rule-strategy trades were fabricated.
- Review found one provenance gap: exact training boundaries/session rows were initially absent from the saved holdout. A manifest regression failed before the fix. Final metadata now includes first/last training feature and execution dates; training_sessions.csv is checksummed. Both known and empty fitting histories are covered. Focused re-review closed the finding with no outstanding issues.
- The actual preview and final artifact creation passed with all network access denied. Every saved file and creator-code checksum was verified; raw, label and mining artifacts were unchanged. Every non-index equity path reconciled with its net trade ledger.
- Synthetic accounting inspection produced allocation 10,000 INR, open 200, close 202, gross 100 INR, fee 10 INR, net 90 INR, and total portfolio equity 20,090 INR. These are test values, not market performance.
- The source modules remain reusable for Day 2: predictions are fixed independently of costs, and the backtester can account for a contiguous sequence of dated decisions across later folds without resetting capital.
