# Day 1 Streamlit Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a responsive five-view Streamlit preview over the verified fixed-holdout results, with honest empty/pending states and no market-data dependency at page load.

**Architecture:** app/results.py loads and verifies a small versioned result bundle and packages new bundles explicitly. app/view_data.py formats saved metrics and filters rules without recomputing research. app/streamlit_app.py renders single-column views; existing src/ research modules remain outside the display runtime. Curated derived artifacts are tracked under app/artifacts/day1_holdout so a fresh checkout can display the experiment without raw prices.

**Tech Stack:** Existing Python 3.13/pandas/numpy plus Streamlit and its chart dependencies. Pin the resolved environment while retaining existing versions. Streamlit AppTest verifies behavior; a local browser preview verifies desktop and narrow layout when supported.

---

## Approved presentation and scope

Follow the already-approved single-column phone-friendly design. Use a light theme, restrained teal accent, readable native controls and an accessible Page selector. No separate design approval is required for these routine implementation choices. This checkpoint is a local Streamlit preview; hosting remains a later project step. Sites' static/Worker hosting cannot execute this Python server, so use the requested Streamlit framework and local preview guidance.

Rules Explorer starts with the short project summary and actual fixed-fit context. It offers target/direction/confidence display filters and shows the genuine zero-rule state. Backtest shows portfolio/stock and cost selectors, indexed close-equity curves, all seven metrics and modeled-cost assumptions. Tuning & Drift identifies the actual fixed parameters and explicitly says walk-forward tuning has not run. Asymmetry shows zero actual U/D forecast counts and undefined precision/lift; no invented chart. Latest Signals clearly says refresh is not enabled and the displayed experiment is historical. Include the requested educational disclaimer.

## Task 1: Environment, curated bundle and failing tests

- [x] Install Streamlit using existing requirements as constraints; freeze exact versions after success.
- [x] Copy only the eight verified derived holdout files into app/artifacts/day1_holdout, preserving bytes and metadata. Keep original raw data and project PDFs excluded from Git.
- [x] Add loader tests for checksums, missing bundles, counts and backend-independent reads. Add tests for percentage/N/A/infinity formatting, display-only rule filters and matching cost/scope selections.
- [x] Add AppTest checks for all five views, honest empty/pending states, cost/scope interactions, no backend provider imports and a missing-results error state.
- [x] Observe expected failures before implementing app/results.py, app/view_data.py and the Streamlit entry point.

```bash
.venv/bin/python -m pytest tests/test_app.py -q --tb=short
```

## Task 2: Read-only adapters and view implementation

- [x] Implement load_bundle with the versioned manifest's exact supported filenames and SHA-256 checks before parsing. Verify saved experiment counts/date coverage. Do not import src/, mlxtend or yfinance.
- [x] Implement explicit package_bundle(source, destination) and a CLI; verify the source first and refuse an existing destination. Page loads only read the already-packaged bundle.
- [x] Implement display helpers. Missing metrics are N/A; positive infinite profit factor is infinity text/symbol; percentage fields are scaled exactly once. Display filters never re-mine rules.
- [x] Build Backtest tables from the selected saved cost/scope, comparing the correct strategy and control paths. Show NIFTY as an uncosted price-index reference. Chart values use initial capital=100 and show test-session closes, with no invented observations.
- [x] Render the five views using native Streamlit widgets and a centered single-column layout. Keep user-facing text focused on experiment meaning, not file paths or implementation internals.
- [x] Provide a clear error if saved results are missing/corrupt. Never substitute fixtures or a previously cached unverified result.
- [x] Run AppTest and the adapter tests with warnings treated as errors; exercise the app with research/provider imports blocked.

## Task 3: Live local preview and QA

- [x] Run the full test suite and pip check. Complete the requesting-code-review checkpoint and address material findings.
- [x] Confirm the selected loopback port is available and start Streamlit in a retained session:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

- [ ] Check the printed local URL and health response, then open the local app using the available user-facing browser. Inspect navigation, Backtest interactions and narrow-screen rendering using supported browser controls. Record any viewport or browser verification limit accurately.
- [x] Run AppTest with external networking denied to verify core display behavior is independent of market APIs. Keep the live preview available for the user's review until project teardown.

## Task 4: Documentation and checkpoint

- [x] Record launch/check/package commands, actual results, pending views and any QA limits in README.md.
- [x] Verify curated artifacts match the original source bundle and existing research inputs/results remain unchanged.
- [x] Commit the app, curated results, theme and pinned dependencies on the current branch. Do not register a Site or publish the app in this local checkpoint.
- [x] Give the working local URL, runnable command, observed tests and a small actual result example with a Key Insight. Pause before Day 2 walk-forward work.

## Observed checkpoint results

- Installed Streamlit 1.67.0 and its dependencies with prior requirements as constraints; existing version pins were preserved and pip check passed.
- Curated eight original derived files under app/artifacts/day1_holdout, totaling 90,375 bytes. This supports app startup from a fresh checkout without raw data. No research or market-provider imports are used by app modules.
- New app checks: 22 passing; full suite: 142 passing with warnings treated as errors. All app tests also passed inside an OS sandbox denying all networking. Separate tests block research/provider imports and exercise Backtest.
- The native-controls, single-column preview has all five requested views. Saved Rules/Backtest outputs are displayed; tuning, asymmetry statistics and latest refresh remain explicitly unavailable rather than fabricated.
- Preview server runs at http://127.0.0.1:8501 in a retained session. Root and health endpoints returned HTTP 200. Keep this local preview available for user review during the ongoing project.
- Automated checks verify the actual empty-rule state, undefined metric formatting, selected stock/cost views and missing-results errors. At c=0.001 the saved basket displays -1.37%; at c=0.005 it displays -1.77%; the cash strategy stays 0.00%.
- Read-only code review found no material issues, independently checked all 12 scope/cost combinations and confirmed curated files match the original bundle.
- Visual QA limitation: CUA reported no available browser provider. Both observed native browser apps, Chrome and Brave, returned cgWindowNotFound. Desktop and narrow-viewport screenshots could not be inspected; record this as unverified rather than complete.
- Sites registration/publication was not performed: this is the explicitly staged local Streamlit preview, with hosting deferred to the planned deployment step. No source data or original experiment output was modified.
