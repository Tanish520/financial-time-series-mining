# Financial time-series mining: proposed Phase 2 design

Status: approved by the team's instruction to start working; implementation proceeds one user-confirmed checkpoint at a time.

Prepared on 18 November 2026 for Tanish Kumar Gupta and Aayush Gupt, CS G520 Advanced Data Mining, BITS Pilani Hyderabad.

## 1. Goal, scope and decisions needing approval

Build a responsive Streamlit application that shows whether interpretable next-session association rules remain useful out of sample after trading costs. The project addresses or investigates G1 (changing market relationships), G2 (manual threshold selection), and G3 (weak economic validation). It does not claim to solve them.

Keep the team's proposed approach: daily stock data, U/D/N labels, Apriori, long-only intraday trades, nested chronological tuning, and precomputed app results. No deep learning, clustering, text mining, database, login, or native mobile application.

Recommended changes to the original brief:

1. Compare all window lengths on the same 63-session validation block. The 250/500/expanding choices refer to fitting history preceding that block, rather than including validation. This replaces a different last-20% block for every candidate.
2. Use one selected parameter configuration per outer fold for the combined portfolio. Avoid a separate 81-configuration search for each stock.
3. Require at least 20 validation trades spread across at least five execution sessions. These are feasibility guards, not statistical-significance guarantees.
4. Tune at a fixed 0.1% total round-trip cost. Apply all three reporting cost levels to the same frozen out-of-sample signals.
5. Add one predetermined, untuned rule-mining baseline to contextualize the benefit or harm of tuning.

Alternatives considered: separate last-20% blocks preserve the wording of the initial prompt but compare candidates on different market periods; a common 100-session block preserves 20% of a 500-session reference window but reserves more data. The common 63-session block is the recommended compromise for a six-year dataset and four-day project.

The instructor's acceptance of a web app and the exact submission deadline have not been supplied. Treat Streamlit and the four relative working days as planning assumptions. Confirm those administrative details separately; do not invent an approval or a calendar deadline.

## 2. System and repository

Data flow: download once -> validate and freeze a data snapshot -> calculate dated features and targets -> tune and mine rules -> generate chronological predictions -> simulate trades -> save results -> display in Streamlit.

| Component | Responsibility | Main output |
| --- | --- | --- |
| src/data.py | Explicit daily download settings, schema checks, calendar alignment, snapshot metadata and cache loading | Cached OHLCV and manifest |
| src/labels.py | Close-to-close features and next-session open-to-close targets, with availability dates | Dated feature/target tables |
| src/mining.py | Boolean transactions, Apriori, rule filtering and deterministic matching | Rules and selected predictions |
| src/walkforward.py | Common-date inner validation, parameter selection and frozen outer tests | Tuning log and out-of-sample predictions |
| src/backtest.py | Capital allocation, modeled costs, benchmarks and metrics | Trade ledger, equity curves and metrics |
| src/pipeline.py | One-command orchestration using the frozen cache | A versioned result bundle |
| app/ | Five single-column Streamlit views reading saved results | Responsive application |
| tests/ | Temporal, rule-direction and accounting checks | Repeatable verification |

Keep data/, src/, app/, tests/, README.md and requirements.txt as requested. Use data/raw/ for the frozen download, data/processed/ for derived labels, and data/results/ for published experiment outputs. The manifest records configuration, data checksum, date coverage, code version when available, and dependency versions. CSV plus JSON is sufficient; avoid adding a storage service.

The experiment command must run with network access disabled after the initial download. Streamlit does not mine or tune during page loads. Failed downloads or absent result bundles produce explicit errors, not invented demonstration results. Any temporary UI fixtures must be clearly marked as synthetic.

## 3. Data contract

Initial integration universe: RELIANCE.NS, TCS.NS and HDFCBANK.NS. Full proposed universe: those three plus INFY.NS, ICICIBANK.NS, SBIN.NS, ITC.NS, LT.NS, SUNPHARMA.NS and MARUTI.NS. Fetch ^NSEI as a benchmark. Verify symbols by successful provider responses and metadata before accepting them into the snapshot; the proposed names are not yet verified.

Request approximately six years of daily data through a fixed, completed trading session. Record the actual start/end dates when the snapshot is created. Explicitly configure adjusted OHLC and save corporate-action information when provided. Never mix adjusted close with unadjusted open in a return calculation. Modern historical adjusted data are not a point-in-time archive; disclose that limitation.

Checks: unique chronological dates, all required price columns, positive prices, consistent high/low bounds, nonnegative stock volume, and adequate coverage. Do not require positive volume from the index. Do not forward-fill missing prices or turn them into neutral labels. Build a shared session calendar from the benchmark, flag missing stock bars, and fail on unexplained missing observations within the accepted common coverage. An explicitly approved coverage change requires a new manifest.

Today's selected companies are a convenience sample, not a reconstruction of historical NIFTY membership. State the resulting survivorship and selection bias. Do not describe results as performance across the historical NIFTY 50 universe.

## 4. Features, targets and eligible rules

For stock s and session t:

- Feature return: close[s,t] / close[s,t-1] - 1.
- Target return attached to feature session t: close[s,t+1] / open[s,t+1] - 1.
- Label U if return > theta, D if return < -theta, otherwise N. Equality is N.
- Theta candidates: 0.0025, 0.005 and 0.01, meaning 0.25%, 0.5% and 1%.

Each sample records feature_session, execution_session and outcome_available_session. A feature is available after its session has completed; its target is available only after the following trading session has completed. The final feature row can support a new signal even though its target is unknown; it cannot be used for training or scored evaluation.

Each training transaction contains one observed label per stock for session t and one NEXT_ target label per stock for session t+1. Sharing transactions across targets keeps computation small. Each extracted rule must have:

- One or two antecedent items, all from the feature session; zero NEXT_ items in the antecedent.
- Exactly one consequent, which must be a NEXT_ item for a target stock.
- Total itemset length at most three, including the consequent.
- Training support and confidence meeting the candidate configuration, and training lift strictly greater than one.

Generate itemsets with mlxtend Apriori and rules with association_rules. For each theta/window pair, mine at the lowest support and confidence thresholds and filter the resulting rules for stricter grid choices. This retains 81 scored configurations while reducing repeated mining work to nine main fits per validation split.

For each stock and signal session, select the matching rule with the highest training confidence. If equally confident rules disagree on the target class, choose cash. Among tied rules for the same class, use lift, support, shorter antecedent and canonical item ordering for deterministic provenance. U means long; D, N, no match or a conflicting tie means cash. D predictions are still recorded for the asymmetry analysis. Never use test outcomes or test lift to choose a rule.

## 5. Chronological evaluation

Define splits using execution sessions, not an ambiguous row-number split.

At an outer cutoff close c, all samples with outcome_available_session <= c are potentially eligible. The next up-to-21 execution sessions form the outer test block. The first test signal uses information through close c and executes in the following session.

Inner selection at cutoff c:

1. Take the most recent 63 eligible execution sessions as the common validation block.
2. Fit each candidate using the 250, 500 or all available eligible samples preceding that validation block. Every candidate ends fitting at the same information cutoff.
3. Freeze candidate rules throughout the 63-session validation block. Select using validation portfolio net return at the predetermined 0.1% round-trip cost and the trade-count guard.
4. Grid: support {0.01, 0.02, 0.05}; confidence {0.55, 0.60, 0.70}; theta {0.0025, 0.005, 0.01}; fitting history {250, 500, expanding}. This gives 81 scored configurations.
5. Break equal validation-return ties by the fixed ordering of the serialized configuration, rather than another performance measure chosen after seeing results.
6. Re-mine with the winning configuration using eligible history through c, retaining the chosen fitting-history length. Validation outcomes are now historical and may be included in this final fit.
7. Freeze these rules for the next 21 execution sessions. At the next cutoff, repeat the full process using only information then available.

If no candidate passes the minimum-trade guard, record an explicit no-eligible-configuration fold and remain in cash for that outer block. Keep the dates in the equity curve. Score a final partial outer block if one remains, without changing the process. Start evaluation only after every window option has enough history for fitting plus the common validation period.

Earlier outer-test outcomes may become ordinary historical training data in later folds once observed. Never use current or future test outcomes to make a decision, and never revise earlier predictions retrospectively.

Record every candidate's validation score, eligibility, trade count and fitting dates, not just the winner. Also record chosen parameters, outer dates, final fitting dates, final rule count and fallback status for every fold. Count window-selection frequencies only among folds with an eligible winner, and display the no-trade fallback count separately.

Freeze the grid and evaluation rules before examining outer-test performance. Searching 81 candidates on a short validation block can overfit validation; outer performance is the check on this optimism. Changes motivated by test results must be disclosed as exploratory. Variable window selection is not itself proof that concept drift was detected.

## 6. Trading, costs, benchmarks and metrics

Use a simple fractional-share, no-leverage simulation. Each stock receives 1/N of current portfolio equity as a daily allocation. A U signal invests that allocation at the next open and exits at that session's close. Unused allocations remain cash; cash earns zero. Never give each simultaneous signal the entire portfolio capital.

For an executed stock trade, modeled return on allocated capital is (close/open - 1) - c, where c is the total round-trip charge expressed as a fraction of entry notional. Portfolio daily return is the sum of stock returns times their 1/N weights, with zero for cash allocations. The ledger stores actual simulated entry notional, gross profit/loss, modeled charge and net profit/loss. These are idealized daily-open/close fills; the cost scenarios approximate execution friction rather than reproduce broker-specific charges.

Report c in {0.0005, 0.001, 0.005}: 0.05%, 0.1%, 0.5%. These are total round-trip assumptions. The cost selector recomputes accounting for the same decisions; it does not retune the model. Verify the original Nair paper's cost convention before claiming the 0.5% scenario directly replicates that paper.

Controls over identical outer-test execution dates:

- Fixed rule-mining baseline: 500-session fitting history, theta=0.005, support=0.02, confidence=0.60, same lift filter and trading policy. Re-mine at the same outer cutoffs, with no inner parameter selection.
- Per-stock buy-and-hold: enter at the first common test open, hold through the last test close, and apply the scenario's round-trip charge once. For daily equity accounting, deduct half the charge at entry and half at final exit, both referenced to initial invested notional.
- Equal-initial-weight buy-and-hold basket: average the normalized per-stock holding paths, without daily rebalancing.
- NIFTY 50 price-index reference: use ^NSEI over the same interval. Label it as a price-index reference, without dividends or an investable fund's implementation costs. Stock buy-and-hold from adjusted OHLC is an adjusted-price proxy, so disclose that distinction.

Report portfolio and per-stock net return, CAGR, daily Sharpe, max drawdown, net profit factor, net win rate and round-trip trade count. Use all test sessions, including cash days. CAGR uses elapsed calendar time; daily Sharpe uses 252 sessions/year, sample standard deviation, and a stated zero risk-free rate. Start drawdown calculations from initial capital, so a first-day loss counts. Calculate win rate and profit factor from the net trade ledger. Zero-trade and zero-variance cases display meaningful zeros or N/A; profit factor is infinite only when there are positive gains and no losses. Do not hide losing trades or empty folds.

The tuned-versus-fixed comparison evaluates the combined tuning procedure. Without an additional ablation, it cannot attribute improvement specifically to G1 rather than G2.

## 7. Asymmetry analysis

Use one selected prediction per stock/session from frozen outer-test rules, including D predictions that do not create a short trade. For U and D separately, show prediction count, correct count, precision, corresponding class base rate and held-out lift. Do not compare training confidence as if it were held-out prediction accuracy.

Because theta may differ across folds and class rates differ across stocks, compute base rates within each fold/stock using that fold's selected theta. Aggregate expected correct counts as the sum of prediction_count * corresponding_test_class_rate. Aggregate lift is total correct predictions divided by that expected count. Display precision and its prediction-weighted base rate alongside lift. Test base rates are used only for descriptive analysis after predictions are frozen.

Use one compact two-panel chart for U versus D precision/base rate and U versus D lift, with counts immediately below. No predictions or a zero base-rate denominator produces N/A, not a fabricated value. Interpret the result as a descriptive test of the earlier observation; do not claim a universal bearish advantage or formal significance.

## 8. Application and result contract

Use one Streamlit app with simple navigation and single-column content.

| View | Required content |
| --- | --- |
| Rules Explorer | Short project summary, selected fold, target/direction filters, antecedent, consequent, support, confidence and training lift; fitting dates clearly visible |
| Tuning & Drift | Fold winners, validation scores and trade counts, window-selection frequencies, no-eligible-configuration count, and an explanation that this investigates adaptive forgetting |
| Backtest | Portfolio/stock selector, cost selector, held-out equity curves against relevant controls, metrics and execution assumptions |
| Asymmetry | Held-out U/D comparison, base rates, counts and a brief interpretation based on computed results |
| Latest Signals | Best-effort recent completed daily bars, signal and rule provenance, data timestamp, rule-fit timestamp, freshness/fallback status and the requested educational disclaimer |

Save final rules by fold, all tuning records, out-of-sample predictions, trade ledgers, equity curves, metric tables, asymmetry summaries and a run manifest. Never present training support/lift and held-out metrics without distinguishing them.

Latest Signals is separate from historical evaluation. Make refresh explicit, cache successful fetches for 15 minutes, reject incomplete/current-session bars, and fall back to the frozen snapshot with a visible stale-data label. All predictor stocks must refer to the same completed session and have a preceding close. If that cannot be established, show that a fresh signal is unavailable. Do not silently backfill predictors.

Apply the rules from the final historical fold, with their recorded theta; this is deliberately a frozen-rule demonstration. Display both rule fitting cutoff and observation timestamp so users can see any age difference. A fresh data fetch does not retrain the model or alter the historical result bundle. Label the intended execution as the next trading session rather than guessing a calendar date on weekends/holidays. Display: "Educational project; not investment advice."

## 9. Verification and reproducibility

Required checks before reporting results:

1. Future-data perturbation: change every bar strictly after session t; all features and the already-frozen model's signal for t must remain unchanged. Future targets may change.
2. Availability boundaries: no sample admitted to a fit has an outcome available after that fit's cutoff; validation/test execution dates remain separated as specified.
3. Direction guard: a rule with any NEXT_ antecedent or multiple consequents cannot reach the matcher.
4. Accounting: known toy prices produce the expected flat round-trip charge; simultaneous signals do not exceed available capital; a first-day loss appears in drawdown.
5. Determinism: fixed seed for any stochastic dependency, stable ordering/tie resolution, pinned dependencies and a frozen input checksum reproduce the same saved numeric outputs.
6. Offline app smoke check: core pages load from saved results with networking disabled, including explicit no-rules/no-trades states. Verify a narrow phone-sized viewport manually.

Run one command for the experiment against the existing snapshot and one command for tests; provide the exact runnable commands when implementing them. Do not claim test passes, timings, returns or successful downloads until observed. Use the same Python version locally and in deployment.

## 10. Four-day execution and checkpoints

| Day | Research/data work | App/document work | Completion evidence |
| --- | --- | --- | --- |
| 1 | Three-stock frozen data, labels, guarded mining, simple chronological holdout backtest | App skeleton against clearly marked synthetic fixtures | Download validation summary, example dated sample/rule, temporal tests, local app |
| 2 | Full walk-forward loop, all 81 candidate scores, chosen windows, fixed baseline | Rules and tuning pages wired to saved results | One inspected fold plus full offline reproducible run |
| 3 | Full universe if healthy, three costs, aligned controls, ledger/metrics, held-out asymmetry | Backtest and asymmetry pages with computed interpretation | Accounting tests and saved results reviewed for leaks |
| 4 | Latest Signals if time remains; final checks | Deployment, concise report, README and live-demo script | Working URL, reproducible repository, report and tested local fallback |

Proposed team split: Tanish owns temporal data flow, tuning and backtest correctness; Aayush owns Streamlit pages, result-file integration and report/demo assembly. Both inspect the same example fold and ledger before interpreting performance. Independent work begins only after agreeing the shared artifact schemas. This proposes human team responsibilities; it does not authorize automated subagents.

Work one checkpoint at a time. After each implementation step, explain the change, supply runnable code/commands, state expected structural output, report actual checks, include a small numerical example with a Key Insight box, and wait for Tanish's confirmation. Refine existing files rather than replacing the project wholesale.

Cut in the agreed order: stretch goals, extra phone polish, Latest Signals, then reduce to five stocks. Preserve walk-forward evaluation, cost-aware comparison with buy-and-hold, and an app displaying those results.

Deployment target: Streamlit Community Cloud, assuming account access is available. Keep an independently runnable local demo as a backup. Confirm the hosting environment's supported Python version and use pinned requirements. Do not claim deployment is complete without visiting and testing the deployed app.

## 11. Worked arithmetic example, not a market result

If hypothetical closes are 200 then 202, the feature return is 1%. With theta=0.5%, the feature label is U. If the following session opens at 100 and closes at 101, the target return is also 1% and its label is U. The latter label becomes available only after the following close.

For a hypothetical 10,000 INR allocated trade at those open/close prices, the model gives 100 INR gross profit. A 0.1% flat round-trip charge on entry notional is 10 INR, leaving 90 INR net profit (0.9%). In the portfolio this is weighted by the stock's allocation. These inputs illustrate definitions and are not observed financial data or claimed experiment performance.

> Key Insight: tomorrow's outcome may label yesterday's training example only after tomorrow has finished. All candidate models must face the same validation dates, and all reported strategies must respect a single finite capital budget.

## 12. Sources and limits

Project requirements and research motivation come from the two PDFs in Project Information and the team's Phase 2 instructions. Original-paper claims, including exact cost conventions, require primary-source checks before appearing as verified facts in the final report. Cite Srivastava et al. (2024), Nair et al. (2015, 2017), and Tusien et al. (2025) for methodological context, without claiming a direct replication on different data.

Primary software references checked during design:

- yfinance download options: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- mlxtend rule definitions: https://rasbt.github.io/mlxtend/api_modules/mlxtend.frequent_patterns/association_rules/
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit dependencies and Python-version guidance: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

This document defines the approved design, not a performance report. See README.md and the per-checkpoint implementation plans for actual progress; empirical performance must come from the implemented evaluation, not this specification.
