# Five-minute live demonstration

Use the deployed URL listed in DEPLOYMENT.md or start `make app`. Keep the PDF and local app ready if the network is unavailable. The live daily refresh is optional; every research result is precomputed.

1. **0:00-0:40 - Problem and contribution.** Explain that Phase 1 motivated adaptive forgetting, automatic thresholds and cost-aware validation. Say the project investigates these gaps. Show ten stocks, 44 folds and the short summary on Rules Explorer.
2. **0:40-1:30 - A dated rule.** Select the final tuned fold. Show its 27 October 2026 fitting cutoff, 250 fitting transactions and training-only confidence/lift. Explain that NEXT_ appears only in the consequent. A high training confidence is not a held-out success rate.
3. **1:30-2:20 - Tuning and drift.** Open Tuning & Drift. Show 27 wins for 250 sessions, 10 for 500, seven expanding and zero full-study fallbacks. Inspect one fold's 81 candidates. Explain common 63-session validation, fixed 0.1% selection cost and 20 trades across five sessions. Window changes do not prove concept drift.
4. **2:20-3:25 - Honest economic result.** Open Backtest, Portfolio, 0.10% cost. Tuned return is -13.30%, fixed +0.04%, holding +45.57%; NIFTY +44.32% is uncosted. Fixed makes only three trades. Select 0.50%: tuned becomes -43.78% while the decisions remain frozen. Explain 1/N capital, next-open entry and same-close exit. Holding has overnight exposure.
5. **3:25-4:10 - Directional observation.** Open Asymmetry. U precision is 38.08% versus weighted base 36.20%; D is 42.85% versus 41.12%. Lifts are about 1.052 and 1.042. Higher raw bearish precision does not establish higher relative lift or a universal advantage.
6. **4:10-4:40 - Latest daily demonstration.** Show model-fit and observation dates. Refresh only if useful; explain the 15-minute successful-fetch cache and dated fallback. Rules stay frozen and the current India session is excluded. A new bar does not produce a new trained model.
7. **4:40-5:00 - Correctness and limitations.** Mention future-data perturbation, chronological maturity, continuous capital, source hashes and tests. Point to the primary-source data correction audit. Disclose convenience-sample bias, static adjusted history and idealized costs/fills.

## Worked example from the actual ledger

On 24 February 2023, the first chronological tuned trade was ICICIBANK.NS. An INR 10,000 allocation entered at adjusted open 816.675700 and exited at 812.563049. Gross PnL was INR -50.358433, the 0.1% charge was INR 10, and net PnL was INR -60.358433. A correct backtest includes this loss.

## Likely questions

- **Why does tuning lose?** The output shows that this selection procedure failed to generalize economically on this sample. Validation search can overfit, patterns change and friction reduces returns. The project does not identify a single causal explanation.
- **Why use daily data?** The study needs interpretable completed observations and reproducible chronology within a four-day scope. Latest Signals is a daily demonstration rather than an intraday price feed.
- **Why not lower confidence until results improve?** Any change motivated by the held-out results would be exploratory and need a new untouched evaluation period.
- **Does the sliding-window frequency prove drift?** No. It records what past validation selected; an ablation or a dedicated drift study would be needed for a stronger claim.
- **What if the app/API fails?** Open the local app or report. Core views do not need Yahoo access. A failed refresh displays cached timestamps and status.

> Key Insight: Present the negative result as evidence about the tested procedure, not as a defect to hide.
