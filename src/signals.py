"""Match frozen rules against observed features without accepting outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import DataValidationError
from src.mining import MiningResult

FEATURE_COLUMNS = {
    "feature_session", "feature_available_session", "ticker",
    "feature_return", "feature_label", "feature_item",
}
SIGNAL_COLUMNS = [
    "feature_session", "ticker", "prediction_label", "action", "reason",
    "matched_antecedents", "consequent", "confidence", "lift", "support",
    "matched_rule_count", "rule_fit_cutoff",
]


def predict_signals(
    model: MiningResult, features: pd.DataFrame, stocks: tuple[str, ...]
) -> pd.DataFrame:
    """Choose the highest-confidence matching class; conflicting exact ties mean cash."""
    if set(features.columns) != FEATURE_COLUMNS:
        raise DataValidationError("The feature interface accepts only observed feature columns, never targets.")
    if not stocks or len(set(stocks)) != len(stocks):
        raise DataValidationError("Provide a nonempty set of distinct stock predictors.")
    cutoff = pd.Timestamp(model.fit_cutoff)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise DataValidationError("The model fit cutoff must be a session-close date.")
    observed = features.copy()
    for name in ("feature_session", "feature_available_session"):
        dates = pd.DatetimeIndex(observed[name])
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise DataValidationError("Feature availability requires nonmissing daily session dates.")
        observed[name] = dates
    if (observed.feature_session < cutoff).any():
        raise DataValidationError("The model fit cutoff is after a requested feature session.")
    if not observed.feature_available_session.equals(observed.feature_session):
        raise DataValidationError("Feature availability does not match its observation session.")
    if observed.duplicated(["feature_session", "ticker"]).any():
        raise DataValidationError("Duplicate feature rows cannot form a predictor cohort.")
    complete = observed.groupby("feature_session").ticker.agg(lambda values: set(values) == set(stocks))
    if not complete.all():
        raise DataValidationError("Every signal requires the complete predictor stock cohort.")
    if not observed.feature_label.isin(["U", "D", "N"]).all():
        raise DataValidationError("Missing or invalid feature labels cannot produce signals.")
    expected = observed.ticker.str.removesuffix(".NS") + "_" + observed.feature_label
    if not observed.feature_item.eq(expected).fillna(False).all():
        raise DataValidationError("Feature items must match their stock's observed label.")
    if not np.isfinite(observed.feature_return.to_numpy(dtype=float, na_value=np.nan)).all():
        raise DataValidationError("Missing or infinite feature returns cannot produce signals.")

    allowed_features = {f"{ticker.removesuffix('.NS')}_{label}" for ticker in stocks for label in "UDN"}
    allowed_targets = {
        f"NEXT_{ticker.removesuffix('.NS')}_{label}": (ticker, label)
        for ticker in stocks for label in "UDN"
    }
    normalized = []
    for row in model.rules.to_dict("records"):
        antecedents = row["antecedents"]
        if not isinstance(antecedents, (tuple, list, set, frozenset)):
            raise DataValidationError("Rule antecedents must be a collection of feature items.")
        antecedents = tuple(sorted(antecedents))
        if not 1 <= len(antecedents) <= 2 or not set(antecedents).issubset(allowed_features):
            raise DataValidationError("Invalid antecedent: NEXT_ and unknown items are forbidden.")
        target = allowed_targets.get(row["consequent"])
        if target is None or target != (row["target_ticker"], row["target_label"]):
            raise DataValidationError("The rule consequent must be one consistent NEXT_ target.")
        if not all(np.isfinite(row[name]) for name in ("confidence", "lift", "support")):
            raise DataValidationError("Rule scores must be finite training metrics.")
        if not 0 <= row["confidence"] <= 1 or not 0 < row["support"] <= 1 or row["lift"] <= 1:
            raise DataValidationError("Rule scores do not satisfy the mining contract.")
        normalized.append({**row, "antecedents": antecedents})

    output = []
    for session, cohort in observed.groupby("feature_session", sort=True):
        items = set(cohort.feature_item)
        for ticker in sorted(stocks):
            matching = [row for row in normalized if row["target_ticker"] == ticker and set(row["antecedents"]).issubset(items)]
            decision = {
                "feature_session": session, "ticker": ticker,
                "prediction_label": pd.NA, "action": "CASH", "reason": "no_match",
                "matched_antecedents": (), "consequent": pd.NA,
                "confidence": np.nan, "lift": np.nan, "support": np.nan,
                "matched_rule_count": len(matching), "rule_fit_cutoff": cutoff,
            }
            if matching:
                confidence = max(row["confidence"] for row in matching)
                winners = [row for row in matching if row["confidence"] == confidence]
                if len({row["target_label"] for row in winners}) > 1:
                    decision.update(reason="confidence_tie", confidence=confidence)
                else:
                    chosen = min(winners, key=lambda row: (
                        -row["lift"], -row["support"], len(row["antecedents"]),
                        row["antecedents"], row["consequent"],
                    ))
                    decision.update(
                        prediction_label=chosen["target_label"],
                        action="LONG" if chosen["target_label"] == "U" else "CASH",
                        reason="matched_rule", matched_antecedents=chosen["antecedents"],
                        consequent=chosen["consequent"], confidence=confidence,
                        lift=chosen["lift"], support=chosen["support"],
                    )
            output.append(decision)
    return pd.DataFrame(output, columns=SIGNAL_COLUMNS)
