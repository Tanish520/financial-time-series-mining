"""Descriptive held-out directional accuracy, with group-weighted base rates."""

from __future__ import annotations

import pandas as pd

from src.data import DataValidationError


def score_asymmetry(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return U/D summaries and auditable fold/stock class-rate contributions.

    Different selected thetas imply different target definitions. Base rates
    therefore come from each fold/stock cohort, weighted by its forecast count.
    Realized labels are for analysis only and never feed rule selection.
    """
    required = {"variant", "fold_id", "ticker", "execution_session", "target_label", "prediction_label"}
    if outcomes.empty or not required.issubset(outcomes.columns):
        raise DataValidationError("Asymmetry requires dated predictions and held-out target labels.")
    if outcomes.duplicated(["variant", "fold_id", "ticker", "execution_session"]).any():
        raise DataValidationError("Duplicate forecast/target observations would bias asymmetry counts.")
    for name in ("target_label", "prediction_label"):
        if not (outcomes[name].isna() | outcomes[name].isin(["U", "D", "N"])).all():
            raise DataValidationError(f"Invalid {name}; expected U/D/N or unknown.")
    if (outcomes.prediction_label.isin(["U", "D"]) & outcomes.target_label.isna()).any():
        raise DataValidationError("A directional forecast requires a known held-out target before scoring.")
    records = []
    for (variant, fold_id, ticker), cohort in outcomes.groupby(["variant", "fold_id", "ticker"], sort=True):
        known = cohort.loc[cohort.target_label.notna()]
        for direction in ("U", "D"):
            forecasts = known.prediction_label.eq(direction).fillna(False)
            count = int(forecasts.sum())
            correct = int((forecasts & known.target_label.eq(direction)).sum())
            base_rate = None if known.empty else float(known.target_label.eq(direction).mean())
            records.append({
                "variant": variant, "fold_id": fold_id, "ticker": ticker, "direction": direction,
                "observations": len(known), "prediction_count": count, "correct_count": correct,
                "class_base_rate": base_rate,
                "expected_correct": 0.0 if not count else count * base_rate,
            })
    groups = pd.DataFrame(records)
    summaries = []
    for variant in sorted(outcomes.variant.unique()):
        for direction in ("U", "D"):
            cohort = groups.loc[groups.variant.eq(variant) & groups.direction.eq(direction)]
            count, correct = int(cohort.prediction_count.sum()), int(cohort.correct_count.sum())
            expected = float(cohort.expected_correct.sum())
            summaries.append({
                "variant": variant, "direction": direction, "prediction_count": count,
                "correct_count": correct, "precision": correct / count if count else None,
                "base_rate": expected / count if count else None, "expected_correct": expected,
                "held_out_lift": correct / expected if expected > 0 else None,
            })
    return pd.DataFrame(summaries), groups
