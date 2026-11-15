"""Expiry-risk scoring: an explicit, configurable rule.

Risk definition
---------------
A batch is **at risk** when the demand FEFO can allocate to it before expiry is
less than its remaining quantity:

    estimated_unsold = max(0, remaining - allocatable)
    at_risk          = estimated_unsold > 0
    risk_score       = estimated_unsold / remaining, in [0, 1]

Severity is a *policy* layered on top, not a learned output. The table below is
the whole rule; there is no hidden heuristic. Day thresholds come from
configuration, so changing them changes the labels and the counts by design.

    unsold ratio        > 90 days      <= 90 days     <= 60 days     <= 30 days
    ----------------    -----------    -----------    -----------    -----------
    (0.00, 0.25]        low            low            medium         high
    (0.25, 0.50]        low            medium         high           critical
    (0.50, 1.00]        medium         high           critical       critical

A batch with zero estimated unsold is always ``low``, whatever its expiry date:
no exposure means no risk, and inflating the label would make precision
meaningless.
"""

from __future__ import annotations

import pandas as pd

from app.schemas.common import RiskLevel

RISK_POLICY_VERSION = "risk-v1"


def classify_severity(
    unsold_ratio: float, days_to_expiry: int, thresholds: dict[str, int]
) -> RiskLevel:
    """Apply the documented 2-D policy table."""
    if unsold_ratio <= 0.0:
        return RiskLevel.LOW

    critical_days = thresholds["critical"]
    high_days = thresholds["high"]
    medium_days = thresholds["medium"]

    if days_to_expiry <= critical_days:
        if unsold_ratio > 0.25:
            return RiskLevel.CRITICAL
        return RiskLevel.HIGH

    if days_to_expiry <= high_days:
        if unsold_ratio > 0.50:
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    if days_to_expiry <= medium_days:
        if unsold_ratio > 0.50:
            return RiskLevel.CRITICAL
        if unsold_ratio > 0.25:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    if unsold_ratio > 0.75:
        return RiskLevel.HIGH
    if unsold_ratio > 0.50:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def assess_batches(allocation: pd.DataFrame, thresholds: dict[str, int]) -> pd.DataFrame:
    """Add ``risk_score``, ``risk_level``, ``at_risk`` and human-readable reasons."""
    if allocation.empty:
        return allocation.assign(
            risk_score=pd.Series(dtype=float),
            risk_level=pd.Series(dtype="object"),
            at_risk=pd.Series(dtype=bool),
            reasons=pd.Series(dtype="object"),
        )

    frame = allocation.copy()
    remaining = frame["remaining_quantity"].replace(0, pd.NA).astype(float)
    frame["risk_score"] = (frame["unsold_quantity"] / remaining).fillna(0.0).clip(0.0, 1.0)
    frame["at_risk"] = frame["unsold_quantity"] > 0

    frame["risk_level"] = [
        classify_severity(float(score), int(days), thresholds).value
        for score, days in zip(frame["risk_score"], frame["days_to_expiry"], strict=True)
    ]

    frame["reasons"] = [_reason(row) for row in frame.itertuples(index=False)]
    return frame


def _reason(row) -> list[str]:
    reasons: list[str] = []
    if row.unsold_quantity <= 0:
        reasons.append("Expected demand before expiry covers the remaining quantity under FEFO.")
        return reasons

    reasons.append(
        f"FEFO can allocate only {row.allocatable_demand:,.0f} of "
        f"{row.remaining_quantity:,.0f} units before expiry "
        f"({row.days_to_expiry} days remain), leaving "
        f"{row.unsold_quantity:,.0f} units exposed."
    )
    if row.days_to_expiry <= 0:
        reasons.append("Batch has already passed its expiry date at the assessment date.")
    elif row.days_to_expiry <= 30:
        reasons.append(
            "Expiry is within 30 days: rotation or supplier return should be considered."
        )
    if row.risk_score > 0.5:
        reasons.append(
            f"{row.risk_score:.0%} of the batch is forecast to remain unsold, "
            "which is a majority of the batch."
        )
    reasons.append("Estimate assumes FEFO discipline and demand continuing at the forecast rate.")
    return reasons


def summarise(assessed: pd.DataFrame) -> dict[str, object]:
    """Aggregate counts used by the API and the dashboard."""
    if assessed.empty:
        return {
            "total_batches": 0,
            "at_risk_batches": 0,
            "estimated_unsold_units": 0.0,
            "by_risk_level": {level.value: 0 for level in RiskLevel},
        }
    return {
        "total_batches": int(len(assessed)),
        "at_risk_batches": int(assessed["at_risk"].sum()),
        "estimated_unsold_units": round(float(assessed["unsold_quantity"].sum()), 4),
        "by_risk_level": {
            level.value: int((assessed["risk_level"] == level.value).sum()) for level in RiskLevel
        },
    }
