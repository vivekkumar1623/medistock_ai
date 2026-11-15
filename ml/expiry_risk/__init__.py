"""Expiry-risk package: FEFO allocation, risk policy, realised labels, metrics."""

from __future__ import annotations

from ml.expiry_risk.evaluate import evaluate_risk
from ml.expiry_risk.fefo import FefoPolicy, allocate_fefo, total_unsold
from ml.expiry_risk.labels import realised_outcomes
from ml.expiry_risk.risk import (
    RISK_POLICY_VERSION,
    assess_batches,
    classify_severity,
    summarise,
)

__all__ = [
    "RISK_POLICY_VERSION",
    "FefoPolicy",
    "allocate_fefo",
    "assess_batches",
    "classify_severity",
    "evaluate_risk",
    "realised_outcomes",
    "summarise",
    "total_unsold",
]
