"""Batch-level expiry-risk contract.

Risk definition (single source of truth, mirrored in ml/expiry_risk/):
  A batch is AT RISK when the demand that FEFO would allocate to it on or
  before its expiry date is less than its remaining quantity.
  estimated_unsold = max(0, remaining_quantity - fefo_allocatable_demand)
  risk_score       = estimated_unsold / remaining_quantity, in [0, 1]
Severity bands (low/medium/high/critical) combine risk_score with
days_to_expiry, using thresholds from configuration. Neither is invented
per-view.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from app.schemas.common import RiskLevel, StrictModel


class BatchRisk(StrictModel):
    batch_id: str
    product_id: str
    medicine_name: str
    remaining_quantity: int = Field(..., ge=0)
    expiry_date: date
    days_to_expiry: int
    expected_demand_before_expiry: float = Field(
        ..., ge=0, description="Forecast demand for the product over [origin, expiry]."
    )
    fefo_allocatable_demand: float = Field(
        ...,
        ge=0,
        description="Demand FEFO assigns to this batch after earlier batches consume theirs.",
    )
    estimated_unsold_units: float = Field(..., ge=0)
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    reasons: list[str] = Field(default_factory=list)
    inputs: dict[str, float | int | str] = Field(
        default_factory=dict, description="The numbers behind the label, for audit."
    )


class ExpiryRiskResponse(StrictModel):
    as_of_date: date
    horizon_days: int
    total_batches: int
    at_risk_batches: int
    estimated_unsold_units: float
    estimated_unsold_value_at_cost: float
    by_risk_level: dict[RiskLevel, int]
    items: list[BatchRisk]
    disclaimer: str = (
        "Predicted expiry risk under a FEFO assumption. Actual unsold inventory can "
        "differ; decisions remain subject to human review."
    )
