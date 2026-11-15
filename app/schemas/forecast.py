"""Demand-forecast contract.

Definition frozen here (and in docs/):
  unit      = one product_id
  origin    = a fixed historical cutoff date; only data <= origin is used
  horizon   = next N days after the origin
  target    = total units demanded for that product across those N days
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import StrictModel


class ForecastRequest(StrictModel):
    product_id: str = Field(..., examples=["MED-0001"])
    horizon_days: int = Field(default=30, ge=1, le=180, examples=[30])
    as_of_date: date | None = Field(
        default=None,
        description="Prediction origin. Omit to use the artifact's training origin.",
    )


class HistoryPoint(StrictModel):
    date: date
    quantity: int = Field(..., ge=0)


class DemandDriver(StrictModel):
    """One contributing factor for *this* prediction.

    ``contribution`` is a signed magnitude in the target's units when the model
    exposes one; otherwise it is null and ``explanation`` carries the evidence.
    We never invent a contribution number.
    """

    feature: str
    value: float | None = None
    reference_value: float | None = Field(
        default=None,
        description=(
            "Population reference value the feature was moved to when measuring the effect."
        ),
    )
    contribution: float | None = None
    direction: Literal["increases_demand", "decreases_demand", "neutral"]
    method: str | None = Field(
        default=None,
        description=(
            "Attribution method. This is not a Shapley value and must not be " "described as one."
        ),
    )
    explanation: str


class PredictionInterval(StrictModel):
    lower: float = Field(..., ge=0)
    upper: float = Field(..., ge=0)
    method: str = Field(
        ...,
        examples=["empirical_residual_quantiles_10_90"],
        description="How the interval was derived. Never a hardcoded +-20%.",
    )
    coverage: float = Field(..., gt=0, lt=1)


class ForecastResponse(StrictModel):
    product_id: str
    medicine_name: str
    as_of_date: date
    horizon_days: int
    predicted_demand: float = Field(..., ge=0, description="Expected units over the horizon.")
    prediction_interval: PredictionInterval | None = None
    model_name: str
    artifact_version: str
    confidence: Literal["low", "medium", "high"]
    drivers: list[DemandDriver] = Field(default_factory=list)
    recent_history: list[HistoryPoint] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list,
        description="Qualifiers a reader must see, e.g. sparse history or wide interval.",
    )
    disclaimer: str = (
        "Demand forecast for inventory decision support only. Not medical advice. "
        "Subject to human review."
    )
