"""Replenishment-recommendation contract."""

from __future__ import annotations

from datetime import date

from pydantic import Field

from app.schemas.common import Priority, RecommendationAction, StrictModel


class Recommendation(StrictModel):
    product_id: str
    medicine_name: str
    action: RecommendationAction
    priority: Priority
    suggested_replenishment_qty: int = Field(
        ..., ge=0, description="0 for non-reorder actions. A suggestion, never an order."
    )
    forecast_demand: float = Field(..., ge=0)
    current_stock: int = Field(..., ge=0)
    safety_stock: int = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=0)
    coverage_days: float | None = None
    excess_units: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    inputs: dict[str, float | int | str | None] = Field(
        default_factory=dict,
        description="The numbers behind the decision. Null means the value is undefined "
        "(e.g. no expiry exposure, or zero forecast demand).",
    )


class RecommendationListResponse(StrictModel):
    generated_at: date
    horizon_days: int
    total: int
    items: list[Recommendation]
    disclaimer: str = (
        "Decision support only. No purchase order or disposal is executed by this system."
    )
