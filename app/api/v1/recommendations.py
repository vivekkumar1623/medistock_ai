"""Replenishment and rotation recommendation endpoints."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.schemas.common import Priority, RecommendationAction
from app.schemas.recommendation import Recommendation, RecommendationListResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

DISCLAIMER = "Decision support only. No purchase order or disposal is executed by this system."


@router.get(
    "",
    response_model=RecommendationListResponse,
    summary="Ordered replenishment and stock-rotation recommendations",
)
def list_recommendations(
    container: ContainerDep,
    action: Annotated[RecommendationAction | None, Query(description="Filter by action")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RecommendationListResponse:
    frame = container.recommendations.build()
    if action is not None:
        frame = frame[frame["action"] == action.value]

    window = frame.iloc[offset : offset + limit]
    items = [
        Recommendation(
            product_id=str(row.product_id),
            medicine_name=str(row.medicine_name),
            action=RecommendationAction(str(row.action)),
            priority=Priority(str(row.priority)),
            suggested_replenishment_qty=int(row.suggested_replenishment_qty),
            forecast_demand=float(row.forecast_demand),
            current_stock=int(row.current_stock),
            safety_stock=int(row.safety_stock),
            lead_time_days=int(row.lead_time_days),
            coverage_days=None if row.coverage_days is None else float(row.coverage_days),
            excess_units=int(row.excess_units),
            reasons=list(row.reasons),
            inputs=dict(row.inputs),
        )
        for row in window.itertuples(index=False)
    ]

    return RecommendationListResponse(
        generated_at=date.fromisoformat(container.bundle.prediction_origin[:10]),
        horizon_days=container.bundle.horizon_days,
        total=int(len(frame)),
        items=items,
        disclaimer=DISCLAIMER,
    )
