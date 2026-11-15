"""Batch-level expiry-risk endpoints, computed from the saved model + FEFO."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.core.exceptions import NotFoundError
from app.schemas.common import RiskLevel
from app.schemas.expiry_risk import BatchRisk, ExpiryRiskResponse

router = APIRouter(prefix="/expiry-risk", tags=["expiry-risk"])

DISCLAIMER = (
    "Predicted expiry risk under a FEFO assumption. Actual unsold inventory can "
    "differ; decisions remain subject to human review."
)


def _to_batch_risk(row) -> BatchRisk:
    return BatchRisk(
        batch_id=str(row.batch_id),
        product_id=str(row.product_id),
        medicine_name=str(row.medicine_name),
        remaining_quantity=int(row.remaining_quantity),
        expiry_date=date.fromisoformat(str(row.expiry_date)[:10]),
        days_to_expiry=int(row.days_to_expiry),
        expected_demand_before_expiry=float(row.expected_demand_before_expiry),
        fefo_allocatable_demand=float(row.allocatable_demand),
        estimated_unsold_units=float(row.unsold_quantity),
        risk_score=float(row.risk_score),
        risk_level=RiskLevel(str(row.risk_level)),
        reasons=list(row.reasons),
        inputs={
            "remaining_quantity": int(row.remaining_quantity),
            "days_to_expiry": int(row.days_to_expiry),
            "expected_demand_before_expiry": float(row.expected_demand_before_expiry),
            "allocated_quantity": float(row.allocated_quantity),
            "risk_score": float(row.risk_score),
            "allocation_policy": "FEFO",
        },
    )


@router.get(
    "",
    response_model=ExpiryRiskResponse,
    summary="Batch-level expiry risk ranking (highest exposure first)",
)
def list_expiry_risk(
    container: ContainerDep,
    risk_level: Annotated[RiskLevel | None, Query(description="Filter by severity")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ExpiryRiskResponse:
    frame = container.expiry.rows()
    summary = container.expiry.assess().summary

    filtered = frame if risk_level is None else frame[frame["risk_level"] == risk_level.value]
    ordered = filtered.sort_values(
        ["risk_score", "days_to_expiry"], ascending=[False, True], kind="mergesort"
    )
    window = ordered.iloc[offset : offset + limit]

    by_level = {RiskLevel(level): int(count) for level, count in summary["by_risk_level"].items()}

    return ExpiryRiskResponse(
        as_of_date=date.fromisoformat(container.bundle.prediction_origin[:10]),
        horizon_days=container.bundle.horizon_days,
        total_batches=int(summary["total_batches"]),
        at_risk_batches=int(summary["at_risk_batches"]),
        estimated_unsold_units=float(summary["estimated_unsold_units"]),
        estimated_unsold_value_at_cost=round(
            float(frame["estimated_unsold_value_at_cost"].sum()), 2
        )
        if not frame.empty
        else 0.0,
        by_risk_level=by_level,
        items=[_to_batch_risk(row) for row in window.itertuples(index=False)],
        disclaimer=DISCLAIMER,
    )


@router.get(
    "/{batch_id}",
    response_model=BatchRisk,
    summary="Single batch risk assessment with the numbers behind the label",
    responses={404: {"description": "Unknown batch"}},
)
def get_batch_risk(batch_id: str, container: ContainerDep) -> BatchRisk:
    try:
        row = container.expiry.batch(batch_id)
    except NotFoundError:
        raise
    return _to_batch_risk(row)
