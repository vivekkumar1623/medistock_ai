"""Demand-forecast endpoints, served from the saved model artifact."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.core.exceptions import InvalidRequestError
from app.schemas.forecast import (
    DemandDriver,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    PredictionInterval,
)
from app.services.forecast_service import ForecastRow

router = APIRouter(prefix="/forecast", tags=["forecast"])

DISCLAIMER = (
    "Demand forecast for inventory decision support only. Not medical advice. "
    "Subject to human review."
)


def _check_horizon(requested: int | None, trained: int) -> int:
    """The artifact answers exactly one question; refuse to answer another."""
    if requested is None or requested == trained:
        return trained
    raise InvalidRequestError(
        f"This model is trained for a {trained}-day horizon and cannot answer a "
        f"{requested}-day question. Retrain with MEDITSTOCK_FORECAST_HORIZON_DAYS={requested}.",
        details={"requested_horizon_days": requested, "trained_horizon_days": trained},
    )


def _check_origin(requested, trained: str) -> str:
    if requested is None:
        return trained
    requested_iso = requested.isoformat()
    if requested_iso != trained:
        raise InvalidRequestError(
            "This artifact holds demand history up to a single prediction origin; "
            "forecasting from an arbitrary date would require rebuilding the feature "
            "snapshot.",
            details={"requested_origin": requested_iso, "artifact_origin": trained},
        )
    return trained


def _to_response(
    row: ForecastRow,
    history: list[dict],
    origin: str,
    horizon: int,
    artifact_version: str,
    model_name: str,
    interval_method: str | None,
    interval_coverage: float,
) -> ForecastResponse:
    interval = None
    if row.lower is not None and row.upper is not None:
        interval = PredictionInterval(
            lower=row.lower,
            upper=row.upper,
            method=interval_method or "empirical_residual_quantiles_10_90",
            coverage=interval_coverage,
        )
    return ForecastResponse(
        product_id=row.product_id,
        medicine_name=row.medicine_name,
        as_of_date=date.fromisoformat(origin[:10]),
        horizon_days=horizon,
        predicted_demand=row.predicted_demand,
        prediction_interval=interval,
        model_name=model_name,
        artifact_version=artifact_version,
        confidence=row.confidence,  # type: ignore[arg-type]
        drivers=[DemandDriver(**driver) for driver in row.drivers],
        recent_history=[HistoryPoint(**point) for point in history],
        notes=row.notes,
        disclaimer=DISCLAIMER,
    )


def _build(
    container: ContainerDep, product_id: str, horizon: int | None, as_of_date
) -> ForecastResponse:
    bundle = container.bundle
    horizon_days = _check_horizon(horizon, bundle.horizon_days)
    origin = _check_origin(as_of_date, bundle.prediction_origin)

    row = container.forecasts.forecast_one(product_id)
    history = container.forecasts.recent_history(product_id, days=60)
    interval_method = (bundle.metrics.get("residual_quantiles") or {}).get("method")

    return _to_response(
        row=row,
        history=history,
        origin=origin,
        horizon=horizon_days,
        artifact_version=bundle.artifact_version,
        model_name=bundle.model_name,
        interval_method=interval_method,
        interval_coverage=0.80,
    )


@router.get(
    "/{product_id}",
    response_model=ForecastResponse,
    summary="30-day demand forecast for one product at the artifact's prediction origin",
    responses={404: {"description": "Unknown product"}, 503: {"description": "No artifact bundle"}},
)
def forecast_product(
    product_id: str,
    container: ContainerDep,
    horizon_days: Annotated[
        int | None, Query(ge=1, le=180, description="Must match the trained horizon")
    ] = None,
) -> ForecastResponse:
    return _build(container, product_id, horizon_days, None)


@router.post(
    "",
    response_model=ForecastResponse,
    summary="Forecast with an explicit product, horizon and prediction origin",
)
def forecast_post(payload: ForecastRequest, container: ContainerDep) -> ForecastResponse:
    return _build(container, payload.product_id, payload.horizon_days, payload.as_of_date)
