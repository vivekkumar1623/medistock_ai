"""Pydantic contracts shared by the API, dashboard and tests."""

from __future__ import annotations

from app.schemas.common import (
    ComponentHealth,
    DataFreshness,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    Priority,
    RecommendationAction,
    RiskLevel,
    StrictModel,
)
from app.schemas.expiry_risk import BatchRisk, ExpiryRiskResponse
from app.schemas.forecast import (
    DemandDriver,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    PredictionInterval,
)
from app.schemas.inventory import InventoryItem, InventoryOverview
from app.schemas.model import (
    ConfusionMatrix,
    CriticalBandMetrics,
    ExpiryMetrics,
    FeatureImportanceItem,
    MetricRecord,
    ModelInfoResponse,
    ModelMetricsResponse,
)
from app.schemas.product import Product, ProductListResponse
from app.schemas.recommendation import Recommendation, RecommendationListResponse

__all__ = [
    "BatchRisk",
    "ComponentHealth",
    "ConfusionMatrix",
    "CriticalBandMetrics",
    "DataFreshness",
    "DemandDriver",
    "ErrorDetail",
    "ErrorResponse",
    "ExpiryMetrics",
    "ExpiryRiskResponse",
    "FeatureImportanceItem",
    "ForecastRequest",
    "ForecastResponse",
    "HealthResponse",
    "HistoryPoint",
    "InventoryItem",
    "InventoryOverview",
    "MetricRecord",
    "ModelInfoResponse",
    "ModelMetricsResponse",
    "PredictionInterval",
    "Priority",
    "Product",
    "ProductListResponse",
    "Recommendation",
    "RecommendationAction",
    "RecommendationListResponse",
    "RiskLevel",
    "StrictModel",
]
