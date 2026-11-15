"""Model cards / evaluation contract exposed by the API and dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import StrictModel


class MetricRecord(StrictModel):
    """One (model, split) evaluation row. All models scored on the same split."""

    model_name: str
    split: str = Field(..., description="validation | test", examples=["test"])
    mae: float = Field(..., ge=0)
    rmse: float = Field(..., ge=0)
    wape: float = Field(..., ge=0, description="sum(|err|) / sum(actual); 0.1 == 10%.")
    n_observations: int = Field(..., ge=1)


class ConfusionMatrix(StrictModel):
    true_positive: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    true_negative: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)


class CriticalBandMetrics(StrictModel):
    """Precision/recall restricted to the critical severity tier.

    The critical tier is the actionable subset: those are the batches a manager is
    expected to act on today, so its error profile matters more than the aggregate.
    """

    n_batches: int = Field(..., ge=0)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    confusion: ConfusionMatrix


class ExpiryMetrics(StrictModel):
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1: float = Field(..., ge=0, le=1)
    accuracy: float = Field(..., ge=0, le=1)
    confusion: ConfusionMatrix
    n_batches: int = Field(..., ge=1)
    positive_class: str = "at_risk"
    threshold_rule: str
    critical_band: CriticalBandMetrics | None = None
    base_rate: float | None = Field(
        default=None, description="Share of evaluable batches that were actually at risk."
    )


class FeatureImportanceItem(StrictModel):
    feature: str
    importance: float
    rank: int = Field(..., ge=1)
    method: str = Field(..., examples=["permutation_importance_test_set"])


class ModelInfoResponse(StrictModel):
    artifact_version: str
    contract_version: str
    created_at_utc: datetime
    selected_model: str
    candidate_models: list[str]
    prediction_origin: str
    horizon_days: int
    n_features: int
    dataset: dict[str, Any]
    training_config: dict[str, Any]
    files: list[str]
    explainability_available: bool
    expiry_evaluation_available: bool
    library_versions: dict[str, str]


class ModelMetricsResponse(StrictModel):
    selected_model: str
    baseline_model: str
    selection_rationale: str
    demand_metrics: list[MetricRecord]
    expiry_metrics: ExpiryMetrics | None = None
    feature_importance: list[FeatureImportanceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
