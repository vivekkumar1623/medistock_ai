"""Model card endpoints. Real: they read the verified artifact bundle."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import BundleDep, StoreDep
from app.schemas.model import (
    ConfusionMatrix,
    CriticalBandMetrics,
    ExpiryMetrics,
    FeatureImportanceItem,
    MetricRecord,
    ModelInfoResponse,
    ModelMetricsResponse,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Artifact identity, provenance and library versions",
)
def model_info(bundle: BundleDep, store: StoreDep) -> ModelInfoResponse:
    manifest = bundle.manifest
    config = bundle.training_config
    return ModelInfoResponse(
        artifact_version=bundle.artifact_version,
        contract_version=manifest.contract_version,
        created_at_utc=manifest.created_at_utc,
        selected_model=bundle.model_name,
        candidate_models=list(config.get("candidate_models", [bundle.model_name])),
        prediction_origin=bundle.prediction_origin,
        horizon_days=bundle.horizon_days,
        n_features=len(bundle.feature_names),
        dataset=manifest.dataset,
        training_config=config,
        files=sorted(manifest.files.keys()),
        explainability_available=bundle.importance is not None,
        expiry_evaluation_available=bundle.metrics_expiry is not None,
        library_versions=dict(manifest.raw.get("library_versions", {})),
    )


def _metric_rows(metrics: dict[str, Any]) -> list[MetricRecord]:
    rows: list[MetricRecord] = []
    for record in metrics.get("models", []):
        rows.append(
            MetricRecord(
                model_name=record["model_name"],
                split=record["split"],
                mae=float(record["mae"]),
                rmse=float(record["rmse"]),
                wape=float(record["wape"]),
                n_observations=int(record["n_observations"]),
            )
        )
    return rows


def _expiry_metrics(payload: dict[str, Any]) -> ExpiryMetrics:
    cm = payload["confusion"]
    critical = payload.get("critical_band") or {}
    critical_matrix = critical.get("confusion") or {}

    def _matrix(source: dict[str, Any]) -> ConfusionMatrix:
        return ConfusionMatrix(
            true_positive=int(source.get("true_positive", 0)),
            false_positive=int(source.get("false_positive", 0)),
            true_negative=int(source.get("true_negative", 0)),
            false_negative=int(source.get("false_negative", 0)),
        )

    return ExpiryMetrics(
        precision=float(payload["precision"]),
        recall=float(payload["recall"]),
        f1=float(payload["f1"]),
        accuracy=float(payload["accuracy"]),
        confusion=_matrix(cm),
        n_batches=int(payload["n_batches"]),
        threshold_rule=str(payload.get("threshold_rule", "documented in expiry_policy.json")),
        base_rate=None if payload.get("base_rate") is None else float(payload["base_rate"]),
        critical_band=CriticalBandMetrics(
            n_batches=int(critical.get("n_batches", 0)),
            precision=float(critical.get("precision", 0.0)),
            recall=float(critical.get("recall", 0.0)),
            confusion=_matrix(critical_matrix),
        )
        if critical
        else None,
    )


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Demand metrics, expiry-risk metrics and global feature importance",
)
def model_metrics(bundle: BundleDep) -> ModelMetricsResponse:
    metrics = bundle.metrics
    importance_items: list[FeatureImportanceItem] = []
    if bundle.importance:
        for rank, item in enumerate(bundle.importance.get("features", []), start=1):
            importance_items.append(
                FeatureImportanceItem(
                    feature=item["feature"],
                    importance=float(item["importance"]),
                    rank=rank,
                    method=str(bundle.importance.get("method", "permutation_importance")),
                )
            )
    return ModelMetricsResponse(
        selected_model=str(metrics.get("selected_model", bundle.model_name)),
        baseline_model=str(metrics.get("baseline_model", "seasonal_naive_30d")),
        selection_rationale=str(metrics.get("selection_rationale", "")),
        demand_metrics=_metric_rows(metrics),
        expiry_metrics=_expiry_metrics(bundle.metrics_expiry) if bundle.metrics_expiry else None,
        feature_importance=importance_items,
        notes=list(metrics.get("notes", [])),
    )
