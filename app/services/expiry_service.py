"""Expiry-risk service: FEFO assessment computed from the bundle.

Serving path: forecast from the saved model -> daily rate -> FEFO allocation ->
severity classification. Nothing is read from a precomputed CSV of risk labels,
so the numbers a dashboard shows are always consistent with the model currently
loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.services.artifact_store import ArtifactBundle
from app.services.forecast_service import ForecastService
from ml.expiry_risk.fefo import FefoPolicy, allocate_fefo
from ml.expiry_risk.risk import assess_batches, summarise


@dataclass
class ExpiryAssessment:
    assessed: pd.DataFrame
    summary: dict[str, Any]
    policy: dict[str, Any]


class ExpiryService:
    def __init__(self, bundle: ArtifactBundle, forecasts: ForecastService) -> None:
        self.bundle = bundle
        self.forecasts = forecasts
        self.settings = get_settings()
        self._assessment: ExpiryAssessment | None = None

    @property
    def snapshot(self) -> pd.Timestamp:
        return pd.Timestamp(self.bundle.prediction_origin)

    def _policy_descriptor(self) -> dict[str, Any]:
        return {
            "policy_version": "risk-v1",
            "allocation": "FEFO",
            "risk_rule": "estimated_unsold_units > 0",
            "severity_bands": self.settings.risk_thresholds,
            "max_extrapolation_days": 365,
        }

    def assess(self) -> ExpiryAssessment:
        if self._assessment is not None:
            return self._assessment

        predictions = self.forecasts.forecast_all()
        daily_rate = (
            predictions.set_index("product_id")["predicted_demand"] / self.bundle.horizon_days
        )
        allocation = allocate_fefo(
            self.bundle.batches,
            daily_rate,
            snapshot=self.snapshot,
            policy=FefoPolicy(max_extrapolation_days=365),
        )
        assessed = assess_batches(allocation, self.settings.risk_thresholds)
        self._assessment = ExpiryAssessment(
            assessed=assessed,
            summary=summarise(assessed),
            policy=self._policy_descriptor(),
        )
        return self._assessment

    def rows(self) -> pd.DataFrame:
        """Assessment joined with product names and costs for presentation."""
        assessed = self.assess().assessed
        if assessed.empty:
            return assessed
        products = self.bundle.products.set_index("product_id")
        frame = assessed.copy()
        frame["medicine_name"] = frame["product_id"].map(products["medicine_name"])
        frame["category"] = frame["product_id"].map(products["category"])
        frame["unit_cost"] = frame["product_id"].map(products["unit_cost"])
        frame["estimated_unsold_value_at_cost"] = (
            frame["unsold_quantity"] * frame["unit_cost"]
        ).round(2)
        return frame

    def batch(self, batch_id: str) -> pd.Series:
        frame = self.rows()
        match = frame[frame["batch_id"] == batch_id]
        if match.empty:
            raise NotFoundError("Unknown batch", details={"batch_id": batch_id})
        return match.iloc[0]

    def product_exposure(self) -> pd.DataFrame:
        """Per-product exposure used by the recommendation rules."""
        frame = self.rows()
        if frame.empty:
            return pd.DataFrame(columns=["product_id", "expiring_units_90d", "nearest_expiry_days"])
        horizon = frame[frame["days_to_expiry"] <= 90]
        grouped = horizon.groupby("product_id", as_index=False).agg(
            expiring_units_90d=("unsold_quantity", "sum"),
            nearest_expiry_days=("days_to_expiry", "min"),
        )
        return grouped
