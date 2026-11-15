"""Forecast service: the one inference path used by the API, the dashboard and scripts.

Design note (D-012): this module imports ``ml.features`` on purpose. The features
used at serving time and the features used at training time must be *the same
code*. Re-implementing feature construction for inference is the classic source
of train/serve skew -- the model receives slightly different inputs and silently
degrades. Sharing the module makes skew a compile-time impossibility instead of a
production surprise.

Nothing here computes a prediction from raw CSVs. It reads the verified
``ArtifactBundle`` and calls ``bundle.model.predict`` -- so "the API uses the
saved model" is true by construction rather than by claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.services.artifact_store import ArtifactBundle
from ml.explainability.drivers import explain_prediction
from ml.features.build import ALL_FEATURES, DemandMatrix, compute_origin_features

logger = get_logger("services.forecast")

FEATURES: list[str] = list(ALL_FEATURES)


@dataclass
class ForecastRow:
    product_id: str
    medicine_name: str
    category: str
    predicted_demand: float
    lower: float | None
    upper: float | None
    confidence: str
    drivers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ForecastService:
    """Batch-scoring facade over one artifact bundle."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self._matrix: DemandMatrix | None = None
        self._features: pd.DataFrame | None = None
        self._predictions: np.ndarray | None = None

    # --- Cached internals ---------------------------------------------------
    @property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.bundle.prediction_origin)

    def _demand_matrix(self) -> DemandMatrix:
        if self._matrix is None:
            self._matrix = DemandMatrix.from_panel(self.bundle.demand_history)
        return self._matrix

    def _feature_frame(self) -> pd.DataFrame:
        if self._features is None:
            self._features = compute_origin_features(
                self._demand_matrix(), self.bundle.products, self.origin
            )
        return self._features

    def _raw_predictions(self) -> np.ndarray:
        if self._predictions is None:
            features = self._feature_frame()
            predicted = np.asarray(self.bundle.model.predict(features[FEATURES]), dtype=float)
            # Demand cannot be negative; a regressor can still emit small negatives.
            self._predictions = np.clip(predicted, 0.0, None)
        return self._predictions

    def prediction_table(self) -> pd.DataFrame:
        """All products with predicted demand, interval and confidence."""
        features = self._feature_frame()
        predictions = self._raw_predictions()
        intervals = self._residual_quantiles()
        table = pd.DataFrame(
            {
                "product_id": features["product_id"].to_numpy(),
                "predicted_demand": np.round(predictions, 4),
                "zero_ratio_90": features["zero_ratio_90"].to_numpy(),
                "mean_30": features["mean_30"].to_numpy(),
                "std_90": features["std_90"].to_numpy(),
            }
        )
        if intervals is not None:
            lower_q, upper_q = intervals
            table["lower"] = np.clip(np.round(predictions + lower_q, 4), 0.0, None)
            table["upper"] = np.round(predictions + upper_q, 4)
        else:
            table["lower"] = np.nan
            table["upper"] = np.nan

        categories = self.bundle.products.set_index("product_id")["category"]
        names = self.bundle.products.set_index("product_id")["medicine_name"]
        table["category"] = table["product_id"].map(categories)
        table["medicine_name"] = table["product_id"].map(names)
        return table

    def _residual_quantiles(self) -> tuple[float, float] | None:
        payload = self.bundle.metrics.get("residual_quantiles")
        if not payload:
            return None
        return float(payload["p10"]), float(payload["p90"])

    # --- Public API ---------------------------------------------------------
    def forecast_all(self) -> pd.DataFrame:
        return self.prediction_table()

    def recent_history(self, product_id: str, days: int = 60) -> list[dict[str, Any]]:
        history = self.bundle.demand_history
        series = history[history["product_id"] == product_id].sort_values("date")
        if series.empty:
            return []
        tail = series.tail(days)
        return [
            {"date": row.date.date(), "quantity": int(row.quantity)}
            for row in tail.itertuples(index=False)
        ]

    def forecast_one(self, product_id: str, include_drivers: bool = True) -> ForecastRow:
        table = self.prediction_table()
        match = table[table["product_id"] == product_id]
        if match.empty:
            raise NotFoundError("Unknown product", details={"product_id": product_id})
        row = match.iloc[0]
        notes = self._notes_for(row)

        drivers: list[dict[str, Any]] = []
        if include_drivers:
            features = self._feature_frame()
            feature_row = features[features["product_id"] == product_id]
            if not feature_row.empty:
                drivers = explain_prediction(
                    model=self.bundle.model,
                    feature_row=feature_row[FEATURES].reset_index(drop=True),
                    reference_frame=features[FEATURES],
                    feature_names=FEATURES,
                    top_k=5,
                )

        return ForecastRow(
            product_id=product_id,
            medicine_name=str(row["medicine_name"]),
            category=str(row["category"]),
            predicted_demand=float(row["predicted_demand"]),
            lower=None if pd.isna(row["lower"]) else float(row["lower"]),
            upper=None if pd.isna(row["upper"]) else float(row["upper"]),
            confidence=self._confidence(row),
            drivers=drivers,
            notes=notes,
        )

    # --- Heuristics, stated plainly ----------------------------------------
    def _confidence(self, row: pd.Series) -> str:
        """Documented heuristic, not a calibrated probability.

        Sparse products (many zero-demand days) and very wide intervals are the
        two situations where an aggregate forecast is least trustworthy.
        """
        zero_ratio = float(row["zero_ratio_90"])
        predicted = float(row["predicted_demand"])
        lower, upper = row["lower"], row["upper"]
        if pd.isna(lower) or pd.isna(upper):
            return "low" if zero_ratio > 0.4 else "medium"
        relative_width = (float(upper) - float(lower)) / max(predicted, 1.0)
        if zero_ratio > 0.5 or predicted < 5.0 or relative_width > 1.5:
            return "low"
        if zero_ratio > 0.2 or relative_width > 0.8:
            return "medium"
        return "high"

    def _notes_for(self, row: pd.Series) -> list[str]:
        notes: list[str] = []
        zero_ratio = float(row["zero_ratio_90"])
        if zero_ratio > 0.4:
            notes.append(
                f"Sparse demand: {zero_ratio:.0%} of the last 90 days had zero sales, "
                "so this forecast is an aggregate estimate with wide uncertainty."
            )
        if float(row["predicted_demand"]) < 5.0:
            notes.append("Very low expected demand; treat the point estimate as indicative.")
        quantiles = self._residual_quantiles()
        if quantiles is not None:
            notes.append(
                "Prediction interval uses empirical residual quantiles from the held-out "
                "test period; coverage is not verified per product."
            )
        return notes

    def explain(self, product_id: str) -> dict[str, Any]:
        """Bundle-level explanation payload used by the API and dashboard."""
        forecast = self.forecast_one(product_id)
        history = self.recent_history(product_id)
        return {
            "forecast": forecast,
            "history": history,
        }


def build_forecast_service(bundle: ArtifactBundle) -> ForecastService:
    """Construct a service for this bundle (cheap; internal work is cached)."""
    return ForecastService(bundle)
