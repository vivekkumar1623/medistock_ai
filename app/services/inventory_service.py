"""Inventory service: stock position joined to forecast coverage.

Status vocabulary is deliberately small and rule-based:

* ``critical``      - no stock on hand.
* ``below_reorder`` - stock is under the reorder point.
* ``excess``        - coverage far beyond the planning horizon.
* ``adequate``      - everything else.

Coverage is ``current_stock / mean_daily_forecast``; when forecast demand is zero
coverage is undefined and reported as ``None`` rather than as a huge number that
would make an idle product look overstocked.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.artifact_store import ArtifactBundle
from app.services.expiry_service import ExpiryService
from app.services.forecast_service import ForecastService

EXCESS_COVERAGE_DAYS = 90


def classify_stock_status(
    current_stock: int, reorder_level: int, coverage_days: float | None
) -> str:
    if current_stock <= 0:
        return "critical"
    if current_stock < reorder_level:
        return "below_reorder"
    if coverage_days is not None and coverage_days > EXCESS_COVERAGE_DAYS:
        return "excess"
    return "adequate"


class InventoryService:
    def __init__(
        self,
        bundle: ArtifactBundle,
        forecasts: ForecastService,
        expiry: ExpiryService,
    ) -> None:
        self.bundle = bundle
        self.forecasts = forecasts
        self.expiry = expiry

    def items(self) -> pd.DataFrame:
        products = self.bundle.products
        stock = self.bundle.inventory_snapshot

        frame = products.merge(
            stock[["product_id", "current_stock", "reorder_level", "supplier_lead_time"]],
            on="product_id",
            how="left",
        )
        frame["current_stock"] = frame["current_stock"].fillna(0).astype(int)
        frame["reorder_level"] = frame["reorder_level"].fillna(0).astype(int)
        frame["supplier_lead_time"] = (
            frame["supplier_lead_time"].fillna(frame["lead_time_days"]).astype(int)
        )

        predictions = self.forecasts.prediction_table()[
            ["product_id", "predicted_demand", "std_90", "lower", "upper", "zero_ratio_90"]
        ]
        frame = frame.merge(predictions, on="product_id", how="left")
        frame["predicted_demand"] = frame["predicted_demand"].fillna(0.0)
        frame["std_90"] = frame["std_90"].fillna(0.0)

        mean_daily = frame["predicted_demand"] / self.bundle.horizon_days
        frame["coverage_days"] = [
            None if rate <= 0 else round(float(stock) / float(rate), 2)
            for stock, rate in zip(frame["current_stock"], mean_daily, strict=True)
        ]
        frame["stock_status"] = [
            classify_stock_status(int(stock), int(reorder), coverage)
            for stock, reorder, coverage in zip(
                frame["current_stock"],
                frame["reorder_level"],
                frame["coverage_days"],
                strict=True,
            )
        ]
        frame["inventory_value_at_cost"] = (frame["current_stock"] * frame["unit_cost"]).round(2)
        return frame.sort_values("product_id").reset_index(drop=True)

    def overview(self) -> dict[str, Any]:
        frame = self.items()
        exposure = self.expiry.rows()
        expiring_90 = (
            int(((exposure["days_to_expiry"] >= 0) & (exposure["days_to_expiry"] <= 90)).sum())
            if not exposure.empty
            else 0
        )
        return {
            "as_of_date": self.bundle.prediction_origin,
            "total_products": int(len(frame)),
            "total_units_on_hand": int(frame["current_stock"].sum()),
            "inventory_value_at_cost": round(float(frame["inventory_value_at_cost"].sum()), 2),
            "products_below_reorder": int((frame["stock_status"] == "below_reorder").sum()),
            "products_excess": int((frame["stock_status"] == "excess").sum()),
            "batches_expiring_next_90d": expiring_90,
            "items": frame,
        }
