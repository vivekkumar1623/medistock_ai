"""Recommendation service: joins forecast, stock and expiry exposure.

One row per product, produced by the ordered rule table in
``ml/inventory/recommendations.py``. Every row carries its inputs, so a manager
can audit the arithmetic without trusting the label.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.services.artifact_store import ArtifactBundle
from app.services.expiry_service import ExpiryService
from app.services.forecast_service import ForecastService
from app.services.inventory_service import InventoryService
from ml.inventory.recommendations import decide_action
from ml.inventory.replenishment import ReplenishmentPolicy, plan_replenishment

ACTION_ORDER = {
    "reorder": 0,
    "urgent_rotation": 1,
    "expiry_exposure": 2,
    "overstock": 3,
    "monitor": 4,
    "sufficient": 5,
}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class RecommendationService:
    def __init__(
        self,
        bundle: ArtifactBundle,
        forecasts: ForecastService,
        expiry: ExpiryService,
        inventory: InventoryService,
    ) -> None:
        self.bundle = bundle
        self.forecasts = forecasts
        self.expiry = expiry
        self.inventory = inventory
        self.settings = get_settings()
        policy = ReplenishmentPolicy(horizon_days=bundle.horizon_days)
        self.policy = policy

    def build(self) -> pd.DataFrame:
        items = self.inventory.items()
        exposure = self.expiry.product_exposure()

        frame = items.merge(exposure, on="product_id", how="left")
        frame["expiring_units_90d"] = frame["expiring_units_90d"].fillna(0.0)
        frame["nearest_expiry_days"] = frame["nearest_expiry_days"].astype("float64")

        records: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            plan = plan_replenishment(
                current_stock=int(row.current_stock),
                forecast_demand=float(row.predicted_demand),
                lead_time_days=int(row.supplier_lead_time),
                sigma_daily=float(row.std_90),
                policy=self.policy,
            )
            nearest = None if pd.isna(row.nearest_expiry_days) else int(row.nearest_expiry_days)
            decision = decide_action(
                current_stock=int(row.current_stock),
                reorder_level=int(row.reorder_level),
                forecast_demand=float(row.predicted_demand),
                coverage_days=plan["coverage_days"],
                lead_time_days=int(row.supplier_lead_time),
                plan=plan,
                expiring_units=float(row.expiring_units_90d),
                nearest_expiry_days=nearest,
                excess_coverage_days=self.policy.excess_coverage_days,
            )
            records.append(
                {
                    "product_id": row.product_id,
                    "medicine_name": row.medicine_name,
                    "category": row.category,
                    "action": decision.action.value,
                    "priority": decision.priority.value,
                    "suggested_replenishment_qty": (
                        plan["suggested_quantity"] if decision.action.value == "reorder" else 0
                    ),
                    "forecast_demand": round(float(row.predicted_demand), 2),
                    "current_stock": int(row.current_stock),
                    "safety_stock": int(plan["safety_stock"]),
                    "lead_time_days": int(row.supplier_lead_time),
                    "coverage_days": plan["coverage_days"],
                    "excess_units": int(plan["excess_units"]),
                    "reasons": decision.reasons,
                    "inputs": {
                        **decision.inputs,
                        **{k: plan[k] for k in ("mean_daily_demand", "target_level")},
                    },
                }
            )

        result = pd.DataFrame(records)
        result["_action_order"] = result["action"].map(ACTION_ORDER).fillna(99)
        result["_priority_order"] = result["priority"].map(PRIORITY_ORDER).fillna(9)
        result = result.sort_values(
            ["_priority_order", "_action_order", "coverage_days"],
            na_position="last",
            kind="mergesort",
        ).drop(columns=["_action_order", "_priority_order"])
        return result.reset_index(drop=True)

    def summary(self) -> dict[str, int]:
        frame = self.build()
        counts = frame["action"].value_counts().to_dict()
        return {action: int(counts.get(action, 0)) for action in ACTION_ORDER}
