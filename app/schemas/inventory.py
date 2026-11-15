"""Inventory overview contract."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.schemas.common import StrictModel


class InventoryItem(StrictModel):
    product_id: str
    medicine_name: str
    category: str
    current_stock: int = Field(..., ge=0)
    reorder_level: int = Field(..., ge=0)
    supplier_lead_time: int = Field(..., ge=0)
    stock_status: Literal["critical", "below_reorder", "adequate", "excess"]
    coverage_days: float | None = Field(
        default=None, description="Current stock / mean daily forecast demand. Null if demand is 0."
    )


class InventoryOverview(StrictModel):
    as_of_date: date
    total_products: int
    total_units_on_hand: int
    inventory_value_at_cost: float
    products_below_reorder: int
    products_excess: int
    batches_expiring_next_90d: int
    items: list[InventoryItem]
