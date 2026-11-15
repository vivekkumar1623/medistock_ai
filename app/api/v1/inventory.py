"""Inventory overview endpoint."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.schemas.inventory import InventoryItem, InventoryOverview

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get(
    "/overview",
    response_model=InventoryOverview,
    summary="Aggregate stock position, coverage and per-product status",
)
def inventory_overview(container: ContainerDep) -> InventoryOverview:
    overview = container.inventory.overview()
    frame = overview["items"]

    items = [
        InventoryItem(
            product_id=str(row.product_id),
            medicine_name=str(row.medicine_name),
            category=str(row.category),
            current_stock=int(row.current_stock),
            reorder_level=int(row.reorder_level),
            supplier_lead_time=int(row.supplier_lead_time),
            stock_status=str(row.stock_status),
            coverage_days=None if row.coverage_days is None else float(row.coverage_days),
        )
        for row in frame.itertuples(index=False)
    ]

    return InventoryOverview(
        as_of_date=date.fromisoformat(str(overview["as_of_date"])[:10]),
        total_products=int(overview["total_products"]),
        total_units_on_hand=int(overview["total_units_on_hand"]),
        inventory_value_at_cost=float(overview["inventory_value_at_cost"]),
        products_below_reorder=int(overview["products_below_reorder"]),
        products_excess=int(overview["products_excess"]),
        batches_expiring_next_90d=int(overview["batches_expiring_next_90d"]),
        items=items,
    )
