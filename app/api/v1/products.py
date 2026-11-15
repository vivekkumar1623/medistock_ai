"""Product catalogue endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import ContainerDep
from app.schemas.product import Product, ProductListResponse

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=ProductListResponse, summary="Product master list")
def list_products(
    container: ContainerDep,
    category: Annotated[str | None, Query(description="Filter by category")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProductListResponse:
    frame = container.bundle.products
    if category:
        frame = frame[frame["category"].str.lower() == category.lower()]

    total = int(len(frame))
    window = frame.sort_values("product_id").iloc[offset : offset + limit]
    items = [
        Product(
            product_id=str(row.product_id),
            medicine_name=str(row.medicine_name),
            category=str(row.category),
            manufacturer=str(row.manufacturer),
            supplier_id=str(row.supplier_id),
            unit_price=float(row.unit_price),
            unit_cost=float(row.unit_cost),
            minimum_stock_level=int(row.minimum_stock_level),
            lead_time_days=int(row.lead_time_days),
        )
        for row in window.itertuples(index=False)
    ]
    return ProductListResponse(items=items, total=total)
