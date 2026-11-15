"""Product master contract."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import StrictModel


class Product(StrictModel):
    product_id: str = Field(..., examples=["MED-0001"])
    medicine_name: str = Field(..., examples=["Amoxicillin 500mg Capsule"])
    category: str = Field(..., examples=["Antibiotics"])
    manufacturer: str = Field(..., examples=["Apex Pharma"])
    supplier_id: str = Field(..., examples=["SUP-03"])
    unit_price: float = Field(..., ge=0, description="Selling price per unit.")
    unit_cost: float = Field(..., ge=0, description="Procurement cost per unit.")
    minimum_stock_level: int = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=0)


class ProductListResponse(StrictModel):
    items: list[Product]
    total: int
