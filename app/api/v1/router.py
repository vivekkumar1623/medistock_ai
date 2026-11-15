"""Aggregate all v1 routers under a single prefix."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import expiry_risk, forecast, inventory, model, products, recommendations

api_router = APIRouter()
api_router.include_router(model.router)
api_router.include_router(products.router)
api_router.include_router(inventory.router)
api_router.include_router(forecast.router)
api_router.include_router(expiry_risk.router)
api_router.include_router(recommendations.router)
