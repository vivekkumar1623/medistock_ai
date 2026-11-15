"""Service container: one construction point for the inference stack.

The bundle is loaded once by ``ArtifactStore``; this container builds the four
business services on top of it and caches them per artifact version. Without this,
every request would rebuild the feature matrix and re-run the FEFO allocation,
which is wasteful and (worse) risks two endpoints reporting subtly different
numbers if anything were non-deterministic.

The container is keyed by artifact version, so retraining and reloading a new
bundle produces a fresh stack instead of silently serving stale cached results.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.services.artifact_store import ArtifactBundle, ArtifactStore
from app.services.expiry_service import ExpiryService
from app.services.forecast_service import ForecastService
from app.services.inventory_service import InventoryService
from app.services.recommendation_service import RecommendationService

logger = get_logger("services.container")


@dataclass
class ServiceContainer:
    bundle: ArtifactBundle
    forecasts: ForecastService
    expiry: ExpiryService
    inventory: InventoryService
    recommendations: RecommendationService


_CACHE: dict[str, ServiceContainer] = {}


def build_container(bundle: ArtifactBundle) -> ServiceContainer:
    forecasts = ForecastService(bundle)
    expiry = ExpiryService(bundle, forecasts)
    inventory = InventoryService(bundle, forecasts, expiry)
    recommendations = RecommendationService(bundle, forecasts, expiry, inventory)
    return ServiceContainer(
        bundle=bundle,
        forecasts=forecasts,
        expiry=expiry,
        inventory=inventory,
        recommendations=recommendations,
    )


def get_container(store: ArtifactStore) -> ServiceContainer:
    """Return the cached container for the store's current artifact version."""
    bundle = store.load()
    key = bundle.artifact_version
    container = _CACHE.get(key)
    if container is None:
        logger.info("Building service container", extra={"artifact_version": key})
        container = build_container(bundle)
        _CACHE[key] = container
    return container


def clear_container_cache() -> None:
    """Drop cached services (tests, or after retraining in-process)."""
    _CACHE.clear()
