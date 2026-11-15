"""FastAPI application factory and entrypoint.

Routing layout (deliberate):
  ``/``            service banner
  ``/health``      root-level health check (probes/load balancers expect this)
  ``/api/v1/...``  versioned business API
  ``/docs``        Swagger UI (generated from the real schemas)

The app never fabricates a prediction: if no verified artifact bundle is
present, data endpoints answer 503 with an actionable message.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.v1 import health
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.artifact_store import get_artifact_store

logger = get_logger("main")

TAGS_METADATA = [
    {"name": "health", "description": "Liveness and readiness of each component."},
    {"name": "model", "description": "Artifact identity, provenance and evaluation metrics."},
    {"name": "products", "description": "Product master data."},
    {"name": "inventory", "description": "Stock position and coverage."},
    {"name": "forecast", "description": "30-day demand forecasts from the saved model artifact."},
    {"name": "expiry-risk", "description": "FEFO-based batch expiry-risk assessment."},
    {"name": "recommendations", "description": "Replenishment and stock-rotation guidance."},
    {"name": "stubs", "description": "Contract-only endpoints, not yet implemented."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    settings.ensure_directories()
    logger.info(
        "Starting application",
        extra={"environment": settings.environment, "artifact_version": settings.artifact_version},
    )
    store = get_artifact_store()
    if store.is_ready():
        try:
            store.load()
        except Exception as exc:  # never block startup on a bad bundle; /health reports it
            logger.error("Artifact bundle present but failed to load", exc_info=exc)
    else:
        logger.warning(
            "No artifact bundle found; data endpoints will return 503 until training runs",
            extra={"expected_path": str(store.manifest_path)},
        )
    yield
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Demand-forecasting and expiry-risk decision support for pharmacy inventory. "
            "Synthetic data; not medical advice; all decisions require human review."
        ),
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
    )
    register_exception_handlers(application)
    application.include_router(health.router)
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    logging.getLogger(__name__).info("Launching uvicorn on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
