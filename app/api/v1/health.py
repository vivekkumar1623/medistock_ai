"""Health and readiness. Real, working, no artifacts required."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.api.deps import SettingsDep, StoreDep
from app.core.config import Settings
from app.schemas.common import ComponentHealth, HealthResponse
from app.services.artifact_store import ArtifactStore

router = APIRouter(tags=["health"])


def _database_component(settings: Settings) -> ComponentHealth:
    """Check the SQLite file is present. Phase 8 upgrades this to a real query."""
    if settings.db_url.startswith("sqlite"):
        raw_path = settings.db_url.removeprefix("sqlite:///")
        from pathlib import Path

        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (settings.data_dir.parent / raw_path).resolve()
        if db_path.is_file():
            return ComponentHealth(name="database", status="ok", detail=str(db_path))
        return ComponentHealth(
            name="database",
            status="degraded",
            detail="SQLite file not initialised yet; run `python -m scripts.init_db`.",
        )
    return ComponentHealth(name="database", status="ok", detail="non-sqlite backend configured")


def build_health(settings: Settings, store: ArtifactStore) -> HealthResponse:
    artifact_status = store.status()
    components = [
        ComponentHealth(name="api", status="ok", detail="process responding"),
        ComponentHealth(
            name="artifacts",
            status="ok" if artifact_status["ready"] else "degraded",
            detail=(
                f"version={artifact_status['version']} ready"
                if artifact_status["ready"]
                else "; ".join(artifact_status["problems"]) or "not trained yet"
            ),
        ),
        _database_component(settings),
    ]
    if all(component.status == "ok" for component in components):
        overall = "ok"
    elif any(component.status == "unavailable" for component in components):
        overall = "unavailable"
    else:
        overall = "degraded"
    return HealthResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checked_at_utc=datetime.now(UTC),
        components=components,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + readiness of API, model artifacts and database",
)
def health(settings: SettingsDep, store: StoreDep) -> HealthResponse:
    return build_health(settings, store)


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "MediStock AI",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
    }
