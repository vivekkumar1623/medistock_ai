"""FastAPI dependencies (request-scoped wiring)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.artifact_store import ArtifactBundle, ArtifactStore, get_artifact_store
from app.services.container import ServiceContainer, get_container


def settings_dep() -> Settings:
    return get_settings()


def store_dep() -> ArtifactStore:
    return get_artifact_store()


def bundle_dep(store: Annotated[ArtifactStore, Depends(store_dep)]) -> ArtifactBundle:
    """Load the verified bundle, or raise ArtifactNotReadyError (HTTP 503)."""
    return store.load()


def container_dep(store: Annotated[ArtifactStore, Depends(store_dep)]) -> ServiceContainer:
    """Load the verified bundle and its cached service stack."""
    return get_container(store)


SettingsDep = Annotated[Settings, Depends(settings_dep)]
StoreDep = Annotated[ArtifactStore, Depends(store_dep)]
BundleDep = Annotated[ArtifactBundle, Depends(bundle_dep)]
ContainerDep = Annotated[ServiceContainer, Depends(container_dep)]
