"""Shared pytest fixtures.

Also makes the repository root importable so a bare ``pytest`` invocation
works identically to ``python -m pytest`` (a classic Windows/WSL papercut).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Before any sklearn import: keep fitting on the sequential path (see module docstring).
from ml.common import runtime  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import Settings, get_settings  # noqa: E402
from app.services.artifact_store import ArtifactStore  # noqa: E402
from tests.factories import build_fake_bundle  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Every test starts from a clean settings cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def fake_settings(tmp_path: Path, artifact_root: Path) -> Settings:
    """Settings pointed at a temporary artifact + data directory."""
    return Settings(
        environment="test",
        data_dir=tmp_path / "data",
        artifact_dir=artifact_root,
        artifact_version="v1",
        forecast_horizon_days=30,
    )


@pytest.fixture
def bundle_dir(artifact_root: Path) -> Path:
    """A valid, contract-compliant bundle under artifacts/v1."""
    return build_fake_bundle(artifact_root / "v1")


@pytest.fixture
def store(fake_settings: Settings) -> ArtifactStore:
    return ArtifactStore(settings=fake_settings, version="v1")
