"""Configuration contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, Settings


def test_defaults_are_sane() -> None:
    settings = Settings()
    assert settings.environment == "development"
    assert settings.random_seed == 42
    assert settings.forecast_horizon_days == 30


def test_paths_resolve_against_project_root() -> None:
    settings = Settings(data_dir="some_relative_dir", artifact_dir="another")
    assert settings.data_dir == (PROJECT_ROOT / "some_relative_dir").resolve()
    assert settings.artifact_dir == (PROJECT_ROOT / "another").resolve()
    assert settings.raw_data_dir == settings.data_dir / "raw"


def test_artifact_path_combines_root_and_version(tmp_path: Path) -> None:
    settings = Settings(artifact_dir=tmp_path / "artifacts", artifact_version="v7")
    assert settings.artifact_path == (tmp_path / "artifacts" / "v7")


def test_env_prefix_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDITSTOCK_RANDOM_SEED", "1234")
    monkeypatch.setenv("MEDITSTOCK_FORECAST_HORIZON_DAYS", "14")
    monkeypatch.setenv("MEDITSTOCK_ARTIFACT_VERSION", "v9")
    settings = Settings()
    assert settings.random_seed == 1234
    assert settings.forecast_horizon_days == 14
    assert settings.artifact_version == "v9"


def test_unprefixed_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray RANDOM_SEED in the environment must not hijack the pipeline."""
    monkeypatch.setenv("RANDOM_SEED", "999")
    assert Settings().random_seed == 42


def test_log_level_is_normalised_and_validated() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")


def test_horizon_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(forecast_horizon_days=0)
    with pytest.raises(ValidationError):
        Settings(forecast_horizon_days=365)


def test_risk_thresholds_are_exposed_as_ordered_map() -> None:
    settings = Settings(risk_critical_days=30, risk_high_days=60, risk_medium_days=90)
    assert settings.risk_thresholds == {"critical": 30, "high": 60, "medium": 90}


def test_ensure_directories_creates_expected_tree(tmp_path: Path) -> None:
    settings = Settings(
        environment="test",
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        artifact_version="v1",
    )
    settings.ensure_directories()
    for path in (
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.interim_data_dir,
        settings.artifact_path,
        settings.figures_dir,
    ):
        assert path.is_dir(), path
