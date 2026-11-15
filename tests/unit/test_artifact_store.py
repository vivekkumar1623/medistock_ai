"""ArtifactBundle contract tests: the integrity guarantees other phases rely on."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import ArtifactIntegrityError, ArtifactNotReadyError
from app.services.artifact_store import ArtifactStore, sha256_file
from tests.factories import build_fake_bundle


def test_not_ready_when_manifest_absent(fake_settings: Settings) -> None:
    store = ArtifactStore(settings=fake_settings, version="v1")
    assert store.is_ready() is False
    with pytest.raises(ArtifactNotReadyError) as excinfo:
        store.load()
    assert excinfo.value.status_code == 503
    assert "train" in excinfo.value.message.lower()


def test_valid_bundle_loads_and_exposes_contract(bundle_dir: Path, fake_settings: Settings) -> None:
    store = ArtifactStore(settings=fake_settings, version="v1")
    assert store.is_ready() is True
    assert store.verify() == []

    bundle = store.load()

    assert bundle.horizon_days == 30
    assert bundle.model_name == "hist_gradient_boosting"
    assert bundle.prediction_origin == "2025-11-01"
    assert bundle.feature_names == [
        "mean_30",
        "sum_30",
        "std_90",
        "zero_ratio_90",
        "ratio_7_30",
        "category",
    ]
    assert bundle.demand_history.shape[0] == 240
    assert bundle.products.shape[0] == 2
    assert bundle.has_expiry_artifacts() is True
    assert bundle.importance is not None


def test_second_load_is_cached(bundle_dir: Path, fake_settings: Settings) -> None:
    store = ArtifactStore(settings=fake_settings, version="v1")
    first = store.load()
    second = store.load()
    assert first is second


def test_checksum_mismatch_is_rejected(artifact_root: Path, fake_settings: Settings) -> None:
    build_fake_bundle(artifact_root / "v1", corrupt_checksums=True)
    store = ArtifactStore(settings=fake_settings, version="v1")
    problems = store.verify()
    assert any("checksum mismatch for model.joblib" in problem for problem in problems)
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.load()
    assert excinfo.value.status_code == 500


def test_post_signing_tamper_is_detected(artifact_root: Path, fake_settings: Settings) -> None:
    build_fake_bundle(artifact_root / "v1", tamper_model_after_manifest=True)
    store = ArtifactStore(settings=fake_settings, version="v1")
    with pytest.raises(ArtifactIntegrityError):
        store.load()


def test_missing_required_file_is_reported(artifact_root: Path, fake_settings: Settings) -> None:
    build_fake_bundle(artifact_root / "v1", omit_files=("demand_history.parquet",))
    store = ArtifactStore(settings=fake_settings, version="v1")
    problems = store.verify()
    assert any("demand_history.parquet" in problem for problem in problems)


def test_contract_major_version_mismatch_is_rejected(
    artifact_root: Path, fake_settings: Settings
) -> None:
    build_fake_bundle(artifact_root / "v1", contract_version="2.0.0")
    store = ArtifactStore(settings=fake_settings, version="v1")
    problems = store.verify()
    assert any("contract version mismatch" in problem for problem in problems)


def test_horizon_mismatch_blocks_serving(artifact_root: Path) -> None:
    """A bundle trained for a 14-day question must not answer a 30-day one."""
    build_fake_bundle(artifact_root / "v1", horizon_days=14)
    settings = Settings(
        environment="test",
        data_dir=artifact_root.parent / "data",
        artifact_dir=artifact_root,
        artifact_version="v1",
        forecast_horizon_days=30,
    )
    store = ArtifactStore(settings=settings, version="v1")
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.load()
    assert "horizon" in excinfo.value.message.lower()


def test_version_mismatch_between_requested_and_manifest(
    artifact_root: Path, fake_settings: Settings
) -> None:
    build_fake_bundle(artifact_root / "v1", version="v1")
    store = ArtifactStore(settings=fake_settings, version="v2")
    assert store.verify() != []


def test_status_reports_problems_without_raising(fake_settings: Settings) -> None:
    store = ArtifactStore(settings=fake_settings, version="v1")
    status = store.status()
    assert status["ready"] is False
    assert status["problems"]


def test_status_of_valid_bundle_includes_provenance(
    bundle_dir: Path, fake_settings: Settings
) -> None:
    store = ArtifactStore(settings=fake_settings, version="v1")
    status = store.status()
    assert status["ready"] is True
    assert status["selected_model"] == "hist_gradient_boosting"
    assert status["dataset"]["is_synthetic"] is True


def test_sha256_helper_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"medistock" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
