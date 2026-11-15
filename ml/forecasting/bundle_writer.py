"""Artifact bundle writer.

This module is the *only* place that creates a bundle. It must produce a
directory that ``app/services/artifact_store.py`` accepts: every required file
present, every hash correct, every manifest key populated.

The manifest is written **last**, after all payload files exist, so a crash
mid-write leaves an obviously incomplete bundle instead of a manifest that
certifies files which were never written. (That ordering bug is listed in the
Phase-1 silent-failure list and is exactly what this file avoids.)
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.common.io import sha256_file, write_json

CONTRACT_VERSION = "1.0.0"


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def write_joblib(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="snappy")
    return path


def build_manifest(
    directory: Path,
    *,
    artifact_version: str,
    selected_model: str,
    candidate_models: list[str],
    prediction_origin: str,
    horizon_days: int,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Hash every file in ``directory`` and assemble the manifest payload."""
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "artifact_version": artifact_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected_model,
        "candidate_models": candidate_models,
        "prediction_origin": prediction_origin,
        "horizon_days": int(horizon_days),
        "dataset": dataset,
        "library_versions": library_versions(),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files
        },
    }


def write_bundle(
    directory: Path,
    *,
    artifact_version: str,
    selected_model: str,
    candidate_models: list[str],
    prediction_origin: str,
    horizon_days: int,
    dataset: dict[str, Any],
    model: Any,
    baseline: Any,
    feature_spec: dict[str, Any],
    metrics: dict[str, Any],
    training_config: dict[str, Any],
    demand_history: pd.DataFrame,
    products: pd.DataFrame,
    inventory_snapshot: pd.DataFrame,
    batches: pd.DataFrame,
    importance: dict[str, Any] | None = None,
    metrics_expiry: dict[str, Any] | None = None,
    expiry_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every payload file, then the manifest. Returns the manifest."""
    directory.mkdir(parents=True, exist_ok=True)

    write_joblib(model, directory / "model.joblib")
    write_joblib(baseline, directory / "baseline.joblib")
    write_json(feature_spec, directory / "feature_spec.json")
    write_json(metrics, directory / "metrics.json")
    write_json(training_config, directory / "training_config.json")
    write_frame(demand_history, directory / "demand_history.parquet")
    write_frame(products, directory / "products.parquet")
    write_frame(inventory_snapshot, directory / "inventory_snapshot.parquet")
    write_frame(batches, directory / "batches.parquet")
    if importance is not None:
        write_json(importance, directory / "importance.json")
    if metrics_expiry is not None:
        write_json(metrics_expiry, directory / "metrics_expiry.json")
    if expiry_policy is not None:
        write_json(expiry_policy, directory / "expiry_policy.json")

    manifest = build_manifest(
        directory,
        artifact_version=artifact_version,
        selected_model=selected_model,
        candidate_models=candidate_models,
        prediction_origin=prediction_origin,
        horizon_days=horizon_days,
        dataset=dataset,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def refresh_manifest(directory: Path) -> dict[str, Any]:
    """Recompute hashes and identity for an existing bundle directory.

    Used when a later phase adds files (e.g. expiry evaluation) to an existing
    bundle. The identity fields are preserved from the previous manifest so the
    provenance does not drift.
    """
    previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest = build_manifest(
        directory,
        artifact_version=previous["artifact_version"],
        selected_model=previous["selected_model"],
        candidate_models=previous.get("candidate_models", [previous["selected_model"]]),
        prediction_origin=previous["prediction_origin"],
        horizon_days=previous["horizon_days"],
        dataset=previous["dataset"],
    )
    manifest["created_at_utc"] = previous["created_at_utc"]
    manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
