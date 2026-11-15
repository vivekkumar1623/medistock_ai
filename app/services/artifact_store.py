"""ArtifactBundle: the single source of truth for inference.

Lever C of the delivery plan is "one artifact, many views". Both the FastAPI
service layer and the Streamlit dashboard load *this* object, from *this*
directory, verified by *this* checksum logic. That is what removes the class of
bug where the API says 42 units and the dashboard charts 39.

Layout of a bundle (``artifacts/<version>/``)::

    manifest.json                    # identity + integrity + provenance
    model.joblib                     # fitted sklearn Pipeline (preprocess + estimator)
    baseline.joblib                  # seasonal-naive parameters (comparison baseline)
    feature_spec.json                # ordered features, dtypes, categories, lookback
    metrics.json                     # per-model metrics on val/test + selection record
    training_config.json             # seed, split dates, hyperparameters
    demand_history.parquet           # product x date demand history (lag source)
    products.parquet                 # product master snapshot at the origin
    inventory_snapshot.parquet       # stock + batch snapshot at the origin
    importance.json                  # global feature importance
    metrics_expiry.json              # precision/recall/F1 + confusion matrix
    expiry_policy.json               # the risk rule + thresholds actually used

Integrity rules enforced on load:
1. ``manifest.json`` exists and parses.
2. Contract major version matches this code's ``CONTRACT_VERSION``.
3. Every file listed in the manifest exists with the recorded SHA-256.
4. Every required file is listed.
5. The declared horizon matches the configured horizon (else the serving layer
   would silently answer a different question than the model was trained for).

Failure modes are *loud*. A missing bundle raises ``ArtifactNotReadyError``
(HTTP 503) and a tampered/corrupt bundle raises ``ArtifactIntegrityError``
(HTTP 500). There is no silent fallback to a default prediction.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import ArtifactIntegrityError, ArtifactNotReadyError
from app.core.logging import get_logger

logger = get_logger("services.artifact_store")

CONTRACT_VERSION = "1.0.0"

REQUIRED_FILES: tuple[str, ...] = (
    "model.joblib",
    "feature_spec.json",
    "metrics.json",
    "training_config.json",
    "demand_history.parquet",
    "products.parquet",
    "inventory_snapshot.parquet",
    "batches.parquet",
)

OPTIONAL_FILES: tuple[str, ...] = (
    "baseline.joblib",
    "importance.json",
    "metrics_expiry.json",
    "expiry_policy.json",
)

REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "contract_version",
    "artifact_version",
    "created_at_utc",
    "dataset",
    "prediction_origin",
    "horizon_days",
    "selected_model",
    "files",
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, streamed to handle large artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactManifest:
    """Typed view over ``manifest.json``. Frozen: manifests are immutable."""

    contract_version: str
    artifact_version: str
    created_at_utc: str
    prediction_origin: str
    horizon_days: int
    selected_model: str
    dataset: dict[str, Any]
    files: dict[str, dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in payload]
        if missing:
            raise ArtifactIntegrityError(
                "manifest.json is missing required keys",
                details={"missing_keys": missing},
            )
        try:
            horizon = int(payload["horizon_days"])
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "manifest.json has a non-integer horizon_days",
                details={"value": payload.get("horizon_days")},
            ) from exc
        return cls(
            contract_version=str(payload["contract_version"]),
            artifact_version=str(payload["artifact_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            prediction_origin=str(payload["prediction_origin"]),
            horizon_days=horizon,
            selected_model=str(payload["selected_model"]),
            dataset=dict(payload["dataset"]),
            files=dict(payload["files"]),
            raw=payload,
        )

    @property
    def contract_major(self) -> int:
        return _major(self.contract_version)


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise ArtifactIntegrityError(
            "Unparseable contract version", details={"version": version}
        ) from exc


@dataclass
class ArtifactBundle:
    """Every precomputed object inference needs, loaded once.

    Kept as a plain dataclass (not a Pydantic model) because it holds live
    Python objects: a fitted sklearn Pipeline and three DataFrames.
    """

    manifest: ArtifactManifest
    directory: Path
    model: Any
    feature_spec: dict[str, Any]
    metrics: dict[str, Any]
    training_config: dict[str, Any]
    demand_history: pd.DataFrame
    products: pd.DataFrame
    inventory_snapshot: pd.DataFrame
    batches: pd.DataFrame
    baseline: Any | None = None
    importance: dict[str, Any] | None = None
    metrics_expiry: dict[str, Any] | None = None
    expiry_policy: dict[str, Any] | None = None

    # --- Convenience accessors used by services -----------------------------
    @property
    def horizon_days(self) -> int:
        return self.manifest.horizon_days

    @property
    def prediction_origin(self) -> str:
        return self.manifest.prediction_origin

    @property
    def model_name(self) -> str:
        return self.manifest.selected_model

    @property
    def artifact_version(self) -> str:
        return self.manifest.artifact_version

    @property
    def feature_names(self) -> list[str]:
        return [item["name"] for item in self.feature_spec.get("features", [])]

    def has_expiry_artifacts(self) -> bool:
        """True when the expiry-risk engine ran and produced a calibrated rule."""
        return self.metrics_expiry is not None and self.expiry_policy is not None


class ArtifactStore:
    """Locates, verifies and caches one artifact bundle."""

    def __init__(self, settings: Settings | None = None, version: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.version = version or self.settings.artifact_version
        self._bundle: ArtifactBundle | None = None
        self._lock = threading.Lock()

    # --- Paths --------------------------------------------------------------
    @property
    def directory(self) -> Path:
        return self.settings.artifact_dir / self.version

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def path_for(self, filename: str) -> Path:
        return self.directory / filename

    def is_ready(self) -> bool:
        """Cheap readiness probe: manifest present and parseable."""
        if not self.manifest_path.is_file():
            return False
        try:
            self._read_manifest()
        except ArtifactIntegrityError:
            return False
        return True

    # --- Verification -------------------------------------------------------
    def _read_manifest(self) -> ArtifactManifest:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ArtifactNotReadyError(
                "No artifact manifest found",
                details={"expected_path": str(self.manifest_path)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError(
                "manifest.json is not valid JSON",
                details={"path": str(self.manifest_path), "error": str(exc)},
            ) from exc
        return ArtifactManifest.from_dict(payload)

    def verify(self, check_checksums: bool = True) -> list[str]:
        """Validate the bundle and return a list of human-readable problems.

        Returns an empty list when the bundle is sound. Never raises for
        *expected* integrity problems, so it can back a health endpoint.
        """
        problems: list[str] = []
        if not self.manifest_path.is_file():
            return [f"manifest.json not found at {self.manifest_path}"]

        try:
            manifest = self._read_manifest()
        except ArtifactIntegrityError as exc:
            return [f"manifest invalid: {exc.message}"]

        if manifest.contract_major != _major(CONTRACT_VERSION):
            problems.append(
                f"contract version mismatch: bundle={manifest.contract_version}, "
                f"code={CONTRACT_VERSION}"
            )

        if manifest.artifact_version != self.version:
            problems.append(
                f"artifact version mismatch: bundle={manifest.artifact_version}, "
                f"requested={self.version}"
            )

        missing_entries = [name for name in REQUIRED_FILES if name not in manifest.files]
        if missing_entries:
            problems.append(f"manifest missing file entries: {missing_entries}")

        for filename, entry in manifest.files.items():
            path = self.path_for(filename)
            if not path.is_file():
                problems.append(f"missing file: {filename}")
                continue
            if check_checksums:
                expected = entry.get("sha256")
                if not expected:
                    problems.append(f"no checksum recorded for {filename}")
                else:
                    actual = sha256_file(path)
                    if actual != expected:
                        problems.append(
                            f"checksum mismatch for {filename}: "
                            f"expected {expected[:12]}..., got {actual[:12]}..."
                        )
        return problems

    # --- Loading ------------------------------------------------------------
    def load(self, force_reload: bool = False) -> ArtifactBundle:
        """Load and verify the bundle. Cached after the first successful load."""
        with self._lock:
            if self._bundle is not None and not force_reload:
                return self._bundle

            if not self.manifest_path.is_file():
                raise ArtifactNotReadyError(
                    "Artifact bundle not found. Run the training pipeline first: "
                    "`python -m scripts.train_pipeline`",
                    details={"expected_path": str(self.manifest_path), "version": self.version},
                )

            problems = self.verify()
            if problems:
                raise ArtifactIntegrityError(
                    "Artifact bundle failed verification",
                    details={"problems": problems, "version": self.version},
                )

            manifest = self._read_manifest()
            if manifest.horizon_days != self.settings.forecast_horizon_days:
                raise ArtifactIntegrityError(
                    "Bundle horizon does not match configured horizon",
                    details={
                        "bundle_horizon_days": manifest.horizon_days,
                        "configured_horizon_days": self.settings.forecast_horizon_days,
                    },
                )

            logger.info(
                "Loading artifact bundle",
                extra={"version": self.version, "selected_model": manifest.selected_model},
            )

            bundle = ArtifactBundle(
                manifest=manifest,
                directory=self.directory,
                model=joblib.load(self.path_for("model.joblib")),
                feature_spec=self._read_json("feature_spec.json"),
                metrics=self._read_json("metrics.json"),
                training_config=self._read_json("training_config.json"),
                demand_history=self._read_parquet("demand_history.parquet"),
                products=self._read_parquet("products.parquet"),
                inventory_snapshot=self._read_parquet("inventory_snapshot.parquet"),
                batches=self._read_parquet("batches.parquet"),
                baseline=self._read_joblib_optional("baseline.joblib"),
                importance=self._read_json_optional("importance.json"),
                metrics_expiry=self._read_json_optional("metrics_expiry.json"),
                expiry_policy=self._read_json_optional("expiry_policy.json"),
            )
            self._bundle = bundle
            logger.info(
                "Artifact bundle loaded",
                extra={
                    "version": self.version,
                    "n_products": int(bundle.products.shape[0]),
                    "demand_rows": int(bundle.demand_history.shape[0]),
                },
            )
            return bundle

    # --- Small typed readers ------------------------------------------------
    def _read_json(self, filename: str) -> dict[str, Any]:
        try:
            return json.loads(self.path_for(filename).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError(
                f"{filename} is not valid JSON", details={"error": str(exc)}
            ) from exc

    def _read_json_optional(self, filename: str) -> dict[str, Any] | None:
        return self._read_json(filename) if self.path_for(filename).is_file() else None

    def _read_parquet(self, filename: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(self.path_for(filename))
        except Exception as exc:  # pyarrow raises several unrelated types
            raise ArtifactIntegrityError(
                f"Could not read {filename}", details={"error": str(exc)}
            ) from exc

    def _read_joblib_optional(self, filename: str) -> Any | None:
        return joblib.load(self.path_for(filename)) if self.path_for(filename).is_file() else None

    # --- Reporting ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Non-raising bundle status, used by /health and /model/info."""
        status: dict[str, Any] = {
            "version": self.version,
            "contract_version": CONTRACT_VERSION,
            "directory": str(self.directory),
            "ready": False,
            "loaded": self._bundle is not None,
            "problems": [],
        }
        if not self.manifest_path.is_file():
            status["problems"] = ["manifest.json missing"]
            return status
        try:
            manifest = self._read_manifest()
        except ArtifactIntegrityError as exc:
            status["problems"] = [exc.message]
            return status

        status.update(
            {
                "ready": True,
                "artifact_version": manifest.artifact_version,
                "created_at_utc": manifest.created_at_utc,
                "selected_model": manifest.selected_model,
                "prediction_origin": manifest.prediction_origin,
                "horizon_days": manifest.horizon_days,
                "dataset": manifest.dataset,
            }
        )
        return status

    def invalidate(self) -> None:
        """Drop the cached bundle (after retraining, or in tests)."""
        with self._lock:
            self._bundle = None


_store_lock = threading.Lock()
_store: ArtifactStore | None = None


def get_artifact_store(force_new: bool = False) -> ArtifactStore:
    """Process-wide store singleton (one bundle loaded per process)."""
    global _store
    with _store_lock:
        if _store is None or force_new:
            _store = ArtifactStore()
        return _store
