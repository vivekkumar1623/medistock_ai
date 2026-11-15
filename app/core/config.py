"""Centralised, environment-driven configuration.

Single source of truth for every tunable in the system. Nothing else in the
codebase may read ``os.environ`` directly: if a value matters, it is declared
here so it is discoverable, typed, validated, and overridable in deployment.

All variables are prefixed ``MEDITSTOCK_`` to avoid collisions with unrelated
process environment variables (a real failure mode on shared CI runners).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# <repo>/app/core/config.py -> parents[2] == <repo>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the API, ML pipeline and dashboard."""

    model_config = SettingsConfigDict(
        env_prefix="MEDITSTOCK_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "MediStock AI"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False

    # --- Reproducibility ---
    random_seed: int = Field(default=42, ge=0)

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    artifact_version: str = "v1"

    # --- Database ---
    db_url: str = "sqlite:///./data/medistock.db"

    # --- Dashboard -> API ---
    api_base_url: str = "http://127.0.0.1:8000"
    api_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Forecasting contract (must match the trained artifact manifest) ---
    forecast_horizon_days: int = Field(default=30, ge=1, le=180)

    # --- Expiry-risk policy thresholds (days remaining before expiry) ---
    risk_critical_days: int = Field(default=30, ge=1)
    risk_high_days: int = Field(default=60, ge=1)
    risk_medium_days: int = Field(default=90, ge=1)

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    def model_post_init(self, __context: object) -> None:
        """Resolve relative paths against the repository root.

        Why: commands may be launched from any working directory (IDE, Docker,
        Windows shortcut). Resolving against ``PROJECT_ROOT`` instead of the
        process CWD removes an entire class of "works on my machine" bugs.
        """
        for field_name in ("data_dir", "artifact_dir"):
            path: Path = getattr(self, field_name)
            if not path.is_absolute():
                object.__setattr__(self, field_name, (PROJECT_ROOT / path).resolve())

    # --- Derived paths -------------------------------------------------------
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def interim_data_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def artifact_path(self) -> Path:
        return self.artifact_dir / self.artifact_version

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def risk_thresholds(self) -> dict[str, int]:
        """Ordered days-remaining cut-offs used by the expiry-risk engine."""
        return {
            "critical": self.risk_critical_days,
            "high": self.risk_high_days,
            "medium": self.risk_medium_days,
        }

    def ensure_directories(self) -> None:
        """Create the writable directories the pipeline expects."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.interim_data_dir,
            self.artifact_path,
            self.figures_dir,
            self.reports_dir / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so repeated calls (FastAPI dependency injection, Streamlit reruns)
    do not re-read and re-validate the environment. Tests call
    ``get_settings.cache_clear()`` after monkeypatching the environment.
    """
    return Settings()
