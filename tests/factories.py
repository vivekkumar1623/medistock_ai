"""Test factories: build throw-away artifact bundles that satisfy the contract.

Why a factory instead of committed fixtures: the contract is the thing under
test. A factory lets each test vary one field (checksum, horizon, missing file)
and assert the store reacts correctly. Every later phase reuses this factory,
so "does inference work against a saved bundle?" is testable without training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_HORIZON = 30
DEFAULT_ORIGIN = "2025-11-01"


class FakeDemandModel:
    """Stand-in with the same ``predict(features)`` contract as the real pipeline.

    Defined here (not in app code) so the inference layers can be exercised
    end-to-end without training a model. It is deliberately trivial and obviously
    *not* a forecast -- tests assert routing, contracts and arithmetic, never
    accuracy.
    """

    def __init__(self, horizon_days: int = DEFAULT_HORIZON) -> None:
        self.horizon_days = horizon_days

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        base = features["mean_30"].to_numpy(dtype=float) * self.horizon_days
        return np.clip(base, 0.0, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_fake_bundle(
    directory: Path,
    *,
    version: str = "v1",
    horizon_days: int = DEFAULT_HORIZON,
    contract_version: str = "1.0.0",
    corrupt_checksums: bool = False,
    tamper_model_after_manifest: bool = False,
    omit_files: tuple[str, ...] = (),
    include_optional: bool = True,
) -> Path:
    """Create a minimal but contract-valid artifact bundle in ``directory``.

    Args:
        corrupt_checksums: record a wrong checksum for ``model.joblib``.
        tamper_model_after_manifest: rewrite ``model.joblib`` after checksums
            were recorded, simulating post-signing corruption.
        omit_files: file names to leave out of the manifest *and* the directory.
    """
    directory.mkdir(parents=True, exist_ok=True)

    products = pd.DataFrame(
        {
            "product_id": ["MED-0001", "MED-0002"],
            "medicine_name": ["Amoxicillin 500mg Capsule", "Cetirizine 10mg Tablet"],
            "category": ["Antibiotics", "Antihistamines"],
            "manufacturer": ["Apex Pharma", "Northwind Labs"],
            "supplier_id": ["SUP-01", "SUP-02"],
            "unit_price": [12.5, 6.0],
            "unit_cost": [8.4, 3.9],
            "minimum_stock_level": [40, 60],
            "lead_time_days": [7, 5],
        }
    )

    dates = pd.date_range(end=DEFAULT_ORIGIN, periods=120, freq="D")
    demand_history = pd.DataFrame(
        {
            "product_id": np.repeat(["MED-0001", "MED-0002"], len(dates)),
            "date": np.tile(dates, 2),
            "quantity": np.tile(np.arange(10, 130), 2).astype("int64"),
        }
    )

    inventory = pd.DataFrame(
        {
            "product_id": ["MED-0001", "MED-0002"],
            "current_stock": [120, 45],
            "reorder_level": [50, 70],
            "supplier_lead_time": [7, 5],
        }
    )

    batches = pd.DataFrame(
        {
            "batch_id": ["BATCH-0001-01", "BATCH-0001-02", "BATCH-0002-01"],
            "product_id": ["MED-0001", "MED-0001", "MED-0002"],
            "batch_quantity": [80, 50, 60],
            "remaining_quantity": [70, 50, 45],
            "manufacturing_date": pd.to_datetime(["2024-11-01", "2025-01-10", "2024-12-01"]),
            "expiry_date": pd.to_datetime(["2025-11-15", "2026-05-20", "2025-12-10"]),
            "batch_arrival_date": pd.to_datetime(["2024-11-20", "2025-01-25", "2024-12-15"]),
        }
    )

    payloads: dict[str, object] = {
        "model.joblib": FakeDemandModel(horizon_days=horizon_days),
        "baseline.joblib": {"type": "fake_seasonal_naive", "season_length": 7},
        "feature_spec.json": {
            "feature_version": "fs-test",
            "target": f"demand_next_{horizon_days}d",
            "lookback_days": 180,
            "features": [
                {"name": "mean_30", "dtype": "float64", "role": "numeric"},
                {"name": "sum_30", "dtype": "float64", "role": "numeric"},
                {"name": "std_90", "dtype": "float64", "role": "numeric"},
                {"name": "zero_ratio_90", "dtype": "float64", "role": "numeric"},
                {"name": "ratio_7_30", "dtype": "float64", "role": "numeric"},
                {"name": "category", "dtype": "category", "role": "categorical"},
            ],
            "numeric_features": ["mean_30", "sum_30", "std_90", "zero_ratio_90", "ratio_7_30"],
            "categorical_features": ["category"],
        },
        "metrics.json": {
            "selected_model": "hist_gradient_boosting",
            "baseline_model": "seasonal_naive_30d",
            "selection_rationale": "Lowest mean rolling-origin WAPE; see docs/model_card.md.",
            "residual_quantiles": {
                "p10": -18.5,
                "p50": 0.4,
                "p90": 22.9,
                "method": "empirical_test_residual_quantiles_10_90",
                "n_residuals": 2,
            },
            "models": [
                {
                    "model_name": "seasonal_naive_30d",
                    "split": "test",
                    "mae": 18.4,
                    "rmse": 24.1,
                    "wape": 0.41,
                    "n_observations": 2,
                },
                {
                    "model_name": "hist_gradient_boosting",
                    "split": "test",
                    "mae": 11.2,
                    "rmse": 15.7,
                    "wape": 0.24,
                    "n_observations": 2,
                },
            ],
            "notes": ["Fixture metrics, not real evaluation output."],
        },
        "training_config.json": {
            "seed": 42,
            "horizon_days": horizon_days,
            "candidate_models": ["seasonal_naive_30d", "random_forest", "hist_gradient_boosting"],
            "split_dates": {
                "train_end": "2025-07-31",
                "validation_end": "2025-09-15",
                "test_end": "2025-11-01",
            },
        },
        "importance.json": {
            "method": "permutation_importance_test_set",
            "features": [
                {"feature": "lag_30_sum", "importance": 0.52},
                {"feature": "lag_7_mean", "importance": 0.31},
                {"feature": "rolling_30_std", "importance": 0.09},
            ],
        },
        "metrics_expiry.json": {
            "precision": 0.72,
            "recall": 0.81,
            "f1": 0.76,
            "accuracy": 0.84,
            "confusion": {
                "true_positive": 34,
                "false_positive": 13,
                "true_negative": 71,
                "false_negative": 8,
            },
            "n_batches": 126,
            "threshold_rule": "at_risk if estimated_unsold_units > 0",
        },
        "expiry_policy.json": {
            "allocation": "FEFO",
            "risk_rule": "estimated_unsold_units > 0",
            "severity_bands": {"critical": 30, "high": 60, "medium": 90},
        },
        "products.parquet": products,
        "demand_history.parquet": demand_history,
        "inventory_snapshot.parquet": inventory,
        "batches.parquet": batches,
    }

    for name in omit_files:
        payloads.pop(name, None)

    # Write every file first, then hash, then manifest (order matters).
    for name, payload in payloads.items():
        path = directory / name
        if name.endswith(".parquet"):
            payload.to_parquet(path, index=False)  # type: ignore[union-attr]
        elif name.endswith(".joblib"):
            joblib.dump(payload, path)
        else:
            _write_json(path, payload)  # type: ignore[arg-type]

    files: dict[str, dict[str, object]] = {}
    for name in payloads:
        path = directory / name
        digest = _sha256(path)
        if corrupt_checksums and name == "model.joblib":
            digest = "0" * 64
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}

    manifest = {
        "contract_version": contract_version,
        "artifact_version": version,
        "created_at_utc": "2025-11-02T09:00:00+00:00",
        "dataset": {
            "version": "ds-test",
            "generator_seed": 42,
            "n_products": int(products.shape[0]),
            "n_sales_rows": int(demand_history.shape[0]),
            "date_min": "2024-11-01",
            "date_max": "2025-11-01",
            "is_synthetic": True,
        },
        "prediction_origin": DEFAULT_ORIGIN,
        "horizon_days": horizon_days,
        "selected_model": "hist_gradient_boosting",
        "candidate_models": ["seasonal_naive_30d", "random_forest", "hist_gradient_boosting"],
        "library_versions": {"scikit_learn": "1.8.0", "pandas": "3.0.2", "numpy": "2.4.4"},
        "files": files,
    }
    _write_json(directory / "manifest.json", manifest)

    if tamper_model_after_manifest:
        joblib.dump({"type": "corrupted_pipeline"}, directory / "model.joblib")

    return directory
