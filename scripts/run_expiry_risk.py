"""Assess expiry risk at the snapshot date and evaluate it against reality.

    python -m scripts.run_expiry_risk

Runs after training (needs the artifact bundle). Writes ``metrics_expiry.json``
and ``expiry_policy.json`` into the bundle, refreshes the manifest, and writes a
standalone report.
"""

from __future__ import annotations

import sys

import pandas as pd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.artifact_store import ArtifactStore
from app.services.forecast_service import build_forecast_service
from ml.common import runtime  # noqa: F401  (thread caps before sklearn import)
from ml.common.io import read_parquet, write_json
from ml.data_generation.config import GeneratorConfig
from ml.expiry_risk.evaluate import evaluate_risk
from ml.expiry_risk.fefo import FefoPolicy, allocate_fefo
from ml.expiry_risk.labels import LABEL_DEFINITION, realised_outcomes
from ml.expiry_risk.risk import RISK_POLICY_VERSION, assess_batches, summarise
from ml.forecasting.bundle_writer import refresh_manifest

logger = get_logger("scripts.run_expiry_risk")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    generator = GeneratorConfig()

    store = ArtifactStore(settings=settings)
    bundle = store.load()
    service = build_forecast_service(bundle)

    snapshot = pd.Timestamp(bundle.prediction_origin)
    policy = FefoPolicy(max_extrapolation_days=365)

    forecast_table = service.forecast_all()
    daily_rate = forecast_table.set_index("product_id")["predicted_demand"] / bundle.horizon_days

    allocation = allocate_fefo(bundle.batches, daily_rate, snapshot=snapshot, policy=policy)
    assessed = assess_batches(allocation, settings.risk_thresholds)
    summary = summarise(assessed)

    # Realised outcomes need data *after* the snapshot; that frame never enters the bundle.
    panel_full = read_parquet(settings.processed_data_dir / "demand_panel.parquet")
    data_end = pd.Timestamp(generator.end_date)
    realised = realised_outcomes(
        bundle.batches,
        panel_full,
        snapshot=snapshot,
        data_end=data_end,
        max_extrapolation_days=policy.max_extrapolation_days,
    )

    policy_payload = {
        "policy_version": RISK_POLICY_VERSION,
        "allocation": "FEFO",
        "tie_break": "batch_id ascending on equal expiry dates",
        "risk_rule": "estimated_unsold_units > 0",
        "severity_table": {
            "rows": "unsold ratio bands",
            "columns": "days to expiry bands",
            "bands": {
                "critical_days": settings.risk_critical_days,
                "high_days": settings.risk_high_days,
                "medium_days": settings.risk_medium_days,
            },
            "cell_rules": {
                "days<=critical": {"ratio>0.25": "critical", "else": "high"},
                "days<=high": {"ratio>0.50": "high", "else": "medium"},
                "days<=medium": {"ratio>0.50": "critical", "ratio>0.25": "medium", "else": "low"},
                "days>medium": {"ratio>0.75": "high", "ratio>0.50": "medium", "else": "low"},
                "ratio==0": "low",
            },
        },
        "demand_extrapolation": {
            "method": "forecast_horizon_total / horizon_days",
            "max_extrapolation_days": policy.max_extrapolation_days,
            "note": "Demand is assumed to continue at the forecast daily rate.",
        },
        "label_definition": LABEL_DEFINITION,
        "evaluation_window": {
            "start": str((snapshot + pd.Timedelta(days=1)).date()),
            "end": str(data_end.date()),
            "days": int((data_end - snapshot).days),
        },
    }

    metrics_expiry = evaluate_risk(
        predicted=assessed,
        realised=realised,
        threshold_rule="at_risk if estimated_unsold_units > 0",
        policy=policy_payload,
    )

    write_json(metrics_expiry, bundle.directory / "metrics_expiry.json")
    write_json(policy_payload, bundle.directory / "expiry_policy.json")
    manifest = refresh_manifest(bundle.directory)

    write_json(
        {
            "summary": summary,
            "metrics": metrics_expiry,
            "top_batches": assessed.sort_values("risk_score", ascending=False)
            .head(20)[
                [
                    "batch_id",
                    "product_id",
                    "remaining_quantity",
                    "days_to_expiry",
                    "allocatable_demand",
                    "unsold_quantity",
                    "risk_score",
                    "risk_level",
                ]
            ]
            .to_dict(orient="records"),
        },
        settings.reports_dir / "expiry_risk_report.json",
    )

    print("\n=== Expiry risk at the snapshot date ===")
    print(f"assessment date      : {snapshot.date()}")
    print(f"batches assessed     : {summary['total_batches']}")
    print(f"batches at risk      : {summary['at_risk_batches']}")
    print(f"estimated unsold     : {summary['estimated_unsold_units']:,.0f} units")
    print(f"by severity          : {summary['by_risk_level']}")

    print("\n=== Realised-outcome evaluation (FEFO label) ===")
    print(f"evaluable batches    : {metrics_expiry['n_batches']}")
    print(f"censored (excluded)  : {metrics_expiry['n_censored_excluded']}")
    print(f"precision            : {metrics_expiry['precision']:.3f}")
    print(f"recall               : {metrics_expiry['recall']:.3f}")
    print(f"f1                   : {metrics_expiry['f1']:.3f}")
    print(f"accuracy             : {metrics_expiry['accuracy']:.3f}")
    print(f"base rate (at risk)  : {metrics_expiry['base_rate']:.3f}")
    print(f"confusion            : {metrics_expiry['confusion']}")
    print(f"critical band        : {metrics_expiry['critical_band']}")
    print(f"\nmanifest files       : {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
