"""Initialise the SQLite database and load the processed dataset.

    python -m scripts.init_db                # schema + master data
    python -m scripts.init_db --with-scores  # also persist current scoring output

Idempotent: safe to run repeatedly. Master data is replaced in a single
transaction, so a half-loaded state is not possible.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime

import pandas as pd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.database.base import Base
from app.database.models import (
    Batch,
    DatasetVersion,
    ForecastResult,
    InventoryRecord,
    Product,
    Recommendation,
    RiskAssessment,
    Sale,
)
from app.database.session import build_engine, build_session_factory, ensure_sqlite_directory
from ml.common import runtime  # noqa: F401
from ml.common.io import read_json, read_parquet

logger = get_logger("scripts.init_db")


def _to_date(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def load_master_data(session, processed_dir, dataset_meta: dict) -> dict[str, int]:
    products = read_parquet(processed_dir / "products.parquet")
    inventory = read_parquet(processed_dir / "inventory.parquet")
    batches = read_parquet(processed_dir / "batches.parquet")
    sales = read_parquet(processed_dir / "sales.parquet")

    # Replace master data atomically (order respects FK dependencies).
    session.query(ForecastResult).delete()
    session.query(RiskAssessment).delete()
    session.query(Recommendation).delete()
    session.query(Batch).delete()
    session.query(Sale).delete()
    session.query(InventoryRecord).delete()
    session.query(Product).delete()
    session.query(DatasetVersion).delete()
    session.flush()

    session.add_all(
        [
            Product(
                product_id=str(row.product_id),
                medicine_name=str(row.medicine_name),
                category=str(row.category),
                manufacturer=str(row.manufacturer),
                supplier_id=str(row.supplier_id),
                unit_price=float(row.unit_price),
                unit_cost=float(row.unit_cost),
                minimum_stock_level=int(row.minimum_stock_level),
                lead_time_days=int(row.lead_time_days),
            )
            for row in products.itertuples(index=False)
        ]
    )
    session.flush()

    session.add_all(
        [
            InventoryRecord(
                product_id=str(row.product_id),
                current_stock=int(row.current_stock),
                reorder_level=int(row.reorder_level),
                last_restock_date=_to_date(row.last_restock_date),
                supplier_lead_time=int(row.supplier_lead_time),
            )
            for row in inventory.itertuples(index=False)
        ]
    )
    session.add_all(
        [
            Batch(
                batch_id=str(row.batch_id),
                product_id=str(row.product_id),
                batch_quantity=int(row.batch_quantity),
                remaining_quantity=int(row.remaining_quantity),
                manufacturing_date=_to_date(row.manufacturing_date),
                expiry_date=_to_date(row.expiry_date),
                batch_arrival_date=_to_date(row.batch_arrival_date),
            )
            for row in batches.itertuples(index=False)
        ]
    )
    session.flush()

    session.add_all(
        [
            Sale(
                transaction_id=str(row.transaction_id),
                product_id=str(row.product_id),
                transaction_date=_to_date(row.transaction_date),
                quantity_sold=int(row.quantity_sold),
                selling_price=float(row.selling_price),
                discount=float(row.discount),
                day_of_week=int(row.day_of_week),
                month=int(row.month),
                season=str(row.season),
                is_weekend=int(row.is_weekend),
                is_festival_window=int(row.is_festival_window),
            )
            for row in sales.itertuples(index=False)
        ]
    )

    session.add(
        DatasetVersion(
            dataset_version=str(dataset_meta["dataset_version"]),
            seed=int(dataset_meta["seed"]),
            is_synthetic=1,
            n_products=int(len(products)),
            n_sales=int(len(sales)),
            n_batches=int(len(batches)),
            date_min=date.fromisoformat(dataset_meta["date_range"]["start"]),
            date_max=date.fromisoformat(dataset_meta["date_range"]["end"]),
            loaded_at_utc=datetime.now(UTC),
        )
    )
    session.commit()

    return {
        "products": len(products),
        "inventory": len(inventory),
        "batches": len(batches),
        "sales": len(sales),
    }


def persist_scores(session, container) -> dict[str, int]:
    """Store the current forecast/risk/recommendation snapshot for audit."""
    bundle = container.bundle
    as_of = date.fromisoformat(bundle.prediction_origin[:10])

    forecasts = container.forecasts.forecast_all()
    session.add_all(
        [
            ForecastResult(
                product_id=str(row.product_id),
                as_of_date=as_of,
                horizon_days=bundle.horizon_days,
                predicted_demand=float(row.predicted_demand),
                lower_bound=None if pd.isna(row.lower) else float(row.lower),
                upper_bound=None if pd.isna(row.upper) else float(row.upper),
                confidence="medium",
                model_name=bundle.model_name,
                artifact_version=bundle.artifact_version,
            )
            for row in forecasts.itertuples(index=False)
        ]
    )

    risks = container.expiry.rows()
    session.add_all(
        [
            RiskAssessment(
                batch_id=str(row.batch_id),
                as_of_date=as_of,
                days_to_expiry=int(row.days_to_expiry),
                estimated_unsold_units=float(row.unsold_quantity),
                risk_score=float(row.risk_score),
                risk_level=str(row.risk_level),
            )
            for row in risks.itertuples(index=False)
        ]
    )

    import json

    recs = container.recommendations.build()
    session.add_all(
        [
            Recommendation(
                product_id=str(row.product_id),
                as_of_date=as_of,
                action=str(row.action),
                priority=str(row.priority),
                suggested_replenishment_qty=int(row.suggested_replenishment_qty),
                coverage_days=None if row.coverage_days is None else float(row.coverage_days),
                reasons_json=json.dumps(list(row.reasons)),
            )
            for row in recs.itertuples(index=False)
        ]
    )
    session.commit()
    return {
        "forecasts": len(forecasts),
        "risk_assessments": len(risks),
        "recommendations": len(recs),
    }


def main(argv: list[str] | None = None) -> int:
    with_scores = "--with-scores" in (argv or sys.argv[1:])

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    ensure_sqlite_directory(settings)

    processed = settings.processed_data_dir
    if not (processed / "products.parquet").is_file():
        logger.error("Processed data missing. Run: python -m scripts.validate_data")
        return 2

    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    dataset_meta = read_json(settings.raw_data_dir / "dataset_meta.json")

    with factory() as session:
        counts = load_master_data(session, processed, dataset_meta)
        print(f"OK  loaded master data: {counts}")

        if with_scores:
            from app.services.artifact_store import ArtifactStore
            from app.services.container import get_container

            store = ArtifactStore(settings=settings)
            if not store.is_ready():
                print("SKIP  no artifact bundle; run training first for --with-scores")
                return 0
            container = get_container(store)
            score_counts = persist_scores(session, container)
            print(f"OK  persisted scoring snapshot: {score_counts}")

    print(f"DB: {settings.db_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
