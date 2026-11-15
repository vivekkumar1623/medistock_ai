"""Validate -> clean -> profile -> write processed parquet.

Run from the repository root:

    python -m scripts.validate_data

Exit code is non-zero when validation finds *errors* that cleaning could not fix,
so the script is usable as a pipeline gate.
"""

from __future__ import annotations

import sys

import pandas as pd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from ml.common.io import read_json, write_json, write_parquet
from ml.data_generation.config import GeneratorConfig
from ml.preprocessing.cleaning import clean_and_revalidate
from ml.preprocessing.profiling import profile_dataset, render_markdown
from ml.preprocessing.validation import validate_dataset

logger = get_logger("scripts.validate_data")

PROCESSED_FILES = {
    "products": "products.parquet",
    "sales": "sales.parquet",
    "inventory": "inventory.parquet",
    "batches": "batches.parquet",
    "demand_panel": "demand_panel.parquet",
}


def load_raw(raw_dir, interim_dir) -> tuple[pd.DataFrame, ...]:
    products = pd.read_csv(raw_dir / "products.csv")
    sales = pd.read_csv(raw_dir / "sales.csv")
    inventory = pd.read_csv(raw_dir / "inventory.csv")
    batches = pd.read_csv(raw_dir / "batches.csv")
    panel = pd.read_parquet(interim_dir / "demand_panel.parquet")

    # Dates are text in CSV. Parse once, here, so validation and cleaning both
    # receive real dtypes (and `errors="coerce"` surfaces bad values as NaT).
    sales["transaction_date"] = pd.to_datetime(sales["transaction_date"], errors="coerce")
    for column in ("manufacturing_date", "expiry_date", "batch_arrival_date"):
        batches[column] = pd.to_datetime(batches[column], errors="coerce")
    inventory["last_restock_date"] = pd.to_datetime(inventory["last_restock_date"], errors="coerce")
    return products, sales, inventory, batches, panel


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    settings.ensure_directories()

    cfg = GeneratorConfig()
    raw_dir, interim_dir, processed_dir = (
        settings.raw_data_dir,
        settings.interim_data_dir,
        settings.processed_data_dir,
    )
    if not (raw_dir / "products.csv").is_file():
        logger.error("Raw dataset missing. Run: python -m scripts.generate_data")
        return 2

    dataset_meta = read_json(raw_dir / "dataset_meta.json")
    products, sales, inventory, batches, panel = load_raw(raw_dir, interim_dir)

    logger.info("Validating raw dataset")
    raw_report = validate_dataset(
        products, sales, inventory, batches, panel, snapshot_date=cfg.snapshot_date
    )
    logger.info(
        "Raw validation complete",
        extra={"errors": len(raw_report.errors), "warnings": len(raw_report.warnings)},
    )

    cleaned, clean_report = clean_and_revalidate(
        products,
        sales,
        inventory,
        batches,
        panel,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        snapshot_date=cfg.snapshot_date,
    )

    write_json(
        {
            "raw": raw_report.to_dict(),
            "after_cleaning": clean_report.to_dict(),
            "cleaning": cleaned.to_dict(),
        },
        settings.reports_dir / "data_quality_report.json",
    )

    write_parquet(cleaned.products, processed_dir / PROCESSED_FILES["products"])
    write_parquet(cleaned.sales, processed_dir / PROCESSED_FILES["sales"])
    write_parquet(cleaned.inventory, processed_dir / PROCESSED_FILES["inventory"])
    write_parquet(cleaned.batches, processed_dir / PROCESSED_FILES["batches"])
    write_parquet(cleaned.panel, processed_dir / PROCESSED_FILES["demand_panel"])

    profile = profile_dataset(
        cleaned.products,
        cleaned.sales,
        cleaned.inventory,
        cleaned.batches,
        cleaned.panel,
        dataset_meta,
    )
    write_json(profile, settings.reports_dir / "data_profile.json")
    (settings.reports_dir / "data_profile.md").write_text(
        render_markdown(profile), encoding="utf-8"
    )

    print(
        f"OK  raw_errors={len(raw_report.errors)} raw_warnings={len(raw_report.warnings)}  "
        f"post_clean_errors={len(clean_report.errors)}  "
        f"cleaning_actions={len(cleaned.actions)}  "
        f"processed_rows={len(cleaned.products)}/{len(cleaned.sales)}/{len(cleaned.batches)}"
    )
    if not clean_report.is_valid:
        for issue in clean_report.errors[:10]:
            print(f"  ERROR {issue.check}: {issue.message} (n={issue.count})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
