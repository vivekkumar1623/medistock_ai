"""Generate the EDA figures and report.

    python -m scripts.run_eda

Requires a trained artifact bundle (for the forecast-vs-coverage figure) and the
processed dataset. Writes ``reports/figures/*.png`` and ``reports/eda_report.md``.
"""

from __future__ import annotations

import sys

import pandas as pd

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from ml.common import runtime  # noqa: F401
from ml.common.io import read_parquet
from ml.data_generation.config import GeneratorConfig
from ml.evaluation.eda import run_eda
from ml.forecasting.bundle_writer import library_versions

logger = get_logger("scripts.run_eda")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    cfg = GeneratorConfig()

    processed = settings.processed_data_dir
    required = [
        "products.parquet",
        "sales.parquet",
        "inventory.parquet",
        "batches.parquet",
        "demand_panel.parquet",
    ]
    missing = [name for name in required if not (processed / name).is_file()]
    if missing:
        logger.error("Processed data missing: %s. Run validate_data first.", missing)
        return 2

    products = read_parquet(processed / "products.parquet")
    sales = read_parquet(processed / "sales.parquet")
    inventory = read_parquet(processed / "inventory.parquet")
    batches = read_parquet(processed / "batches.parquet")
    panel = read_parquet(processed / "demand_panel.parquet")

    # Forecasts for the coverage figure: read from the artifact-backed service so the
    # EDA uses the same numbers the application serves.
    from app.services.artifact_store import ArtifactStore
    from app.services.forecast_service import ForecastService

    store = ArtifactStore(settings=settings)
    if store.is_ready():
        forecast = ForecastService(store.load()).forecast_all()[["product_id", "predicted_demand"]]
    else:
        logger.warning("No artifact bundle; coverage figure uses trailing 30-day demand")
        forecast = (
            panel[panel["date"] > pd.Timestamp(cfg.snapshot_date) - pd.Timedelta(days=30)]
            .groupby("product_id")["quantity"]
            .sum()
            .rename("predicted_demand")
            .reset_index()
        )

    report = run_eda(
        products=products,
        sales=sales,
        inventory=inventory,
        batches=batches,
        panel=panel,
        forecast=forecast,
        snapshot=cfg.snapshot_date,
        figures_dir=settings.figures_dir,
        report_path=settings.reports_dir / "eda_report.md",
    )

    print(f"OK  EDA report: {report}")
    print(f"OK  figures: {len(list(settings.figures_dir.glob('*.png')))}")
    print(f"OK  library versions: {library_versions()['pandas']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
