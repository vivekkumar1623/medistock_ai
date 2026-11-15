"""Train, select and bundle the demand forecaster.

python -m scripts.train_pipeline
"""

from __future__ import annotations

import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Must be imported before scikit-learn: caps OpenMP/joblib worker pools.
from ml.common import runtime  # noqa: F401
from ml.forecasting.train import TrainingConfig, train_forecaster

logger = get_logger("scripts.train_pipeline")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    settings.ensure_directories()

    if not (settings.processed_data_dir / "training_frame.parquet").is_file():
        logger.error("Processed data missing. Run: python -m scripts.build_features")
        return 2

    result = train_forecaster(settings, TrainingConfig())

    metrics = result.metrics
    print("\n=== Demand forecast evaluation (identical rows per split) ===")
    header = f"{'model':<26}{'split':<12}{'MAE':>9}{'RMSE':>9}{'WAPE':>8}"
    print(header)
    print("-" * len(header))
    for row in sorted(metrics["models"], key=lambda r: (r["split"], r["wape"])):
        print(
            f"{row['model_name']:<26}{row['split']:<12}"
            f"{row['mae']:>9.2f}{row['rmse']:>9.2f}{row['wape']:>8.3f}"
        )

    if metrics.get("rolling_origin"):
        print("\n=== Rolling-origin validation (WAPE) ===")
        folds = sorted({row["fold"] for row in metrics["rolling_origin"]})
        names = sorted({row["model_name"] for row in metrics["rolling_origin"]})
        print(f"{'model':<26}" + "".join(f"{'fold ' + str(f):>10}" for f in folds))
        for name in names:
            values = []
            for fold in folds:
                match = next(
                    r
                    for r in metrics["rolling_origin"]
                    if r["fold"] == fold and r["model_name"] == name
                )
                values.append(f"{match['wape']:>10.3f}")
            print(f"{name:<26}" + "".join(values))

    print(f"\nSelected model: {result.selected_model}")
    print(f"Bundle: {result.bundle_dir}")
    print(f"Files: {', '.join(sorted(result.manifest['files']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
