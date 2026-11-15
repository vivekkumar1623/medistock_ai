"""Produce the error-analysis report from the served model.

    python -m scripts.run_error_analysis

Uses the model inside the artifact bundle and the identical chronological test
split defined in ``TrainingConfig``, then slices the residuals.
"""

from __future__ import annotations

import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.artifact_store import ArtifactStore
from ml.common import runtime  # noqa: F401
from ml.common.io import read_parquet, write_json
from ml.data_generation.config import GeneratorConfig
from ml.evaluation.error_analysis import analyse, render_markdown
from ml.features.build import TARGET_TEMPLATE
from ml.forecasting.splits import chronological_split
from ml.forecasting.train import FEATURES, TrainingConfig

logger = get_logger("scripts.run_error_analysis")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    cfg = GeneratorConfig()
    training_cfg = TrainingConfig()

    store = ArtifactStore(settings=settings)
    bundle = store.load()

    frame = read_parquet(settings.processed_data_dir / "training_frame.parquet")
    target = TARGET_TEMPLATE.format(horizon=cfg.horizon_days)
    _, _, test, _ = chronological_split(
        frame, target, training_cfg.validation_start, training_cfg.test_start
    )

    test = test.copy()
    test["prediction"] = bundle.model.predict(test[FEATURES])

    analysis = analyse(test, target)
    write_json(analysis, settings.reports_dir / "error_analysis.json")
    report = render_markdown(analysis, settings.reports_dir / "error_analysis.md")

    print(f"OK  error analysis: {report}")
    print(f"    overall: {analysis['overall']}")
    worst_category = analysis["by_category"][0]["group"] if analysis["by_category"] else "n/a"
    print(f"    worst category: {worst_category}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
