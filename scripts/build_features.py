"""Build the leakage-safe modelling dataset.

    python -m scripts.build_features

Writes ``data/processed/training_frame.parquet`` and a feature specification
draft. Every row is one (product, origin) pair with a forward-looking target.
"""

from __future__ import annotations

import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from ml.common.io import read_parquet, write_json, write_parquet
from ml.data_generation.config import GeneratorConfig
from ml.features.build import (
    ALL_FEATURES,
    assert_no_target_features,
    build_feature_spec,
    build_training_frame,
    origin_grid,
)

logger = get_logger("scripts.build_features")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    cfg = GeneratorConfig()

    panel = read_parquet(settings.processed_data_dir / "demand_panel.parquet")
    products = read_parquet(settings.processed_data_dir / "products.parquet")

    origins = origin_grid(cfg.origin_start, cfg.origin_end, cfg.origin_step_days)
    logger.info(
        "Building feature frame",
        extra={"origins": len(origins), "horizon_days": cfg.horizon_days},
    )

    frame = build_training_frame(
        panel=panel,
        products=products,
        origins=origins,
        horizon_days=cfg.horizon_days,
    )

    target = f"demand_next_{cfg.horizon_days}d"
    assert_no_target_features(list(ALL_FEATURES), target)

    write_parquet(frame, settings.processed_data_dir / "training_frame.parquet")
    write_json(
        {
            **build_feature_spec(cfg.horizon_days, lookback_days=180),
            "row_count": int(len(frame)),
            "n_products": int(frame["product_id"].nunique()),
            "n_origins": int(frame["origin_date"].nunique()),
            "origin_min": str(frame["origin_date"].min().date()),
            "origin_max": str(frame["origin_date"].max().date()),
            "target_mean": round(float(frame[target].mean()), 4),
            "target_zero_share": round(float((frame[target] == 0).mean()), 4),
        },
        settings.processed_data_dir / "feature_spec.json",
    )

    print(
        f"OK  rows={len(frame)}  products={frame['product_id'].nunique()}  "
        f"origins={frame['origin_date'].nunique()}  "
        f"target_mean={frame[target].mean():.2f}  "
        f"zero_target_share={(frame[target] == 0).mean():.1%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
