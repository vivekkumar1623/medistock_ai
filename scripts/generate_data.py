"""Operator entrypoint: generate the synthetic pharmacy dataset.

Usage (from the repository root):

    python -m scripts.generate_data

Deterministic: the same ``MEDITSTOCK_RANDOM_SEED`` and generator version produce
byte-identical CSV/parquet files, asserted by
``tests/regression/test_reproducibility.py``.
"""

from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from ml.data_generation import GeneratorConfig, generate_dataset, write_dataset

logger = get_logger("scripts.generate_data")


def build_config(seed: int | None, n_products: int | None) -> GeneratorConfig:
    base = GeneratorConfig()
    return GeneratorConfig(
        seed=seed if seed is not None else base.seed,
        n_products=n_products if n_products is not None else base.n_products,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the MediStock AI synthetic dataset.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument(
        "--products", type=int, default=None, help="Override the number of products."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json, settings.environment)
    settings.ensure_directories()

    cfg = build_config(args.seed, args.products)
    logger.info(
        "Generating dataset",
        extra={
            "seed": cfg.seed,
            "n_products": cfg.n_products,
            "start": cfg.start_date,
            "end": cfg.end_date,
            "snapshot": cfg.snapshot_date,
        },
    )

    dataset = generate_dataset(cfg)
    meta = write_dataset(dataset, settings.raw_data_dir, settings.interim_data_dir)

    logger.info(
        "Dataset written",
        extra={
            "products": meta["n_products"],
            "sales": meta["n_sales_transactions"],
            "batches": meta["n_batches"],
            "panel_rows": meta["n_demand_panel_rows"],
            "raw_dir": str(settings.raw_data_dir),
        },
    )
    print(
        f"OK  products={meta['n_products']}  sales={meta['n_sales_transactions']}  "
        f"batches={meta['n_batches']}  panel_rows={meta['n_demand_panel_rows']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
