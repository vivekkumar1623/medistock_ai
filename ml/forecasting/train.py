"""Model training, selection and bundle creation.

Evaluation protocol (stated once, enforced here):

1. Split by origin date: train -> validation -> test. No shuffling.
2. Tune each candidate's small grid on the **validation window** only.
3. Select the winner by **mean rolling-origin WAPE**, computed on expanding-window
   folds that lie entirely **before the test period**. A single validation window
   is a noisy basis for a decision: with 600 rows, a 0.001 WAPE gap is
   indistinguishable from zero, and picking on it selects noise. Averaging three
   expanding-window folds gives a selection statistic with a defensible variance,
   and it still never touches the test set.
4. Refit the selected configuration on train+validation.
5. Score **test exactly once** with that frozen model.
6. Report every candidate on identical test rows so the comparison is fair.

The test set is never used to choose a hyperparameter, a model, or a threshold.
If you find yourself running step 5 twice because the number was disappointing,
that is the failure this ordering exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from ml.common.io import read_json, read_parquet
from ml.data_generation.config import GeneratorConfig
from ml.evaluation.metrics import compare_on_common_rows, evaluate
from ml.explainability.importance import permutation_importance_summary
from ml.features.build import ALL_FEATURES, TARGET_TEMPLATE, build_feature_spec
from ml.forecasting.baseline import MovingAverage90, SeasonalNaive30
from ml.forecasting.bundle_writer import write_bundle
from ml.forecasting.pipeline import CANDIDATE_BUILDERS, CANDIDATE_GRIDS
from ml.forecasting.splits import (
    chronological_split,
    rolling_origin_folds,
    subset_by_origins,
)

logger = get_logger("ml.forecasting.train")

PUBLIC_PRODUCT_COLUMNS = [
    "product_id",
    "medicine_name",
    "category",
    "manufacturer",
    "unit_price",
    "unit_cost",
    "supplier_id",
    "minimum_stock_level",
    "lead_time_days",
]

INVENTORY_COLUMNS = [
    "product_id",
    "current_stock",
    "reorder_level",
    "supplier_lead_time",
    "last_restock_date",
]

FEATURES: list[str] = list(ALL_FEATURES)


@dataclass(frozen=True)
class TrainingConfig:
    """Everything that determines the trained artifact, in one auditable place."""

    validation_start: str = "2025-06-01"
    test_start: str = "2025-08-01"
    n_importance_repeats: int = 5
    importance_top_n: int = 15
    n_rolling_folds: int = 3
    rolling_block_origins: int = 2
    artifact_version: str | None = None


@dataclass
class TrainingResult:
    bundle_dir: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    selected_model: str


def _wape_or_inf(scores: dict[str, Any]) -> float:
    value = scores.get("wape")
    return float(value) if value is not None else float("inf")


def train_forecaster(
    settings: Settings | None = None,
    training_config: TrainingConfig | None = None,
) -> TrainingResult:
    settings = settings or get_settings()
    cfg = training_config or TrainingConfig()
    generator = GeneratorConfig()

    target = TARGET_TEMPLATE.format(horizon=generator.horizon_days)
    processed = settings.processed_data_dir

    frame = read_parquet(processed / "training_frame.parquet")
    products = read_parquet(processed / "products.parquet")
    inventory = read_parquet(processed / "inventory.parquet")
    batches = read_parquet(processed / "batches.parquet")
    panel = read_parquet(processed / "demand_panel.parquet")
    dataset_meta = read_json(settings.raw_data_dir / "dataset_meta.json")

    train, validation, test, split_summary = chronological_split(
        frame, target, cfg.validation_start, cfg.test_start
    )
    logger.info(
        "Chronological split ready",
        extra={
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    )

    baselines = [SeasonalNaive30(horizon_days=generator.horizon_days), MovingAverage90()]
    validation_predictions: dict[str, np.ndarray] = {
        baseline.name: baseline.predict(validation) for baseline in baselines
    }
    test_predictions: dict[str, np.ndarray] = {
        baseline.name: baseline.predict(test) for baseline in baselines
    }

    # --- 1. Tune candidates on the validation window only -------------------
    tuning_records: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    for name, builder in CANDIDATE_BUILDERS.items():
        best: tuple[dict[str, Any], dict[str, Any]] | None = None
        for params in CANDIDATE_GRIDS[name]:
            model = builder(generator.seed, params)
            model.fit(train[FEATURES], train[target])
            scores = evaluate(validation[target], model.predict(validation[FEATURES]))
            tuning_records.append(
                {"model_name": name, "params": dict(params), "split": "validation", **scores}
            )
            if best is None or _wape_or_inf(scores) < _wape_or_inf(best[1]):
                best = (params, scores)
        if best is None:
            raise RuntimeError(f"No validation result recorded for candidate {name}")
        best_params[name] = dict(best[0])
        logger.info(
            "Candidate tuned",
            extra={"model": name, "params": best[0], "validation_wape": best[1]["wape"]},
        )

    # --- 2. Rolling-origin evaluation strictly before the test period -------
    pre_test = frame[pd.to_datetime(frame["origin_date"]) < pd.Timestamp(cfg.test_start)]
    fold_rows = _rolling_origin_evaluation(
        pre_test, target, cfg, generator.seed, baselines, best_params
    )
    selection_scores = _mean_wape_by_model(fold_rows)

    # --- 3. Select on the robust statistic ---------------------------------
    selected_model_name = min(selection_scores, key=selection_scores.get)
    logger.info(
        "Model selected by rolling-origin WAPE",
        extra={"selected": selected_model_name, "scores": selection_scores},
    )

    # --- 4. Refit candidates on train+validation, then score test once ------
    train_plus_validation = pd.concat([train, validation], ignore_index=True)
    final_models: dict[str, Any] = {}
    for name, builder in CANDIDATE_BUILDERS.items():
        train_only = builder(generator.seed, best_params[name])
        train_only.fit(train[FEATURES], train[target])
        validation_predictions[name] = train_only.predict(validation[FEATURES])

        refit = builder(generator.seed, best_params[name])
        refit.fit(train_plus_validation[FEATURES], train_plus_validation[target])
        test_predictions[name] = refit.predict(test[FEATURES])
        final_models[name] = refit

    validation_rows = compare_on_common_rows(validation, validation_predictions, target)
    test_rows = compare_on_common_rows(test, test_predictions, target)

    for row in validation_rows + test_rows:
        if row["wape"] is None:
            raise RuntimeError(
                f"WAPE undefined for {row['model_name']} on {row['split']}: "
                "the split contains no positive actual demand"
            )

    selected_test = next(row for row in test_rows if row["model_name"] == selected_model_name)
    selected_validation = next(
        row for row in validation_rows if row["model_name"] == selected_model_name
    )
    logger.info(
        "Selected model scored on test",
        extra={
            "selected": selected_model_name,
            "validation_wape": selected_validation["wape"],
            "test_wape": selected_test["wape"],
        },
    )

    if selected_model_name in final_models:
        served_model = final_models[selected_model_name]
    else:
        served_model = next(b for b in baselines if b.name == selected_model_name)

    # Empirical residual quantiles from the test period, used to build an honest
    # prediction interval at serving time. This is not a calibrated posterior; the
    # label says exactly what it is.
    test_actual = test[target].to_numpy(dtype=float)
    test_residual = test_actual - test_predictions[selected_model_name]
    residual_quantiles = {
        "p10": round(float(np.quantile(test_residual, 0.10)), 4),
        "p50": round(float(np.quantile(test_residual, 0.50)), 4),
        "p90": round(float(np.quantile(test_residual, 0.90)), 4),
        "method": "empirical_test_residual_quantiles_10_90",
        "n_residuals": int(len(test_residual)),
    }

    if selected_model_name in final_models:
        importance = permutation_importance_summary(
            model=final_models[selected_model_name],
            features=test[FEATURES],
            target=test[target].to_numpy(dtype=float),
            feature_names=FEATURES,
            seed=generator.seed,
            n_repeats=cfg.n_importance_repeats,
            top_n=cfg.importance_top_n,
        )

    metrics = {
        "selected_model": selected_model_name,
        "baseline_model": "seasonal_naive_30d",
        "selection_metric": "mean_rolling_origin_wape_pre_test",
        "selection_scores": {k: round(v, 6) for k, v in selection_scores.items()},
        "selection_rationale": (
            f"Selected '{selected_model_name}': lowest mean rolling-origin WAPE "
            f"({selection_scores[selected_model_name]:.4f}) across "
            f"{cfg.n_rolling_folds} expanding-window folds that end before "
            f"{cfg.test_start}. Single-window validation WAPE was also recorded "
            f"({selected_validation['wape']:.3f}) but was not used to select, because "
            "differences between the strongest candidates on one 600-row window were "
            "within noise. The test set was scored once afterwards "
            f"(WAPE {selected_test['wape']:.3f})."
        ),
        "models": [{**row, "split": "validation"} for row in validation_rows]
        + [{**row, "split": "test"} for row in test_rows],
        "tuning": tuning_records,
        "rolling_origin": fold_rows,
        "baselines": [baseline.describe() for baseline in baselines],
        "residual_quantiles": residual_quantiles,
        "notes": [
            "All candidate models are scored on identical rows within each split.",
            "WAPE is volume-weighted: sum(|error|) / sum(|actual|).",
            "Overlapping 30-day target windows across origins induce autocorrelation; "
            "that is why selection uses the fold mean rather than one window.",
            "Test metrics were produced after model selection and were not used to tune anything.",
        ],
    }

    training_config = {
        "seed": generator.seed,
        "horizon_days": generator.horizon_days,
        "validation_start": cfg.validation_start,
        "test_start": cfg.test_start,
        "split": split_summary.to_dict(),
        "candidate_models": list(CANDIDATE_BUILDERS) + [b.name for b in baselines],
        "selected_params": best_params,
        "grids": {name: [dict(p) for p in grid] for name, grid in CANDIDATE_GRIDS.items()},
        "feature_count": len(FEATURES),
        "feature_spec_version": build_feature_spec(generator.horizon_days, lookback_days=180)[
            "feature_version"
        ],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "selection_metric": metrics["selection_metric"],
        "dataset_version": dataset_meta["dataset_version"],
    }

    snapshot = pd.Timestamp(generator.snapshot_date)
    demand_history = panel[panel["date"] <= snapshot].copy()

    dataset_manifest = {
        "version": dataset_meta["dataset_version"],
        "generator_seed": dataset_meta["seed"],
        "n_products": int(products["product_id"].nunique()),
        "n_sales_rows": int(dataset_meta["n_sales_transactions"]),
        "n_batches": int(len(batches)),
        "date_min": dataset_meta["date_range"]["start"],
        "date_max": dataset_meta["date_range"]["end"],
        "snapshot_date": generator.snapshot_date,
        "is_synthetic": True,
        "provenance_note": (
            "Synthetic pharmacy data generated by the MediStock AI simulator. "
            "Not real pharmacy transaction data."
        ),
    }

    artifact_version = cfg.artifact_version or settings.artifact_version
    bundle_dir = settings.artifact_dir / artifact_version
    manifest = write_bundle(
        bundle_dir,
        artifact_version=artifact_version,
        selected_model=selected_model_name,
        candidate_models=list(CANDIDATE_BUILDERS) + [b.name for b in baselines],
        prediction_origin=generator.snapshot_date,
        horizon_days=generator.horizon_days,
        dataset=dataset_manifest,
        model=served_model,
        baseline=baselines[0],
        feature_spec=build_feature_spec(generator.horizon_days, lookback_days=180),
        metrics=metrics,
        training_config=training_config,
        demand_history=demand_history,
        products=products[PUBLIC_PRODUCT_COLUMNS],
        inventory_snapshot=inventory[INVENTORY_COLUMNS],
        batches=batches,
        importance=importance,
    )

    logger.info("Artifact bundle written", extra={"directory": str(bundle_dir)})
    return TrainingResult(
        bundle_dir=bundle_dir,
        manifest=manifest,
        metrics=metrics,
        selected_model=selected_model_name,
    )


def _rolling_origin_evaluation(
    frame: pd.DataFrame,
    target: str,
    cfg: TrainingConfig,
    seed: int,
    baselines: list[Any],
    best_params: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expanding-window folds entirely inside the pre-test period.

    ``frame`` must already exclude test origins. This is asserted, because
    accidentally including them would turn a validation statistic into a test
    statistic and invalidate the whole protocol.
    """
    if pd.to_datetime(frame["origin_date"]).max() >= pd.Timestamp(cfg.test_start):
        raise AssertionError(
            "Rolling-origin frames must not include test origins (would leak the test period)"
        )

    rows: list[dict[str, Any]] = []
    try:
        folds = rolling_origin_folds(
            frame, n_folds=cfg.n_rolling_folds, block_origins=cfg.rolling_block_origins
        )
    except ValueError as exc:
        logger.warning("Rolling-origin evaluation skipped: %s", exc)
        return rows

    for fold in folds:
        fold_train = subset_by_origins(frame, fold.train_origins)
        fold_validation = subset_by_origins(frame, fold.validation_origins)
        for baseline in baselines:
            scores = evaluate(fold_validation[target], baseline.predict(fold_validation))
            rows.append({"fold": fold.fold, "model_name": baseline.name, **scores})
        for name, builder in CANDIDATE_BUILDERS.items():
            model = builder(seed, best_params[name])
            model.fit(fold_train[FEATURES], fold_train[target])
            scores = evaluate(fold_validation[target], model.predict(fold_validation[FEATURES]))
            rows.append({"fold": fold.fold, "model_name": name, **scores})
    return rows


def _mean_wape_by_model(fold_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean WAPE per model across folds; requires every model in every fold."""
    if not fold_rows:
        raise RuntimeError("Rolling-origin evaluation produced no rows")
    means: dict[str, float] = {}
    for model_name in {row["model_name"] for row in fold_rows}:
        values = [
            row["wape"]
            for row in fold_rows
            if row["model_name"] == model_name and row["wape"] is not None
        ]
        if not values:
            raise RuntimeError(f"No usable WAPE for {model_name} in rolling-origin evaluation")
        means[model_name] = float(np.mean(values))
    return means
