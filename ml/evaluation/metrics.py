"""Forecast evaluation metrics.

Why these three:
* **MAE** — average absolute error in *units*, the unit a pharmacy manager
  reasons in ("we will be about 18 packs out").
* **RMSE** — punishes large misses more than MAE. A model that is usually close
  but occasionally wildly wrong is dangerous for expiry decisions, and RMSE
  surfaces that.
* **WAPE** — ``sum(|error|) / sum(|actual|)``, a volume-weighted relative error.
  Chosen over MAPE because this dataset contains genuine zero demand, where MAPE
  is undefined; WAPE only needs the *aggregate* denominator to be non-zero.

WAPE is weighted by volume, so high-selling products dominate it. That is a
deliberate choice for an inventory decision: being 20 units wrong on a product
selling 2000 units matters less than being 20 wrong on one selling 40, and the
per-product breakdown in the error analysis covers the latter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray):
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("Cannot evaluate on an empty array")
    return true, pred


def mae(y_true, y_pred) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def wape(y_true, y_pred) -> float | None:
    """Volume-weighted absolute percentage error.

    Returns ``None`` when every actual is zero: the metric is undefined there, and
    returning 0.0 would silently look like a perfect model.
    """
    true, pred = _as_arrays(y_true, y_pred)
    denominator = float(np.sum(np.abs(true)))
    if denominator == 0.0:
        return None
    return float(np.sum(np.abs(true - pred)) / denominator)


def bias(y_true, y_pred) -> float:
    """Mean signed error (pred - actual): positive means systematic over-forecast."""
    true, pred = _as_arrays(y_true, y_pred)
    return float(np.mean(pred - true))


def evaluate(y_true, y_pred) -> dict[str, float | None]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n_observations": int(len(np.asarray(y_true))),
    }


def compare_on_common_rows(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], target: str
) -> list[dict]:
    """Evaluate several models on identical rows.

    Fairness rule: all candidates are scored on the same target vector. Any model
    missing a prediction for a row invalidates the comparison, so we assert rather
    than silently drop rows (dropping rows is how comparisons get flattered).
    """
    y_true = frame[target].to_numpy(dtype=float)
    rows: list[dict] = []
    for name, pred in predictions.items():
        if len(pred) != len(frame):
            raise ValueError(
                f"{name}: {len(pred)} predictions for {len(frame)} rows; "
                "all candidates must be scored on the same rows"
            )
        scores = evaluate(y_true, pred)
        rows.append({"model_name": name, **scores})
    return rows
