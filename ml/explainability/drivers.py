"""Per-prediction explanations by one-at-a-time ablation.

Method: for the prediction in question, replace **one** feature value with a
population reference value (median for numeric, mode for categorical), re-predict,
and record the change. The contribution answers a precise, checkable question:

    "If this product's 30-day demand history were at the typical level across all
     products, how would the forecast for *this* product move?"

Why this rather than SHAP:
* It requires no extra dependency and no background-data sampling design.
* It is exact for the model (each number is a real model output difference), not an
  approximation of a Shapley value.
* Its limitation is also clear and disclosable: contributions are **not additive**
  and they are measured **relative to the population reference**, not relative to a
  model expectation. We label them accordingly and never call them SHAP values.

What we refuse to do: invent a "contribution" for a feature the model cannot react
to. If ablation moves the prediction by 0, the driver is not reported as
influential just because a global importance chart ranked it highly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METHOD = "one_at_a_time_ablation_vs_population_reference"


def _reference_row(reference_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    reference: dict[str, Any] = {}
    for name in feature_names:
        column = reference_frame[name]
        if pd.api.types.is_numeric_dtype(column):
            reference[name] = float(column.median())
        else:
            modes = column.mode(dropna=True)
            reference[name] = modes.iloc[0] if not modes.empty else None
    return pd.DataFrame([reference])


def explain_prediction(
    model,
    feature_row: pd.DataFrame,
    reference_frame: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-``top_k`` features by absolute ablation effect."""
    if feature_row.empty:
        return []

    base_features = feature_row[feature_names].reset_index(drop=True)
    reference = _reference_row(reference_frame, feature_names)

    try:
        base_prediction = float(np.asarray(model.predict(base_features))[0])
    except Exception:  # pragma: no cover - model contract broken
        return []

    contributions: list[dict[str, Any]] = []
    for name in feature_names:
        if base_features[name].iloc[0] is None:
            continue
        ablated = base_features.copy()
        ablated[name] = reference[name].iloc[0]
        try:
            ablated_prediction = float(np.asarray(model.predict(ablated))[0])
        except Exception:  # pragma: no cover
            continue
        delta = base_prediction - ablated_prediction
        if abs(delta) < 1e-9:
            continue
        contributions.append(
            {
                "feature": name,
                "value": _as_float(base_features[name].iloc[0]),
                "reference_value": _as_float(reference[name].iloc[0]),
                "contribution": round(delta, 4),
                "direction": "increases_demand" if delta > 0 else "decreases_demand",
                "method": METHOD,
                "explanation": (
                    f"For this product, {name}={_format_value(base_features[name].iloc[0])} "
                    f"versus a population reference of {_format_value(reference[name].iloc[0])}; "
                    f"holding everything else fixed, setting {name} to the reference moves the "
                    f"forecast by {-delta:+.1f} units."
                ),
            }
        )

    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return contributions[:top_k]


def _as_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:,.2f}"
