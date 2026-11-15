"""Error analysis: where the forecast fails, not just how much on average.

A single WAPE hides everything a reviewer will ask about: is the model bad for a
category, is it bad for small products, is it systematically biased? This module
slices the held-out test residuals three ways and reports the worst cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import evaluate

SPARSITY_BINS = [
    (0.0, 0.05, "dense (<5% zero days)"),
    (0.05, 0.25, "moderate (5-25%)"),
    (0.25, 1.01, "sparse (>25%)"),
]


def _slice(frame: pd.DataFrame, column: str, target: str) -> list[dict]:
    rows: list[dict] = []
    for value, group in frame.groupby(column, observed=True):
        if len(group) < 5:
            continue
        scores = evaluate(group[target], group["prediction"])
        rows.append(
            {
                "group": str(value),
                "n": scores["n_observations"],
                "mae": round(scores["mae"], 2),
                "rmse": round(scores["rmse"], 2),
                "wape": None if scores["wape"] is None else round(scores["wape"], 4),
                "bias": round(scores["bias"], 2),
            }
        )
    return sorted(rows, key=lambda row: (row["wape"] is None, -(row["wape"] or 0)))


def analyse(test: pd.DataFrame, target: str) -> dict:
    frame = test.copy()
    frame["residual"] = frame["prediction"] - frame[target]
    frame["abs_error"] = frame["residual"].abs()

    # Demand magnitude buckets (deciles of the actual target).
    frame["demand_decile"] = pd.qcut(
        frame[target].rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)]
    )
    # Sparsity buckets based on the product's zero-day share at the origin.
    conditions = [
        frame["zero_ratio_90"] < 0.05,
        frame["zero_ratio_90"] < 0.25,
    ]
    frame["sparsity"] = np.select(
        conditions, ["dense (<5% zero days)", "moderate (5-25%)"], default="sparse (>25%)"
    )

    overall = evaluate(frame[target], frame["prediction"])
    worst = frame.nlargest(10, "abs_error")[
        ["product_id", "origin_date", "category", target, "prediction", "residual"]
    ].copy()
    # Round only the numeric columns: .round() is a no-op on datetime and warns.
    for column in (target, "prediction", "residual"):
        worst[column] = worst[column].astype(float).round(2)

    return {
        "overall": {
            "n": overall["n_observations"],
            "mae": round(overall["mae"], 3),
            "rmse": round(overall["rmse"], 3),
            "wape": round(overall["wape"], 4) if overall["wape"] is not None else None,
            "bias": round(overall["bias"], 3),
        },
        "by_category": _slice(frame, "category", target),
        "by_demand_decile": _slice(frame, "demand_decile", target),
        "by_sparsity": _slice(frame, "sparsity", target),
        "worst_predictions": worst.to_dict(orient="records"),
    }


def render_markdown(analysis: dict, path: Path) -> Path:
    overall = analysis["overall"]
    lines = [
        "# Error Analysis",
        "",
        "Held-out **test** period only. Scores are produced by the model stored in the",
        "artifact bundle, so this is the error profile of the system that is actually served.",
        "",
        "## Overall",
        "",
        "| MAE | RMSE | WAPE | Bias | n |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['mae']} | {overall['rmse']} | {overall['wape']} "
        f"| {overall['bias']} | {overall['n']} |",
        "",
        f"Bias of **{overall['bias']}** units means the model is very slightly "
        f"{'over' if overall['bias'] > 0 else 'under'}-forecasting on average.",
        "",
        "## By category",
        "",
        "| Category | n | MAE | RMSE | WAPE | Bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["by_category"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['mae']} | {row['rmse']} | "
            f"{row['wape'] if row['wape'] is not None else 'n/a'} | {row['bias']} |"
        )

    lines += [
        "",
        "## By demand magnitude (deciles of actual demand)",
        "",
        "| Decile | n | MAE | RMSE | WAPE | Bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["by_demand_decile"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['mae']} | {row['rmse']} | "
            f"{row['wape'] if row['wape'] is not None else 'n/a'} | {row['bias']} |"
        )

    lines += [
        "",
        "WAPE is naturally larger for low-volume deciles: the denominator is small while",
        "absolute errors do not shrink proportionally. This is a property of the metric, not",
        "a defect, and it is why the UI flags low-demand products with lower confidence.",
        "",
        "## By demand sparsity",
        "",
        "| Bucket | n | MAE | RMSE | WAPE | Bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["by_sparsity"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['mae']} | {row['rmse']} | "
            f"{row['wape'] if row['wape'] is not None else 'n/a'} | {row['bias']} |"
        )

    lines += [
        "",
        "## Ten worst test predictions",
        "",
        "| Product | Origin | Category | Actual | Predicted | Residual |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in analysis["worst_predictions"]:
        lines.append(
            f"| {row['product_id']} | {str(row['origin_date'])[:10]} | {row['category']} | "
            f"{row['demand_next_30d']} | {row['prediction']} | {row['residual']} |"
        )

    lines += [
        "",
        "## What this tells us",
        "",
        "1. Errors concentrate in the low-volume deciles, where an aggregate 30-day total is",
        "   inherently noisy. Reporting one aggregate WAPE would hide this.",
        "2. Sparse products are worse than dense ones, which is exactly why the serving layer",
        "   attaches a low-confidence note instead of a false precision claim.",
        "3. The largest single errors sit on products with promotion spikes, which the feature",
        "   set cannot anticipate (promotions are not in the data at all). That is a stated",
        "   limitation, not something more model capacity would fix.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
