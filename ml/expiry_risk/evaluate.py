"""Expiry-risk classifier evaluation.

Positive class = ``at_risk`` (estimated unsold > 0). The business meaning of each
error type is asymmetric and worth stating, because it drives which metric to
optimise:

* **False positive** (flagged at risk, actually sold through): a wasted rotation
  or a premature discount. Cost: staff time, some margin.
* **False negative** (flagged safe, actually expires): the full cost of the
  expired units plus disposal handling. Much more expensive.

So recall matters more than precision for the critical tier. We report both and
also report precision/recall restricted to the ``critical`` severity band, which
is the actionable subset.
"""

from __future__ import annotations

import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion(predicted_at_risk: pd.Series, actual_at_risk: pd.Series) -> dict[str, int]:
    tp = int(((predicted_at_risk) & (actual_at_risk)).sum())
    fp = int(((predicted_at_risk) & (~actual_at_risk)).sum())
    tn = int(((~predicted_at_risk) & (~actual_at_risk)).sum())
    fn = int(((~predicted_at_risk) & (actual_at_risk)).sum())
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn}


def evaluate_risk(
    predicted: pd.DataFrame,
    realised: pd.DataFrame,
    threshold_rule: str,
    policy: dict,
) -> dict:
    """Join predictions to realised labels on evaluable batches and score them."""
    merged = predicted.merge(
        realised[
            [
                "batch_id",
                "label_known",
                "realised_unsold_units",
                "realised_at_risk",
                "observed_window_days",
            ]
        ],
        on="batch_id",
        how="inner",
        validate="one_to_one",
    )

    evaluable = merged[merged["label_known"]].copy()
    if evaluable.empty:
        raise RuntimeError(
            "No evaluable batches: every batch expires after the observed data window. "
            "Extend the dataset or shorten the assessment window."
        )

    matrix = confusion(evaluable["at_risk"], evaluable["realised_at_risk"])
    tp, fp, tn, fn = (
        matrix["true_positive"],
        matrix["false_positive"],
        matrix["true_negative"],
        matrix["false_negative"],
    )
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = _safe_ratio(tp + tn, tp + fp + tn + fn)

    critical = evaluable[evaluable["risk_level"] == "critical"]
    critical_matrix = confusion(critical["at_risk"], critical["realised_at_risk"])
    critical_precision = _safe_ratio(
        critical_matrix["true_positive"],
        critical_matrix["true_positive"] + critical_matrix["false_positive"],
    )
    critical_recall = _safe_ratio(
        critical_matrix["true_positive"],
        critical_matrix["true_positive"] + critical_matrix["false_negative"],
    )

    error_framing = {
        "false_positive_cost": (
            "Unnecessary rotation or markdown of stock that would have sold. "
            "Costs staff time and margin."
        ),
        "false_negative_cost": (
            "Expired stock that was never flagged: full unit cost plus disposal. "
            "The expensive error for a pharmacy."
        ),
        "asymmetry_note": (
            "Recall is prioritised for the critical band because a missed expiry "
            "costs more than an unnecessary rotation."
        ),
    }

    return {
        "positive_class": "at_risk",
        "threshold_rule": threshold_rule,
        "policy_version": policy.get("policy_version", "risk-v1"),
        "n_batches": int(len(evaluable)),
        "n_censored_excluded": int((~merged["label_known"]).sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "confusion": matrix,
        "base_rate": round(float(evaluable["realised_at_risk"].mean()), 4),
        "critical_band": {
            "n_batches": int(len(critical)),
            "precision": round(critical_precision, 4),
            "recall": round(critical_recall, 4),
            "confusion": critical_matrix,
        },
        "error_framing": error_framing,
        "evaluation_window": policy.get("evaluation_window", {}),
        "label_definition": policy.get("label_definition", ""),
    }
