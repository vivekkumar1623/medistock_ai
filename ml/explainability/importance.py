"""Global feature importance.

Method: permutation importance on the **test** set, scored by MAE.

Why permutation rather than impurity importance: impurity-based importance is
biased toward high-cardinality and high-variance features and is computed on the
training data, so it can rank a leaky or overfit feature highly. Permutation
measures the actual increase in error when a feature's values are shuffled, which
answers the question a reviewer asks: "what happens to error if this signal is
destroyed?"

Cost: ``n_features x n_repeats`` extra predictions. At ~30 features and 480 test
rows this is trivial, which is why we can afford the honest method.

This is **global** importance. Per-prediction attributions are computed
separately (``ml/explainability/drivers.py``) and must never be presented as the
same thing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_summary(
    model: Any,
    features: pd.DataFrame,
    target: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_repeats: int = 5,
    top_n: int = 15,
) -> dict:
    """Return the top-``top_n`` features by mean permutation importance."""
    result = permutation_importance(
        model,
        features[feature_names],
        target,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    rows = [
        {
            "feature": name,
            "importance": float(mean),
            "importance_std": float(std),
        }
        for name, mean, std in zip(
            feature_names, result.importances_mean, result.importances_std, strict=True
        )
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    rows = rows[:top_n]

    total = sum(max(row["importance"], 0.0) for row in rows) or 1.0
    for row in rows:
        row["share"] = round(max(row["importance"], 0.0) / total, 4)

    return {
        "method": "permutation_importance_test_set",
        "scoring": "neg_mean_absolute_error",
        "n_repeats": n_repeats,
        "n_features": len(feature_names),
        "features": rows,
    }
