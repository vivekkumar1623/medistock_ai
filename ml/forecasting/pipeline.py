"""Candidate model pipelines.

Every candidate is a single sklearn ``Pipeline`` that contains its own
preprocessing. That is the leakage defence for R4: imputation medians and
category vocabularies are learned inside ``pipeline.fit(X_train)`` only, so there
is no code path where a scaler or encoder can see validation/test data.

Trees are chosen deliberately:
* They capture non-linear interactions (e.g. "high volume AND zero-sales days")
  without hand-specifying them.
* They need no scaling, so the preprocessing is limited to median imputation and
  category encoding -- fewer places for a subtle bug.
* They handle the many zero-prevalence features produced by sparse demand.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Median imputation for numerics; frozen-vocabulary one-hot for categories."""
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median", add_indicator=False))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",  # unknown category -> all-zero row, never a crash
                    sparse_output=False,
                    dtype=float,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_random_forest(seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    # n_jobs=1 deliberately: parallel fitting spawns worker pools, which are both
    # forbidden in this sandbox and a source of non-determinism across machines.
    # At ~2k rows the single-threaded fit takes under a second.
    settings = {"n_estimators": 300, "min_samples_leaf": 2, "n_jobs": 1}
    settings.update(params or {})
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    random_state=seed,
                    max_features=0.6,
                    **settings,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting(seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    settings = {
        "learning_rate": 0.06,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 15,
        "l2_regularization": 0.0,
    }
    settings.update(params or {})
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=seed,
                    early_stopping=False,
                    **settings,
                ),
            ),
        ]
    )


CANDIDATE_BUILDERS: dict[str, Any] = {
    "random_forest": build_random_forest,
    "hist_gradient_boosting": build_hist_gradient_boosting,
}

# Small, defensible grids. Tuned on validation only; the test set is touched once.
CANDIDATE_GRIDS: dict[str, list[dict[str, Any]]] = {
    "random_forest": [
        {"max_depth": 12, "min_samples_leaf": 2},
        {"max_depth": None, "min_samples_leaf": 4},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.06, "max_leaf_nodes": 15},
        {"learning_rate": 0.10, "max_leaf_nodes": 31},
    ],
}
