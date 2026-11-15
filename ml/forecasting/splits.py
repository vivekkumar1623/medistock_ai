"""Chronological splitting and rolling-origin validation.

Never a random split. A random split lets the model memorise the same demand
regime in train and test, which inflates every metric and teaches nothing about
future performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitSummary:
    train_origins: list[str]
    validation_origins: list[str]
    test_origins: list[str]
    train_rows: int
    validation_rows: int
    test_rows: int
    target_column: str

    def to_dict(self) -> dict:
        return {
            "train_origins": self.train_origins,
            "validation_origins": self.validation_origins,
            "test_origins": self.test_origins,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "target_column": self.target_column,
        }


def chronological_split(
    frame: pd.DataFrame,
    target: str,
    validation_start: str,
    test_start: str,
    origin_column: str = "origin_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitSummary]:
    """Split by origin date: train < validation_start <= validation < test_start <= test."""
    origins = pd.to_datetime(frame[origin_column])
    validation_start_ts = pd.Timestamp(validation_start)
    test_start_ts = pd.Timestamp(test_start)

    train = frame[origins < validation_start_ts]
    validation = frame[(origins >= validation_start_ts) & (origins < test_start_ts)]
    test = frame[origins >= test_start_ts]

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty partition: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    if train[origin_column].max() >= validation[origin_column].min():
        raise AssertionError("Train/validation origins overlap: leakage risk")
    if validation[origin_column].max() >= test[origin_column].min():
        raise AssertionError("Validation/test origins overlap: leakage risk")

    summary = SplitSummary(
        train_origins=sorted({str(d.date()) for d in train[origin_column].unique()}),
        validation_origins=sorted({str(d.date()) for d in validation[origin_column].unique()}),
        test_origins=sorted({str(d.date()) for d in test[origin_column].unique()}),
        train_rows=int(len(train)),
        validation_rows=int(len(validation)),
        test_rows=int(len(test)),
        target_column=target,
    )
    return train, validation, test, summary


@dataclass(frozen=True)
class RollingFold:
    fold: int
    train_origins: list[pd.Timestamp]
    validation_origins: list[pd.Timestamp]


def rolling_origin_folds(
    frame: pd.DataFrame,
    n_folds: int = 3,
    block_origins: int = 2,
    origin_column: str = "origin_date",
) -> list[RollingFold]:
    """Expanding-window folds: always train on the past, validate on the next block.

    Used to check that the single-window ranking is not an artefact of one lucky
    period. Origins are ordered and never shuffled.
    """
    origins = sorted(pd.to_datetime(frame[origin_column].unique()))
    if len(origins) < n_folds * block_origins + 2:
        raise ValueError(
            f"Need at least {n_folds * block_origins + 2} distinct origins, got {len(origins)}"
        )

    folds: list[RollingFold] = []
    for fold_index in range(n_folds):
        block_end = len(origins) - (n_folds - 1 - fold_index) * block_origins
        block_start = block_end - block_origins
        validation_origins = origins[block_start:block_end]
        train_origins = origins[:block_start]
        if train_origins:
            folds.append(
                RollingFold(
                    fold=fold_index + 1,
                    train_origins=list(train_origins),
                    validation_origins=list(validation_origins),
                )
            )
    return folds


def subset_by_origins(
    frame: pd.DataFrame, origins: list[pd.Timestamp], origin_column: str = "origin_date"
) -> pd.DataFrame:
    return frame[pd.to_datetime(frame[origin_column]).isin(origins)]
