"""Data generation package."""

from __future__ import annotations

from ml.data_generation.config import GeneratorConfig
from ml.data_generation.generate import GeneratedDataset, generate_dataset, write_dataset

__all__ = ["GeneratedDataset", "GeneratorConfig", "generate_dataset", "write_dataset"]
