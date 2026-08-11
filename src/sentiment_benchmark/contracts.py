"""Small typed contracts shared by independent model implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    """Frames assigned once and reused unchanged by every model family."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelResult:
    """Comparable predictions and resource evidence from one fitted model."""

    name: str
    family: str
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray
    train_seconds: float
    inference_seconds: float
    inference_samples: int
    parameter_count: int
    trainable_parameter_count: int
    artifact_path: Path
    best_epoch: int | None = None
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def artifact_bytes(self) -> int:
        if self.artifact_path.is_file():
            return self.artifact_path.stat().st_size
        if self.artifact_path.is_dir():
            return sum(
                path.stat().st_size for path in self.artifact_path.rglob("*") if path.is_file()
            )
        return 0

    @property
    def inference_ms_per_sample(self) -> float:
        if self.inference_samples <= 0:
            return 0.0
        return 1_000 * self.inference_seconds / self.inference_samples
