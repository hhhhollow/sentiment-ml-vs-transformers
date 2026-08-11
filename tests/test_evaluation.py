from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentiment_benchmark.config import ExperimentConfig
from sentiment_benchmark.contracts import DatasetSplits, ModelResult
from sentiment_benchmark.evaluation import (
    evaluate_model_results,
    expected_calibration_error,
    probability_metrics,
)


def test_probability_metrics_and_calibration_are_exact() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = probability_metrics(y, probability)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert expected_calibration_error(y, probability, bins=2) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="finite values"):
        probability_metrics(y, np.array([0.1, np.nan, 0.8, 0.9]))


def _split(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": [f"{prefix}-{index}" for index in range(8)],
            "text": [f"text {index}" for index in range(8)],
            "label": [0, 1] * 4,
            "source": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )


def test_common_evaluation_selects_on_validation_and_writes_strict_evidence(tmp_path) -> None:
    splits = DatasetSplits(_split("train"), _split("val"), _split("test"))
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    first = ModelResult(
        name="tfidf_logistic_regression",
        family="traditional",
        validation_probabilities=np.array([0.1, 0.9] * 4),
        test_probabilities=np.array([0.1, 0.9] * 4),
        validation_ids=splits.validation["sentence_id"].to_numpy(),
        test_ids=splits.test["sentence_id"].to_numpy(),
        train_seconds=0.1,
        inference_seconds=0.01,
        inference_samples=8,
        parameter_count=10,
        trainable_parameter_count=10,
        artifact_path=artifact,
    )
    second = ModelResult(
        name="pytorch_textcnn",
        family="deep_learning_from_scratch",
        validation_probabilities=np.array([0.4, 0.6] * 4),
        test_probabilities=np.array([0.9, 0.1] * 4),
        validation_ids=splits.validation["sentence_id"].to_numpy(),
        test_ids=splits.test["sentence_id"].to_numpy(),
        train_seconds=1.0,
        inference_seconds=0.02,
        inference_samples=8,
        parameter_count=100,
        trainable_parameter_count=100,
        artifact_path=artifact,
    )

    output = evaluate_model_results(
        [first, second],
        splits,
        ExperimentConfig(fast=True),
        tmp_path / "reports",
    )

    assert output["champion_model"] == "pytorch_textcnn"
    comparison = output["comparison"].set_index("model")
    assert comparison.loc["pytorch_textcnn", "test_macro_f1"] == pytest.approx(0.0)
    assert (tmp_path / "reports" / "paired_deltas_vs_tfidf.csv").is_file()
    raw = (tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["champion_model"] == "pytorch_textcnn"

    second.test_ids = second.test_ids[::-1]
    with pytest.raises(ValueError, match="not aligned to split IDs"):
        evaluate_model_results(
            [first, second],
            splits,
            ExperimentConfig(fast=True),
            tmp_path / "misaligned-reports",
        )
