from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths, SplitConfig
from sentiment_benchmark.contracts import DatasetSplits, ModelResult
from sentiment_benchmark.evaluation import evaluate_model_results
from sentiment_benchmark.reporting import (
    create_data_evidence,
    create_evaluation_plots,
    write_portfolio_documents,
    write_run_manifest,
)


def _frame(prefix: str, count: int = 12) -> pd.DataFrame:
    sources = ["amazon", "imdb", "yelp"] * 4
    return pd.DataFrame(
        {
            "sentence_id": [f"{prefix}-{index}" for index in range(count)],
            "source": sources,
            "text": [f"unique {prefix} sentence {index}" for index in range(count)],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_reporting_writes_machine_and_human_readable_evidence(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    paths.raw_archive.write_bytes(b"archive")
    paths.split_assignments.write_text("sentence_id,split\na,train\n", encoding="utf-8")
    splits = DatasetSplits(_frame("train"), _frame("val"), _frame("test"))
    frame = pd.concat((splits.train, splits.validation, splits.test), ignore_index=True)
    frame.to_csv(paths.prepared_data, index=False)
    artifact = paths.models_dir / "artifact.bin"
    artifact.write_bytes(b"weights")
    probabilities = np.array([0.1, 0.9] * 6)
    results = [
        ModelResult(
            name="tfidf_logistic_regression",
            family="traditional",
            validation_probabilities=probabilities,
            test_probabilities=probabilities,
            validation_ids=splits.validation["sentence_id"].to_numpy(),
            test_ids=splits.test["sentence_id"].to_numpy(),
            train_seconds=0.1,
            inference_seconds=0.01,
            inference_samples=12,
            parameter_count=12,
            trainable_parameter_count=12,
            artifact_path=artifact,
        ),
        ModelResult(
            name="pytorch_textcnn",
            family="deep_learning_from_scratch",
            validation_probabilities=probabilities,
            test_probabilities=probabilities,
            validation_ids=splits.validation["sentence_id"].to_numpy(),
            test_ids=splits.test["sentence_id"].to_numpy(),
            train_seconds=1.0,
            inference_seconds=0.02,
            inference_samples=12,
            parameter_count=120,
            trainable_parameter_count=120,
            artifact_path=artifact,
            best_epoch=1,
            history=pd.DataFrame({"epoch": [1], "validation_macro_f1": [1.0]}),
        ),
        ModelResult(
            name="distilbert",
            family="pretrained_transformer",
            validation_probabilities=probabilities,
            test_probabilities=probabilities,
            validation_ids=splits.validation["sentence_id"].to_numpy(),
            test_ids=splits.test["sentence_id"].to_numpy(),
            train_seconds=2.0,
            inference_seconds=0.03,
            inference_samples=12,
            parameter_count=1_200,
            trainable_parameter_count=1_200,
            artifact_path=artifact,
            best_epoch=1,
            history=pd.DataFrame({"epoch": [1], "validation_macro_f1": [1.0]}),
            metadata={"pretraining_compute_included": False},
        ),
    ]
    config = ExperimentConfig(fast=True)
    split_config = SplitConfig()
    evaluation = evaluate_model_results(results, splits, config, paths.reports_dir)

    create_data_evidence(frame, splits, paths)
    create_evaluation_plots(evaluation, results, splits, paths)
    write_portfolio_documents(frame, splits, evaluation, results, paths, config, split_config)
    manifest = write_run_manifest(frame, splits, evaluation, paths, config, split_config)
    assert manifest["dataset"]["prepared_canonical_sha256"]
    assert manifest["dataset"]["prepared_file_sha256"]

    assert manifest["selection"]["test_excluded_from_selection"] is True
    assert json.loads((paths.reports_dir / "run_manifest.json").read_text()) == manifest
    assert (paths.reports_dir / "EXPERIMENT_REPORT.md").is_file()
    assert (paths.reports_dir / "DATA_CARD.md").is_file()
    assert (paths.reports_dir / "MODEL_CARDS.md").is_file()
    assert (paths.figures_dir / "performance_comparison.png").is_file()
    assert (paths.figures_dir / "performance_cost_frontier.png").is_file()
