from __future__ import annotations

import json

import numpy as np
import pandas as pd
from safetensors.torch import load_file

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths
from sentiment_benchmark.contracts import DatasetSplits
from sentiment_benchmark.predict import predict_frame
from sentiment_benchmark.textcnn import train_textcnn


def _balanced_splits() -> DatasetSplits:
    positive_train = [
        "I loved this warm delightful story",
        "A brilliant and charming little movie",
        "Excellent service and friendly staff",
        "The product works perfectly and feels sturdy",
        "Wonderful acting with a satisfying ending",
        "Fast delivery and fantastic quality",
        "This was fun uplifting and memorable",
        "A smart script with excellent performances",
    ]
    negative_train = [
        "I hated this dull disappointing story",
        "A terrible and boring little movie",
        "Awful service and rude staff",
        "The product broke immediately and feels flimsy",
        "Poor acting with a frustrating ending",
        "Slow delivery and dreadful quality",
        "This was tedious depressing and forgettable",
        "A weak script with terrible performances",
    ]
    train = pd.DataFrame(
        {
            "text": positive_train + negative_train,
            "label": [1] * len(positive_train) + [0] * len(negative_train),
        }
    )
    validation = pd.DataFrame(
        {
            "text": [
                "validationonly pleasant and enjoyable",
                "validationonly excellent and touching",
                "validationonly grim and boring",
                "validationonly awful and irritating",
            ],
            "label": [1, 1, 0, 0],
        }
    )
    test = pd.DataFrame(
        {
            "text": [
                "testonly delightful quality",
                "testonly smart and charming",
                "testonly disappointing quality",
                "testonly weak and tedious",
            ],
            "label": [1, 1, 0, 0],
        }
    )
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.insert(0, "sentence_id", [f"{split_name}-{index}" for index in range(len(frame))])
    return DatasetSplits(train=train, validation=validation, test=test)


def _fast_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=17,
        textcnn_epochs=5,
        textcnn_patience=1,
        textcnn_batch_size=4,
        textcnn_learning_rate=2e-3,
        textcnn_max_tokens=16,
        textcnn_vocab_size=100,
        latency_repeats=2,
        fast=True,
    )


def test_train_textcnn_is_reproducible_and_persists_auditable_artifacts(tmp_path) -> None:
    splits = _balanced_splits()
    config = _fast_config()
    first_paths = ProjectPaths(tmp_path / "first")
    second_paths = ProjectPaths(tmp_path / "second")

    first = train_textcnn(splits, first_paths, config, device="cpu")
    second = train_textcnn(splits, second_paths, config, device="cpu")

    assert first.name == "pytorch_textcnn"
    assert first.family == "deep_learning_from_scratch"
    assert first.validation_probabilities.shape == (len(splits.validation),)
    assert first.test_probabilities.shape == (len(splits.test),)
    assert np.isfinite(first.validation_probabilities).all()
    assert np.isfinite(first.test_probabilities).all()
    assert ((0.0 <= first.test_probabilities) & (first.test_probabilities <= 1.0)).all()
    np.testing.assert_allclose(
        first.validation_probabilities,
        second.validation_probabilities,
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        first.test_probabilities,
        second.test_probabilities,
        rtol=0,
        atol=1e-7,
    )

    assert first.parameter_count > 0
    assert first.trainable_parameter_count == first.parameter_count
    assert first.inference_seconds >= 0
    assert first.inference_samples == len(splits.test) * config.latency_repeats
    assert 1 <= first.best_epoch <= config.effective_textcnn_epochs
    assert first.artifact_path == first_paths.models_dir / "textcnn"
    assert first.artifact_bytes > 0

    weights_path = first.artifact_path / "model.safetensors"
    vocab_path = first.artifact_path / "vocab.json"
    model_config_path = first.artifact_path / "config.json"
    history_path = first_paths.reports_dir / "textcnn_history.csv"
    assert weights_path.is_file()
    assert vocab_path.is_file()
    assert model_config_path.is_file()
    assert history_path.is_file()
    assert load_file(str(weights_path))

    vocabulary_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocabulary = vocabulary_payload["token_to_index"]
    assert vocabulary_payload["fit_split"] == "train"
    assert vocabulary["<PAD>"] == 0
    assert vocabulary["<UNK>"] == 1
    assert "validationonly" not in vocabulary
    assert "testonly" not in vocabulary

    saved_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    assert saved_config["model_name"] == "pytorch_textcnn"
    assert saved_config["tokenizer"]["fit_split"] == "train"
    assert saved_config["training"]["selection_metric"] == "validation_macro_f1"
    assert saved_config["runtime"]["resolved_device"] == "cpu"
    assert saved_config["files"]["weights"] == "model.safetensors"

    expected_history_columns = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_macro_f1",
        "is_best",
    ]
    assert first.history.columns.tolist() == expected_history_columns
    saved_history = pd.read_csv(history_path)
    assert saved_history.columns.tolist() == expected_history_columns
    assert len(saved_history) == len(first.history)
    assert saved_history["is_best"].any()

    artifact_predictions = predict_frame(
        splits.test[["sentence_id", "text"]],
        "pytorch_textcnn",
        first_paths,
        device="cpu",
    )
    np.testing.assert_allclose(
        artifact_predictions["positive_probability"],
        first.test_probabilities,
        rtol=0,
        atol=1e-7,
    )
