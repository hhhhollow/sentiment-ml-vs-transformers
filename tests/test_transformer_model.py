from __future__ import annotations

import pandas as pd
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths
from sentiment_benchmark.contracts import DatasetSplits
from sentiment_benchmark.predict import predict_frame
from sentiment_benchmark.transformer_model import train_transformer


def _frame(prefix: str, rows: int = 8) -> pd.DataFrame:
    texts = [
        "good product",
        "bad product",
        "excellent service",
        "awful service",
        "great movie",
        "terrible movie",
        "love this",
        "hate this",
    ][:rows]
    return pd.DataFrame(
        {
            "sentence_id": [f"{prefix}-{index}" for index in range(rows)],
            "text": texts,
            "label": [1, 0, 1, 0, 1, 0, 1, 0][:rows],
            "source": ["fixture"] * rows,
        }
    )


def _tiny_local_distilbert(path) -> None:
    path.mkdir(parents=True)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "good",
        "bad",
        "excellent",
        "awful",
        "service",
        "product",
        "great",
        "terrible",
        "movie",
        "love",
        "hate",
        "this",
    ]
    (path / "vocab.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(path)
    model = DistilBertForSequenceClassification(
        DistilBertConfig(
            vocab_size=len(vocabulary),
            max_position_embeddings=128,
            n_layers=1,
            n_heads=2,
            dim=24,
            hidden_dim=48,
            num_labels=2,
        )
    )
    model.save_pretrained(path, safe_serialization=True)


def test_tiny_local_transformer_runs_without_network(tmp_path) -> None:
    base_model = tmp_path / "tiny-base"
    _tiny_local_distilbert(base_model)
    paths = ProjectPaths(tmp_path / "run")
    splits = DatasetSplits(_frame("train"), _frame("validation"), _frame("test"))
    config = ExperimentConfig(
        fast=True,
        transformer_batch_size=4,
        transformer_max_tokens=16,
        latency_repeats=1,
    )

    result = train_transformer(
        splits,
        paths,
        config,
        device="cpu",
        model_name=str(base_model),
        revision=None,
        local_files_only=True,
    )

    assert result.name == "distilbert"
    assert result.family == "pretrained_transformer"
    assert result.validation_probabilities.shape == (len(splits.validation),)
    assert result.test_probabilities.shape == (len(splits.test),)
    assert ((0 <= result.test_probabilities) & (result.test_probabilities <= 1)).all()
    assert result.parameter_count == result.trainable_parameter_count > 0
    assert result.artifact_path.is_dir()
    assert (result.artifact_path / "model.safetensors").is_file()
    assert (result.artifact_path / "benchmark_metadata.json").is_file()
    assert (paths.reports_dir / "distilbert_history.csv").is_file()
    assert result.best_epoch == int(
        result.history.loc[result.history["validation_macro_f1"].idxmax(), "epoch"]
    )

    artifact_predictions = predict_frame(
        splits.test[["sentence_id", "text"]],
        "distilbert",
        paths,
        device="cpu",
    )
    pd.testing.assert_series_equal(
        artifact_predictions["positive_probability"],
        pd.Series(result.test_probabilities, name="positive_probability"),
        check_dtype=False,
        check_exact=False,
        atol=1e-7,
        rtol=0,
    )
