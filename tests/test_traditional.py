from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths
from sentiment_benchmark.contracts import DatasetSplits
from sentiment_benchmark.traditional import train_tfidf_logistic


def _labelled_frame(negative: list[str], positive: list[str]) -> pd.DataFrame:
    rows = [(text, 0) for text in negative] + [(text, 1) for text in positive]
    return pd.DataFrame(rows, columns=["text", "label"])


def _small_splits(test_text: list[str] | None = None) -> DatasetSplits:
    train = _labelled_frame(
        negative=[
            "terrible product completely broken",
            "awful service never recommend",
            "poor quality and slow delivery",
            "hated this unpleasant experience",
            "horrible value very disappointed",
            "bad item failed expectations",
            "waste purchase angry customer",
            "dreadful performance hate it",
        ],
        positive=[
            "excellent product works perfectly",
            "wonderful service highly recommend",
            "great quality and fast delivery",
            "loved this pleasant experience",
            "amazing value very satisfied",
            "fantastic item exceeded expectations",
            "good purchase happy customer",
            "brilliant performance love it",
        ],
    )
    validation = _labelled_frame(
        negative=[
            "terrible quality and awful service",
            "bad broken item",
            "very disappointed and angry",
            "poor unpleasant purchase",
        ],
        positive=[
            "excellent quality and wonderful service",
            "good perfect item",
            "very satisfied and happy",
            "great pleasant purchase",
        ],
    )
    test = pd.DataFrame(
        {
            "text": test_text
            or [
                "excellent product and fast service",
                "terrible purchase and broken item",
                "wonderful experience",
                "awful disappointing experience",
            ]
        }
    )
    for split_name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.insert(0, "sentence_id", [f"{split_name}-{index}" for index in range(len(frame))])
    return DatasetSplits(train=train, validation=validation, test=test)


def test_training_returns_probabilities_and_persists_auditable_search(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    splits = _small_splits()
    config = ExperimentConfig(seed=17, latency_repeats=3, fast=True)

    result = train_tfidf_logistic(splits, paths, config)

    assert result.name == "tfidf_logistic_regression"
    assert result.family == "traditional"
    assert result.validation_probabilities.shape == (len(splits.validation),)
    assert result.test_probabilities.shape == (len(splits.test),)
    assert np.isfinite(result.validation_probabilities).all()
    assert np.isfinite(result.test_probabilities).all()
    assert np.logical_and(
        result.validation_probabilities >= 0, result.validation_probabilities <= 1
    ).all()
    assert np.logical_and(result.test_probabilities >= 0, result.test_probabilities <= 1).all()
    assert result.train_seconds > 0
    assert result.inference_seconds > 0
    assert result.inference_samples == len(splits.test) * config.latency_repeats
    assert result.parameter_count == result.trainable_parameter_count > 0
    assert result.artifact_path.is_file()
    assert result.artifact_bytes > 0

    search_path = paths.reports_dir / "traditional_validation_search.csv"
    search = pd.read_csv(search_path)
    assert search_path.is_file()
    assert len(search) == 9
    assert set(search["representation"]) == {"word_unigram", "word_bigram", "word_char"}
    assert set(search["C"]) == {0.5, 1.0, 2.0}
    assert search["selected"].sum() == 1
    assert set(search["trained_rows"]) == {"train_only"}
    selected = search.loc[search["selected"]].iloc[0]
    assert selected["validation_macro_f1"] == search["validation_macro_f1"].max()
    assert result.metadata["selected_candidate"] == selected["candidate"]
    assert result.metadata["selection_metric"] == "validation_macro_f1"
    assert result.metadata["trained_on"] == "train_only"

    artifact = joblib.load(result.artifact_path)
    assert artifact["selected_candidate"] == selected["candidate"]
    assert artifact["selection_metric"] == "validation_macro_f1"
    assert artifact["trained_on"] == "train_only"
    assert artifact["training_rows"] == len(splits.train)
    assert artifact["validation_rows"] == len(splits.validation)
    assert artifact["solver"] == "liblinear"
    assert artifact["max_iter"] == 2_000
    direct = artifact["pipeline"].predict_proba(splits.validation["text"])[:, 1]
    np.testing.assert_allclose(result.validation_probabilities, direct)


def test_candidate_selection_does_not_use_test_text_or_labels(tmp_path) -> None:
    first_splits = _small_splits(
        ["all excellent wonderful", "all terrible awful", "neutral unseen words"]
    )
    second_splits = _small_splits(["completely different document", "another unrelated sentence"])
    # Neither test frame has a label column: the route may score test text only after selection.
    assert "label" not in first_splits.test
    assert "label" not in second_splits.test
    config = ExperimentConfig(seed=23, latency_repeats=1, fast=True)
    first_paths = ProjectPaths(tmp_path / "first")
    second_paths = ProjectPaths(tmp_path / "second")

    first = train_tfidf_logistic(first_splits, first_paths, config)
    second = train_tfidf_logistic(second_splits, second_paths, config)

    assert first.metadata["selected_candidate"] == second.metadata["selected_candidate"]
    np.testing.assert_allclose(first.validation_probabilities, second.validation_probabilities)
    first_search = pd.read_csv(first_paths.reports_dir / "traditional_validation_search.csv")
    second_search = pd.read_csv(second_paths.reports_dir / "traditional_validation_search.csv")
    stable_columns = [
        "candidate_index",
        "candidate",
        "representation",
        "C",
        "validation_macro_f1",
        "feature_count",
        "trained_rows",
        "selected",
    ]
    pd.testing.assert_frame_equal(first_search[stable_columns], second_search[stable_columns])
