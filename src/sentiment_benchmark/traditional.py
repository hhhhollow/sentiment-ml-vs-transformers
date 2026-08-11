"""TF-IDF plus logistic-regression baseline selected on validation data only."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from sentiment_benchmark.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    ExperimentConfig,
    ProjectPaths,
)
from sentiment_benchmark.contracts import DatasetSplits, ModelResult

MODEL_NAME = "tfidf_logistic_regression"
ARTIFACT_FILENAME = "tfidf_logistic.joblib"
SEARCH_FILENAME = "traditional_validation_search.csv"


@dataclass(frozen=True)
class _Candidate:
    """A deterministic preprocessing and regularisation configuration."""

    name: str
    representation: str
    c: float


def _candidate_specs() -> tuple[_Candidate, ...]:
    """Return a small, explicit search space shared by every run."""
    return tuple(
        _Candidate(name=f"{representation}_c{c:g}", representation=representation, c=c)
        for representation in ("word_unigram", "word_bigram", "word_char")
        for c in (0.5, 1.0, 2.0)
    )


def _word_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )


def _build_pipeline(candidate: _Candidate, seed: int) -> Pipeline:
    if candidate.representation == "word_unigram":
        features: Any = _word_vectorizer((1, 1))
    elif candidate.representation == "word_bigram":
        features = _word_vectorizer((1, 2))
    elif candidate.representation == "word_char":
        features = FeatureUnion(
            transformer_list=[
                ("word", _word_vectorizer((1, 2))),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    else:  # pragma: no cover - candidate definitions are private and exhaustive.
        raise ValueError(f"unknown representation: {candidate.representation}")

    return Pipeline(
        steps=[
            ("tfidf", features),
            (
                "classifier",
                LogisticRegression(
                    C=candidate.c,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _validated_text(frame: pd.DataFrame, split_name: str) -> pd.Series:
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(f"{split_name} split is missing required column: {TEXT_COLUMN}")
    if frame.empty:
        raise ValueError(f"{split_name} split is empty")
    if frame[TEXT_COLUMN].isna().any():
        raise ValueError(f"{split_name} split contains missing text")
    text = frame[TEXT_COLUMN].astype(str)
    if text.str.strip().eq("").any():
        raise ValueError(f"{split_name} split contains blank text")
    return text


def _validated_labels(frame: pd.DataFrame, split_name: str) -> pd.Series:
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{split_name} split is missing required column: {TARGET_COLUMN}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{split_name} split contains missing labels")
    labels = frame[TARGET_COLUMN].astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} labels must be binary 0/1")
    return labels


def _positive_probabilities(pipeline: Pipeline, text: pd.Series) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    positive_indexes = np.flatnonzero(classes == 1)
    if len(positive_indexes) != 1:
        raise RuntimeError("fitted classifier does not contain the positive class label 1")
    return np.asarray(pipeline.predict_proba(text)[:, int(positive_indexes[0])], dtype=float)


def train_tfidf_logistic(
    splits: DatasetSplits,
    paths: ProjectPaths,
    config: ExperimentConfig,
) -> ModelResult:
    """Fit candidates on train, select by validation macro-F1, and score test once selected.

    The winning estimator is deliberately not refitted on train plus validation. This keeps
    its training rows identical to the neural and transformer routes that use validation for
    model selection or early stopping.
    """
    paths.ensure_directories()
    train_text = _validated_text(splits.train, "train")
    validation_text = _validated_text(splits.validation, "validation")
    test_text = _validated_text(splits.test, "test")
    train_labels = _validated_labels(splits.train, "train")
    validation_labels = _validated_labels(splits.validation, "validation")
    if train_labels.nunique() != 2:
        raise ValueError("train split must contain both classes")
    if config.latency_repeats < 1:
        raise ValueError("latency_repeats must be at least 1")

    candidate_rows: list[dict[str, Any]] = []
    fitted_candidates: list[Pipeline] = []
    training_started = time.perf_counter()

    for candidate_index, candidate in enumerate(_candidate_specs()):
        pipeline = _build_pipeline(candidate, config.seed)
        fit_started = time.perf_counter()
        pipeline.fit(train_text, train_labels)
        fit_seconds = time.perf_counter() - fit_started
        validation_prediction = pipeline.predict(validation_text)
        validation_macro_f1 = float(
            f1_score(validation_labels, validation_prediction, average="macro", zero_division=0)
        )
        feature_count = int(pipeline.named_steps["tfidf"].transform(train_text.iloc[:1]).shape[1])
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate.name,
                "representation": candidate.representation,
                "C": candidate.c,
                "validation_macro_f1": validation_macro_f1,
                "feature_count": feature_count,
                "fit_seconds": float(fit_seconds),
                "trained_rows": "train_only",
            }
        )
        fitted_candidates.append(pipeline)

    best_index = max(
        range(len(candidate_rows)),
        key=lambda index: (candidate_rows[index]["validation_macro_f1"], -index),
    )
    best_pipeline = fitted_candidates[best_index]
    candidate_rows[best_index]["selected"] = True
    for index, row in enumerate(candidate_rows):
        if index != best_index:
            row["selected"] = False

    search = pd.DataFrame(candidate_rows)
    search_path = paths.reports_dir / SEARCH_FILENAME
    search.to_csv(search_path, index=False)

    classifier = best_pipeline.named_steps["classifier"]
    trainable_parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
    selected = candidate_rows[best_index]
    artifact_path = paths.models_dir / ARTIFACT_FILENAME
    search_report = str(search_path.relative_to(paths.root))
    artifact = {
        "pipeline": best_pipeline,
        "model_name": MODEL_NAME,
        "family": "traditional",
        "selected_candidate": selected["candidate"],
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": selected["validation_macro_f1"],
        "trained_on": "train_only",
        "training_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "classes": classifier.classes_.tolist(),
        "feature_count": selected["feature_count"],
        "trainable_parameter_count": trainable_parameter_count,
        "seed": config.seed,
        "solver": classifier.solver,
        "max_iter": classifier.max_iter,
        "search_report": search_report,
    }
    joblib.dump(artifact, artifact_path)
    train_seconds = time.perf_counter() - training_started

    validation_probabilities = _positive_probabilities(best_pipeline, validation_text)

    # Warm caches before timing; this prediction is intentionally not included in latency.
    _positive_probabilities(best_pipeline, test_text)
    inference_seconds = 0.0
    test_probabilities: np.ndarray | None = None
    for _ in range(config.latency_repeats):
        inference_started = time.perf_counter()
        test_probabilities = _positive_probabilities(best_pipeline, test_text)
        inference_seconds += time.perf_counter() - inference_started
    if test_probabilities is None:  # guarded by the latency_repeats validation above.
        raise RuntimeError("test inference did not run")

    return ModelResult(
        name=MODEL_NAME,
        family="traditional",
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        validation_ids=splits.validation[ID_COLUMN].to_numpy(copy=True),
        test_ids=splits.test[ID_COLUMN].to_numpy(copy=True),
        train_seconds=float(train_seconds),
        inference_seconds=float(inference_seconds),
        inference_samples=len(splits.test) * config.latency_repeats,
        parameter_count=trainable_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        artifact_path=artifact_path,
        metadata={
            "selected_candidate": selected["candidate"],
            "selection_metric": "validation_macro_f1",
            "validation_macro_f1": selected["validation_macro_f1"],
            "trained_on": "train_only",
            "feature_count": selected["feature_count"],
            "candidate_count": len(candidate_rows),
            "search_report": search_report,
            "latency_repeats": config.latency_repeats,
            "training_cost_scope": "representation_setup_through_selected_artifact",
        },
    )
