from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_benchmark.config import ProjectPaths
from sentiment_benchmark.predict import predict_frame


def _write_tfidf_artifact(paths: ProjectPaths) -> None:
    paths.ensure_directories()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]
    ).fit(["good product", "great service", "bad product", "awful service"], [1, 1, 0, 0])
    joblib.dump(
        {"model_name": "tfidf_logistic_regression", "pipeline": pipeline},
        paths.models_dir / "tfidf_logistic.joblib",
    )


def test_tfidf_batch_prediction_uses_artifact_and_omits_raw_text(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    _write_tfidf_artifact(paths)
    frame = pd.DataFrame({"text": ["great product", "awful product"]})

    output = predict_frame(frame, "tfidf_logistic_regression", paths)

    assert list(output.columns) == [
        "sentence_id",
        "positive_probability",
        "predicted_label",
        "predicted_sentiment",
        "model",
    ]
    assert output["sentence_id"].tolist() == ["input_000000", "input_000001"]
    assert output["positive_probability"].between(0, 1).all()
    assert "text" not in output


def test_prediction_validation_fails_closed(tmp_path) -> None:
    paths = ProjectPaths(tmp_path)
    _write_tfidf_artifact(paths)

    with pytest.raises(ValueError, match="non-missing and non-blank"):
        predict_frame(pd.DataFrame({"text": [""]}), "tfidf_logistic_regression", paths)
    with pytest.raises(ValueError, match="unknown model"):
        predict_frame(pd.DataFrame({"text": ["fine"]}), "unknown", paths)
