"""Paired performance, uncertainty, domain, and cost evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentiment_benchmark.config import (
    ID_COLUMN,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    ExperimentConfig,
)
from sentiment_benchmark.contracts import DatasetSplits, ModelResult

DECISION_THRESHOLD = 0.5


def dump_json(payload: object, path: Path) -> None:
    """Write interoperable JSON and reject NaN/Infinity rather than hiding them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("labels and probabilities must have equal non-zero length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def probability_metrics(
    y_true: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float | int]:
    """Compute ranking, calibration, and threshold metrics from positive probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not len(y):
        raise ValueError("labels and probabilities must have equal non-zero length")
    if set(np.unique(y)).difference({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("labels must contain both binary classes")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    predicted = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_precision": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack((1 - p, p)), labels=[0, 1])),
        "ece_10_bin": float(expected_calibration_error(y, p, bins=10)),
        "predicted_positive_rate": float(predicted.mean()),
        "correct": int((predicted == y).sum()),
    }


def _bootstrap_interval(
    y: np.ndarray,
    probability: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        indexes = rng.integers(0, len(y), size=len(y))
        if np.unique(y[indexes]).size < 2:
            continue
        values.append(metric(y[indexes], probability[indexes]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _macro_f1(y: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(y, probability >= DECISION_THRESHOLD, average="macro"))


def _accuracy(y: np.ndarray, probability: np.ndarray) -> float:
    return float(accuracy_score(y, probability >= DECISION_THRESHOLD))


def _paired_delta_interval(
    y: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    point = metric(y, candidate) - metric(y, reference)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        indexes = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indexes]
        if np.unique(sampled_y).size < 2:
            continue
        deltas.append(metric(sampled_y, candidate[indexes]) - metric(sampled_y, reference[indexes]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(low), float(high)


def evaluate_model_results(
    results: list[ModelResult],
    splits: DatasetSplits,
    config: ExperimentConfig,
    reports_dir: Path,
) -> dict[str, Any]:
    """Persist the common test evaluation after validation-only model selection."""
    if len({result.name for result in results}) != len(results):
        raise ValueError("model names must be unique")
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_y = splits.validation[TARGET_COLUMN].to_numpy(dtype=int)
    test_y = splits.test[TARGET_COLUMN].to_numpy(dtype=int)
    iterations = config.effective_bootstrap_iterations
    rows: list[dict[str, Any]] = []
    prediction_columns: dict[str, Any] = {
        ID_COLUMN: splits.test[ID_COLUMN].to_numpy(),
        SOURCE_COLUMN: splits.test[SOURCE_COLUMN].to_numpy(),
        TARGET_COLUMN: test_y,
    }
    expected_validation_ids = splits.validation[ID_COLUMN].to_numpy()
    expected_test_ids = splits.test[ID_COLUMN].to_numpy()

    for index, result in enumerate(results):
        if not np.array_equal(result.validation_ids, expected_validation_ids):
            raise ValueError(f"{result.name} validation predictions are not aligned to split IDs")
        if not np.array_equal(result.test_ids, expected_test_ids):
            raise ValueError(f"{result.name} test predictions are not aligned to split IDs")
        validation_metrics = probability_metrics(validation_y, result.validation_probabilities)
        test_metrics = probability_metrics(test_y, result.test_probabilities)
        f1_low, f1_high = _bootstrap_interval(
            test_y,
            result.test_probabilities,
            _macro_f1,
            iterations,
            config.seed + index,
        )
        accuracy_low, accuracy_high = _bootstrap_interval(
            test_y,
            result.test_probabilities,
            _accuracy,
            iterations,
            config.seed + 100 + index,
        )
        rows.append(
            {
                "model": result.name,
                "family": result.family,
                "selection_split": "validation",
                "validation_macro_f1": validation_metrics["macro_f1"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "test_macro_f1_ci_low": f1_low,
                "test_macro_f1_ci_high": f1_high,
                "test_accuracy_ci_low": accuracy_low,
                "test_accuracy_ci_high": accuracy_high,
                "validation_to_test_macro_f1_delta": (
                    float(test_metrics["macro_f1"]) - float(validation_metrics["macro_f1"])
                ),
                "train_seconds": result.train_seconds,
                "inference_ms_per_sample": result.inference_ms_per_sample,
                "parameter_count": result.parameter_count,
                "trainable_parameter_count": result.trainable_parameter_count,
                "artifact_bytes": result.artifact_bytes,
                "artifact_mib": result.artifact_bytes / (1024**2),
                "best_epoch": result.best_epoch,
                "device": result.metadata.get("device", "cpu"),
                "pretrained": result.family == "pretrained_transformer",
                "pretraining_compute_included": bool(
                    result.metadata.get("pretraining_compute_included", False)
                ),
            }
        )
        prediction_columns[f"{result.name}_probability"] = result.test_probabilities
        prediction_columns[f"{result.name}_prediction"] = (
            result.test_probabilities >= DECISION_THRESHOLD
        ).astype(int)

    comparison = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "model"], ascending=[False, True]
    )
    champion_name = str(comparison.iloc[0]["model"])
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    predictions = pd.DataFrame(prediction_columns)
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    prediction_names = [column for column in predictions if column.endswith("_prediction")]
    disagreement = predictions[predictions[prediction_names].nunique(axis=1) > 1]
    disagreement.to_csv(reports_dir / "model_disagreements.csv", index=False)

    domain_rows: list[dict[str, Any]] = []
    for source in sorted(splits.test[SOURCE_COLUMN].unique()):
        mask = splits.test[SOURCE_COLUMN].to_numpy() == source
        for result in results:
            metrics = probability_metrics(test_y[mask], result.test_probabilities[mask])
            domain_rows.append({"source": source, "model": result.name, **metrics})
    domain_metrics = pd.DataFrame(domain_rows)
    domain_metrics.to_csv(reports_dir / "source_metrics.csv", index=False)

    reference = next(
        (result for result in results if result.name == "tfidf_logistic_regression"), None
    )
    delta_rows: list[dict[str, Any]] = []
    if reference is not None:
        for candidate_index, candidate in enumerate(results):
            if candidate is reference:
                continue
            for metric_name, metric in (("macro_f1", _macro_f1), ("accuracy", _accuracy)):
                point, low, high = _paired_delta_interval(
                    test_y,
                    candidate.test_probabilities,
                    reference.test_probabilities,
                    metric,
                    iterations,
                    config.seed + 1_000 + candidate_index,
                )
                delta_rows.append(
                    {
                        "candidate": candidate.name,
                        "reference": reference.name,
                        "metric": metric_name,
                        "delta": point,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_iterations": iterations,
                    }
                )
    paired_deltas = pd.DataFrame(delta_rows)
    paired_deltas.to_csv(reports_dir / "paired_deltas_vs_tfidf.csv", index=False)

    payload = {
        "champion_model": champion_name,
        "champion_selection_rule": "highest validation macro-F1; test excluded",
        "decision_threshold": DECISION_THRESHOLD,
        "bootstrap_iterations": iterations,
        "models": {
            str(row["model"]): {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in row.items()
                if key != "model"
            }
            for row in rows
        },
    }
    dump_json(payload, reports_dir / "metrics.json")
    return {
        "champion_model": champion_name,
        "comparison": comparison,
        "predictions": predictions,
        "domain_metrics": domain_metrics,
        "paired_deltas": paired_deltas,
    }
