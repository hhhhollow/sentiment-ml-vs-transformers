"""Batch prediction from each persisted benchmark artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_benchmark.config import ID_COLUMN, TEXT_COLUMN, ProjectPaths
from sentiment_benchmark.textcnn import (
    EncodedTextDataset,
    PadCollator,
    TextCNN,
)
from sentiment_benchmark.transformer_model import select_device

MODEL_CHOICES = ("tfidf_logistic_regression", "pytorch_textcnn", "distilbert")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(f"inference input is missing required column: {TEXT_COLUMN}")
    if frame.empty:
        raise ValueError("inference input is empty")
    if frame[TEXT_COLUMN].isna().any() or frame[TEXT_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("inference text must be non-missing and non-blank")
    validated = frame.copy()
    if ID_COLUMN not in validated:
        validated.insert(0, ID_COLUMN, [f"input_{index:06d}" for index in range(len(frame))])
    if validated[ID_COLUMN].isna().any() or validated[ID_COLUMN].duplicated().any():
        raise ValueError("sentence_id must be non-missing and unique when supplied")
    return validated


def _predict_tfidf(texts: pd.Series, artifact_path: Path) -> np.ndarray:
    artifact = joblib.load(artifact_path)
    if artifact.get("model_name") != "tfidf_logistic_regression":
        raise ValueError("artifact is not the expected TF-IDF model bundle")
    return np.asarray(artifact["pipeline"].predict_proba(texts)[:, 1], dtype=float)


def _predict_textcnn(
    texts: pd.Series,
    artifact_dir: Path,
    requested_device: str | None,
) -> np.ndarray:
    model_config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    vocabulary_payload = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
    if model_config.get("model_name") != "pytorch_textcnn":
        raise ValueError("artifact is not the expected TextCNN bundle")
    architecture = model_config["architecture"]
    model = TextCNN(
        vocab_size=int(architecture["vocab_size"]),
        embedding_dim=int(architecture["embedding_dim"]),
        filters_per_kernel=int(architecture["filters_per_kernel"]),
        kernel_sizes=tuple(int(value) for value in architecture["kernel_sizes"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(load_file(str(artifact_dir / "model.safetensors"), device="cpu"))
    device = select_device(requested_device)
    model.to(device).eval()
    inference_frame = pd.DataFrame(
        {TEXT_COLUMN: texts.to_numpy(), "label": np.zeros(len(texts), dtype=int)}
    )
    dataset = EncodedTextDataset(
        inference_frame,
        vocabulary_payload["token_to_index"],
        int(model_config["input"]["maximum_tokens"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(model_config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=PadCollator(),
    )
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for token_ids, _ in loader:
            logits = model(token_ids.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def _predict_distilbert(
    texts: pd.Series,
    artifact_dir: Path,
    requested_device: str | None,
) -> np.ndarray:
    metadata = json.loads((artifact_dir / "benchmark_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir, local_files_only=True)
    device = select_device(requested_device)
    model.to(device).eval()
    probabilities: list[np.ndarray] = []
    batch_size = int(metadata["batch_size"])
    text_values = texts.astype(str).tolist()
    with torch.inference_mode():
        for start in range(0, len(text_values), batch_size):
            encoded = tokenizer(
                text_values[start : start + batch_size],
                padding="max_length",
                truncation=True,
                max_length=int(metadata["max_tokens"]),
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            logits = model(**model_inputs).logits
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def predict_frame(
    frame: pd.DataFrame,
    model_name: str,
    paths: ProjectPaths | None = None,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    """Score text while keeping original text out of the prediction output."""
    paths = paths or ProjectPaths.discover()
    validated = _validate_frame(frame)
    if model_name == "tfidf_logistic_regression":
        probability = _predict_tfidf(
            validated[TEXT_COLUMN], paths.models_dir / "tfidf_logistic.joblib"
        )
    elif model_name == "pytorch_textcnn":
        probability = _predict_textcnn(validated[TEXT_COLUMN], paths.models_dir / "textcnn", device)
    elif model_name == "distilbert":
        probability = _predict_distilbert(
            validated[TEXT_COLUMN], paths.models_dir / "distilbert", device
        )
    else:
        raise ValueError(f"unknown model {model_name!r}; choose one of {MODEL_CHOICES}")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise RuntimeError("model returned invalid probabilities")
    predicted = (probability >= 0.5).astype(int)
    return pd.DataFrame(
        {
            ID_COLUMN: validated[ID_COLUMN].to_numpy(),
            "positive_probability": probability,
            "predicted_label": predicted,
            "predicted_sentiment": np.where(predicted == 1, "positive", "negative"),
            "model": model_name,
        }
    )


def predict_csv(
    input_path: Path,
    output_path: Path,
    model_name: str,
    paths: ProjectPaths | None = None,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    output = predict_frame(frame, model_name, paths, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
