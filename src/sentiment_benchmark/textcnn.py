"""Auditable native-PyTorch TextCNN training for the sentiment benchmark."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from sentiment_benchmark.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    ExperimentConfig,
    ProjectPaths,
)
from sentiment_benchmark.contracts import DatasetSplits, ModelResult

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1

# The benchmark data are English sentences. Keeping the rule deliberately small makes
# tokenization inspectable and reproducible without an external model or downloaded asset.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")

KERNEL_SIZES = (3, 4, 5)
EMBEDDING_DIM = 64
FILTERS_PER_KERNEL = 64
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
DECISION_THRESHOLD = 0.5


def tokenize(text: str) -> list[str]:
    """Lowercase and split one sentence with the documented regex tokenizer."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: Iterable[str], max_size: int) -> dict[str, int]:
    """Fit a deterministic frequency vocabulary on the supplied training texts only."""
    if max_size < 2:
        raise ValueError("textcnn_vocab_size must leave room for PAD and UNK")

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    # Alphabetical tie-breaking means row ordering cannot change token IDs.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    for token, _ in ranked[: max_size - len(vocabulary)]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_tokens: int) -> list[int]:
    """Map one sentence to bounded token IDs, preserving an explicit empty-text signal."""
    if max_tokens < 1:
        raise ValueError("textcnn_max_tokens must be positive")
    encoded = [vocabulary.get(token, UNK_INDEX) for token in tokenize(text)[:max_tokens]]
    return encoded or [UNK_INDEX]


class EncodedTextDataset(Dataset[tuple[list[int], float]]):
    """A minimal map-style dataset containing pre-tokenized IDs and binary labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: dict[str, int],
        max_tokens: int,
    ) -> None:
        texts = frame[TEXT_COLUMN].fillna("").astype(str)
        self.encoded = [encode_text(text, vocabulary, max_tokens) for text in texts]
        self.labels = frame[TARGET_COLUMN].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> tuple[list[int], float]:
        return self.encoded[index], self.labels[index]


class PadCollator:
    """Pad each batch to its longest sentence and at least the largest convolution."""

    def __init__(self, minimum_width: int = max(KERNEL_SIZES)) -> None:
        self.minimum_width = minimum_width

    def __call__(self, batch: Sequence[tuple[list[int], float]]) -> tuple[Tensor, Tensor]:
        if not batch:
            raise ValueError("cannot collate an empty batch")
        width = max(self.minimum_width, max(len(token_ids) for token_ids, _ in batch))
        tokens = torch.full((len(batch), width), PAD_INDEX, dtype=torch.long)
        labels = torch.empty(len(batch), dtype=torch.float32)
        for row, (token_ids, label) in enumerate(batch):
            tokens[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            labels[row] = label
        return tokens, labels


class TextCNN(nn.Module):
    """Kim-style sentence CNN: embedding, parallel Conv1d, max pool, and linear head."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        filters_per_kernel: int = FILTERS_PER_KERNEL,
        kernel_sizes: Sequence[int] = KERNEL_SIZES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if len(kernel_sizes) < 2:
            raise ValueError("TextCNN requires multiple convolution kernel sizes")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters_per_kernel, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters_per_kernel * len(kernel_sizes), 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = [
            torch.relu(convolution(embedded)).amax(dim=2) for convolution in self.convolutions
        ]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def _validate_splits(splits: DatasetSplits) -> None:
    for split_name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        missing = sorted({TEXT_COLUMN, TARGET_COLUMN}.difference(frame.columns))
        if missing:
            raise ValueError(f"{split_name} split is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{split_name} split cannot be empty")
        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(f"{split_name} labels cannot be missing")
        labels = set(frame[TARGET_COLUMN].unique().tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"{split_name} labels must be binary 0/1")
    if splits.train[TARGET_COLUMN].nunique() != 2:
        raise ValueError("train split must contain both classes")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_device(requested: str | None) -> tuple[torch.device, str | None]:
    """Prefer MPS automatically, with a documented CPU fallback when unavailable."""
    if requested is None:
        name = "mps" if torch.backends.mps.is_available() else "cpu"
        return torch.device(name), None

    candidate = torch.device(requested)
    if candidate.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu"), "requested MPS was unavailable; used CPU"
    if candidate.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "requested CUDA was unavailable; used CPU"
    return candidate, None


def _make_loaders(
    splits: DatasetSplits,
    vocabulary: dict[str, int],
    config: ExperimentConfig,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if config.textcnn_batch_size < 1:
        raise ValueError("textcnn_batch_size must be positive")

    datasets = [
        EncodedTextDataset(frame, vocabulary, config.textcnn_max_tokens)
        for frame in (splits.train, splits.validation)
    ]
    collator = PadCollator()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        datasets[0],
        batch_size=config.textcnn_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        datasets[1],
        batch_size=config.textcnn_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return train_loader, validation_loader


def _make_evaluation_loader(
    frame: pd.DataFrame,
    vocabulary: dict[str, int],
    config: ExperimentConfig,
) -> DataLoader[Any]:
    return DataLoader(
        EncodedTextDataset(frame, vocabulary, config.textcnn_max_tokens),
        batch_size=config.textcnn_batch_size,
        shuffle=False,
        collate_fn=PadCollator(),
        num_workers=0,
    )


def _predict_logits_and_loss(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    total_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for token_ids, labels in loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            logits = model(token_ids)
            loss = loss_function(logits, labels)
            batch_size = len(labels)
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            logits_parts.append(logits.detach().cpu().numpy())
            label_parts.append(labels.detach().cpu().numpy())
    return (
        np.concatenate(logits_parts),
        np.concatenate(label_parts),
        total_loss / sample_count,
    )


def _probabilities(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for token_ids, _ in loader:
            logits = model(token_ids.to(device))
            parts.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_inference(
    model: nn.Module,
    frame: pd.DataFrame,
    vocabulary: dict[str, int],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, int]:
    repeats = config.latency_repeats
    if repeats < 1:
        raise ValueError("latency_repeats must be positive")

    def predict_raw_text() -> np.ndarray:
        dataset = EncodedTextDataset(frame, vocabulary, config.textcnn_max_tokens)
        loader = DataLoader(
            dataset,
            batch_size=config.textcnn_batch_size,
            shuffle=False,
            collate_fn=PadCollator(),
            num_workers=0,
        )
        return _probabilities(model, loader, device)

    # Warm the model backend, but time raw-text tokenization, batching, and forward passes.
    predict_raw_text()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        predict_raw_text()
        _synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, len(frame) * repeats


def _cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _write_json(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def train_textcnn(
    splits: DatasetSplits,
    paths: ProjectPaths,
    config: ExperimentConfig,
    device: str | None = None,
) -> ModelResult:
    """Fit, select, persist, and benchmark a native PyTorch TextCNN."""
    _validate_splits(splits)
    if config.textcnn_patience < 1:
        raise ValueError("textcnn_patience must be positive")
    if config.textcnn_learning_rate <= 0:
        raise ValueError("textcnn_learning_rate must be positive")
    if config.effective_textcnn_epochs < 1:
        raise ValueError("effective TextCNN epochs must be positive")

    paths.ensure_directories()
    _seed_everything(config.seed)
    resolved_device, device_note = _resolve_device(device)

    training_started = time.perf_counter()
    # This is the only vocabulary fit call: validation and test text are never observed here.
    train_texts = splits.train[TEXT_COLUMN].fillna("").astype(str).tolist()
    vocabulary = build_vocabulary(train_texts, config.textcnn_vocab_size)
    train_loader, validation_loader = _make_loaders(splits, vocabulary, config)

    model = TextCNN(vocab_size=len(vocabulary)).to(resolved_device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.textcnn_learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    history_rows: list[dict[str, int | float | bool]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_macro_f1 = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.effective_textcnn_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for token_ids, labels in train_loader:
            token_ids = token_ids.to(resolved_device)
            labels = labels.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = len(labels)
            train_loss_sum += float(loss.item()) * batch_size
            train_samples += batch_size

        validation_logits, validation_labels, validation_loss = _predict_logits_and_loss(
            model,
            validation_loader,
            loss_function,
            resolved_device,
        )
        bounded_logits = np.clip(validation_logits, -80.0, 80.0)
        validation_probabilities = 1.0 / (1.0 + np.exp(-bounded_logits))
        validation_predictions = (validation_probabilities >= DECISION_THRESHOLD).astype(int)
        validation_macro_f1 = float(
            f1_score(
                validation_labels,
                validation_predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        improved = validation_macro_f1 > best_macro_f1 + 1e-12
        if improved:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_samples,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_macro_f1,
                "is_best": improved,
            }
        )
        if epochs_without_improvement >= config.textcnn_patience:
            break
    if best_state is None:
        raise RuntimeError("TextCNN training did not produce a checkpoint")

    artifact_dir = paths.models_dir / "textcnn"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "model.safetensors"
    save_file(
        best_state,
        str(weights_path),
        metadata={"model": "pytorch_textcnn", "format": "pt"},
    )

    history = pd.DataFrame(
        history_rows,
        columns=[
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_macro_f1",
            "is_best",
        ],
    )
    history_path = paths.reports_dir / "textcnn_history.csv"
    history.to_csv(history_path, index=False)

    vocabulary_payload = {
        "schema_version": "1.0",
        "fit_split": "train",
        "pad_token": PAD_TOKEN,
        "pad_index": PAD_INDEX,
        "unk_token": UNK_TOKEN,
        "unk_index": UNK_INDEX,
        "token_to_index": vocabulary,
    }
    _write_json(vocabulary_payload, artifact_dir / "vocab.json")

    model_config = {
        "schema_version": "1.0",
        "model_name": "pytorch_textcnn",
        "family": "deep_learning_from_scratch",
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "tokenizer": {
            "name": "lowercase_regex",
            "pattern": TOKEN_PATTERN.pattern,
            "fit_split": "train",
        },
        "architecture": {
            "vocab_size": len(vocabulary),
            "embedding_dim": EMBEDDING_DIM,
            "filters_per_kernel": FILTERS_PER_KERNEL,
            "kernel_sizes": list(KERNEL_SIZES),
            "dropout": DROPOUT,
            "padding_index": PAD_INDEX,
        },
        "training": {
            "seed": config.seed,
            "batch_size": config.textcnn_batch_size,
            "learning_rate": config.textcnn_learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "maximum_epochs": config.effective_textcnn_epochs,
            "patience": config.textcnn_patience,
            "selection_metric": "validation_macro_f1",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "decision_threshold": DECISION_THRESHOLD,
        },
        "input": {
            "maximum_tokens": config.textcnn_max_tokens,
            "requested_vocabulary_size": config.textcnn_vocab_size,
        },
        "runtime": {
            "requested_device": device,
            "resolved_device": str(resolved_device),
            "device_note": device_note,
            "latency_repeats": config.latency_repeats,
        },
        "files": {
            "weights": weights_path.name,
            "vocabulary": "vocab.json",
            "history": str(history_path.relative_to(paths.root)),
        },
    }
    _write_json(model_config, artifact_dir / "config.json")
    train_seconds = time.perf_counter() - training_started

    # Reload from the safe, tensor-only checkpoint so final predictions prove restoration works.
    restored_state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(restored_state)
    model.to(resolved_device)

    test_loader = _make_evaluation_loader(splits.test, vocabulary, config)
    validation_probabilities = _probabilities(model, validation_loader, resolved_device)
    test_probabilities = _probabilities(model, test_loader, resolved_device)
    inference_seconds, inference_samples = _measure_inference(
        model,
        splits.test,
        vocabulary,
        config,
        resolved_device,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata: dict[str, Any] = {
        "device": str(resolved_device),
        "device_note": device_note,
        "vocabulary_size": len(vocabulary),
        "latency_repeats": config.latency_repeats,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": best_macro_f1,
        "history_path": str(history_path.relative_to(paths.root)),
        "weights_format": "safetensors",
        "training_cost_scope": "representation_setup_through_selected_artifact",
    }
    return ModelResult(
        name="pytorch_textcnn",
        family="deep_learning_from_scratch",
        validation_probabilities=validation_probabilities,
        test_probabilities=test_probabilities,
        validation_ids=splits.validation[ID_COLUMN].to_numpy(copy=True),
        test_ids=splits.test[ID_COLUMN].to_numpy(copy=True),
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        inference_samples=inference_samples,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        artifact_path=artifact_dir,
        best_epoch=best_epoch,
        history=history,
        metadata=metadata,
    )
