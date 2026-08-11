"""Pinned DistilBERT fine-tuning with an explicit PyTorch training loop."""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_benchmark.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TRANSFORMER_MODEL,
    TRANSFORMER_REVISION,
    ExperimentConfig,
    ProjectPaths,
)
from sentiment_benchmark.contracts import DatasetSplits, ModelResult


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def select_device(requested: str | None = None) -> torch.device:
    """Prefer an accelerator when available, with a deterministic CPU fallback."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokenize_frame(tokenizer: object, frame: pd.DataFrame, max_tokens: int) -> TensorDataset:
    encoded = tokenizer(
        frame[TEXT_COLUMN].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    labels = torch.as_tensor(frame[TARGET_COLUMN].to_numpy(dtype=np.int64, copy=True))
    tensors: list[torch.Tensor] = [encoded["input_ids"], encoded["attention_mask"]]
    tensors.append(labels)
    return TensorDataset(*tensors)


def _forward_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    moved = [tensor.to(device) for tensor in batch]
    labels = moved[-1]
    model_inputs = {"input_ids": moved[0], "attention_mask": moved[1], "labels": labels}
    output = model(**model_inputs)
    return output.loss, output.logits


def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            moved = [tensor.to(device) for tensor in batch]
            inputs = {"input_ids": moved[0], "attention_mask": moved[1]}
            logits = model(**inputs).logits
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probabilities)


def _predict_raw_frame(
    model: torch.nn.Module,
    tokenizer: object,
    frame: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    dataset = _tokenize_frame(tokenizer, frame, config.transformer_max_tokens)
    loader = DataLoader(
        dataset,
        batch_size=config.transformer_batch_size,
        shuffle=False,
    )
    return _predict_loader(model, loader, device)


def _save_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prefetch_assets(
    model_name: str,
    revision: str | None,
    cache_dir: Path,
    local_files_only: bool,
) -> float:
    """Resolve remote assets outside the comparable downstream-training timer."""
    if local_files_only:
        return 0.0
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
            num_labels=2,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=False,
            num_labels=2,
        )
    del tokenizer, model
    gc.collect()
    return time.perf_counter() - started


def train_transformer(
    splits: DatasetSplits,
    paths: ProjectPaths,
    config: ExperimentConfig,
    *,
    device: str | None = None,
    model_name: str = TRANSFORMER_MODEL,
    revision: str | None = TRANSFORMER_REVISION,
    local_files_only: bool = False,
) -> ModelResult:
    """Fine-tune a pinned pretrained Transformer without using the test set for selection."""
    paths.ensure_directories()
    selected_device = select_device(device)
    asset_resolution_seconds = _prefetch_assets(
        model_name,
        revision,
        paths.hf_cache,
        local_files_only,
    )
    _seed_everything(config.seed)

    training_started = time.perf_counter()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=paths.hf_cache,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=paths.hf_cache,
        local_files_only=True,
        num_labels=2,
    )
    load_seconds = time.perf_counter() - load_started
    model.to(selected_device)

    train_data = _tokenize_frame(tokenizer, splits.train, config.transformer_max_tokens)
    validation_data = _tokenize_frame(tokenizer, splits.validation, config.transformer_max_tokens)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.transformer_batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=config.transformer_batch_size, shuffle=False
    )
    optimizer = AdamW(model.parameters(), lr=config.transformer_learning_rate, weight_decay=0.01)
    model_dir = paths.models_dir / "distilbert"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []
    for epoch in range(1, config.effective_transformer_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_batch(model, batch, selected_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_examples = len(batch[-1])
            total_loss += float(loss.detach().cpu()) * batch_examples
            examples += batch_examples

        validation_probability = _predict_loader(model, validation_loader, selected_device)
        validation_f1 = float(
            f1_score(
                splits.validation[TARGET_COLUMN],
                (validation_probability >= 0.5).astype(int),
                average="macro",
            )
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(examples, 1),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1 + 1e-12:
            best_f1 = validation_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.transformer_patience:
                break

    history = pd.DataFrame(history_rows)
    history.to_csv(paths.reports_dir / "distilbert_history.csv", index=False)

    metadata = {
        "base_model": model_name,
        "revision": revision,
        "device": str(selected_device),
        "asset_resolution_seconds": asset_resolution_seconds,
        "local_model_load_seconds": load_seconds,
        "asset_resolution_included_in_train_seconds": False,
        "max_tokens": config.transformer_max_tokens,
        "batch_size": config.transformer_batch_size,
        "learning_rate": config.transformer_learning_rate,
        "epochs_completed": len(history),
        "best_validation_macro_f1": best_f1,
        "pretraining_compute_included": False,
        "training_cost_scope": "representation_setup_through_selected_artifact",
    }
    _save_metadata(model_dir / "benchmark_metadata.json", metadata)
    train_seconds = time.perf_counter() - training_started

    del model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(selected_device)
    validation_data = _tokenize_frame(tokenizer, splits.validation, config.transformer_max_tokens)
    test_data = _tokenize_frame(tokenizer, splits.test, config.transformer_max_tokens)
    validation_loader = DataLoader(
        validation_data, batch_size=config.transformer_batch_size, shuffle=False
    )
    test_loader = DataLoader(test_data, batch_size=config.transformer_batch_size, shuffle=False)
    validation_probability = _predict_loader(model, validation_loader, selected_device)
    test_probability = _predict_loader(model, test_loader, selected_device)

    # Warm the backend, then time raw tokenization, batching, and model forward passes.
    _predict_raw_frame(model, tokenizer, splits.test, config, selected_device)
    inference_seconds = 0.0
    for _ in range(max(1, config.latency_repeats)):
        started = time.perf_counter()
        _predict_raw_frame(model, tokenizer, splits.test, config, selected_device)
        inference_seconds += time.perf_counter() - started

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelResult(
        name="distilbert",
        family="pretrained_transformer",
        validation_probabilities=validation_probability,
        test_probabilities=test_probability,
        validation_ids=splits.validation[ID_COLUMN].to_numpy(copy=True),
        test_ids=splits.test[ID_COLUMN].to_numpy(copy=True),
        train_seconds=train_seconds,
        inference_seconds=float(inference_seconds),
        inference_samples=len(splits.test) * max(1, config.latency_repeats),
        parameter_count=parameters,
        trainable_parameter_count=trainable,
        artifact_path=model_dir,
        best_epoch=best_epoch,
        history=history,
        metadata=metadata,
    )
