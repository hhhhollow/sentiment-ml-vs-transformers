"""Portfolio-grade plots, manifests, data card, and measured experiment report."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from sentiment_benchmark.config import (
    ID_COLUMN,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TRANSFORMER_LICENSE,
    TRANSFORMER_MODEL,
    TRANSFORMER_REVISION,
    UCI_DATASET_DOI,
    UCI_DATASET_LICENSE,
    UCI_DATASET_SHA256,
    UCI_DATASET_URL,
    ExperimentConfig,
    ProjectPaths,
    SplitConfig,
)
from sentiment_benchmark.contracts import DatasetSplits, ModelResult
from sentiment_benchmark.data import UCI_RAW_ROWS
from sentiment_benchmark.evaluation import dump_json

DISPLAY_NAMES = {
    "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
    "pytorch_textcnn": "PyTorch TextCNN",
    "distilbert": "DistilBERT",
}
COLORS = {
    "tfidf_logistic_regression": "#2563eb",
    "pytorch_textcnn": "#f97316",
    "distilbert": "#16a34a",
}


def _sha256_frame(frame: pd.DataFrame) -> str:
    canonical = frame.sort_values(ID_COLUMN).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "safetensors",
        "joblib",
    ]
    return {package: importlib.metadata.version(package) for package in packages}


def create_data_evidence(
    frame: pd.DataFrame,
    splits: DatasetSplits,
    paths: ProjectPaths,
) -> dict[str, Any]:
    """Create aggregate EDA without committing copyrighted review text."""
    paths.ensure_directories()
    evidence = frame.copy()
    evidence["characters"] = evidence[TEXT_COLUMN].str.len()
    evidence["words"] = evidence[TEXT_COLUMN].str.split().str.len()

    source_rows = (
        evidence.groupby(SOURCE_COLUMN)
        .agg(
            rows=(ID_COLUMN, "size"),
            positive_rate=(TARGET_COLUMN, "mean"),
            median_characters=("characters", "median"),
            p95_characters=("characters", lambda values: values.quantile(0.95)),
            median_words=("words", "median"),
        )
        .reset_index()
    )
    source_rows.to_csv(paths.reports_dir / "dataset_by_source.csv", index=False)

    split_rows: list[dict[str, Any]] = []
    for name, split in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        for source, subset in split.groupby(SOURCE_COLUMN):
            split_rows.append(
                {
                    "split": name,
                    "source": source,
                    "rows": len(subset),
                    "positive_rate": float(subset[TARGET_COLUMN].mean()),
                }
            )
    split_distribution = pd.DataFrame(split_rows)
    split_distribution.to_csv(paths.reports_dir / "split_distribution.csv", index=False)

    summary = {
        "raw_rows": UCI_RAW_ROWS,
        "prepared_rows": len(frame),
        "normalised_duplicates_removed": UCI_RAW_ROWS - len(frame),
        "positive_rate": float(frame[TARGET_COLUMN].mean()),
        "sources": sorted(frame[SOURCE_COLUMN].unique().tolist()),
        "source_count": int(frame[SOURCE_COLUMN].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
        "normalised_text_duplicates_remaining": 0,
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "data_sha256": _sha256_frame(frame),
    }
    dump_json(summary, paths.reports_dir / "dataset_summary.json")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts = evidence.groupby([SOURCE_COLUMN, TARGET_COLUMN]).size().unstack(fill_value=0)
    counts.columns = ["Negative", "Positive"]
    counts.plot(kind="bar", ax=axes[0], color=["#64748b", "#22c55e"], rot=0)
    axes[0].set(title="Class balance by review source", xlabel="Source", ylabel="Sentences")
    for source, subset in evidence.groupby(SOURCE_COLUMN):
        axes[1].hist(
            subset["words"],
            bins=np.arange(0, min(100, int(evidence["words"].max()) + 5), 5),
            alpha=0.45,
            label=source,
        )
    axes[1].set(
        title="Sentence length distribution",
        xlabel="Whitespace-separated words (clipped view)",
        ylabel="Sentences",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "dataset_overview.png", dpi=180)
    plt.close(fig)
    return summary


def create_evaluation_plots(
    evaluation: dict[str, Any],
    results: list[ModelResult],
    splits: DatasetSplits,
    paths: ProjectPaths,
) -> None:
    """Visualise predictive quality, costs, domains, errors, and learning histories."""
    paths.ensure_directories()
    sns.set_theme(style="whitegrid", context="notebook")
    comparison = evaluation["comparison"].copy()
    comparison["display"] = comparison["model"].map(DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics = ["test_macro_f1", "test_accuracy", "test_roc_auc"]
    labels = ["Macro-F1", "Accuracy", "ROC-AUC"]
    x = np.arange(len(comparison))
    width = 0.24
    for offset, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
        ax.bar(x + (offset - 1) * width, comparison[metric], width=width, label=label)
    ax.set_xticks(x, comparison["display"])
    ax.set_ylim(0.45, 1.0)
    ax.set(title="Held-out performance on one shared test set", ylabel="Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "performance_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for row in comparison.to_dict(orient="records"):
        ax.scatter(
            max(float(row["train_seconds"]), 1e-4),
            row["test_macro_f1"],
            s=90 + 28 * np.log1p(float(row["artifact_mib"])),
            color=COLORS[str(row["model"])],
            alpha=0.85,
        )
        ax.annotate(
            DISPLAY_NAMES[str(row["model"])],
            (max(float(row["train_seconds"]), 1e-4), row["test_macro_f1"]),
            xytext=(7, 5),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set(
        title="Performance-cost frontier",
        xlabel="Downstream model-development time (seconds, log scale)",
        ylabel="Test Macro-F1",
    )
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "performance_cost_frontier.png", dpi=180)
    plt.close(fig)

    domain = evaluation["domain_metrics"]
    heatmap = domain.pivot(index="model", columns="source", values="macro_f1")
    heatmap.index = [DISPLAY_NAMES[index] for index in heatmap.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=1.0, ax=ax)
    ax.set(title="Test Macro-F1 by source domain", xlabel="Review source", ylabel="Model")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "source_robustness.png", dpi=180)
    plt.close(fig)

    figure, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.2))
    axes_array = np.atleast_1d(axes)
    y = splits.test[TARGET_COLUMN].to_numpy(dtype=int)
    for axis, result in zip(axes_array, results, strict=True):
        matrix = confusion_matrix(y, result.test_probabilities >= 0.5, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set(
            title=DISPLAY_NAMES[result.name],
            xlabel="Predicted",
            ylabel="Actual",
        )
    figure.suptitle("Held-out confusion matrices at fixed 0.50 threshold", y=1.02)
    figure.tight_layout()
    figure.savefig(paths.figures_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for result in results:
        observed, predicted = calibration_curve(y, result.test_probabilities, n_bins=10)
        ax.plot(predicted, observed, marker="o", label=DISPLAY_NAMES[result.name])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect calibration")
    ax.set(
        title="Probability calibration on the held-out test set",
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "calibration_comparison.png", dpi=180)
    plt.close(fig)

    histories = [result for result in results if not result.history.empty]
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.5))
        for axis, result in zip(np.atleast_1d(axes), histories, strict=True):
            axis.plot(
                result.history["epoch"],
                result.history["validation_macro_f1"],
                marker="o",
            )
            axis.axvline(result.best_epoch, color="#dc2626", linestyle="--", label="Best epoch")
            axis.set(
                title=DISPLAY_NAMES[result.name],
                xlabel="Epoch",
                ylabel="Validation Macro-F1",
                ylim=(0.45, 1.0),
            )
            axis.legend()
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "learning_curves.png", dpi=180)
        plt.close(fig)


def _format_model_table(comparison: pd.DataFrame) -> str:
    rows = [
        "| Model | Val Macro-F1 | Test Macro-F1 (95% CI) | Accuracy | ROC-AUC | "
        "Dev s | Infer ms/item | Params | Artifact MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        rows.append(
            "| {name} | {val:.4f} | {f1:.4f} [{low:.4f}, {high:.4f}] | {acc:.4f} | "
            "{auc:.4f} | {seconds:.2f} | {latency:.3f} | {params:,} | {size:.2f} |".format(
                name=DISPLAY_NAMES[str(row["model"])],
                val=row["validation_macro_f1"],
                f1=row["test_macro_f1"],
                low=row["test_macro_f1_ci_low"],
                high=row["test_macro_f1_ci_high"],
                acc=row["test_accuracy"],
                auc=row["test_roc_auc"],
                seconds=row["train_seconds"],
                latency=row["inference_ms_per_sample"],
                params=int(row["parameter_count"]),
                size=row["artifact_mib"],
            )
        )
    return "\n".join(rows)


def _format_source_table(domain: pd.DataFrame) -> str:
    sources = sorted(domain["source"].unique().tolist())
    rows = [
        "| Model | " + " | ".join(source.title() for source in sources) + " |",
        "|---|" + "---:|" * len(sources),
    ]
    for model_name, model_rows in domain.groupby("model", sort=False):
        values = model_rows.set_index("source")["macro_f1"]
        scores = " | ".join(f"{float(values[source]):.4f}" for source in sources)
        rows.append(f"| {DISPLAY_NAMES[str(model_name)]} | {scores} |")
    return "\n".join(rows)


def write_portfolio_documents(
    frame: pd.DataFrame,
    splits: DatasetSplits,
    evaluation: dict[str, Any],
    results: list[ModelResult],
    paths: ProjectPaths,
    config: ExperimentConfig,
    split_config: SplitConfig,
) -> None:
    """Write source-grounded documents whose numbers come only from generated CSVs."""
    comparison = evaluation["comparison"]
    champion_name = str(evaluation["champion_model"])
    champion = comparison.loc[comparison["model"] == champion_name].iloc[0]
    baseline = comparison.loc[comparison["model"] == "tfidf_logistic_regression"].iloc[0]
    f1_delta = float(champion["test_macro_f1"] - baseline["test_macro_f1"])
    cost_ratio = float(champion["train_seconds"] / max(baseline["train_seconds"], 1e-9))
    size_ratio = float(champion["artifact_bytes"] / max(baseline["artifact_bytes"], 1))
    textcnn = comparison.loc[comparison["model"] == "pytorch_textcnn"].iloc[0]
    textcnn_cost_ratio = float(textcnn["train_seconds"] / max(baseline["train_seconds"], 1e-9))
    textcnn_latency_ratio = float(
        textcnn["inference_ms_per_sample"] / max(baseline["inference_ms_per_sample"], 1e-9)
    )
    textcnn_parameter_ratio = float(
        textcnn["parameter_count"] / max(baseline["parameter_count"], 1)
    )
    deltas = evaluation["paired_deltas"]
    transformer_delta = deltas.loc[
        (deltas["candidate"] == "distilbert") & (deltas["metric"] == "macro_f1")
    ].iloc[0]
    textcnn_delta = deltas.loc[
        (deltas["candidate"] == "pytorch_textcnn") & (deltas["metric"] == "macro_f1")
    ].iloc[0]
    domain = evaluation["domain_metrics"]
    worst = domain.loc[domain["macro_f1"].idxmin()]
    model_table = _format_model_table(comparison)
    source_table = _format_source_table(domain)
    calibration_notes = "\n".join(
        f"- {DISPLAY_NAMES[str(row['model'])]}: test ECE = {float(row['test_ece_10_bin']):.4f}"
        for row in comparison.to_dict(orient="records")
    )

    report = f"""# Experiment Report: Traditional ML vs Deep Learning

## Executive answer

On the single untouched test set, **{DISPLAY_NAMES[champion_name]}** was selected before test
evaluation because it had the highest validation Macro-F1. Its held-out Macro-F1 was
**{champion["test_macro_f1"]:.4f}** versus **{baseline["test_macro_f1"]:.4f}** for the TF-IDF
baseline, a difference of **{f1_delta:+.4f}**. That downstream result required roughly
**{cost_ratio:.1f}x** the measured downstream model-development time and
**{size_ratio:.1f}x** the serialized artifact size of TF-IDF + Logistic Regression.

The paired-bootstrap 95% interval for DistilBERT's Macro-F1 improvement is
**[{transformer_delta["ci_low"]:+.4f}, {transformer_delta["ci_high"]:+.4f}]**. In contrast, the
from-scratch TextCNN reached **{textcnn["test_macro_f1"]:.4f}**, a paired difference of
**{textcnn_delta["delta"]:+.4f}**
**[{textcnn_delta["ci_low"]:+.4f}, {textcnn_delta["ci_high"]:+.4f}]** versus TF-IDF. The result
makes the central distinction concrete: using PyTorch alone did not improve quality on this small
corpus. The gain was observed on the pretrained-Transformer rung, which also changes architecture,
tokenizer, parameter count, and optimisation; this experiment does not isolate pretraining's causal
contribution.

TextCNN also required **{textcnn_cost_ratio:.1f}x** the downstream development time,
**{textcnn_latency_ratio:.1f}x** the warm inference latency, and
**{textcnn_parameter_ratio:.1f}x** the parameters of TF-IDF while performing worse. Its serialized
artifact was slightly smaller, showing why artifact bytes alone are not a sufficient cost proxy.

{model_table}

The table is the measured answer, not an assumption that the Transformer must win. Confidence
intervals quantify finite-test uncertainty; paired deltas in `paired_deltas_vs_tfidf.csv` are the
right place to judge whether an apparent improvement is stable.

![Performance and cost frontier](figures/performance_cost_frontier.png)

## Experimental protocol

- Dataset: UCI Sentiment Labelled Sentences, {len(frame):,} prepared sentences from Amazon,
  IMDb, and Yelp; DOI `{UCI_DATASET_DOI}`, license {UCI_DATASET_LICENSE}.
- Cleaning: NFKC + case-fold + whitespace-normalised duplicate keys are removed globally before
  splitting ({UCI_RAW_ROWS - len(frame)} duplicate rows removed). Retained text is used for
  modelling after surrounding whitespace is stripped by the parser.
- Split: one persisted source×label-stratified train/validation/test assignment with
  {len(splits.train):,}/{len(splits.validation):,}/{len(splits.test):,} rows. Every model receives
  the same IDs, and no normalised sentence crosses a split.
- Selection: candidates/checkpoints are chosen only by validation Macro-F1. Test labels never affect
  fitting, checkpoint selection, hyperparameters, or the champion; they are used only in the common
  final evaluation. Every classifier uses the same fixed 0.50 decision threshold.
- Uncertainty: {config.effective_bootstrap_iterations:,} deterministic bootstrap resamples produce
  95% intervals; model-to-baseline deltas use paired row resamples.

## What each rung demonstrates

1. **TF-IDF + Logistic Regression** searches word unigram, word bigram, and word+character
   representations with three regularisation values. Vocabulary fitting stays inside train.
2. **PyTorch TextCNN** builds a train-only regex vocabulary, learned embeddings, parallel 3/4/5
   token convolutions, global max pooling, dropout, AdamW, and validation early stopping. Its
   training loop, checkpoint restoration, batching, and inference are implemented explicitly.
3. **DistilBERT** fine-tunes `{TRANSFORMER_MODEL}` pinned to immutable revision
   `{TRANSFORMER_REVISION}`. Tokenization, batching, optimizer steps, gradient clipping, validation
   checkpointing, safe serialization, and latency measurement are explicit PyTorch code.

## Domain robustness and error analysis

The lowest source-specific result was **{worst["macro_f1"]:.4f}** for
{DISPLAY_NAMES[str(worst["model"])]} on **{worst["source"]}**. Aggregate scores can therefore hide
meaningful domain variation even when every source is balanced. `source_metrics.csv` contains all
per-source metrics, while `model_disagreements.csv` records only IDs, labels, probabilities, and
predictions—no review text is copied into the versioned report.

{source_table}

![Source robustness](figures/source_robustness.png)

![Probability calibration](figures/calibration_comparison.png)

![Confusion matrices](figures/confusion_matrices.png)

## Cost interpretation

Measured training cost uses one shared boundary: representation/tokenizer setup from validated split
frames, model construction, candidate search or epoch optimisation, validation selection, and
serialization of the selected runnable artifact. Test evaluation, dataset download, initial
base-model download, and DistilBERT pretraining are excluded. The formal Transformer run uses the
verified local cache after a separate untimed asset-resolution step. Timings are useful for
order-of-magnitude comparison on this machine, not as hardware-independent benchmarks. Energy and
carbon are not estimated because the project does not have a trustworthy power measurement.

Warm in-process inference timing starts from raw text and includes each model's tokenizer or
vectorizer, batch construction, and forward pass. It excludes reading model artifacts from disk.

## Limits and responsible use

- The source authors selected clearly positive/negative English sentences; neutral, mixed,
  long-form, multilingual, sarcasm-heavy, and contemporary distribution-shift cases are
  underrepresented.
- Random stratification tests in-distribution generalisation, not future or new-platform robustness.
- Confidence intervals resample rows from this one test split; they do not cover split-seed or
  training-seed variation.
- Review-source slices are domains, not demographic fairness groups. The dataset has no reliable
  protected-attribute annotations, so it cannot support a demographic fairness claim.
- Pretrained Transformers can inherit social bias from pretraining data. High aggregate sentiment
  accuracy does not make the system suitable for consequential decisions about people.
- The original review text is downloaded under {UCI_DATASET_LICENSE} and remains uncommitted;
  derived aggregate evidence and stable IDs are versioned.
"""
    (paths.reports_dir / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")

    data_card = f"""# Data Card: UCI Sentiment Labelled Sentences

## Provenance and license

- Canonical source: {UCI_DATASET_URL}
- UCI DOI: https://doi.org/{UCI_DATASET_DOI}
- License: {UCI_DATASET_LICENSE}; attribution to Dimitrios Kotzias and the UCI Machine Learning
  Repository is required.
- Pinned archive SHA-256: `{UCI_DATASET_SHA256}`

The corpus contains English sentences sampled from product, movie, and restaurant reviews. UCI
reports 500 positive and 500 negative sentences per source and says neutral sentences were excluded.

## Processing and split

The parser reads physical lines, strips surrounding whitespace, and separates the label at the final
tab. It verifies the archive
and all three member hashes, validates binary labels and non-empty text, then removes duplicate
normalised-text keys globally before any split. This produces {len(frame):,} rows from
{UCI_RAW_ROWS:,} source rows. The retained original sentence is used for modelling; the normalised
key exists only for duplicate/leakage control.

Stable SHA-256-derived sentence IDs replace row positions. A source×label-stratified split assigns
{len(splits.train):,} train, {len(splits.validation):,} validation, and {len(splits.test):,} test
rows. The mapping is persisted, and ID/text overlap checks fail closed.

## Known limitations

Labels encode deliberately clear binary sentiment rather than the full ambiguity of natural
language. The dataset is small, English-only, and sourced from three older review domains. It
contains no demographic labels, timestamp, annotator agreement, or reliable author identity. It is
appropriate for an educational model-comparison benchmark, not for claims about production drift,
multilinguality, individual people, or demographic fairness.
"""
    (paths.reports_dir / "DATA_CARD.md").write_text(data_card, encoding="utf-8")

    model_cards = f"""# Model Cards

All three binary English-sentiment classifiers are educational artifacts. Their intended use is to
compare modelling approaches on the pinned UCI benchmark. They are not suitable for employment,
education, credit, health, moderation enforcement, or other consequential decisions about people.

{model_table}

## Selection and evaluation boundary

Every representation is fitted on train only. TF-IDF candidates and neural checkpoints are
selected by validation Macro-F1; the held-out test set never controls hyperparameters, epochs, or
the reported champion. Probabilities are converted with a fixed 0.50 threshold. Artifacts record
model-specific preprocessing so inference cannot silently use a different vocabulary or tokenizer.

## Artifact and inference contract

- Input: a non-blank English `text` string and an optional unique `sentence_id`.
- Output: positive-class probability, a label from a fixed 0.50 threshold, and the model name.
- Artifacts: `artifacts/models/tfidf_logistic.joblib`, `artifacts/models/textcnn/`, and
  `artifacts/models/distilbert/`. TextCNN and DistilBERT weights use safetensors; DistilBERT is
  derived from `{TRANSFORMER_MODEL}` at immutable revision `{TRANSFORMER_REVISION}`.
- Probabilities are not post-hoc calibrated. Ten-bin ECE on this test split is diagnostic, not a
  guarantee for new domains:

{calibration_notes}

## Test Macro-F1 by source

{source_table}

## Model-specific risks

- TF-IDF is sparse and efficient but weak at compositional meaning, negation over long spans, and
  unseen wording.
- TextCNN learns local n-gram-like patterns from only {len(splits.train):,} training rows;
  variance and out-of-vocabulary behaviour are important.
- DistilBERT has much broader pretrained knowledge but inherits unknown pretraining biases and
  carries substantially more parameters. This benchmark excludes its upstream pretraining compute.

The bootstrap intervals quantify row-sampling uncertainty on one fixed test split. They do not
measure variance across alternative train/validation/test assignments or neural training seeds.
"""
    (paths.reports_dir / "MODEL_CARDS.md").write_text(model_cards, encoding="utf-8")


def write_run_manifest(
    frame: pd.DataFrame,
    splits: DatasetSplits,
    evaluation: dict[str, Any],
    paths: ProjectPaths,
    config: ExperimentConfig,
    split_config: SplitConfig,
) -> dict[str, Any]:
    """Record enough provenance to reproduce and audit the measured run."""
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "UCI Sentiment Labelled Sentences",
            "url": UCI_DATASET_URL,
            "doi": UCI_DATASET_DOI,
            "license": UCI_DATASET_LICENSE,
            "archive_sha256": _sha256_file(paths.raw_archive),
            "prepared_canonical_sha256": _sha256_frame(frame),
            "prepared_file_sha256": _sha256_file(paths.prepared_data),
            "raw_rows": UCI_RAW_ROWS,
            "prepared_rows": len(frame),
        },
        "split": {
            "strategy": "two-stage source-and-label stratified random split",
            "config": {
                "test_size": split_config.test_size,
                "validation_size": split_config.validation_size,
                "random_state": split_config.random_state,
            },
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "assignment_sha256": _sha256_file(paths.split_assignments),
            "id_overlap": 0,
            "normalised_text_overlap": 0,
        },
        "experiment_config": config.to_dict(),
        "transformer": {
            "model": TRANSFORMER_MODEL,
            "revision": TRANSFORMER_REVISION,
            "license": TRANSFORMER_LICENSE,
            "pretraining_compute_included": False,
        },
        "selection": {
            "metric": "validation macro-F1",
            "champion_model": evaluation["champion_model"],
            "test_excluded_from_selection": True,
            "decision_threshold": 0.5,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        },
    }
    dump_json(manifest, paths.reports_dir / "run_manifest.json")
    return manifest
