"""One command that materialises data, trains all models, and writes the comparison."""

from __future__ import annotations

import time
from typing import Any

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths, SplitConfig
from sentiment_benchmark.data import download_and_prepare
from sentiment_benchmark.evaluation import dump_json, evaluate_model_results
from sentiment_benchmark.reporting import (
    create_data_evidence,
    create_evaluation_plots,
    write_portfolio_documents,
    write_run_manifest,
)
from sentiment_benchmark.splits import create_splits
from sentiment_benchmark.textcnn import train_textcnn
from sentiment_benchmark.traditional import train_tfidf_logistic
from sentiment_benchmark.transformer_model import train_transformer


def run_benchmark(
    paths: ProjectPaths | None = None,
    config: ExperimentConfig | None = None,
    split_config: SplitConfig | None = None,
    *,
    force_download: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute the frozen data and evaluation protocol end to end."""
    paths = paths or ProjectPaths.discover()
    config = config or ExperimentConfig()
    split_config = split_config or SplitConfig(random_state=config.seed)
    paths.ensure_directories()
    started = time.perf_counter()

    frame = download_and_prepare(
        paths.raw_archive,
        paths.prepared_data,
        force_download=force_download,
    )
    splits = create_splits(frame, split_config, assignments_path=paths.split_assignments)
    data_summary = create_data_evidence(frame, splits, paths)

    results = [
        train_tfidf_logistic(splits, paths, config),
        train_textcnn(splits, paths, config, device=device),
        train_transformer(splits, paths, config, device=device),
    ]
    evaluation = evaluate_model_results(results, splits, config, paths.reports_dir)
    create_evaluation_plots(evaluation, results, splits, paths)
    write_portfolio_documents(
        frame,
        splits,
        evaluation,
        results,
        paths,
        config,
        split_config,
    )
    manifest = write_run_manifest(
        frame,
        splits,
        evaluation,
        paths,
        config,
        split_config,
    )
    comparison = evaluation["comparison"]
    champion = comparison.loc[comparison["model"] == evaluation["champion_model"]].iloc[0]
    payload = {
        "champion_model": evaluation["champion_model"],
        "champion_test_macro_f1": float(champion["test_macro_f1"]),
        "champion_test_accuracy": float(champion["test_accuracy"]),
        "models_compared": [result.name for result in results],
        "prepared_rows": len(frame),
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "data_sha256": data_summary["data_sha256"],
        "manifest": str((paths.reports_dir / "run_manifest.json").relative_to(paths.root)),
        "comparison": str((paths.reports_dir / "model_comparison.csv").relative_to(paths.root)),
        "report": str((paths.reports_dir / "EXPERIMENT_REPORT.md").relative_to(paths.root)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "fast": config.fast,
        "transformer_revision": manifest["transformer"]["revision"],
    }
    dump_json(payload, paths.reports_dir / "benchmark_summary.json")
    return payload
