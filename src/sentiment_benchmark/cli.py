"""Command line interface for data provenance and the complete benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentiment_benchmark.config import ExperimentConfig, ProjectPaths
from sentiment_benchmark.data import (
    download_dataset,
    prepare_dataset,
    save_dataset,
    sha256_file,
)
from sentiment_benchmark.predict import MODEL_CHOICES


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        prog="sentiment-benchmark",
        description="Compare TF-IDF, a PyTorch TextCNN, and pinned DistilBERT fairly.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download and verify the pinned UCI archive")
    download.add_argument("--output", type=Path, default=paths.raw_archive)
    download.add_argument("--force", action="store_true")

    prepare = commands.add_parser("prepare", help="parse, de-duplicate, validate, and save data")
    prepare.add_argument("--archive", type=Path, default=paths.raw_archive)
    prepare.add_argument("--output", type=Path, default=paths.prepared_data)

    run_all = commands.add_parser("run-all", help="run all three models and generate evidence")
    run_all.add_argument("--fast", action="store_true", help="short smoke-test epochs/intervals")
    run_all.add_argument("--force-download", action="store_true")
    run_all.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="override automatic PyTorch device selection",
    )

    predict = commands.add_parser("predict", help="score a text CSV with one saved artifact")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=paths.predictions)
    predict.add_argument("--model", choices=MODEL_CHOICES, default="distilbert")
    predict.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.discover()
    paths.ensure_directories()
    if args.command == "download":
        output = download_dataset(args.output, force=args.force)
        payload = {"output": str(output), "sha256": sha256_file(output)}
    elif args.command == "prepare":
        frame = prepare_dataset(args.archive)
        save_dataset(frame, args.output)
        payload = {
            "output": str(args.output),
            "rows": len(frame),
            "positive_rate": float(frame["label"].mean()),
        }
    elif args.command == "run-all":
        from sentiment_benchmark.experiment import run_benchmark

        payload = run_benchmark(
            paths,
            ExperimentConfig(fast=args.fast),
            force_download=args.force_download,
            device=args.device,
        )
    else:
        from sentiment_benchmark.predict import predict_csv

        output = predict_csv(
            args.input,
            args.output,
            args.model,
            paths,
            device=args.device,
        )
        payload = {
            "output": str(args.output),
            "rows": len(output),
            "model": args.model,
            "positive_rate": float(output["predicted_label"].mean()),
        }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
