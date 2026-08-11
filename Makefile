.DEFAULT_GOAL := help
UV_CACHE_DIR ?= .uv-cache
HF_HOME ?= artifacts/huggingface
MPLCONFIGDIR ?= .mpl-cache

.PHONY: help setup data benchmark benchmark-fast predict test lint check

help:
	@echo "setup          Install the locked environment"
	@echo "data           Download, verify, and prepare UCI data"
	@echo "benchmark      Run the formal three-model benchmark"
	@echo "benchmark-fast Run a short end-to-end integration benchmark"
	@echo "predict        Score examples/inference_sample.csv with DistilBERT"
	@echo "test           Run the offline test suite with coverage"
	@echo "lint           Check Ruff lint and formatting"
	@echo "check          Run lock, lint, and tests"

setup:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --frozen --all-groups

data:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen sentiment-benchmark download
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen sentiment-benchmark prepare

benchmark:
	UV_CACHE_DIR=$(UV_CACHE_DIR) HF_HOME=$(HF_HOME) MPLCONFIGDIR=$(MPLCONFIGDIR) \
		TOKENIZERS_PARALLELISM=false uv run --frozen sentiment-benchmark run-all --device cpu

benchmark-fast:
	UV_CACHE_DIR=$(UV_CACHE_DIR) HF_HOME=$(HF_HOME) MPLCONFIGDIR=$(MPLCONFIGDIR) \
		TOKENIZERS_PARALLELISM=false uv run --frozen sentiment-benchmark run-all --fast --device cpu

predict:
	UV_CACHE_DIR=$(UV_CACHE_DIR) HF_HOME=$(HF_HOME) uv run --frozen sentiment-benchmark predict \
		--input examples/inference_sample.csv --model distilbert --device cpu

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) MPLCONFIGDIR=$(MPLCONFIGDIR) uv run --frozen pytest \
		--cov=sentiment_benchmark --cov-report=term-missing --cov-fail-under=80

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen ruff check .
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --frozen ruff format --check .

check:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv lock --check
	$(MAKE) lint
	$(MAKE) test
