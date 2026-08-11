"""Immutable experiment configuration and project paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

RANDOM_STATE = 42
ID_COLUMN = "sentence_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
SOURCE_COLUMN = "source"

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
)
UCI_DATASET_SHA256 = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
UCI_DATASET_DOI = "10.24432/C57604"
UCI_DATASET_LICENSE = "CC BY 4.0"
UCI_MEMBER_SHA256 = {
    "amazon": "47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3",
    "imdb": "aef2e49e3da25714d61175e3a6e68eeef74a20a2f914318dc3be9947ea86512d",
    "yelp": "c76468b7b5c6e56a0804d728345c5f84aa2142ddb214420f61cc9cfd4c00d2ea",
}

TRANSFORMER_MODEL = "distilbert/distilbert-base-uncased"
TRANSFORMER_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
TRANSFORMER_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class SplitConfig:
    """One shared, untouched split contract for every model family."""

    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = RANDOM_STATE


@dataclass(frozen=True)
class ExperimentConfig:
    """Training settings; fast mode only shortens smoke-test work."""

    seed: int = RANDOM_STATE
    textcnn_epochs: int = 18
    textcnn_patience: int = 4
    textcnn_batch_size: int = 64
    textcnn_learning_rate: float = 1e-3
    textcnn_max_tokens: int = 96
    textcnn_vocab_size: int = 12_000
    transformer_epochs: int = 3
    transformer_patience: int = 2
    transformer_batch_size: int = 16
    transformer_learning_rate: float = 2e-5
    transformer_max_tokens: int = 96
    bootstrap_iterations: int = 1_000
    latency_repeats: int = 7
    fast: bool = False

    @property
    def effective_textcnn_epochs(self) -> int:
        return 2 if self.fast else self.textcnn_epochs

    @property
    def effective_transformer_epochs(self) -> int:
        return 1 if self.fast else self.transformer_epochs

    @property
    def effective_bootstrap_iterations(self) -> int:
        return 50 if self.fast else self.bootstrap_iterations

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    """All local inputs, generated evidence, and non-versioned weights."""

    root: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        return cls(Path(__file__).resolve().parents[2])

    @property
    def raw_archive(self) -> Path:
        return self.root / "data" / "raw" / "sentiment_labelled_sentences.zip"

    @property
    def prepared_data(self) -> Path:
        return self.root / "data" / "processed" / "sentences.csv"

    @property
    def split_assignments(self) -> Path:
        return self.root / "data" / "processed" / "split_assignments.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "data" / "predictions" / "sentiment_predictions.csv"

    @property
    def hf_cache(self) -> Path:
        return self.root / "artifacts" / "huggingface"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_archive.parent,
            self.prepared_data.parent,
            self.predictions.parent,
            self.models_dir,
            self.hf_cache,
            self.reports_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
