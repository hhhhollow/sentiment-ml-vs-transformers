"""One deterministic source-and-label-stratified split shared by every model."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_benchmark.config import (
    ID_COLUMN,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    SplitConfig,
)
from sentiment_benchmark.contracts import DatasetSplits
from sentiment_benchmark.data import (
    normalised_text_key,
    validate_dataset,
)

SPLIT_COLUMN = "split"
SPLIT_NAMES = ("train", "validation", "test")
ASSIGNMENT_COLUMNS = (ID_COLUMN, SPLIT_COLUMN)


class SplitIntegrityError(ValueError):
    """Raised when saved assignments or split isolation violate the contract."""


def _split_proportions(config: SplitConfig) -> dict[str, float]:
    validation = float(config.validation_size)
    test = float(config.test_size)
    train = 1.0 - validation - test
    proportions = {"train": train, "validation": validation, "test": test}
    if not all(math.isfinite(value) and value > 0.0 for value in proportions.values()):
        raise ValueError("train, validation, and test proportions must all be positive and finite")
    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split proportions must sum to one")
    return proportions


def _strata(frame: pd.DataFrame) -> pd.Series:
    return frame[SOURCE_COLUMN].astype(str) + "\0" + frame[TARGET_COLUMN].astype(str)


def create_split_assignments(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Create order-independent deterministic assignments within each source-label stratum."""

    validate_dataset(frame)
    config = SplitConfig() if config is None else config
    proportions = _split_proportions(config)
    stratum_sizes = frame.groupby([SOURCE_COLUMN, TARGET_COLUMN], sort=True).size()
    if (stratum_sizes < len(SPLIT_NAMES)).any():
        raise SplitIntegrityError(
            "each source-label stratum needs at least three rows so every split is represented"
        )

    ordered = frame.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
    try:
        train_validation, test = train_test_split(
            ordered,
            test_size=proportions["test"],
            random_state=config.random_state,
            shuffle=True,
            stratify=_strata(ordered),
        )
        relative_validation_size = proportions["validation"] / (
            proportions["train"] + proportions["validation"]
        )
        train, validation = train_test_split(
            train_validation,
            test_size=relative_validation_size,
            random_state=config.random_state,
            shuffle=True,
            stratify=_strata(train_validation),
        )
    except ValueError as error:
        raise SplitIntegrityError(
            f"unable to create source-label stratified splits: {error}"
        ) from error

    pieces: list[pd.DataFrame] = []
    for split_name, split_frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        piece = split_frame.loc[:, [ID_COLUMN]].copy()
        piece[SPLIT_COLUMN] = split_name
        pieces.append(piece)
    assignments = pd.concat(pieces, ignore_index=True).sort_values(ID_COLUMN, kind="stable")
    assignments = assignments.reset_index(drop=True)
    validate_split_assignments(assignments, expected_ids=set(frame[ID_COLUMN]))
    return assignments


def validate_split_assignments(
    assignments: pd.DataFrame,
    *,
    expected_ids: set[str] | None = None,
) -> None:
    """Validate the persisted ID-to-split mapping."""

    if list(assignments.columns) != list(ASSIGNMENT_COLUMNS):
        raise SplitIntegrityError(
            "assignment columns must be exactly "
            f"{list(ASSIGNMENT_COLUMNS)}, observed {list(assignments.columns)}"
        )
    if assignments.empty:
        raise SplitIntegrityError("split assignments are empty")
    if assignments.isna().any().any():
        raise SplitIntegrityError("split assignments contain missing values")
    if (
        not assignments[ID_COLUMN]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise SplitIntegrityError("assignment IDs must be non-empty strings")
    if assignments[ID_COLUMN].duplicated().any():
        raise SplitIntegrityError("each sentence ID must have exactly one split assignment")
    observed_splits = set(assignments[SPLIT_COLUMN].tolist())
    if observed_splits != set(SPLIT_NAMES):
        raise SplitIntegrityError(
            "assignments must contain exactly "
            f"{list(SPLIT_NAMES)}, observed {sorted(observed_splits)}"
        )
    if expected_ids is not None and set(assignments[ID_COLUMN]) != expected_ids:
        missing = expected_ids - set(assignments[ID_COLUMN])
        extra = set(assignments[ID_COLUMN]) - expected_ids
        raise SplitIntegrityError(
            f"assignment IDs do not match dataset IDs: {len(missing)} missing, {len(extra)} extra"
        )


def apply_split_assignments(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
) -> DatasetSplits:
    """Materialise the three immutable frames from one validated assignment table."""

    validate_dataset(frame)
    validate_split_assignments(assignments, expected_ids=set(frame[ID_COLUMN]))
    assigned = frame.merge(assignments, on=ID_COLUMN, how="left", sort=False, validate="one_to_one")

    def select(name: str) -> pd.DataFrame:
        selected = assigned.loc[assigned[SPLIT_COLUMN] == name, frame.columns].copy()
        selected = selected.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)
        validate_dataset(selected)
        return selected

    splits = DatasetSplits(
        train=select("train"),
        validation=select("validation"),
        test=select("test"),
    )
    assert_no_split_leakage(splits)
    return splits


def assert_no_split_leakage(splits: DatasetSplits) -> None:
    """Assert that neither IDs nor normalised texts cross any split boundary."""

    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    for frame in frames.values():
        validate_dataset(frame)

    id_sets = {name: set(frame[ID_COLUMN]) for name, frame in frames.items()}
    text_key_sets = {
        name: set(frame[TEXT_COLUMN].map(normalised_text_key)) for name, frame in frames.items()
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            if id_sets[left_name] & id_sets[right_name]:
                raise SplitIntegrityError(
                    f"sentence IDs overlap across {left_name} and {right_name}"
                )
            if text_key_sets[left_name] & text_key_sets[right_name]:
                raise SplitIntegrityError(
                    f"normalised review text overlaps across {left_name} and {right_name}"
                )


def save_split_assignments(assignments: pd.DataFrame, path: Path) -> Path:
    """Atomically save the reusable mapping that every model consumes."""

    validate_split_assignments(assignments)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            assignments.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_split_assignments(path: Path) -> pd.DataFrame:
    """Load and validate a previously persisted split mapping."""

    assignments = pd.read_csv(path, dtype={ID_COLUMN: "string", SPLIT_COLUMN: "string"})
    assignments[ID_COLUMN] = assignments[ID_COLUMN].astype(str)
    assignments[SPLIT_COLUMN] = assignments[SPLIT_COLUMN].astype(str)
    validate_split_assignments(assignments)
    return assignments


def create_splits(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    assignments_path: Path | None = None,
) -> DatasetSplits:
    """Create one shared 60/20/20 split and optionally persist its assignments."""

    assignments = create_split_assignments(frame, config)
    splits = apply_split_assignments(frame, assignments)
    if assignments_path is not None:
        save_split_assignments(assignments, assignments_path)
    return splits


def load_splits(frame: pd.DataFrame, assignments_path: Path) -> DatasetSplits:
    """Recreate model inputs from the saved experiment assignment contract."""

    return apply_split_assignments(frame, load_split_assignments(assignments_path))
