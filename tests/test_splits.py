from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.config import (
    ID_COLUMN,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    SplitConfig,
)
from sentiment_benchmark.contracts import DatasetSplits
from sentiment_benchmark.data import DATASET_COLUMNS, normalised_text_key, validate_dataset
from sentiment_benchmark.splits import (
    SPLIT_COLUMN,
    SplitIntegrityError,
    apply_split_assignments,
    assert_no_split_leakage,
    create_split_assignments,
    create_splits,
    load_split_assignments,
    load_splits,
    save_split_assignments,
    validate_split_assignments,
)


def _balanced_frame(rows_per_stratum: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in ("amazon", "imdb", "yelp"):
        for label in (0, 1):
            for number in range(rows_per_stratum):
                sentence_id = f"{source}-{label}-{number:03d}"
                rows.append(
                    {
                        ID_COLUMN: sentence_id,
                        SOURCE_COLUMN: source,
                        TEXT_COLUMN: f"Unique review {sentence_id}",
                        TARGET_COLUMN: label,
                    }
                )
    frame = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    validate_dataset(frame)
    return frame


def _official_shape_frame() -> pd.DataFrame:
    counts = {
        ("amazon", 0): 496,
        ("amazon", 1): 491,
        ("imdb", 0): 498,
        ("imdb", 1): 498,
        ("yelp", 0): 497,
        ("yelp", 1): 499,
    }
    rows: list[dict[str, object]] = []
    for (source, label), count in counts.items():
        for number in range(count):
            sentence_id = f"{source}-{label}-{number:03d}"
            rows.append(
                {
                    ID_COLUMN: sentence_id,
                    SOURCE_COLUMN: source,
                    TEXT_COLUMN: f"Official-shape review {sentence_id}",
                    TARGET_COLUMN: label,
                }
            )
    frame = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    validate_dataset(frame)
    return frame


def _assert_same_splits(left: DatasetSplits, right: DatasetSplits) -> None:
    pd.testing.assert_frame_equal(left.train, right.train)
    pd.testing.assert_frame_equal(left.validation, right.validation)
    pd.testing.assert_frame_equal(left.test, right.test)


def test_assignments_are_deterministic_order_independent_and_stratified() -> None:
    frame = _balanced_frame()
    first = create_split_assignments(frame)
    repeated = create_split_assignments(frame.sample(frac=1.0, random_state=91))

    pd.testing.assert_frame_equal(first, repeated)
    joined = frame.merge(first, on=ID_COLUMN, validate="one_to_one")
    counts = joined.groupby([SOURCE_COLUMN, TARGET_COLUMN, SPLIT_COLUMN]).size()
    for source in ("amazon", "imdb", "yelp"):
        for label in (0, 1):
            assert counts[source, label, "train"] == 6
            assert counts[source, label, "validation"] == 2
            assert counts[source, label, "test"] == 2

    changed_seed = create_split_assignments(frame, SplitConfig(random_state=7))
    assert not first.equals(changed_seed)


def test_create_splits_has_expected_sizes_and_no_id_or_text_overlap(tmp_path: Path) -> None:
    frame = _balanced_frame()
    assignment_path = tmp_path / "split_assignments.csv"

    splits = create_splits(frame, assignments_path=assignment_path)

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (36, 12, 12)
    assert assignment_path.is_file()
    assert_no_split_leakage(splits)

    ids = [set(part[ID_COLUMN]) for part in (splits.train, splits.validation, splits.test)]
    keys = [
        set(part[TEXT_COLUMN].map(normalised_text_key))
        for part in (splits.train, splits.validation, splits.test)
    ]
    assert ids[0].isdisjoint(ids[1] | ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert keys[0].isdisjoint(keys[1] | keys[2])
    assert keys[1].isdisjoint(keys[2])


def test_official_prepared_shape_has_exact_global_60_20_20_counts() -> None:
    splits = create_splits(_official_shape_frame())

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (1787, 596, 596)


def test_saved_assignments_recreate_identical_shared_model_inputs(tmp_path: Path) -> None:
    frame = _balanced_frame()
    assignments = create_split_assignments(frame)
    path = save_split_assignments(assignments, tmp_path / "nested" / "assignments.csv")

    loaded_assignments = load_split_assignments(path)
    direct = apply_split_assignments(frame, assignments)
    restored = load_splits(frame.sample(frac=1.0, random_state=19), path)

    pd.testing.assert_frame_equal(loaded_assignments, assignments)
    _assert_same_splits(direct, restored)


def test_assert_no_split_leakage_detects_normalised_text_crossing_boundary() -> None:
    splits = create_splits(_balanced_frame())
    validation = splits.validation.copy()
    validation.loc[0, TEXT_COLUMN] = splits.train.loc[0, TEXT_COLUMN].swapcase()
    contaminated = DatasetSplits(splits.train, validation, splits.test)

    with pytest.raises(SplitIntegrityError, match="normalised review text overlaps"):
        assert_no_split_leakage(contaminated)


def test_assert_no_split_leakage_detects_id_crossing_boundary() -> None:
    splits = create_splits(_balanced_frame())
    validation = splits.validation.copy()
    validation.loc[0, ID_COLUMN] = splits.train.loc[0, ID_COLUMN]
    contaminated = DatasetSplits(splits.train, validation, splits.test)

    with pytest.raises(SplitIntegrityError, match="sentence IDs overlap"):
        assert_no_split_leakage(contaminated)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda assignments: assignments.iloc[:-1].copy(),
        lambda assignments: pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True),
        lambda assignments: assignments.assign(split="train"),
    ],
)
def test_apply_rejects_incomplete_duplicate_or_invalid_assignments(mutation: object) -> None:
    frame = _balanced_frame()
    assignments = create_split_assignments(frame)
    invalid = mutation(assignments)  # type: ignore[operator]

    with pytest.raises(SplitIntegrityError):
        apply_split_assignments(frame, invalid)


def test_validate_assignments_rejects_unexpected_columns() -> None:
    assignments = create_split_assignments(_balanced_frame()).assign(extra="not allowed")

    with pytest.raises(SplitIntegrityError, match="assignment columns must be exactly"):
        validate_split_assignments(assignments)


def test_tiny_source_label_stratum_is_rejected() -> None:
    frame = _balanced_frame(rows_per_stratum=3)
    drop_ids = {"amazon-0-001", "amazon-0-002"}
    frame = frame.loc[~frame[ID_COLUMN].isin(drop_ids)].reset_index(drop=True)
    validate_dataset(frame)

    with pytest.raises(SplitIntegrityError, match="at least three rows"):
        create_split_assignments(frame)


@pytest.mark.parametrize(
    "config",
    [
        SplitConfig(test_size=0.5, validation_size=0.5),
        SplitConfig(test_size=-0.1, validation_size=0.2),
        SplitConfig(test_size=float("nan"), validation_size=0.2),
    ],
)
def test_invalid_split_proportions_are_rejected(config: SplitConfig) -> None:
    with pytest.raises(ValueError, match="proportions"):
        create_split_assignments(_balanced_frame(), config)
