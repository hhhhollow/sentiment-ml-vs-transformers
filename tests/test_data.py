from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sentiment_benchmark.config import ID_COLUMN, SOURCE_COLUMN, TARGET_COLUMN, TEXT_COLUMN
from sentiment_benchmark.data import (
    DATASET_COLUMNS,
    DatasetIntegrityError,
    download_dataset,
    load_dataset,
    normalised_text_key,
    prepare_dataset,
    save_dataset,
    sha256_bytes,
    sha256_file,
    validate_dataset,
)


def _fixture_members(*, conflict: bool = False) -> dict[str, bytes]:
    yelp_duplicate_label = "0" if conflict else "1"
    return {
        "amazon": (
            "Great phone\t1\n"
            "Bad battery\t0\n"
            "Internal\ttab preserved\t1\n"
            "Vertical\vspace stays physical\t0\n"
            "ＣＬＥＡＮ   Text\t1\n"
        ).encode(),
        "imdb": (b"great   PHONE\t1\nAwful film\t0\nWonderful film\t1\n"),
        "yelp": (f"clean text\t{yelp_duplicate_label}\nTerrible food\t0\nTasty food\t1\n").encode(),
    }


def _write_fixture_archive(
    tmp_path: Path,
    *,
    members: dict[str, bytes] | None = None,
    filename: str = "fixture.zip",
) -> tuple[Path, dict[str, str]]:
    payloads = _fixture_members() if members is None else members
    archive_path = tmp_path / filename
    member_names = {
        "amazon": "nested/sentiment labelled sentences/amazon_cells_labelled.txt",
        "imdb": "nested/sentiment labelled sentences/imdb_labelled.txt",
        "yelp": "nested/sentiment labelled sentences/yelp_labelled.txt",
    }
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, payload in payloads.items():
            archive.writestr(member_names[source], payload)
        archive.writestr("nested/readme.txt", b"offline fixture only")
    return archive_path, {source: sha256_bytes(payload) for source, payload in payloads.items()}


def test_download_uses_injected_file_url_and_reuses_verified_existing_file(tmp_path: Path) -> None:
    source, _ = _write_fixture_archive(tmp_path)
    destination = tmp_path / "raw" / "dataset.zip"
    expected = sha256_file(source)

    result = download_dataset(
        destination,
        url=source.as_uri(),
        expected_sha256=expected,
        chunk_size=17,
    )

    assert result == destination
    assert destination.read_bytes() == source.read_bytes()
    reused = download_dataset(
        destination,
        url="file:///this/path/must/not/be/opened.zip",
        expected_sha256=expected,
    )
    assert reused == destination


def test_download_never_replaces_existing_file_until_forced_bytes_verify(tmp_path: Path) -> None:
    source, _ = _write_fixture_archive(tmp_path)
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"keep this existing file")

    with pytest.raises(DatasetIntegrityError, match="existing UCI archive SHA-256 mismatch"):
        download_dataset(
            destination,
            url=source.as_uri(),
            expected_sha256=sha256_file(source),
        )
    assert destination.read_bytes() == b"keep this existing file"

    with pytest.raises(DatasetIntegrityError, match="downloaded UCI archive SHA-256 mismatch"):
        download_dataset(
            destination,
            url=source.as_uri(),
            expected_sha256="0" * 64,
            force=True,
        )
    assert destination.read_bytes() == b"keep this existing file"

    download_dataset(
        destination,
        url=source.as_uri(),
        expected_sha256=sha256_file(source),
        force=True,
    )
    assert destination.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob(".dataset.zip.*.tmp"))


def test_prepare_dataset_parses_last_tab_and_globally_deduplicates(tmp_path: Path) -> None:
    archive_path, member_hashes = _write_fixture_archive(tmp_path)

    first = prepare_dataset(
        archive_path,
        expected_member_sha256=member_hashes,
        expected_raw_rows=11,
        expected_prepared_rows=9,
    )
    second = prepare_dataset(
        archive_path,
        expected_member_sha256=member_hashes,
        expected_raw_rows=11,
        expected_prepared_rows=9,
    )

    assert list(first.columns) == list(DATASET_COLUMNS)
    assert len(first) == 9
    assert first[ID_COLUMN].is_unique
    assert first[TEXT_COLUMN].map(normalised_text_key).is_unique
    assert "Internal\ttab preserved" in set(first[TEXT_COLUMN])
    assert "Vertical\vspace stays physical" in set(first[TEXT_COLUMN])
    assert "ＣＬＥＡＮ   Text" in set(first[TEXT_COLUMN])
    assert "great   PHONE" not in set(first[TEXT_COLUMN])
    assert "clean text" not in set(first[TEXT_COLUMN])
    pd.testing.assert_frame_equal(first, second)

    for source, group in first.groupby(SOURCE_COLUMN):
        assert source in {"amazon", "imdb", "yelp"}
        assert set(group[TARGET_COLUMN]) == {0, 1}


def test_save_and_load_prepared_dataset_round_trip(tmp_path: Path) -> None:
    archive_path, member_hashes = _write_fixture_archive(tmp_path)
    frame = prepare_dataset(
        archive_path,
        expected_member_sha256=member_hashes,
        expected_raw_rows=11,
        expected_prepared_rows=9,
    )
    output = save_dataset(frame, tmp_path / "processed" / "sentences.csv")

    loaded = load_dataset(output)

    pd.testing.assert_frame_equal(loaded, frame)


def test_prepare_rejects_member_hash_mismatch(tmp_path: Path) -> None:
    archive_path, member_hashes = _write_fixture_archive(tmp_path)
    member_hashes["imdb"] = "f" * 64

    with pytest.raises(DatasetIntegrityError, match="imdb member SHA-256 mismatch"):
        prepare_dataset(
            archive_path,
            expected_member_sha256=member_hashes,
            expected_raw_rows=None,
            expected_prepared_rows=None,
        )


def test_prepare_rejects_normalised_duplicates_with_conflicting_labels(tmp_path: Path) -> None:
    members = _fixture_members(conflict=True)
    archive_path, member_hashes = _write_fixture_archive(tmp_path, members=members)

    with pytest.raises(DatasetIntegrityError, match="conflicting labels"):
        prepare_dataset(
            archive_path,
            expected_member_sha256=member_hashes,
            expected_raw_rows=11,
            expected_prepared_rows=None,
        )


@pytest.mark.parametrize(
    ("invalid_line", "message"),
    [
        ("\t1\n", "empty review text"),
        ("not labelled\n", "no tab-delimited label"),
        ("review\t2\n", "invalid binary label"),
        ("\n", "is empty"),
    ],
)
def test_prepare_rejects_malformed_physical_rows(
    tmp_path: Path,
    invalid_line: str,
    message: str,
) -> None:
    members = _fixture_members()
    members["amazon"] = invalid_line.encode()
    archive_path, member_hashes = _write_fixture_archive(tmp_path, members=members)

    with pytest.raises(DatasetIntegrityError, match=message):
        prepare_dataset(
            archive_path,
            expected_member_sha256=member_hashes,
            expected_raw_rows=None,
            expected_prepared_rows=None,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(sentence_id="duplicate"), "IDs must be unique"),
        (
            lambda frame: frame.assign(text=frame[TEXT_COLUMN].mask(frame.index == 0, "   ")),
            "non-empty string",
        ),
        (lambda frame: frame.assign(label=1), "both binary classes"),
        (
            lambda frame: frame.assign(
                source=frame[SOURCE_COLUMN].mask(frame[SOURCE_COLUMN] == "yelp", "imdb")
            ),
            "source must contain exactly",
        ),
        (
            lambda frame: frame.assign(
                text=frame[TEXT_COLUMN].mask(frame.index == 1, frame.loc[0, TEXT_COLUMN].upper())
            ),
            "normalised review text must be globally unique",
        ),
    ],
)
def test_validate_dataset_rejects_contract_violations(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    archive_path, member_hashes = _write_fixture_archive(tmp_path)
    frame = prepare_dataset(
        archive_path,
        expected_member_sha256=member_hashes,
        expected_raw_rows=11,
        expected_prepared_rows=9,
    )

    invalid = mutation(frame)  # type: ignore[operator]
    with pytest.raises(DatasetIntegrityError, match=message):
        validate_dataset(invalid)
