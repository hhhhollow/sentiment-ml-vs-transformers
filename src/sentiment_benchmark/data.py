"""Pinned UCI download, parsing, de-duplication, and dataset validation."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pandas as pd
from pandas.api.types import is_integer_dtype

from sentiment_benchmark.config import (
    ID_COLUMN,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    UCI_DATASET_SHA256,
    UCI_DATASET_URL,
    UCI_MEMBER_SHA256,
)

UCI_RAW_ROWS = 3_000
UCI_PREPARED_ROWS = 2_979
UCI_DUPLICATE_ROWS = UCI_RAW_ROWS - UCI_PREPARED_ROWS

SOURCE_ORDER = ("amazon", "imdb", "yelp")
UCI_MEMBER_BASENAMES = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}
DATASET_COLUMNS = (ID_COLUMN, SOURCE_COLUMN, TEXT_COLUMN, TARGET_COLUMN)


class DatasetIntegrityError(ValueError):
    """Raised when pinned bytes or the prepared dataset violate their contract."""


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for in-memory bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading the archive into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_expected_hash(value: str, *, artifact: str) -> str:
    expected = value.casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{artifact} expected SHA-256 must contain 64 hexadecimal characters")
    return expected


def _verify_file(path: Path, expected_sha256: str, *, artifact: str) -> None:
    expected = _normalise_expected_hash(expected_sha256, artifact=artifact)
    actual = sha256_file(path)
    if actual != expected:
        raise DatasetIntegrityError(
            f"{artifact} SHA-256 mismatch: expected {expected}, observed {actual}"
        )


def download_dataset(
    destination: Path,
    *,
    url: str = UCI_DATASET_URL,
    expected_sha256: str = UCI_DATASET_SHA256,
    force: bool = False,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download the pinned UCI archive after verifying bytes in a temporary file.

    An existing destination is verified and reused by default. It is never replaced unless
    ``force=True`` is explicit, and even then replacement happens atomically only after the
    newly downloaded bytes pass their checksum.
    """

    destination = Path(destination)
    expected = _normalise_expected_hash(expected_sha256, artifact="UCI archive")
    if destination.exists() and not force:
        if not destination.is_file():
            raise DatasetIntegrityError(f"download destination is not a file: {destination}")
        _verify_file(destination, expected, artifact="existing UCI archive")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "sentiment-ml-vs-transformers/0.1"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                while chunk := response.read(chunk_size):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        _verify_file(temporary_path, expected, artifact="downloaded UCI archive")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def normalised_text_key(text: str) -> str:
    """Canonical key used only for global duplicate and leakage detection."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _stable_sentence_id(text_key: str) -> str:
    return f"sentence_{hashlib.sha256(text_key.encode('utf-8')).hexdigest()}"


def _resolve_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    by_basename: dict[str, list[zipfile.ZipInfo]] = {}
    for member in archive.infolist():
        if member.is_dir():
            continue
        by_basename.setdefault(PurePosixPath(member.filename).name, []).append(member)

    resolved: dict[str, zipfile.ZipInfo] = {}
    for source in SOURCE_ORDER:
        basename = UCI_MEMBER_BASENAMES[source]
        matches = by_basename.get(basename, [])
        if len(matches) != 1:
            raise DatasetIntegrityError(
                f"archive must contain exactly one {basename!r}; found {len(matches)}"
            )
        resolved[source] = matches[0]
    return resolved


def _parse_member(payload: bytes, source: str) -> list[dict[str, object]]:
    try:
        physical_lines = payload.decode("utf-8-sig").split("\n")
    except UnicodeDecodeError as error:
        raise DatasetIntegrityError(f"{source} member is not valid UTF-8") from error

    rows: list[dict[str, object]] = []
    for line_number, physical_line in enumerate(physical_lines, start=1):
        line = physical_line[:-1] if physical_line.endswith("\r") else physical_line
        if not line:
            if line_number == len(physical_lines):
                continue
            raise DatasetIntegrityError(f"{source} line {line_number} is empty")
        if "\t" not in line:
            raise DatasetIntegrityError(f"{source} line {line_number} has no tab-delimited label")

        text, raw_label = line.rsplit("\t", maxsplit=1)
        text = text.strip()
        if not text:
            raise DatasetIntegrityError(f"{source} line {line_number} has empty review text")
        if raw_label not in {"0", "1"}:
            raise DatasetIntegrityError(
                f"{source} line {line_number} has invalid binary label {raw_label!r}"
            )
        rows.append(
            {
                SOURCE_COLUMN: source,
                TEXT_COLUMN: text,
                TARGET_COLUMN: int(raw_label),
                "_source_order": SOURCE_ORDER.index(source),
                "_line_number": line_number,
                "_text_key": normalised_text_key(text),
            }
        )
    return rows


def prepare_dataset(
    archive_path: Path,
    *,
    expected_member_sha256: Mapping[str, str] = UCI_MEMBER_SHA256,
    expected_raw_rows: int | None = UCI_RAW_ROWS,
    expected_prepared_rows: int | None = UCI_PREPARED_ROWS,
) -> pd.DataFrame:
    """Parse and globally de-duplicate the three pinned UCI review sources.

    Lines are separated using the physical ``\n`` byte representation after UTF-8 decoding,
    then split at the final tab so tabs inside a review remain part of its text. Duplicate
    comparison uses NFKC, case-folding, and whitespace collapsing; the original cleaned text
    from the first deterministic occurrence is retained.
    """

    expected_sources = set(SOURCE_ORDER)
    if set(expected_member_sha256) != expected_sources:
        raise ValueError(f"member checksum mapping must contain exactly {sorted(expected_sources)}")

    rows: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _resolve_members(archive)
            for source in SOURCE_ORDER:
                payload = archive.read(members[source])
                expected = _normalise_expected_hash(
                    expected_member_sha256[source], artifact=f"{source} member"
                )
                actual = sha256_bytes(payload)
                if actual != expected:
                    raise DatasetIntegrityError(
                        f"{source} member SHA-256 mismatch: expected {expected}, observed {actual}"
                    )
                rows.extend(_parse_member(payload, source))
    except zipfile.BadZipFile as error:
        raise DatasetIntegrityError(f"invalid UCI zip archive: {archive_path}") from error

    raw = pd.DataFrame.from_records(rows)
    if raw.empty:
        raise DatasetIntegrityError("UCI archive produced no review rows")
    if expected_raw_rows is not None and len(raw) != expected_raw_rows:
        raise DatasetIntegrityError(
            f"unexpected raw row count: expected {expected_raw_rows}, observed {len(raw)}"
        )

    conflicting = raw.groupby("_text_key", sort=False)[TARGET_COLUMN].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        example = str(conflicting.index[0])
        raise DatasetIntegrityError(
            f"normalised duplicate reviews have conflicting labels; first key: {example!r}"
        )

    prepared = (
        raw.sort_values(["_source_order", "_line_number"], kind="stable")
        .drop_duplicates("_text_key", keep="first")
        .copy()
    )
    if expected_prepared_rows is not None and len(prepared) != expected_prepared_rows:
        removed = len(raw) - len(prepared)
        raise DatasetIntegrityError(
            "unexpected prepared row count: "
            f"expected {expected_prepared_rows}, observed {len(prepared)} "
            f"({removed} duplicates removed)"
        )

    prepared[ID_COLUMN] = prepared["_text_key"].map(_stable_sentence_id)
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype("int64")
    prepared = prepared.loc[:, list(DATASET_COLUMNS)].reset_index(drop=True)
    validate_dataset(prepared)
    return prepared


def validate_dataset(frame: pd.DataFrame) -> None:
    """Enforce the complete prepared-data contract without mutating the frame."""

    if list(frame.columns) != list(DATASET_COLUMNS):
        raise DatasetIntegrityError(
            "dataset columns must be exactly "
            f"{list(DATASET_COLUMNS)}, observed {list(frame.columns)}"
        )
    if frame.empty:
        raise DatasetIntegrityError("dataset is empty")
    if frame.isna().any().any():
        raise DatasetIntegrityError("dataset contains missing values")

    if not frame[ID_COLUMN].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise DatasetIntegrityError("sentence IDs must be non-empty strings")
    if frame[ID_COLUMN].duplicated().any():
        raise DatasetIntegrityError("sentence IDs must be unique")

    if (
        not frame[TEXT_COLUMN]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise DatasetIntegrityError("review text must be a non-empty string")

    if not is_integer_dtype(frame[TARGET_COLUMN].dtype):
        raise DatasetIntegrityError("label must use an integer dtype")
    if set(frame[TARGET_COLUMN].unique().tolist()) != {0, 1}:
        raise DatasetIntegrityError("label must contain both binary classes 0 and 1")

    sources = set(frame[SOURCE_COLUMN].tolist())
    if sources != set(SOURCE_ORDER):
        raise DatasetIntegrityError(
            f"source must contain exactly {list(SOURCE_ORDER)}, observed {sorted(sources)}"
        )
    for source, group in frame.groupby(SOURCE_COLUMN, sort=False):
        if set(group[TARGET_COLUMN].tolist()) != {0, 1}:
            raise DatasetIntegrityError(f"source {source!r} must contain both binary classes")

    text_keys = frame[TEXT_COLUMN].map(normalised_text_key)
    if (text_keys == "").any():
        raise DatasetIntegrityError("normalised review text must not be empty")
    if text_keys.duplicated().any():
        raise DatasetIntegrityError("normalised review text must be globally unique")


def save_dataset(frame: pd.DataFrame, path: Path) -> Path:
    """Validate and atomically persist the prepared dataset as UTF-8 CSV."""

    validate_dataset(frame)
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
            frame.to_csv(temporary, index=False, lineterminator="\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a previously prepared CSV and re-run every integrity check."""

    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame


def download_and_prepare(
    archive_path: Path,
    prepared_path: Path,
    *,
    url: str = UCI_DATASET_URL,
    archive_sha256: str = UCI_DATASET_SHA256,
    member_sha256: Mapping[str, str] = UCI_MEMBER_SHA256,
    force_download: bool = False,
    expected_raw_rows: int | None = UCI_RAW_ROWS,
    expected_prepared_rows: int | None = UCI_PREPARED_ROWS,
) -> pd.DataFrame:
    """Materialise the pinned archive and its validated prepared representation."""

    archive = download_dataset(
        archive_path,
        url=url,
        expected_sha256=archive_sha256,
        force=force_download,
    )
    frame = prepare_dataset(
        archive,
        expected_member_sha256=member_sha256,
        expected_raw_rows=expected_raw_rows,
        expected_prepared_rows=expected_prepared_rows,
    )
    save_dataset(frame, prepared_path)
    return frame
