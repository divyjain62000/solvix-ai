
import pytest

from pathlib import Path

from backend.core.ingestion.validator import CSVValidator
from backend.core.ingestion.exceptions import (
    IngestionError,
    EmptyFileError,
    FileSizeLimitExceededError,
    InvalidCSVError,
    CSVEncodingError,
    CSVValidationError,
    UnsupportedFileFormatError,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def create_csv(directory, filename, content, encoding="utf-8"):
    file_path = directory / filename
    file_path.write_text(content, encoding=encoding)
    return file_path


def test_valid_csv(temp_dir):
    file = create_csv(
        temp_dir,
        "valid.csv",
        "Name,Age,Salary\nJohn,25,50000\nDavid,30,60000\n",
    )

    result = CSVValidator.validate(file)

    assert result["headers"] == ["Name", "Age", "Salary"]
    assert result["row_count"] == 2
    assert result["column_count"] == 3
    assert result["file_size_bytes"] > 0


def test_empty_file(temp_dir):
    file = create_csv(temp_dir, "empty.csv", "")

    with pytest.raises(EmptyFileError):
        CSVValidator.validate(file)


def test_duplicate_headers(temp_dir):
    file = create_csv(
        temp_dir,
        "duplicate.csv",
        "Name,Age,Name\nJohn,25,John\n",
    )

    with pytest.raises(CSVValidationError):
        CSVValidator.validate(file)


def test_empty_column_name(temp_dir):
    file = create_csv(
        temp_dir,
        "empty_header.csv",
        "Name,,Salary\nJohn,25,50000\n",
    )

    with pytest.raises(CSVValidationError):
        CSVValidator.validate(file)


def test_inconsistent_columns(temp_dir):
    file = create_csv(
        temp_dir,
        "invalid_columns.csv",
        "Name,Age,Salary\nJohn,25,50000\nDavid,30\n",
    )

    with pytest.raises(CSVValidationError):
        CSVValidator.validate(file)


def test_wrong_extension(temp_dir):
    file = create_csv(
        temp_dir,
        "data.txt",
        "Name,Age\nJohn,25\n",
    )

    with pytest.raises(UnsupportedFileFormatError):
        CSVValidator.validate(file)


def test_file_size_limit(temp_dir):
    file = create_csv(
        temp_dir,
        "large.csv",
        "Name,Age\nJohn,25\n",
    )

    with pytest.raises(FileSizeLimitExceededError):
        CSVValidator.validate(file, max_file_size_mb=0)


def test_utf8_bom(temp_dir):
    file = create_csv(
        temp_dir,
        "bom.csv",
        "Name,Age\nJohn,25\n",
        encoding="utf-8-sig",
    )

    result = CSVValidator.validate(file)

    assert result["headers"] == ["Name", "Age"]


def test_invalid_encoding(temp_dir):
    file = temp_dir / "invalid_encoding.csv"
    file.write_bytes(b"Name,Age\nJohn,\xff\n")

    with pytest.raises(CSVEncodingError):
        CSVValidator.validate(file)


def test_headers_without_records(temp_dir):
    file = create_csv(
        temp_dir,
        "headers_only.csv",
        "Name,Age,Salary\n",
    )

    with pytest.raises(CSVValidationError):
        CSVValidator.validate(file)


def test_missing_file(temp_dir):
    file = temp_dir / "missing.csv"

    with pytest.raises(IngestionError):
        CSVValidator.validate(file)


def test_case_insensitive_extension(temp_dir):
    file = create_csv(
        temp_dir,
        "data.CSV",
        "Name,Age\nJohn,25\n",
    )

    result = CSVValidator.validate(file)

    assert result["row_count"] == 1
