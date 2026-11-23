
import csv
from pathlib import Path

from .exceptions import (
    EmptyFileError,
    FileSizeLimitExceededError,
    InvalidCSVError,
    CSVEncodingError,
    CSVValidationError,
    UnsupportedFileFormatError,
    IngestionError,
)


class CSVValidator:
    """Validates CSV files without loading the complete dataset into memory."""

    @staticmethod
    def validate_extension(file_path: Path) -> None:
        if Path(file_path).suffix.lower() != ".csv":
            raise UnsupportedFileFormatError(
                f"Unsupported file format: {Path(file_path).suffix}. "
                "Only CSV files are supported."
            )

    @staticmethod
    def validate_file_exists(file_path: Path) -> None:
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            raise IngestionError(f"File does not exist: {path}")

    @staticmethod
    def validate_file_not_empty(file_path: Path) -> None:
        path = Path(file_path)

        if path.stat().st_size == 0:
            raise EmptyFileError(f"File is empty: {path}")

    @staticmethod
    def validate_file_size(
        file_path: Path,
        max_file_size_mb: int | None = 500,
    ) -> int:
        """Validate file size and return its size in bytes."""

        file_size = Path(file_path).stat().st_size

        if max_file_size_mb is not None:
            max_size_bytes = max_file_size_mb * 1024 * 1024

            if file_size > max_size_bytes:
                raise FileSizeLimitExceededError(
                    f"File size ({file_size / (1024 * 1024):.2f} MB) "
                    f"exceeds the configured limit of {max_file_size_mb} MB."
                )

        return file_size

    @staticmethod
    def validate_headers(headers: list[str] | None) -> list[str]:
        """Validate and return normalized CSV headers."""

        if not headers:
            raise CSVValidationError("CSV file does not contain headers.")

        normalized_headers = [header.strip() for header in headers]

        if any(not header for header in normalized_headers):
            raise CSVValidationError("CSV contains empty column names.")

        if len(normalized_headers) != len(set(normalized_headers)):
            raise CSVValidationError("CSV contains duplicate column names.")

        return normalized_headers

    @staticmethod
    def validate_csv_structure(
        file_path: Path,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
    ) -> dict:
        """
        Stream through CSV and validate its structure.

        Does not load the complete dataset into memory.
        """

        path = Path(file_path)

        row_count = 0

        try:
            with path.open(
                mode="r",
                encoding=encoding,
                newline="",
            ) as file:

                reader = csv.reader(
                    file,
                    delimiter=delimiter,
                    strict=True,
                )

                headers = CSVValidator.validate_headers(
                    next(reader, None)
                )

                expected_columns = len(headers)

                for line_number, row in enumerate(reader, start=2):

                    if not row:
                        raise CSVValidationError(
                            f"Empty row detected at line {line_number}."
                        )

                    if len(row) != expected_columns:
                        raise CSVValidationError(
                            f"Invalid column count at line {line_number}. "
                            f"Expected {expected_columns}, got {len(row)}."
                        )

                    row_count += 1

                if row_count == 0:
                    raise CSVValidationError(
                        "CSV contains headers but no data records."
                    )

                return {
                    "headers": headers,
                    "row_count": row_count,
                    "column_count": expected_columns,
                }

        except UnicodeDecodeError as exc:
            raise CSVEncodingError(
                f"Unable to decode CSV using {encoding}: {exc}"
            ) from exc

        except csv.Error as exc:
            raise InvalidCSVError(
                f"Malformed CSV file: {exc}"
            ) from exc

    @classmethod
    def validate(
        cls,
        file_path: Path,
        max_file_size_mb: int | None = 500,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
    ) -> dict:
        """Run all CSV validation checks."""

        cls.validate_file_exists(file_path)
        cls.validate_extension(file_path)
        cls.validate_file_not_empty(file_path)

        file_size = cls.validate_file_size(
            file_path,
            max_file_size_mb,
        )

        structure = cls.validate_csv_structure(
            file_path,
            encoding,
            delimiter,
        )

        return {
            "file_size_bytes": file_size,
            **structure,
        }
