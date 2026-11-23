
class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""


class UnsupportedFileFormatError(IngestionError):
    """Raised when the uploaded file format is unsupported."""


class FileSizeLimitExceededError(IngestionError):
    """Raised when a file exceeds the configured size limit."""


class EmptyFileError(IngestionError):
    """Raised when the uploaded file is empty."""


class InvalidCSVError(IngestionError):
    """Raised when a CSV file cannot be parsed."""


class CSVEncodingError(IngestionError):
    """Raised when the CSV file cannot be decoded."""


class CSVValidationError(IngestionError):
    """Raised when CSV structure or headers are invalid."""


class StorageError(IngestionError):
    """Raised when storing or accessing the uploaded file fails."""
