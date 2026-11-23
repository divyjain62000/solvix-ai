class CleaningError(Exception):
    """Base exception for all cleaning-related errors."""


class MissingValueError(CleaningError):
    """Raised when missing-value handling fails."""


class DuplicateHandlingError(CleaningError):
    """Raised when duplicate detection or removal fails."""


class OutlierHandlingError(CleaningError):
    """Raised when outlier detection or treatment fails."""


class InvalidCleaningConfigError(CleaningError):
    """Raised when cleaning configuration is invalid."""


class UnsupportedCleaningStrategyError(CleaningError):
    """Raised when an unsupported cleaning strategy is requested."""


class CleaningExecutionError(CleaningError):
    """Raised when a cleaning operation fails during execution."""


class CleaningStorageError(CleaningError):
    """Raised when reading or writing cleaned data or change files fails."""


class CleaningValidationError(CleaningError):
    """Raised when the input dataset is invalid for cleaning."""
