
"""Data models for dataset cleaning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Literal

import pandas as pd


MissingValueStrategy = Literal[
    "drop", "mean", "median", "mode", "constant"
]
DuplicateKeep = Literal["first", "last"] | None
OutlierMethod = Literal["iqr", "zscore"]
OutlierAction = Literal["remove", "cap"]


@dataclass
class CleaningConfig:
    """Configuration for dataset cleaning operations."""

    handle_missing_values: bool = True
    missing_value_strategy: MissingValueStrategy = "median"
    missing_value_constant: Any = None

    remove_duplicates: bool = True
    duplicate_keep: DuplicateKeep = "first"

    handle_outliers: bool = False
    outlier_method: OutlierMethod = "iqr"
    outlier_action: OutlierAction = "cap"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    columns_to_clean: list[str] | None = None

    def __post_init__(self):
        if self.missing_value_strategy not in {
            "drop", "mean", "median", "mode", "constant"
        }:
            raise ValueError("Unsupported missing value strategy")

        if self.duplicate_keep not in {"first", "last", None}:
            raise ValueError("duplicate_keep must be first, last, or None")

        if self.outlier_method not in {"iqr", "zscore"}:
            raise ValueError("Unsupported outlier detection method")

        if self.outlier_action not in {"remove", "cap"}:
            raise ValueError("Unsupported outlier action")

        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be greater than zero")

        if self.zscore_threshold <= 0:
            raise ValueError("zscore_threshold must be greater than zero")

        if (
            self.handle_missing_values
            and self.missing_value_strategy == "constant"
            and self.missing_value_constant is None
        ):
            raise ValueError(
                "missing_value_constant is required for constant strategy"
            )

        if self.columns_to_clean is not None:
            if not self.columns_to_clean:
                raise ValueError("columns_to_clean cannot be empty")

            if any(
                not isinstance(column, str) or not column.strip()
                for column in self.columns_to_clean
            ):
                raise ValueError("Column names must be non-empty strings")

            if len(set(self.columns_to_clean)) != len(self.columns_to_clean):
                raise ValueError("columns_to_clean contains duplicate names")


@dataclass
class CleaningOperationResult:
    """Result and change-storage reference for one cleaning operation."""

    operation_id: str
    operation_type: str
    description: str

    affected_columns: list[str] = field(default_factory=list)

    rows_before: int = 0
    rows_after: int = 0
    rows_removed: int = 0
    values_changed: int = 0

    parameters: dict[str, Any] = field(default_factory=dict)

    # Reference to reversible changes stored outside memory.
    change_reference: str | None = None

    executed_at: datetime = field(default_factory=datetime.now)


@dataclass
class CleaningSummary:
    """Aggregated summary of cleaning operations."""

    original_rows: int = 0
    final_rows: int = 0

    original_columns: int = 0
    final_columns: int = 0

    total_rows_removed: int = 0
    total_values_changed: int = 0

    operations_performed: int = 0

    operation_results: list[CleaningOperationResult] = field(
        default_factory=list
    )


@dataclass
class CleaningResult:
    """Dataset returned after cleaning."""

    dataset_id: str

    dataframe: pd.DataFrame | None = None
    chunks: Iterator[pd.DataFrame] | None = None

    summary: CleaningSummary = field(default_factory=CleaningSummary)

    # Assigned by the versioning layer after persistence.
    version_id: str | None = None

    def __post_init__(self):
        if (self.dataframe is None) == (self.chunks is None):
            raise ValueError(
                "CleaningResult must contain exactly one of "
                "'dataframe' or 'chunks'"
            )


@dataclass
class ColumnMissingInfo:
    column_name: str
    total_values: int = 0
    missing_values: int = 0
    missing_percentage: float = 0.0


@dataclass
class MissingValueAnalysisResult:
    total_rows: int = 0
    total_missing_values: int = 0
    columns_with_missing: int = 0
    columns: list[ColumnMissingInfo] = field(default_factory=list)


@dataclass
class DuplicateAnalysisResult:
    """Analysis result for duplicate rows."""

    total_rows: int = 0
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0


@dataclass
class ColumnOutlierInfo:
    """Outlier information for an individual numerical column."""

    column_name: str
    total_values: int = 0
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass
class OutlierAnalysisResult:
    """Result of dataset outlier analysis."""

    total_rows: int = 0
    total_outlier_values: int = 0
    columns_with_outliers: int = 0
    columns: list[ColumnOutlierInfo] = field(default_factory=list)
