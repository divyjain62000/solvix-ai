from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ProfilingConfig:
    """Configuration controlling dataset profiling."""

    include_numeric_statistics: bool = True
    include_categorical_statistics: bool = True
    include_missing_values: bool = True
    include_duplicate_rows: bool = True

    # Optional sampling limit for large datasets.
    sample_size: int | None = None

    unique_count_mode: Literal["exact", "approximate"] = "approximate"

    hll_precision: int = 14

    def __post_init__(self):
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be greater than zero")


@dataclass
class ColumnProfile:
    """Profiling information for an individual column."""

    name: str

    # Actual Pandas dtype: int64, float64, object, bool, etc.
    data_type: str

    # Semantic type: numerical, categorical, boolean, datetime, identifier.
    inferred_type: str

    total_values: int = 0
    missing_values: int = 0
    missing_percentage: float = 0.0
    unique_values: int = 0

    # Numerical statistics: mean, median, std, min, max, etc.
    numeric_statistics: dict[str, Any] = field(default_factory=dict)

    # Categorical statistics: top values, frequencies, etc.
    categorical_statistics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAnalysisResult:
    """Result of dataset quality analysis."""

    columns: list[ColumnProfile] = field(default_factory=list)

    total_missing_values: int = 0
    duplicate_rows: int = 0


@dataclass
class DatasetProfile:
    """Complete profiling result for a dataset."""

    dataset_id: str

    total_rows: int = 0
    total_columns: int = 0

    total_missing_values: int = 0
    duplicate_rows: int = 0

    memory_usage_bytes: int = 0

    columns: list[ColumnProfile] = field(default_factory=list)
