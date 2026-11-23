
import os

import pandas as pd
import pytest

from backend.core.profiling.analyzers.chunked_quality_analyzer import (
    ChunkedQualityAnalyzer,
)
from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)


def create_profiles(dataframe):
    """Create column profiles for testing."""

    return [
        ColumnProfile(
            name=column,
            data_type=series.dtype.name,
            inferred_type="categorical",
        )
        for column, series in dataframe.items()
    ]


def test_missing_values_across_multiple_chunks():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk1 = pd.DataFrame({
        "name": ["A", None, "C"],
        "age": [20, 30, None],
    })

    chunk2 = pd.DataFrame({
        "name": [None, "E"],
        "age": [40, None],
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    result = analyzer.finalize(create_profiles(chunk1))

    assert result.total_missing_values == 4
    assert result.columns[0].missing_values == 2
    assert result.columns[1].missing_values == 2


def test_missing_percentages():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk = pd.DataFrame({
        "name": ["A", None, "C", None],
        "age": [20, 30, None, 40],
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.columns[0].missing_percentage == 50.0
    assert result.columns[1].missing_percentage == 25.0


def test_duplicate_rows_within_same_chunk():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk = pd.DataFrame({
        "name": ["A", "B", "A"],
        "age": [20, 30, 20],
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.duplicate_rows == 1


def test_duplicate_rows_across_chunks():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk1 = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30],
    })

    chunk2 = pd.DataFrame({
        "name": ["A", "C"],
        "age": [20, 40],
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    result = analyzer.finalize(create_profiles(chunk1))

    assert result.duplicate_rows == 1


def test_duplicate_detection_disabled():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    assert analyzer.sqlite_connection is None
    assert analyzer.database_path is None

    chunk = pd.DataFrame({
        "name": ["A", "A"],
        "age": [20, 20],
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.duplicate_rows == 0


def test_missing_value_detection_disabled():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(
            include_missing_values=False,
            include_duplicate_rows=False,
        )
    )

    chunk = pd.DataFrame({
        "name": ["A", None, "C"],
        "age": [20, None, 30],
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.total_missing_values == 0
    assert result.columns[0].missing_values == 0
    assert result.columns[1].missing_values == 0


def test_schema_mismatch_raises_error():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk1 = pd.DataFrame({
        "name": ["A"],
        "age": [20],
    })

    chunk2 = pd.DataFrame({
        "name": ["B"],
        "salary": [50000],
    })

    analyzer.update(chunk1)

    with pytest.raises(ValueError, match="Schema mismatch"):
        analyzer.update(chunk2)

    analyzer.close()


def test_invalid_input_raises_error():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    with pytest.raises(TypeError, match="pandas DataFrame"):
        analyzer.update(["A", "B"])

    analyzer.close()


def test_finalize_cleans_up_sqlite_resources():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30],
    })

    analyzer.update(chunk)

    database_path = analyzer.database_path

    assert database_path is not None
    assert os.path.exists(database_path)

    analyzer.finalize(create_profiles(chunk))

    assert analyzer.sqlite_connection is None
    assert analyzer.database_path is None
    assert not os.path.exists(database_path)


def test_two_analyzers_have_independent_duplicate_counts():
    config = ProfilingConfig(include_duplicate_rows=True)

    analyzer1 = ChunkedQualityAnalyzer(config)
    analyzer2 = ChunkedQualityAnalyzer(config)

    chunk1 = pd.DataFrame({
        "name": ["A", "A"],
        "age": [20, 20],
    })

    chunk2 = pd.DataFrame({
        "name": ["B", "C"],
        "age": [30, 40],
    })

    analyzer1.update(chunk1)
    analyzer2.update(chunk2)

    result1 = analyzer1.finalize(create_profiles(chunk1))
    result2 = analyzer2.finalize(create_profiles(chunk2))

    assert result1.duplicate_rows == 1
    assert result2.duplicate_rows == 0




def test_empty_chunk_with_valid_schema():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk1 = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30],
    })

    empty_chunk = pd.DataFrame({
        "name": pd.Series(dtype="object"),
        "age": pd.Series(dtype="int64"),
    })

    analyzer.update(chunk1)
    analyzer.update(empty_chunk)

    result = analyzer.finalize(create_profiles(chunk1))

    assert analyzer.total_rows == 2
    assert result.duplicate_rows == 0
    assert result.total_missing_values == 0


def test_empty_dataset():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk = pd.DataFrame({
        "name": pd.Series(dtype="object"),
        "age": pd.Series(dtype="int64"),
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0

    for profile in result.columns:
        assert profile.total_values == 0
        assert profile.missing_percentage == 0.0


def test_all_null_columns():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk = pd.DataFrame({
        "name": [None, None, None],
        "age": [None, None, None],
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.total_missing_values == 6
    assert result.columns[0].missing_values == 3
    assert result.columns[1].missing_values == 3

    assert result.columns[0].missing_percentage == 100.0
    assert result.columns[1].missing_percentage == 100.0


def test_equivalent_numeric_values_are_duplicates():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk = pd.DataFrame({
        "value": pd.Series([1, 1.0, 2], dtype="object")
    })

    analyzer.update(chunk)

    result = analyzer.finalize(create_profiles(chunk))

    assert result.duplicate_rows == 1


def test_finalize_error_cleans_up_sqlite():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=True)
    )

    chunk = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30],
    })

    analyzer.update(chunk)

    database_path = analyzer.database_path

    invalid_profiles = [
        ColumnProfile(
            name="wrong_column",
            data_type="object",
            inferred_type="categorical",
        )
    ]

    with pytest.raises(ValueError, match="do not match"):
        analyzer.finalize(invalid_profiles)

    assert analyzer.sqlite_connection is None
    assert analyzer.database_path is None
    assert not os.path.exists(database_path)


def test_update_after_finalize_raises_error():
    analyzer = ChunkedQualityAnalyzer(
        ProfilingConfig(include_duplicate_rows=False)
    )

    chunk = pd.DataFrame({
        "name": ["A"],
        "age": [20],
    })

    analyzer.update(chunk)
    analyzer.finalize(create_profiles(chunk))

    with pytest.raises(RuntimeError, match="after finalization"):
        analyzer.update(chunk)
