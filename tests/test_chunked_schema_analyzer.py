
import pandas as pd
import pytest

from backend.core.profiling.models import ProfilingConfig
from backend.core.profiling.analyzers.chunked_schema_analyzer import (
    ChunkedSchemaAnalyzer,
)


@pytest.fixture
def analyzer():
    return ChunkedSchemaAnalyzer(ProfilingConfig())


def test_schema_initialization(analyzer):
    chunk = pd.DataFrame({
        "age": [25, 30, 35],
        "city": ["Indore", "Bhopal", "Ujjain"],
    })

    analyzer.update(chunk)

    assert analyzer.initialized is True
    assert analyzer.column_names == ["age", "city"]
    assert analyzer.total_rows == 3


def test_multiple_chunks(analyzer):
    chunk1 = pd.DataFrame({
        "age": [25, 30],
        "city": ["Indore", "Bhopal"],
    })

    chunk2 = pd.DataFrame({
        "age": [35, 40],
        "city": ["Ujjain", "Delhi"],
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    assert analyzer.total_rows == 4


def test_unique_values_across_chunks(analyzer):
    chunk1 = pd.DataFrame({
        "city": ["Indore", "Bhopal", "Ujjain"],
    })

    chunk2 = pd.DataFrame({
        "city": ["Indore", "Delhi", "Bhopal"],
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    profiles = analyzer.finalize()
    city_profile = profiles[0]

    assert city_profile.unique_values == 4


def test_missing_values_excluded_from_unique_count(analyzer):
    chunk1 = pd.DataFrame({
        "city": ["Indore", None, "Bhopal"],
    })

    chunk2 = pd.DataFrame({
        "city": [None, "Indore", "Ujjain"],
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    profile = analyzer.finalize()[0]

    assert profile.unique_values == 3
    assert profile.total_values == 6


def test_numeric_type_promotion(analyzer):
    chunk1 = pd.DataFrame({
        "age": pd.Series([25, 30], dtype="int64"),
    })

    chunk2 = pd.DataFrame({
        "age": pd.Series([35.5, 40.5], dtype="float64"),
    })

    analyzer.update(chunk1)
    analyzer.update(chunk2)

    assert analyzer.column_types["age"] == "float64"


def test_incompatible_type_raises_error(analyzer):
    chunk1 = pd.DataFrame({
        "age": [25, 30],
    })

    chunk2 = pd.DataFrame({
        "age": ["thirty", "forty"],
    })

    analyzer.update(chunk1)

    with pytest.raises(ValueError, match="Incompatible data type"):
        analyzer.update(chunk2)


def test_schema_mismatch_raises_error(analyzer):
    chunk1 = pd.DataFrame({
        "age": [25, 30],
        "city": ["Indore", "Bhopal"],
    })

    chunk2 = pd.DataFrame({
        "age": [35, 40],
        "country": ["India", "India"],
    })

    analyzer.update(chunk1)

    with pytest.raises(ValueError, match="Schema mismatch"):
        analyzer.update(chunk2)


def test_invalid_input(analyzer):
    with pytest.raises(TypeError, match="chunk must be a pandas DataFrame"):
        analyzer.update("invalid")


def test_empty_dataframe_without_columns(analyzer):
    chunk = pd.DataFrame()

    with pytest.raises(ValueError, match="at least one column"):
        analyzer.update(chunk)


def test_empty_chunk_with_valid_schema(analyzer):
    chunk1 = pd.DataFrame({
        "age": [25, 30],
        "city": ["Indore", "Bhopal"],
    })

    empty_chunk = pd.DataFrame({
        "age": pd.Series(dtype="int64"),
        "city": pd.Series(dtype="object"),
    })

    analyzer.update(chunk1)
    analyzer.update(empty_chunk)

    assert analyzer.total_rows == 2
    assert len(analyzer.finalize()) == 2


def test_finalize_before_processing_chunks(analyzer):
    profiles = analyzer.finalize()

    assert profiles == []


def test_duplicate_column_names():
    analyzer = ChunkedSchemaAnalyzer(ProfilingConfig())

    chunk = pd.DataFrame(
        [[25, 30]],
        columns=["age", "age"],
    )

    with pytest.raises(ValueError, match="Duplicate column names"):
        analyzer.update(chunk)


def test_hll_precision_validation():
    config = ProfilingConfig()
    config.hll_precision = 3

    with pytest.raises(ValueError, match="hll_precision"):
        ChunkedSchemaAnalyzer(config)
