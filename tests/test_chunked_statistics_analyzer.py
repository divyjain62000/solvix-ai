
import numpy as np
import pandas as pd
import pytest
import os
from backend.core.profiling.analyzers.chunked_statistics_analyzer import (
    ChunkedStatisticsAnalyzer,
)
from backend.core.profiling.analyzers.chunked_schema_analyzer import (
    ChunkedSchemaAnalyzer,
)
from backend.core.profiling.models import ProfilingConfig
from backend.core.profiling.analyzers.schema_analyzer import SchemaAnalyzer
from backend.core.profiling.models import ProfilingConfig
from backend.core.profiling.analyzers.statistics_analyzer import StatisticsAnalyzer



def get_column_profiles(chunks):
    config = ProfilingConfig()

    analyzer = ChunkedSchemaAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    return analyzer.finalize()


def test_numeric_statistics_across_multiple_chunks():
    chunks = [
        pd.DataFrame({"age": [10, 20, 30]}),
        pd.DataFrame({"age": [40, 50, 60]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 6
    assert statistics["mean"] == pytest.approx(35.0)
    assert statistics["variance"] == pytest.approx(350.0)
    assert statistics["std"] == pytest.approx(np.sqrt(350.0))
    assert statistics["min"] == 10.0
    assert statistics["max"] == 60.0


def test_numeric_statistics_exclude_missing_values():
    chunks = [
        pd.DataFrame({"score": [10.0, np.nan, 30.0]}),
        pd.DataFrame({"score": [np.nan, 50.0, 70.0]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 4
    assert statistics["mean"] == pytest.approx(40.0)
    assert statistics["min"] == 10.0
    assert statistics["max"] == 70.0


def test_numeric_statistics_exclude_infinity():
    chunks = [
        pd.DataFrame({"value": [10.0, np.inf]}),
        pd.DataFrame({"value": [-np.inf, 30.0]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 2
    assert statistics["mean"] == pytest.approx(20.0)
    assert statistics["min"] == 10.0
    assert statistics["max"] == 30.0



def test_categorical_frequencies_across_multiple_chunks():
    chunks = [
        pd.DataFrame({"city": ["Indore", "Pune", "Indore"]}),
        pd.DataFrame({"city": ["Pune", "Mumbai", "Indore"]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    frequencies = result[0].categorical_statistics["top_values"]

    assert frequencies == {
        "Indore": 3,
        "Pune": 2,
        "Mumbai": 1,
    }


def test_categorical_frequencies_exclude_missing_values():
    chunks = [
        pd.DataFrame({"city": ["Indore", None, "Pune"]}),
        pd.DataFrame({"city": ["Indore", None, "Mumbai"]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    frequencies = result[0].categorical_statistics["top_values"]

    assert frequencies == {
        "Indore": 2,
        "Pune": 1,
        "Mumbai": 1,
    }


def test_boolean_columns_have_categorical_frequencies():
    chunks = [
        pd.DataFrame({"is_active": [True, False, True]}),
        pd.DataFrame({"is_active": [False, True, True]}),
    ]

    analyzer = ChunkedStatisticsAnalyzer(
        ProfilingConfig(unique_count_mode="exact")
    )

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    frequencies = result[0].categorical_statistics["top_values"]

    assert frequencies == {
        True: 4,
        False: 2,
    }


def test_exact_unique_count_across_multiple_chunks():
    chunks = [
        pd.DataFrame({"city": ["Indore", "Pune", "Indore"]}),
        pd.DataFrame({"city": ["Mumbai", "Pune", "Delhi"]}),
    ]

    config = ProfilingConfig(unique_count_mode="exact")
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    assert result[0].unique_values == 4


def test_exact_unique_count_handles_numeric_equivalence():
    chunks = [
        pd.DataFrame({"value": [1, 2, 1]}),
        pd.DataFrame({"value": [1.0, 3.0, 2.0]}),
    ]

    config = ProfilingConfig(unique_count_mode="exact")
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    assert result[0].unique_values == 3


def test_exact_unique_count_excludes_missing_values():
    chunks = [
        pd.DataFrame({"city": ["Indore", None, "Pune"]}),
        pd.DataFrame({"city": ["Indore", None, "Mumbai"]}),
    ]

    config = ProfilingConfig(unique_count_mode="exact")
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    assert result[0].unique_values == 3


def test_approximate_unique_count_across_multiple_chunks():
    chunks = [
        pd.DataFrame({"value": list(range(0, 500))}),
        pd.DataFrame({"value": list(range(500, 1000))}),
        pd.DataFrame({"value": list(range(500, 750))}),
    ]

    config = ProfilingConfig(
        unique_count_mode="approximate",
        hll_precision=14,
    )
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    estimated_count = result[0].unique_values

    assert abs(estimated_count - 1000) / 1000 < 0.05


def test_database_is_cleaned_after_finalize():
    chunks = [
        pd.DataFrame({"city": ["Indore", "Pune"]}),
        pd.DataFrame({"city": ["Mumbai", "Indore"]}),
    ]

    config = ProfilingConfig(unique_count_mode="exact")
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    database_path = analyzer.database_path

    assert database_path is not None
    assert os.path.exists(database_path)

    profiles = get_column_profiles(chunks)
    analyzer.finalize(profiles)

    assert not os.path.exists(database_path)



def test_exact_unique_count_preserves_large_integer_precision():
    large_value = 2**53

    chunks = [
        pd.DataFrame({"value": [large_value, large_value + 1]}),
        pd.DataFrame({"value": [large_value, large_value + 2]}),
    ]

    config = ProfilingConfig(unique_count_mode="exact")
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    assert result[0].unique_values == 3


def test_numeric_statistics_are_skipped_when_disabled():
    chunks = [
        pd.DataFrame({"value": [10, 20, 30]}),
        pd.DataFrame({"value": [40, 50, 60]}),
    ]

    config = ProfilingConfig(
        include_numeric_statistics=False,
        unique_count_mode="exact",
    )
    analyzer = ChunkedStatisticsAnalyzer(config)

    for chunk in chunks:
        analyzer.update(chunk)

    assert analyzer.numeric_states == {}

    profiles = get_column_profiles(chunks)
    result = analyzer.finalize(profiles)

    assert result[0].numeric_statistics == {}


def test_chunked_numeric_statistics_match_in_memory():
    # Arrange
    dataframe = pd.DataFrame({
        "age": [10, 20, 20, 30, 40, 50, 60]
    })

    chunks = [
        dataframe.iloc[:2].copy(),
        dataframe.iloc[2:5].copy(),
        dataframe.iloc[5:].copy(),
    ]

    config = ProfilingConfig(unique_count_mode="exact")

    # Generate profiles for in-memory analysis
    schema_analyzer = SchemaAnalyzer()
    column_profiles = schema_analyzer.analyze(dataframe)

    # In-memory analysis
    in_memory_result = StatisticsAnalyzer().analyze(
        dataframe,
        column_profiles,
        config,
    )

    # Chunked analysis
    chunked_analyzer = ChunkedStatisticsAnalyzer(config)

    try:
        for chunk in chunks:
            chunked_analyzer.update(chunk)

        chunked_profiles = get_column_profiles(chunks)
        chunked_result = chunked_analyzer.finalize(chunked_profiles)

    finally:
        chunked_analyzer.close()

    # Compare results
    in_memory_stats = in_memory_result[0].numeric_statistics
    chunked_stats = chunked_result[0].numeric_statistics

    numeric_fields = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "variance",
        "mode",
    ]

    for field in numeric_fields:
        assert chunked_stats[field] == pytest.approx(
            in_memory_stats[field]
        ), f"Mismatch in {field}"

    assert in_memory_stats["mode"] == 20
    assert chunked_stats["mode"] == 20
