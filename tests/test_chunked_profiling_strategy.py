
import pandas as pd
import pytest

from backend.core.profiling.models import ProfilingConfig
from backend.core.profiling.strategies.chunked import ChunkedProfilingStrategy


@pytest.fixture
def strategy():
    return ChunkedProfilingStrategy()


@pytest.fixture
def sample_chunks():
    return iter([
        pd.DataFrame({
            "age": [25, 30, 35],
            "city": ["Indore", "Bhopal", "Indore"],
        }),
        pd.DataFrame({
            "age": [40, 30],
            "city": ["Ujjain", "Bhopal"],
        }),
    ])


def test_complete_profiling(strategy, sample_chunks):
    result = strategy.profile(
        data=sample_chunks,
        dataset_id="dataset_001",
    )

    assert result.dataset_id == "dataset_001"
    assert result.total_rows == 5
    assert result.total_columns == 2
    assert result.duplicate_rows == 1
    assert result.total_missing_values == 0
    assert len(result.columns) == 2
    assert {column.name for column in result.columns} == {"age", "city"}


def test_multiple_chunks_aggregate_correctly(strategy):
    chunks = iter([
        pd.DataFrame({"value": [10, 20]}),
        pd.DataFrame({"value": [30, 40]}),
        pd.DataFrame({"value": [50]}),
    ])

    result = strategy.profile(chunks, "dataset_002")

    assert result.total_rows == 5
    assert result.total_columns == 1

    numeric_stats = result.columns[0].numeric_statistics

    assert numeric_stats["count"] == 5
    assert numeric_stats["mean"] == 30.0


def test_missing_values_across_chunks(strategy):
    chunks = iter([
        pd.DataFrame({
            "age": [25, None],
            "city": ["Indore", "Bhopal"],
        }),
        pd.DataFrame({
            "age": [None, 40],
            "city": [None, "Ujjain"],
        }),
    ])

    result = strategy.profile(chunks, "dataset_003")

    assert result.total_rows == 4
    assert result.total_missing_values == 3

    profiles = {column.name: column for column in result.columns}

    assert profiles["age"].missing_values == 2
    assert profiles["city"].missing_values == 1


def test_duplicate_rows_across_chunks(strategy):
    chunks = iter([
        pd.DataFrame({
            "id": [1, 2],
            "city": ["Indore", "Bhopal"],
        }),
        pd.DataFrame({
            "id": [2, 3],
            "city": ["Bhopal", "Ujjain"],
        }),
    ])

    result = strategy.profile(chunks, "dataset_004")

    assert result.total_rows == 4
    assert result.duplicate_rows == 1


def test_empty_iterator_raises_value_error(strategy):
    with pytest.raises(ValueError):
        strategy.profile(iter([]), "dataset_005")


def test_iterator_consumed_only_once(strategy):
    consumed = []

    def chunk_generator():
        for index in range(2):
            consumed.append(index)
            yield pd.DataFrame({"value": [index + 1]})

    result = strategy.profile(
        chunk_generator(),
        "dataset_006",
    )

    assert result.total_rows == 2
    assert consumed == [0, 1]


def test_custom_configuration():
    config = ProfilingConfig(
        include_numeric_statistics=False,
        include_categorical_statistics=False,
        include_missing_values=False,
        include_duplicate_rows=False,
    )

    strategy = ChunkedProfilingStrategy(config)

    chunks = iter([
        pd.DataFrame({
            "age": [25, 25],
            "city": ["Indore", "Indore"],
        }),
        pd.DataFrame({
            "age": [30],
            "city": ["Bhopal"],
        }),
    ])

    result = strategy.profile(chunks, "dataset_007")

    assert result.total_rows == 3
    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0

    for column in result.columns:
        assert column.numeric_statistics == {}
        assert column.categorical_statistics == {}


def test_invalid_chunk_input(strategy):
    chunks = iter([
        pd.DataFrame({"value": [1]}),
        "invalid chunk",
    ])

    with pytest.raises(TypeError):
        strategy.profile(chunks, "dataset_008")


def test_cleanup_after_successful_profiling():
    strategy = ChunkedProfilingStrategy()

    chunks = iter([
        pd.DataFrame({"value": [1, 2, 3]}),
    ])

    strategy.profile(chunks, "dataset_009")

    # Quality analyzer cleanup
    assert strategy.quality_analyzer.sqlite_connection is None
    assert strategy.quality_analyzer.database_path is None

    # Statistics analyzer cleanup
    assert strategy.statistics_analyzer.sqlite_connection is None
    assert strategy.statistics_analyzer.database_path is None


def test_cleanup_after_profiling_error():
    strategy = ChunkedProfilingStrategy()

    chunks = iter([
        pd.DataFrame({"value": [1, 2]}),
        pd.DataFrame({"different_column": [3, 4]}),
    ])

    with pytest.raises(ValueError):
        strategy.profile(chunks, "dataset_010")

    # Quality analyzer cleanup
    assert strategy.quality_analyzer.sqlite_connection is None
    assert strategy.quality_analyzer.database_path is None

    # Statistics analyzer cleanup
    assert strategy.statistics_analyzer.sqlite_connection is None
    assert strategy.statistics_analyzer.database_path is None


def test_analyzers_are_independent():
    strategy_one = ChunkedProfilingStrategy()
    strategy_two = ChunkedProfilingStrategy()

    result_one = strategy_one.profile(
        iter([pd.DataFrame({"value": [1, 2]})]),
        "dataset_011",
    )

    result_two = strategy_two.profile(
        iter([pd.DataFrame({"value": [10, 20]})]),
        "dataset_012",
    )

    mean_one = result_one.columns[0].numeric_statistics["mean"]
    mean_two = result_two.columns[0].numeric_statistics["mean"]

    assert mean_one == 1.5
    assert mean_two == 15.0
