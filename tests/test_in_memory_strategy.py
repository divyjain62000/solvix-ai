
import pandas as pd
import pytest

from backend.core.profiling.models import ProfilingConfig, DatasetProfile
from backend.core.profiling.strategies.in_memory import (
    InMemoryProfilingStrategy,
)


@pytest.fixture
def strategy():
    return InMemoryProfilingStrategy()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 30],
        "salary": [30000, 40000, 50000, 60000, 40000],
        "city": ["Indore", "Bhopal", "Indore", "Ujjain", "Bhopal"],
    })


def test_complete_profiling(strategy, sample_dataframe):
    result = strategy.profile(
        data=sample_dataframe,
        dataset_id="dataset_001",
        config=ProfilingConfig(),
    )

    assert isinstance(result, DatasetProfile)
    assert result.dataset_id == "dataset_001"
    assert result.total_rows == 5
    assert result.total_columns == 3
    assert len(result.columns) == 3


def test_default_configuration(strategy, sample_dataframe):
    result = strategy.profile(
        data=sample_dataframe,
        dataset_id="dataset_001",
        config=None,
    )

    assert result.total_rows == 5
    assert result.total_columns == 3


def test_custom_configuration(strategy, sample_dataframe):
    config = ProfilingConfig(
        include_numeric_statistics=False,
        include_categorical_statistics=False,
        include_missing_values=False,
        include_duplicate_rows=False,
    )

    result = strategy.profile(
        data=sample_dataframe,
        dataset_id="dataset_001",
        config=config,
    )

    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0

    for column in result.columns:
        assert column.numeric_statistics == {}
        assert column.categorical_statistics == {}


def test_missing_values(strategy):
    dataframe = pd.DataFrame({
        "age": [25, None, 35, None],
        "city": ["Indore", "Bhopal", None, "Ujjain"],
    })

    result = strategy.profile(
        data=dataframe,
        dataset_id="dataset_002",
        config=ProfilingConfig(),
    )

    assert result.total_missing_values == 3

    age_profile = next(
        column for column in result.columns
        if column.name == "age"
    )
    assert age_profile.missing_values == 2
    assert age_profile.missing_percentage == 50.0


def test_duplicate_rows(strategy):
    dataframe = pd.DataFrame({
        "age": [25, 30, 30, 40],
        "city": ["Indore", "Bhopal", "Bhopal", "Ujjain"],
    })

    result = strategy.profile(
        data=dataframe,
        dataset_id="dataset_003",
        config=ProfilingConfig(),
    )

    assert result.duplicate_rows == 1


def test_empty_dataframe(strategy):
    dataframe = pd.DataFrame({
        "age": pd.Series(dtype="int64"),
        "city": pd.Series(dtype="object"),
    })

    result = strategy.profile(
        data=dataframe,
        dataset_id="dataset_004",
        config=ProfilingConfig(),
    )

    assert result.total_rows == 0
    assert result.total_columns == 2
    assert len(result.columns) == 2
    assert result.memory_usage_bytes == int(
        dataframe.memory_usage(deep=True).sum()
    )


def test_invalid_input(strategy):
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        strategy.profile(
            data="invalid data",
            dataset_id="dataset_005",
            config=ProfilingConfig(),
        )
