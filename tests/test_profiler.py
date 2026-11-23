from unittest import result

import pandas as pd
import pytest

from backend.core.profiling.profiler import Profiler
from backend.core.profiling.models import ProfilingConfig


@pytest.fixture
def profiler():
    return Profiler()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 30],
            "salary": [30000, 40000, 50000, 60000, 40000],
            "city": ["Indore", "Bhopal", "Indore", "Ujjain", "Bhopal"],
        }
    )


# --------------------------------------------------
# Complete Profiling Tests
# --------------------------------------------------


def test_complete_profiling(profiler, sample_dataframe):
    result = profiler.profile(
        dataframe=sample_dataframe,
        dataset_id="dataset_001",
    )

    assert result.dataset_id == "dataset_001"
    assert result.total_rows == 5
    assert result.total_columns == 3

    assert result.total_missing_values == 0
    assert result.duplicate_rows == 1

    assert len(result.columns) == 3

    age_profile = next(profile for profile in result.columns if profile.name == "age")

    city_profile = next(profile for profile in result.columns if profile.name == "city")

    assert age_profile.numeric_statistics["mean"] == 32
    assert city_profile.categorical_statistics["unique"] == 3


# --------------------------------------------------
# Configuration Tests
# --------------------------------------------------


def test_default_configuration(profiler, sample_dataframe):
    result = profiler.profile(
        dataframe=sample_dataframe,
        dataset_id="dataset_002",
        config=None,
    )

    assert len(result.columns) == 3

    assert result.columns[0].numeric_statistics != {}
    assert result.columns[2].categorical_statistics != {}


def test_custom_configuration(profiler, sample_dataframe):
    config = ProfilingConfig(
        include_numeric_statistics=False,
        include_categorical_statistics=False,
        include_missing_values=False,
        include_duplicate_rows=False,
    )

    result = profiler.profile(
        dataframe=sample_dataframe,
        dataset_id="dataset_003",
        config=config,
    )

    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0

    for column in result.columns:
        assert column.numeric_statistics == {}
        assert column.categorical_statistics == {}


# --------------------------------------------------
# Data Quality Tests
# --------------------------------------------------


def test_missing_values(profiler):
    dataframe = pd.DataFrame(
        {
            "age": [25, None, 35, None],
            "city": ["Indore", "Bhopal", None, "Ujjain"],
        }
    )

    result = profiler.profile(dataframe, "dataset_004")

    assert result.total_missing_values == 3

    age_profile = next(profile for profile in result.columns if profile.name == "age")

    city_profile = next(profile for profile in result.columns if profile.name == "city")

    assert age_profile.missing_values == 2
    assert city_profile.missing_values == 1


def test_duplicate_rows(profiler):
    dataframe = pd.DataFrame(
        {
            "age": [25, 30, 25, 40],
            "city": ["Indore", "Bhopal", "Indore", "Ujjain"],
        }
    )

    result = profiler.profile(dataframe, "dataset_005")

    assert result.duplicate_rows == 1


# --------------------------------------------------
# Empty DataFrame Test
# --------------------------------------------------


def test_empty_dataframe(profiler):
    dataframe = pd.DataFrame()

    result = profiler.profile(
        dataframe=dataframe,
        dataset_id="dataset_006",
    )

    assert result.dataset_id == "dataset_006"
    assert result.total_rows == 0
    assert result.total_columns == 0
    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0
    assert result.columns == []
    expected_memory = int(dataframe.memory_usage(deep=True).sum())

    assert result.memory_usage_bytes == expected_memory


# --------------------------------------------------
# Dataset Metadata Test
# --------------------------------------------------


def test_dataset_metadata(profiler, sample_dataframe):
    result = profiler.profile(
        dataframe=sample_dataframe,
        dataset_id="dataset_007",
    )

    expected_memory = int(sample_dataframe.memory_usage(deep=True).sum())

    assert result.dataset_id == "dataset_007"
    assert result.total_rows == len(sample_dataframe)
    assert result.total_columns == len(sample_dataframe.columns)
    assert result.memory_usage_bytes == expected_memory
    assert isinstance(result.memory_usage_bytes, int)
