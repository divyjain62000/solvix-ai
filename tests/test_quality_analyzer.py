
import pandas as pd
import pytest

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)
from backend.core.profiling.analyzers.quality_analyzer import (
    QualityAnalyzer,
)


@pytest.fixture
def analyzer():
    return QualityAnalyzer()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "name": ["Alice", "Bob", None, "Alice"],
        "age": [25, None, 30, 25],
        "department": ["IT", "HR", "IT", "IT"],
    })


@pytest.fixture
def sample_profiles(sample_dataframe):
    return [
        ColumnProfile(
            name=column,
            data_type=sample_dataframe[column].dtype.name,
            inferred_type="categorical",
            total_values=len(sample_dataframe),
            unique_values=sample_dataframe[column].nunique(),
        )
        for column in sample_dataframe.columns
    ]


def test_analyze_missing_values(
    analyzer, sample_dataframe, sample_profiles
):
    result = analyzer.analyze(
        sample_dataframe,
        sample_profiles,
        ProfilingConfig(),
    )

    profiles = {profile.name: profile for profile in result.columns}

    assert profiles["name"].missing_values == 1
    assert profiles["age"].missing_values == 1
    assert profiles["department"].missing_values == 0

    assert result.total_missing_values == 2


def test_calculate_missing_percentage(
    analyzer, sample_dataframe, sample_profiles
):
    result = analyzer.analyze(
        sample_dataframe,
        sample_profiles,
        ProfilingConfig(),
    )

    profiles = {profile.name: profile for profile in result.columns}

    assert profiles["name"].missing_percentage == 25.0
    assert profiles["age"].missing_percentage == 25.0
    assert profiles["department"].missing_percentage == 0.0


def test_count_duplicate_rows(
    analyzer, sample_dataframe, sample_profiles
):
    result = analyzer.analyze(
        sample_dataframe,
        sample_profiles,
        ProfilingConfig(),
    )

    assert result.duplicate_rows == 1


def test_no_missing_values(analyzer):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })

    profiles = [
        ColumnProfile(
            name=column,
            data_type=dataframe[column].dtype.name,
            inferred_type="categorical",
        )
        for column in dataframe.columns
    ]

    result = analyzer.analyze(
        dataframe,
        profiles,
        ProfilingConfig(),
    )

    assert result.total_missing_values == 0

    for profile in result.columns:
        assert profile.missing_values == 0
        assert profile.missing_percentage == 0.0


def test_no_duplicate_rows(analyzer):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })

    result = analyzer.analyze(
        dataframe,
        [],
        ProfilingConfig(),
    )

    assert result.duplicate_rows == 0


def test_empty_dataframe(analyzer):
    dataframe = pd.DataFrame({
        "name": pd.Series(dtype="object"),
        "age": pd.Series(dtype="float64"),
    })

    profiles = [
        ColumnProfile(
            name="name",
            data_type="object",
            inferred_type="categorical",
        ),
        ColumnProfile(
            name="age",
            data_type="float64",
            inferred_type="numerical",
        ),
    ]

    result = analyzer.analyze(
        dataframe,
        profiles,
        ProfilingConfig(),
    )

    assert result.total_missing_values == 0
    assert result.duplicate_rows == 0

    for profile in result.columns:
        assert profile.missing_percentage == 0.0


def test_missing_value_analysis_disabled(analyzer):
    dataframe = pd.DataFrame({
        "name": ["Alice", None, "Bob"],
    })

    profiles = [
        ColumnProfile(
            name="name",
            data_type="object",
            inferred_type="categorical",
        )
    ]

    config = ProfilingConfig(
        include_missing_values=False,
        include_duplicate_rows=True,
    )

    result = analyzer.analyze(dataframe, profiles, config)

    assert result.total_missing_values == 0
    assert result.columns[0].missing_values == 0
    assert result.columns[0].missing_percentage == 0.0


def test_duplicate_analysis_disabled(analyzer):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Alice", "Bob"],
    })

    config = ProfilingConfig(
        include_missing_values=True,
        include_duplicate_rows=False,
    )

    result = analyzer.analyze(dataframe, [], config)

    assert result.duplicate_rows == 0


def test_original_profiles_remain_unchanged(
    analyzer, sample_dataframe, sample_profiles
):
    original_profiles = [
        (
            profile.missing_values,
            profile.missing_percentage,
        )
        for profile in sample_profiles
    ]

    result = analyzer.analyze(
        sample_dataframe,
        sample_profiles,
        ProfilingConfig(),
    )

    # Original objects should not be modified.
    for profile, original in zip(sample_profiles, original_profiles):
        assert profile.missing_values == original[0]
        assert profile.missing_percentage == original[1]

    # Returned profiles should be updated.
    assert result.columns[0].missing_values == 1


def test_duplicate_rows_excludes_first_occurrence(analyzer):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Alice", "Alice", "Bob"],
        "age": [25, 25, 25, 30],
    })

    result = analyzer.analyze(
        dataframe,
        [],
        ProfilingConfig(),
    )

    assert result.duplicate_rows == 2
