
import numpy as np
import pandas as pd
import pytest

from backend.core.profiling.models import (
    ColumnProfile,
    ProfilingConfig,
)
from backend.core.profiling.analyzers.statistics_analyzer import (
    StatisticsAnalyzer,
)


@pytest.fixture
def analyzer():
    return StatisticsAnalyzer()


@pytest.fixture
def config():
    return ProfilingConfig()


def create_profile(name, inferred_type):
    return ColumnProfile(
        name=name,
        data_type="float64" if inferred_type == "numerical" else "object",
        inferred_type=inferred_type,
    )


# --------------------------------------------------
# Numerical Statistics Tests
# --------------------------------------------------

def test_numeric_statistics(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [10, 20, 30, 40, 50]
    })

    profiles = [create_profile("age", "numerical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 5
    assert statistics["mean"] == 30
    assert statistics["50%"] == 30
    assert statistics["min"] == 10
    assert statistics["max"] == 50
    assert statistics["variance"] == 250


def test_numeric_statistics_with_missing_values(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [10, 20, np.nan, 40, 50]
    })

    profiles = [create_profile("age", "numerical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 4
    assert statistics["mean"] == 30
    assert statistics["min"] == 10
    assert statistics["max"] == 50


def test_numeric_statistics_excludes_infinite_values(analyzer, config):
    dataframe = pd.DataFrame({
        "score": [10, 20, np.inf, -np.inf, 30]
    })

    profiles = [create_profile("score", "numerical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 3
    assert statistics["mean"] == 20
    assert statistics["min"] == 10
    assert statistics["max"] == 30


def test_numeric_statistics_all_null_values(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [np.nan, np.nan, np.nan]
    })

    profiles = [create_profile("age", "numerical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 0
    assert statistics["mean"] is None
    assert statistics["variance"] is None


def test_numeric_statistics_single_value(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [25]
    })

    profiles = [create_profile("age", "numerical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].numeric_statistics

    assert statistics["count"] == 1
    assert statistics["mean"] == 25
    assert statistics["variance"] is None
    assert statistics["std"] is None


# --------------------------------------------------
# Categorical Statistics Tests
# --------------------------------------------------

def test_categorical_statistics(analyzer, config):
    dataframe = pd.DataFrame({
        "language": [
            "Python", "Java", "Python",
            "C++", "Java", "Python"
        ]
    })

    profiles = [create_profile("language", "categorical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].categorical_statistics

    assert statistics["count"] == 6
    assert statistics["unique"] == 3
    assert statistics["top"] == "Python"
    assert statistics["freq"] == 3

    assert statistics["value_counts"] == {
        "Python": 3,
        "Java": 2,
        "C++": 1,
    }


def test_categorical_statistics_with_missing_values(analyzer, config):
    dataframe = pd.DataFrame({
        "city": ["Indore", None, "Bhopal", "Indore", None]
    })

    profiles = [create_profile("city", "categorical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].categorical_statistics

    assert statistics["count"] == 3
    assert statistics["unique"] == 2
    assert statistics["top"] == "Indore"
    assert statistics["freq"] == 2


def test_categorical_statistics_all_null_values(analyzer, config):
    dataframe = pd.DataFrame({
        "city": [None, None, None]
    })

    profiles = [create_profile("city", "categorical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].categorical_statistics

    assert statistics["count"] == 0
    assert statistics["unique"] == 0
    assert statistics["top"] is None
    assert statistics["freq"] == 0
    assert statistics["value_counts"] == {}


def test_categorical_statistics_top_10_limit(analyzer, config):
    dataframe = pd.DataFrame({
        "category": [f"Category_{i}" for i in range(20)]
    })

    profiles = [create_profile("category", "categorical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].categorical_statistics

    assert len(statistics["value_counts"]) == 10
    assert statistics["unique"] == 20


def test_categorical_statistics_preserves_original_types(analyzer, config):
    dataframe = pd.DataFrame({
        "category": [1, 2, 1, 3, 1]
    })

    profiles = [create_profile("category", "categorical")]

    result = analyzer.analyze(dataframe, profiles, config)

    statistics = result[0].categorical_statistics

    assert statistics["top"] == 1
    assert isinstance(statistics["top"], (int, np.integer))


# --------------------------------------------------
# Configuration Tests
# --------------------------------------------------

def test_numeric_statistics_disabled(analyzer):
    dataframe = pd.DataFrame({
        "age": [10, 20, 30]
    })

    profiles = [create_profile("age", "numerical")]

    config = ProfilingConfig(
        include_numeric_statistics=False
    )

    result = analyzer.analyze(dataframe, profiles, config)

    assert result[0].numeric_statistics == {}


def test_categorical_statistics_disabled(analyzer):
    dataframe = pd.DataFrame({
        "city": ["Indore", "Bhopal", "Indore"]
    })

    profiles = [create_profile("city", "categorical")]

    config = ProfilingConfig(
        include_categorical_statistics=False
    )

    result = analyzer.analyze(dataframe, profiles, config)

    assert result[0].categorical_statistics == {}


# --------------------------------------------------
# Profile Immutability Tests
# --------------------------------------------------

def test_original_profiles_remain_unchanged(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [10, 20, 30]
    })

    profiles = [create_profile("age", "numerical")]

    original_profiles = profiles.copy()

    result = analyzer.analyze(dataframe, profiles, config)

    assert profiles[0].numeric_statistics == {}
    assert result[0].numeric_statistics != {}
    assert result[0] is not original_profiles[0]


# --------------------------------------------------
# Multiple Column Tests
# --------------------------------------------------

def test_multiple_column_statistics(analyzer, config):
    dataframe = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["Indore", "Bhopal", "Indore"],
        "active": [True, False, True],
    })

    profiles = [
        create_profile("age", "numerical"),
        create_profile("city", "categorical"),
        create_profile("active", "boolean"),
    ]

    result = analyzer.analyze(dataframe, profiles, config)

    assert result[0].numeric_statistics["mean"] == 30
    assert result[1].categorical_statistics["top"] == "Indore"
    assert result[2].numeric_statistics == {}
    assert result[2].categorical_statistics == {
        "count": 3,
        "unique": 2,
        "top": True,
        "freq": 2,
        "mode": True,
        "value_counts": {True: 2, False: 1},
        "top_values": {True: 2, False: 1},
    }
