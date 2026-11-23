
import pandas as pd
import pytest

from backend.core.profiling.analyzers.schema_analyzer import SchemaAnalyzer


@pytest.fixture
def analyzer():
    return SchemaAnalyzer()


def test_analyze_numeric_columns(analyzer):
    dataframe = pd.DataFrame({
        "age": [22, 25, 30, 35],
        "salary": [25000.5, 30000.0, 45000.5, 50000.0],
    })

    profiles = analyzer.analyze(dataframe)

    assert len(profiles) == 2

    age_profile = next(p for p in profiles if p.name == "age")
    salary_profile = next(p for p in profiles if p.name == "salary")

    assert age_profile.data_type == "int64"
    assert age_profile.inferred_type == "numerical"
    assert age_profile.total_values == 4
    assert age_profile.unique_values == 4

    assert salary_profile.data_type == "float64"
    assert salary_profile.inferred_type == "numerical"


def test_analyze_boolean_column(analyzer):
    dataframe = pd.DataFrame({
        "is_active": [True, False, True, False]
    })

    profiles = analyzer.analyze(dataframe)

    assert profiles[0].inferred_type == "boolean"
    assert profiles[0].unique_values == 2


def test_analyze_datetime_column(analyzer):
    dataframe = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ])
    })

    profiles = analyzer.analyze(dataframe)

    assert profiles[0].inferred_type == "datetime"
    assert profiles[0].data_type.startswith("datetime64")


def test_analyze_categorical_column(analyzer):
    dataframe = pd.DataFrame({
        "department": ["HR", "IT", "Finance", "IT"]
    })

    profiles = analyzer.analyze(dataframe)

    assert profiles[0].inferred_type == "categorical"
    assert profiles[0].unique_values == 3


@pytest.mark.parametrize(
    "column_name",
    ["id", "employee_id", "customer_id", "user_key", "postal_code"],
)
def test_identifies_identifier_columns(analyzer, column_name):
    dataframe = pd.DataFrame({
        column_name: [101, 102, 103, 104]
    })

    profiles = analyzer.analyze(dataframe)

    assert profiles[0].inferred_type == "identifier"


def test_identifies_uuid_column(analyzer):
    dataframe = pd.DataFrame({
        "reference": [
            "550e8400-e29b-41d4-a716-446655440000",
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
        ]
    })

    profiles = analyzer.analyze(dataframe)

    assert profiles[0].inferred_type == "identifier"


def test_handles_missing_values(analyzer):
    dataframe = pd.DataFrame({
        "age": [20, 25, None, 35],
        "department": ["HR", None, "IT", "Finance"],
    })

    profiles = analyzer.analyze(dataframe)

    age_profile = next(p for p in profiles if p.name == "age")
    department_profile = next(
        p for p in profiles if p.name == "department"
    )

    # Schema analyzer counts total rows and non-null unique values.
    assert age_profile.total_values == 4
    assert age_profile.unique_values == 3

    assert department_profile.total_values == 4
    assert department_profile.unique_values == 3


def test_handles_empty_dataframe(analyzer):
    dataframe = pd.DataFrame()

    profiles = analyzer.analyze(dataframe)

    assert profiles == []


def test_handles_all_null_column(analyzer):
    dataframe = pd.DataFrame({
        "unknown": [None, None, None]
    })

    profiles = analyzer.analyze(dataframe)

    assert len(profiles) == 1
    assert profiles[0].name == "unknown"
    assert profiles[0].total_values == 3
    assert profiles[0].unique_values == 0


def test_preserves_column_order(analyzer):
    dataframe = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 30],
        "is_active": [True, False],
    })

    profiles = analyzer.analyze(dataframe)

    assert [profile.name for profile in profiles] == [
        "name",
        "age",
        "is_active",
    ]
