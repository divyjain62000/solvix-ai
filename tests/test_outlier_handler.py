
import pandas as pd
import pytest

from backend.core.cleaning.handlers.outlier_handler import OutlierHandler
from backend.core.cleaning.models import CleaningConfig
from backend.core.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)


@pytest.fixture
def handler():
    return OutlierHandler()


@pytest.fixture
def dataframe():
    return pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 100],
        "salary": [30000, 32000, 34000, 36000, 38000, 500000],
    })


@pytest.fixture
def profile():
    return DatasetProfile(
        dataset_id="test-dataset",
        total_rows=6,
        total_columns=2,
        columns=[
            ColumnProfile(
                name="age",
                data_type="int64",
                inferred_type="numerical",
                total_values=6,
                missing_values=0,
                numeric_statistics={
                    "count": 6,
                    "mean": 36.67,
                    "std": 30.43,
                    "min": 20,
                    "25%": 23,
                    "50%": 25,
                    "75%": 27,
                    "max": 100,
                    "variance": 926.67,
                },
            ),
            ColumnProfile(
                name="salary",
                data_type="int64",
                inferred_type="numerical",
                total_values=6,
                missing_values=0,
                numeric_statistics={
                    "count": 6,
                    "mean": 86000,
                    "std": 187000,
                    "min": 30000,
                    "25%": 32500,
                    "50%": 35000,
                    "75%": 37500,
                    "max": 500000,
                    "variance": 34969000000,
                },
            ),
        ],
    )


# ---------------------------------------------------------
# ANALYZE TESTS
# ---------------------------------------------------------

def test_analyze_detects_iqr_outliers(handler, dataframe, profile):
    config = CleaningConfig(
        outlier_method="iqr",
        columns_to_clean=["age"],
    )

    result = handler.analyze(dataframe, profile, config)

    assert result.total_rows == 6
    assert result.total_outlier_values == 1
    assert result.columns_with_outliers == 1

    age_result = result.columns[0]

    assert age_result.column_name == "age"
    assert age_result.outlier_count == 1
    assert age_result.lower_bound == 17.0
    assert age_result.upper_bound == 33.0


def test_analyze_detects_zscore_outliers(handler, dataframe, profile):
    config = CleaningConfig(
        outlier_method="zscore",
        zscore_threshold=1.0,
        columns_to_clean=["age"],
    )

    result = handler.analyze(dataframe, profile, config)

    assert result.total_outlier_values == 1
    assert result.columns[0].outlier_count == 1


def test_analyze_ignores_missing_values(handler, profile):
    dataframe = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 100, None],
        "salary": [30000, 32000, 34000, 36000, 38000, 500000, None],
    })

    config = CleaningConfig(
        columns_to_clean=["age"],
    )

    result = handler.analyze(dataframe, profile, config)

    assert result.total_outlier_values == 1
    assert result.columns[0].total_values == 6


def test_analyze_uses_all_numerical_columns(handler, dataframe, profile):
    config = CleaningConfig()

    result = handler.analyze(dataframe, profile, config)

    assert len(result.columns) == 2
    assert result.columns_with_outliers == 2


def test_analyze_rejects_missing_profile_column(handler, dataframe, profile):
    config = CleaningConfig(
        columns_to_clean=["unknown"],
    )

    with pytest.raises(ValueError):
        handler.analyze(dataframe, profile, config)


def test_analyze_rejects_non_numerical_column(handler, dataframe, profile):
    profile.columns[0].inferred_type = "categorical"

    config = CleaningConfig(
        columns_to_clean=["age"],
    )

    with pytest.raises(ValueError):
        handler.analyze(dataframe, profile, config)


def test_analyze_rejects_missing_dataframe_column(handler, profile):
    dataframe = pd.DataFrame({"salary": [30000, 40000]})

    config = CleaningConfig(
        columns_to_clean=["age"],
    )

    with pytest.raises(ValueError):
        handler.analyze(dataframe, profile, config)


# ---------------------------------------------------------
# CLEAN TESTS
# ---------------------------------------------------------

def test_clean_removes_entire_rows_with_outliers(
    handler,
    dataframe,
    profile,
):
    config = CleaningConfig(
        outlier_method="iqr",
        outlier_action="remove",
        columns_to_clean=["age"],
    )

    result, operation = handler.clean(
        dataframe,
        config,
        profile,
    )

    assert len(result) == 5
    assert 100 not in result["age"].values
    assert operation.rows_before == 6
    assert operation.rows_after == 5
    assert operation.rows_removed == 1


def test_clean_caps_outlier_values(handler, dataframe, profile):
    config = CleaningConfig(
        outlier_method="iqr",
        outlier_action="cap",
        columns_to_clean=["age"],
    )

    result, operation = handler.clean(
        dataframe,
        config,
        profile,
    )

    assert result["age"].max() == 33.0
    assert len(result) == 6
    assert operation.rows_removed == 0
    assert operation.values_changed == 1


def test_clean_preserves_missing_values(handler, profile):
    dataframe = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 100, None],
        "salary": [30000, 32000, 34000, 36000, 38000, 500000, None],
    })

    config = CleaningConfig(
        outlier_action="cap",
        columns_to_clean=["age"],
    )

    result, operation = handler.clean(
        dataframe,
        config,
        profile,
    )

    assert result["age"].isna().sum() == 1
    assert operation.values_changed == 1


def test_clean_does_not_modify_original_dataframe(
    handler,
    dataframe,
    profile,
):
    original = dataframe.copy(deep=True)

    config = CleaningConfig(
        outlier_action="cap",
        columns_to_clean=["age"],
    )

    handler.clean(dataframe, config, profile)

    pd.testing.assert_frame_equal(dataframe, original)


# ---------------------------------------------------------
# CHUNKED ANALYZE TESTS
# ---------------------------------------------------------

def test_analyze_chunks_detects_outliers(handler, dataframe, profile):
    chunks = [
        dataframe.iloc[:3].copy(),
        dataframe.iloc[3:].copy(),
    ]

    config = CleaningConfig(
        outlier_method="iqr",
        columns_to_clean=["age"],
    )

    result = handler.analyze_chunks(
        chunks,
        profile,
        config,
    )

    assert result.total_rows == 6
    assert result.total_outlier_values == 1
    assert result.columns[0].outlier_count == 1


def test_analyze_chunks_matches_in_memory_analysis(
    handler,
    dataframe,
    profile,
):
    config = CleaningConfig()

    in_memory_result = handler.analyze(
        dataframe,
        profile,
        config,
    )

    chunks_result = handler.analyze_chunks(
        [
            dataframe.iloc[:2].copy(),
            dataframe.iloc[2:4].copy(),
            dataframe.iloc[4:].copy(),
        ],
        profile,
        config,
    )

    assert (
        chunks_result.total_outlier_values
        == in_memory_result.total_outlier_values
    )

    assert (
        chunks_result.columns_with_outliers
        == in_memory_result.columns_with_outliers
    )

    for memory_column, chunk_column in zip(
        in_memory_result.columns,
        chunks_result.columns,
    ):
        assert memory_column.outlier_count == chunk_column.outlier_count


# ---------------------------------------------------------
# CHUNKED CLEAN TESTS
# ---------------------------------------------------------

def test_clean_chunks_removes_outlier_rows(handler, dataframe, profile):
    chunks = [
        dataframe.iloc[:3].copy(),
        dataframe.iloc[3:].copy(),
    ]

    config = CleaningConfig(
        outlier_action="remove",
        columns_to_clean=["age"],
    )

    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        config,
        profile,
    )

    result = pd.concat(
        list(cleaned_chunks),
        ignore_index=True,
    )

    assert len(result) == 5
    assert 100 not in result["age"].values
    assert operation.rows_before == 6
    assert operation.rows_after == 5
    assert operation.rows_removed == 1
    assert operation.values_changed == 0


def test_clean_chunks_caps_outlier_values(handler, dataframe, profile):
    chunks = [
        dataframe.iloc[:3].copy(),
        dataframe.iloc[3:].copy(),
    ]

    config = CleaningConfig(
        outlier_action="cap",
        columns_to_clean=["age"],
    )

    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        config,
        profile,
    )

    result = pd.concat(
        list(cleaned_chunks),
        ignore_index=True,
    )

    assert result["age"].max() == 33.0
    assert len(result) == 6
    assert operation.rows_before == 6
    assert operation.rows_after == 6
    assert operation.rows_removed == 0
    assert operation.values_changed == 1


def test_clean_chunks_preserves_row_order(handler, dataframe, profile):
    chunks = [
        dataframe.iloc[:2].copy(),
        dataframe.iloc[2:4].copy(),
        dataframe.iloc[4:].copy(),
    ]

    config = CleaningConfig(
        outlier_action="cap",
        columns_to_clean=["age"],
    )

    cleaned_chunks, _ = handler.clean_chunks(
        chunks,
        config,
        profile,
    )

    result = pd.concat(
        list(cleaned_chunks),
        ignore_index=True,
    )

    assert result["age"].tolist() == [20, 22, 24, 26, 28, 33.0]


# ---------------------------------------------------------
# VALIDATION TESTS
# ---------------------------------------------------------

def test_clean_rejects_invalid_dataframe(handler, profile):
    config = CleaningConfig()

    with pytest.raises(ValueError):
        handler.clean(
            None,
            config,
            profile,
        )


def test_analyze_rejects_invalid_profile(handler, dataframe):
    config = CleaningConfig()

    with pytest.raises(ValueError):
        handler.analyze(
            dataframe,
            None,
            config,
        )
