import pandas as pd
import pytest

from backend.core.cleaning.exceptions import MissingValueError
from backend.core.cleaning.handlers.missing_value_handler import (
    MissingValueHandler,
)
from backend.core.cleaning.models import (
    CleaningConfig,
    MissingValueAnalysisResult,
)
from backend.core.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)


@pytest.fixture
def handler():
    return MissingValueHandler()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [20, 30, None, 40],
            "salary": [1000.0, None, 3000.0, 4000.0],
            "city": ["Indore", "Ujjain", None, "Bhopal"],
        }
    )


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        dataset_id="dataset-123",
        total_rows=4,
        total_columns=3,
        total_missing_values=3,
        columns=[
            ColumnProfile(
                name="age",
                data_type="float64",
                inferred_type="numerical",
                total_values=4,
                missing_values=1,
                missing_percentage=25.0,
                numeric_statistics={
                    "mean": 30.0,
                    "50%": 30.0,
                    "mode": 20.0,
                },
            ),
            ColumnProfile(
                name="salary",
                data_type="float64",
                inferred_type="numerical",
                total_values=4,
                missing_values=1,
                missing_percentage=25.0,
                numeric_statistics={
                    "mean": 2666.67,
                    "50%": 3000.0,
                    "mode": 1000.0,
                }
            ),
            ColumnProfile(
                name="city",
                data_type="object",
                inferred_type="categorical",
                total_values=4,
                missing_values=1,
                missing_percentage=25.0,
            ),
        ],
    )


# ---------------- ANALYZE TESTS ----------------


def test_analyze_missing_values(handler, sample_profile):
    result = handler.analyze(sample_profile)

    assert isinstance(result, MissingValueAnalysisResult)
    assert result.total_rows == 4
    assert result.total_missing_values == 3
    assert result.columns_with_missing == 3
    assert len(result.columns) == 3


def test_analyze_profile_without_missing_values(handler):
    profile = DatasetProfile(
        dataset_id="dataset-123",
        total_rows=4,
        total_columns=1,
        total_missing_values=0,
        columns=[
            ColumnProfile(
                name="age",
                data_type="int64",
                inferred_type="numerical",
                total_values=4,
                missing_values=0,
            ),
        ],
    )

    result = handler.analyze(profile)

    assert result.total_missing_values == 0
    assert result.columns_with_missing == 0
    assert result.columns == []


def test_analyze_rejects_none_profile(handler):
    with pytest.raises(MissingValueError):
        handler.analyze(None)


# ---------------- CLEAN TESTS ----------------


def test_clean_using_mean(handler, sample_dataframe, sample_profile):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="mean",
            columns_to_clean=["age", "salary"],
        ),
        sample_profile,
    )

    assert result["age"].isna().sum() == 0
    assert result["salary"].isna().sum() == 0
    assert result.loc[2, "age"] == 30.0
    assert result.loc[1, "salary"] == 2666.67
    assert operation.values_changed == 2
    assert operation.rows_removed == 0


def test_clean_using_median(handler, sample_dataframe, sample_profile):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="median",
            columns_to_clean=["age", "salary"],
        ),
        sample_profile,
    )

    assert result["age"].isna().sum() == 0
    assert result["salary"].isna().sum() == 0
    assert result.loc[2, "age"] == 30.0
    assert result.loc[1, "salary"] == 3000.0
    assert operation.values_changed == 2
    assert operation.rows_removed == 0


def test_clean_using_mode(handler, sample_dataframe):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="mode",
            columns_to_clean=["city"],
        ),
    )

    assert result["city"].isna().sum() == 0
    assert result.loc[2, "city"] == "Bhopal"
    assert operation.values_changed == 1
    assert operation.rows_removed == 0


def test_clean_using_constant(handler, sample_dataframe):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="constant",
            missing_value_constant="Unknown",
            columns_to_clean=["city"],
        ),
    )

    assert result["city"].isna().sum() == 0
    assert result.loc[2, "city"] == "Unknown"
    assert operation.values_changed == 1
    assert operation.rows_removed == 0


def test_clean_using_drop(handler, sample_dataframe):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="drop",
        ),
    )

    assert len(result) == 2
    assert result.isna().sum().sum() == 0
    assert operation.rows_removed == 2
    assert operation.rows_after == 2


def test_drop_only_selected_columns(handler):
    dataframe = pd.DataFrame(
        {
            "age": [20, None, 30],
            "city": ["Indore", "Ujjain", None],
        }
    )

    result, operation = handler.clean(
        dataframe,
        CleaningConfig(
            missing_value_strategy="drop",
            columns_to_clean=["age"],
        ),
    )

    assert len(result) == 2
    assert result["age"].isna().sum() == 0
    assert result["city"].isna().sum() == 1
    assert operation.rows_removed == 1


def test_clean_only_selected_columns(handler, sample_dataframe):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="constant",
            missing_value_constant="Unknown",
            columns_to_clean=["city"],
        ),
    )

    assert result["city"].isna().sum() == 0
    assert result["age"].isna().sum() == 1
    assert result["salary"].isna().sum() == 1
    assert operation.affected_columns == ["city"]


def test_clean_preserves_original_dataframe(handler, sample_dataframe):
    original = sample_dataframe.copy(deep=True)

    handler.clean(
        sample_dataframe,
        CleaningConfig(
            missing_value_strategy="constant",
            missing_value_constant="Unknown",
            columns_to_clean=["city"],
        ),
    )

    pd.testing.assert_frame_equal(sample_dataframe, original)


def test_clean_disabled(handler, sample_dataframe):
    result, operation = handler.clean(
        sample_dataframe,
        CleaningConfig(handle_missing_values=False),
    )

    pd.testing.assert_frame_equal(result, sample_dataframe)

    assert operation.values_changed == 0
    assert operation.rows_removed == 0


def test_clean_rejects_invalid_dataframe(handler):
    with pytest.raises(MissingValueError):
        handler.clean(
            None,
            CleaningConfig(),
        )


def test_clean_rejects_invalid_columns(handler, sample_dataframe):
    with pytest.raises(MissingValueError, match="Columns not found"):
        handler.clean(
            sample_dataframe,
            CleaningConfig(
                columns_to_clean=["invalid_column"],
            ),
        )


def test_mean_rejects_categorical_column(handler):
    dataframe = pd.DataFrame(
        {
            "city": ["Indore", None, "Ujjain"],
        }
    )

    with pytest.raises(MissingValueError, match="numerical column"):
        handler.clean(
            dataframe,
            CleaningConfig(
                missing_value_strategy="mean",
                columns_to_clean=["city"],
            ),
        )


def test_median_rejects_categorical_column(handler):
    dataframe = pd.DataFrame(
        {
            "city": ["Indore", None, "Ujjain"],
        }
    )

    with pytest.raises(MissingValueError, match="numerical column"):
        handler.clean(
            dataframe,
            CleaningConfig(
                missing_value_strategy="median",
                columns_to_clean=["city"],
            ),
        )
