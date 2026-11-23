
import pandas as pd
import pytest

from backend.core.cleaning.exceptions import MissingValueError
from backend.core.cleaning.handlers.missing_value_handler import (
    MissingValueHandler,
)
from backend.core.cleaning.models import CleaningConfig
from backend.core.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)


@pytest.fixture
def handler():
    return MissingValueHandler()


@pytest.fixture
def chunks():
    return iter([
        pd.DataFrame({
            "age": [20.0, None],
            "city": ["Indore", None],
        }),
        pd.DataFrame({
            "age": [30.0, None],
            "city": [None, "Ujjain"],
        }),
    ])


@pytest.fixture
def profile():
    return DatasetProfile(
        dataset_id="dataset-123",
        total_rows=4,
        total_columns=2,
        total_missing_values=4,
        columns=[
            ColumnProfile(
                name="age",
                data_type="float64",
                inferred_type="numerical",
                total_values=4,
                missing_values=2,
                missing_percentage=50.0,
                numeric_statistics={
                    "mean": 25.0,
                    "50%": 25.0,
                    "mode": 20.0,
                },
            ),
            ColumnProfile(
                name="city",
                data_type="object",
                inferred_type="categorical",
                total_values=4,
                missing_values=2,
                missing_percentage=50.0,
            ),
        ],
    )


def test_clean_chunks_using_constant(handler, chunks):
    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        CleaningConfig(
            missing_value_strategy="constant",
            missing_value_constant="Unknown",
            columns_to_clean=["city"],
        ),
    )

    results = list(cleaned_chunks)

    assert len(results) == 2
    assert results[0]["city"].isna().sum() == 0
    assert results[1]["city"].isna().sum() == 0
    assert operation.rows_before == 4
    assert operation.rows_after == 4
    assert operation.values_changed == 2
    assert operation.affected_columns == ["city"]


def test_clean_chunks_using_mean(handler, chunks, profile):
    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        CleaningConfig(
            missing_value_strategy="mean",
            columns_to_clean=["age"],
        ),
        profile,
    )

    results = list(cleaned_chunks)

    assert len(results) == 2
    assert results[0].loc[1, "age"] == 25.0
    assert results[1].loc[1, "age"] == 25.0
    assert operation.values_changed == 2
    assert operation.rows_removed == 0


def test_clean_chunks_using_median(handler, chunks, profile):
    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        CleaningConfig(
            missing_value_strategy="median",
            columns_to_clean=["age"],
        ),
        profile,
    )

    results = list(cleaned_chunks)

    assert len(results) == 2
    assert results[0].loc[1, "age"] == 25.0
    assert results[1].loc[1, "age"] == 25.0
    assert operation.values_changed == 2
    assert operation.rows_removed == 0


def test_clean_chunks_using_drop(handler, chunks):
    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        CleaningConfig(
            missing_value_strategy="drop",
            columns_to_clean=["age"],
        ),
    )

    results = list(cleaned_chunks)

    assert sum(len(chunk) for chunk in results) == 2
    assert operation.rows_before == 4
    assert operation.rows_after == 2
    assert operation.rows_removed == 2


def test_clean_chunks_requires_profile_for_mean(handler, chunks):
    with pytest.raises(
        MissingValueError,
        match="Dataset profile is required",
    ):
        handler.clean_chunks(
            chunks,
            CleaningConfig(
                missing_value_strategy="mean",
            ),
        )


def test_clean_chunks_rejects_none(handler):
    with pytest.raises(
        MissingValueError,
        match="Chunks cannot be None",
    ):
        handler.clean_chunks(
            None,
            CleaningConfig(),
        )


def test_clean_chunks_rejects_invalid_chunk(handler):
    cleaned_chunks, _ = handler.clean_chunks(
        [pd.DataFrame({"age": [20]}), "invalid"],
        CleaningConfig(
            missing_value_strategy="constant",
            missing_value_constant=0,
        ),
    )

    next(cleaned_chunks)

    with pytest.raises(
        MissingValueError,
        match="Every chunk",
    ):
        next(cleaned_chunks)


def test_clean_chunks_disabled(handler, chunks):
    cleaned_chunks, operation = handler.clean_chunks(
        chunks,
        CleaningConfig(handle_missing_values=False),
    )

    results = list(cleaned_chunks)

    assert len(results) == 2
    assert operation.values_changed == 0
    assert operation.rows_removed == 0
    assert operation.rows_before == 4
    assert operation.rows_after == 4
