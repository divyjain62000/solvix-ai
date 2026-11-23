
import pandas as pd
import pytest

from backend.core.cleaning.handlers.duplicate_handler import DuplicateHandler
from backend.core.cleaning.models import CleaningConfig
from backend.core.profiling.models import DatasetProfile


@pytest.fixture
def handler():
    return DuplicateHandler()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "id": [1, 2, 2, 3, 3, 4],
        "name": ["A", "B", "B", "C", "C", "D"],
    })


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        dataset_id="test-dataset",
        total_rows=6,
        total_columns=2,
        duplicate_rows=2,
    )


# -------------------------------------------------------------------
# Analyze tests
# -------------------------------------------------------------------

def test_analyze_returns_duplicate_analysis(handler, sample_profile):
    result = handler.analyze(sample_profile)

    assert result.total_rows == 6
    assert result.duplicate_rows == 2
    assert result.duplicate_percentage == pytest.approx(33.333333, rel=1e-5)


def test_analyze_without_duplicates(handler):
    profile = DatasetProfile(
        dataset_id="test-dataset",
        total_rows=5,
        total_columns=2,
        duplicate_rows=0,
    )

    result = handler.analyze(profile)

    assert result.duplicate_rows == 0
    assert result.duplicate_percentage == 0.0


# -------------------------------------------------------------------
# In-memory cleaning tests
# -------------------------------------------------------------------

def test_clean_keeps_first_duplicate(handler, sample_dataframe):
    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep="first",
    )

    cleaned_df, result = handler.clean(sample_dataframe, config)

    assert len(cleaned_df) == 4
    assert cleaned_df["id"].tolist() == [1, 2, 3, 4]
    assert result.rows_before == 6
    assert result.rows_after == 4
    assert result.rows_removed == 2


def test_clean_keeps_last_duplicate(handler, sample_dataframe):
    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep="last",
    )

    cleaned_df, result = handler.clean(sample_dataframe, config)

    assert len(cleaned_df) == 4
    assert cleaned_df["id"].tolist() == [1, 2, 3, 4]
    assert cleaned_df.index.tolist() == [0, 2, 4, 5]
    assert result.rows_removed == 2


def test_clean_removes_all_duplicate_occurrences(handler, sample_dataframe):
    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep=None,
    )

    cleaned_df, result = handler.clean(sample_dataframe, config)

    assert cleaned_df["id"].tolist() == [1, 4]
    assert len(cleaned_df) == 2
    assert result.rows_removed == 4


def test_clean_disabled(handler, sample_dataframe):
    config = CleaningConfig(remove_duplicates=False)

    cleaned_df, result = handler.clean(sample_dataframe, config)

    pd.testing.assert_frame_equal(cleaned_df, sample_dataframe)
    assert result.rows_before == 6
    assert result.rows_after == 6
    assert result.rows_removed == 0


# -------------------------------------------------------------------
# Chunked cleaning tests
# -------------------------------------------------------------------

def test_clean_chunks_duplicates_within_same_chunk(handler):
    chunks = [
        pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": ["A", "B", "B", "C"],
        })
    ]

    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep="first",
    )

    cleaned_chunks, result = handler.clean_chunks(chunks, config)
    output = list(cleaned_chunks)

    assert len(output) == 1
    assert output[0]["id"].tolist() == [1, 2, 3]
    assert result.rows_before == 4
    assert result.rows_after == 3
    assert result.rows_removed == 1


def test_clean_chunks_duplicates_across_chunks(handler):
    chunks = [
        pd.DataFrame({
            "id": [1, 2],
            "name": ["A", "B"],
        }),
        pd.DataFrame({
            "id": [2, 3],
            "name": ["B", "C"],
        }),
    ]

    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep="first",
    )

    cleaned_chunks, result = handler.clean_chunks(chunks, config)
    output = list(cleaned_chunks)

    combined = pd.concat(output, ignore_index=True)

    assert combined["id"].tolist() == [1, 2, 3]
    assert result.rows_before == 4
    assert result.rows_after == 3
    assert result.rows_removed == 1


@pytest.mark.parametrize(
    ("keep", "expected_ids", "expected_removed"),
    [
        ("first", [1, 2, 3], 1),
        ("last", [1, 2, 3], 1),
        (None, [1, 3], 2),
    ],
)
def test_clean_chunks_keep_strategy_across_chunks(
    handler,
    keep,
    expected_ids,
    expected_removed,
):
    chunks = [
        pd.DataFrame({
            "id": [1, 2],
            "name": ["A", "B"],
        }),
        pd.DataFrame({
            "id": [2, 3],
            "name": ["B", "C"],
        }),
    ]

    config = CleaningConfig(
        remove_duplicates=True,
        duplicate_keep=keep,
    )

    cleaned_chunks, result = handler.clean_chunks(chunks, config)
    output = list(cleaned_chunks)

    combined = pd.concat(output, ignore_index=True)

    assert combined["id"].tolist() == expected_ids
    assert result.rows_removed == expected_removed


def test_clean_chunks_disabled(handler):
    chunks = [
        pd.DataFrame({"id": [1, 2]}),
        pd.DataFrame({"id": [2, 3]}),
    ]

    config = CleaningConfig(remove_duplicates=False)

    cleaned_chunks, result = handler.clean_chunks(chunks, config)
    output = list(cleaned_chunks)

    assert len(output) == 2
    assert output[0]["id"].tolist() == [1, 2]
    assert output[1]["id"].tolist() == [2, 3]
    assert result.rows_removed == 0
    assert result.rows_before == 4
    assert result.rows_after == 4


def test_clean_chunks_empty_input(handler):
    config = CleaningConfig(remove_duplicates=True)

    cleaned_chunks, result = handler.clean_chunks([], config)
    output = list(cleaned_chunks)

    assert output == []
    assert result.rows_before == 0
    assert result.rows_after == 0
    assert result.rows_removed == 0
