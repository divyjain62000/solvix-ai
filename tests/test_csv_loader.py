
import pandas as pd
import pytest

from backend.core.ingestion.loader import CSVLoader


@pytest.fixture
def csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "employees.csv"

    df = pd.DataFrame({
        " Name ": ["Amit", "Rahul", "Priya", "Neha", "Vikas"],
        " Age ": [25, 30, 28, 32, 27],
        " Department ": [
            "IT", "HR", "IT", "Finance", "HR"
        ],
        " Salary ": [50000, 60000, 55000, 70000, 52000],
    })

    df.to_csv(file_path, index=False)
    return file_path


# --------------------------------------------------
# In-memory loading
# --------------------------------------------------

def test_load_in_memory(csv_file):
    df = CSVLoader.load_in_memory(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert len(df.columns) == 4


def test_load_in_memory_normalizes_columns(csv_file):
    df = CSVLoader.load_in_memory(csv_file)

    assert list(df.columns) == [
        "Name", "Age", "Department", "Salary"
    ]


def test_load_in_memory_with_dtype_optimization(csv_file):
    df = CSVLoader.load_in_memory(
        csv_file,
        optimize=True
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


# --------------------------------------------------
# Chunk loading
# --------------------------------------------------

def test_get_chunk_iterator(csv_file):
    reader = CSVLoader.get_chunk_iterator(
        csv_file,
        chunk_size=2
    )

    assert isinstance(reader, pd.io.parsers.TextFileReader)

    reader.close()


def test_load_in_chunks(csv_file):
    chunks = list(
        CSVLoader.load_in_chunks(
            csv_file,
            chunk_size=2
        )
    )

    assert len(chunks) == 3
    assert sum(len(chunk) for chunk in chunks) == 5

    for chunk in chunks:
        assert isinstance(chunk, pd.DataFrame)


def test_load_in_chunks_normalizes_columns(csv_file):
    chunks = list(
        CSVLoader.load_in_chunks(
            csv_file,
            chunk_size=2
        )
    )

    assert list(chunks[0].columns) == [
        "Name", "Age", "Department", "Salary"
    ]


def test_load_in_chunks_invalid_chunk_size(csv_file):
    with pytest.raises(ValueError):
        CSVLoader.load_in_chunks(
            csv_file,
            chunk_size=0
        )


# --------------------------------------------------
# Memory estimation
# --------------------------------------------------

def test_estimate_memory_usage(csv_file):
    estimated_memory = CSVLoader.estimate_memory_usage(
        csv_file,
        total_rows=5,
        sample_rows=3
    )

    assert isinstance(estimated_memory, int)
    assert estimated_memory > 0


def test_estimate_memory_usage_invalid_total_rows(csv_file):
    with pytest.raises(ValueError):
        CSVLoader.estimate_memory_usage(
            csv_file,
            total_rows=0
        )


def test_estimate_memory_usage_invalid_sample_rows(csv_file):
    with pytest.raises(ValueError):
        CSVLoader.estimate_memory_usage(
            csv_file,
            total_rows=5,
            sample_rows=0
        )


def test_estimate_memory_usage_invalid_safety_factor(csv_file):
    with pytest.raises(ValueError):
        CSVLoader.estimate_memory_usage(
            csv_file,
            total_rows=5,
            safety_factor=0.5
        )


# --------------------------------------------------
# Data type optimization
# --------------------------------------------------

def test_optimize_dtypes_downcasts_integer():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40]
    })

    optimized_df = CSVLoader.optimize_dtypes(df)

    assert optimized_df["Age"].dtype.itemsize < df["Age"].dtype.itemsize


def test_optimize_dtypes_converts_low_cardinality_column():
    df = pd.DataFrame({
        "Department": [
            "IT", "HR", "IT", "HR", "IT"
        ]
    })

    optimized_df = CSVLoader.optimize_dtypes(df)

    assert isinstance(
        optimized_df["Department"].dtype,
        pd.CategoricalDtype
    )


def test_optimize_dtypes_preserves_row_count():
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "Department": ["IT", "HR", "IT"]
    })

    optimized_df = CSVLoader.optimize_dtypes(df)

    assert len(optimized_df) == len(df)


def test_optimize_dtypes_invalid_category_threshold():
    df = pd.DataFrame({"Department": ["IT", "HR"]})

    with pytest.raises(ValueError):
        CSVLoader.optimize_dtypes(
            df,
            category_threshold=0
        )


def test_optimize_dtypes_invalid_max_categories():
    df = pd.DataFrame({"Department": ["IT", "HR"]})

    with pytest.raises(ValueError):
        CSVLoader.optimize_dtypes(
            df,
            max_categories=0
        )
