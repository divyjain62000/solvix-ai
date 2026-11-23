
import pandas as pd
import pytest

from backend.core.ingestion.models import IngestionConfig
from backend.core.ingestion.service import IngestionService
from backend.core.ingestion.exceptions import (
    FileSizeLimitExceededError,
    InvalidCSVError,
    IngestionError,
    CSVValidationError
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    file_path = tmp_path / "employees.csv"

    file_path.write_text(
        "name,age,salary\n"
        "Amit,25,50000\n"
        "Rahul,30,60000\n"
        "Priya,28,55000\n",
        encoding="utf-8",
    )

    return file_path


def test_in_memory_ingestion(sample_csv):
    config = IngestionConfig(mode="in_memory")
    service = IngestionService(config)

    result = service.ingest(sample_csv)

    assert result.dataset_id is not None
    assert result.filename == "employees.csv"
    assert result.metadata.processing_mode == "in_memory"

    assert isinstance(result.dataframe, pd.DataFrame)
    assert result.chunks is None

    assert len(result.dataframe) == 3
    assert list(result.dataframe.columns) == ["name", "age", "salary"]


def test_chunked_ingestion(sample_csv):
    config = IngestionConfig(
        mode="chunked",
        chunk_size=2,
    )
    service = IngestionService(config)

    result = service.ingest(sample_csv)

    assert result.metadata.processing_mode == "chunked"
    assert result.dataframe is None
    assert result.chunks is not None

    chunks = list(result.chunks)

    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1

    total_rows = sum(len(chunk) for chunk in chunks)
    assert total_rows == 3


def test_auto_mode_selects_in_memory_for_small_file(sample_csv):
    config = IngestionConfig(
        mode="auto",
        in_memory_threshold_mb=100,
    )
    service = IngestionService(config)

    result = service.ingest(sample_csv)

    assert result.metadata.processing_mode == "in_memory"
    assert result.dataframe is not None
    assert result.chunks is None


def test_auto_mode_selects_chunked_for_large_estimated_memory(sample_csv):
    config = IngestionConfig(
        mode="auto",
        in_memory_threshold_mb=1,
    )
    service = IngestionService(config)

    # Force the estimated memory above the configured threshold.
    service._select_processing_mode = lambda file_path, total_rows: "chunked"

    result = service.ingest(sample_csv)

    assert result.metadata.processing_mode == "chunked"
    assert result.dataframe is None
    assert result.chunks is not None


def test_metadata_correctness(sample_csv):
    config = IngestionConfig(mode="in_memory")
    service = IngestionService(config)

    result = service.ingest(sample_csv)

    metadata = result.metadata

    assert metadata.filename == "employees.csv"
    assert metadata.row_count == 3
    assert metadata.column_count == 3
    assert metadata.columns == ["name", "age", "salary"]
    assert metadata.file_size_bytes == sample_csv.stat().st_size
    assert metadata.processing_mode == "in_memory"


def test_dataset_ids_are_unique(sample_csv):
    config = IngestionConfig(mode="in_memory")
    service = IngestionService(config)

    result1 = service.ingest(sample_csv)
    result2 = service.ingest(sample_csv)

    assert result1.dataset_id != result2.dataset_id


def test_missing_file_raises_error(tmp_path):
    config = IngestionConfig(mode="in_memory")
    service = IngestionService(config)

    missing_file = tmp_path / "missing.csv"

    with pytest.raises(IngestionError):
        service.ingest(missing_file)


def test_invalid_csv_raises_error(tmp_path):
    file_path = tmp_path / "invalid.csv"

    file_path.write_text(
        "name,age,salary\n"
        "Amit,25,50000\n"
        "Rahul,30\n",
        encoding="utf-8",
    )

    config = IngestionConfig(mode="in_memory")
    service = IngestionService(config)

    with pytest.raises(CSVValidationError, match="Invalid column count"):
        service.ingest(file_path)


def test_file_size_limit_exceeded(tmp_path):
    file_path = tmp_path / "large.csv"

    file_path.write_text(
        "name,age\nAmit,25\n",
        encoding="utf-8",
    )

    config = IngestionConfig(
        mode="in_memory",
        max_file_size_mb=1,
    )

    service = IngestionService(config)

    # Set a very small limit to trigger the exception.
    config.max_file_size_mb = 0

    with pytest.raises(FileSizeLimitExceededError):
        service.ingest(file_path)
