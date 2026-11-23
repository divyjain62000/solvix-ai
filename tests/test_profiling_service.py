import pytest
import pandas as pd

from pathlib import Path
from unittest.mock import MagicMock

from backend.core.ingestion.models import (
    IngestionResult,
    DatasetMetadata,
)

from backend.core.profiling.models import (
    DatasetProfile,
    ProfilingConfig,
)

from backend.core.profiling.service import ProfilingService


@pytest.fixture
def config():
    return ProfilingConfig()


@pytest.fixture
def sample_profile():
    return DatasetProfile(
        dataset_id="dataset_001",
        total_rows=3,
        total_columns=2,
        total_missing_values=0,
        duplicate_rows=0,
        memory_usage_bytes=1024,
        columns=[],
    )


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "salary": [30000, 40000, 50000],
        }
    )


@pytest.fixture
def in_memory_result(sample_dataframe):
    return IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            row_count=3,
            column_count=2,
            processing_mode="in_memory",
        ),
        dataframe=sample_dataframe,
    )


@pytest.fixture
def chunked_result(sample_dataframe):
    return IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            row_count=3,
            column_count=2,
            processing_mode="chunked",
        ),
        chunks=iter([sample_dataframe]),
    )


@pytest.fixture
def service(config):
    profiling_service = ProfilingService(config)

    profiling_service.in_memory_strategy = MagicMock()
    profiling_service.chunked_strategy = MagicMock()

    return profiling_service


def test_in_memory_profiling(
    service,
    in_memory_result,
    sample_profile,
    config,
):
    service.in_memory_strategy.profile.return_value = sample_profile

    result = service.profile(in_memory_result)

    assert result == sample_profile

    service.in_memory_strategy.profile.assert_called_once_with(
        in_memory_result.dataframe,
        in_memory_result.dataset_id,
        config,
    )

    service.chunked_strategy.profile.assert_not_called()


def test_chunked_profiling(
    service,
    chunked_result,
    sample_profile,
    config,
):
    service.chunked_strategy.profile.return_value = sample_profile

    result = service.profile(chunked_result)

    assert result == sample_profile

    args = service.chunked_strategy.profile.call_args.args

    assert args[1] == chunked_result.dataset_id
    assert args[2] == config

    service.chunked_strategy.profile.assert_called_once()
    service.in_memory_strategy.profile.assert_not_called()


def test_none_ingestion_result(service):
    with pytest.raises(
        ValueError,
        match="ingestion_result cannot be None",
    ):
        service.profile(None)


def test_missing_data_sources(service):
    result = IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            processing_mode="in_memory",
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        service.profile(result)


def test_both_data_sources_populated(
    service,
    sample_dataframe,
):
    result = IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            processing_mode="in_memory",
        ),
        dataframe=sample_dataframe,
        chunks=iter([sample_dataframe]),
    )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        service.profile(result)


def test_in_memory_mode_with_chunks_only(
    service,
    sample_dataframe,
):
    result = IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            processing_mode="in_memory",
        ),
        chunks=iter([sample_dataframe]),
    )

    with pytest.raises(
        ValueError,
        match="In-memory mode requires a dataframe",
    ):
        service.profile(result)


def test_chunked_mode_with_dataframe_only(
    service,
    sample_dataframe,
):
    result = IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            processing_mode="chunked",
        ),
        dataframe=sample_dataframe,
    )

    with pytest.raises(
        ValueError,
        match="Chunked mode requires a chunks iterator",
    ):
        service.profile(result)


def test_unsupported_processing_mode(
    service,
    sample_dataframe,
):
    result = IngestionResult(
        dataset_id="dataset_001",
        filename="sample.csv",
        file_path=Path("data/raw/sample.csv"),
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=1024,
            processing_mode="invalid",
        ),
        dataframe=sample_dataframe,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported processing mode",
    ):
        service.profile(result)


def test_strategy_configuration_forwarding(
    config,
    in_memory_result,
    sample_profile,
):
    service = ProfilingService(config)

    service.in_memory_strategy = MagicMock()
    service.chunked_strategy = MagicMock()

    service.in_memory_strategy.profile.return_value = sample_profile

    service.profile(in_memory_result)

    service.in_memory_strategy.profile.assert_called_once_with(
        in_memory_result.dataframe,
        in_memory_result.dataset_id,
        config,
    )
