
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.application.dataset_ingestion_service import (
    DatasetIngestionService,
)
from backend.core.ingestion.models import (
    DatasetMetadata,
    IngestionConfig,
    IngestionResult,
    PersistentDatasetMetadata,
)


@pytest.fixture
def config():
    return IngestionConfig(mode="in_memory")


@pytest.fixture
def ingestion_result(tmp_path):
    source_file = tmp_path / "sample.csv"
    source_file.write_text("name,age\nA,20\nB,30\n")

    return IngestionResult(
        dataset_id="dataset-123",
        filename="sample.csv",
        file_path=source_file,
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=source_file.stat().st_size,
            row_count=2,
            column_count=2,
            columns=["name", "age"],
            processing_mode="in_memory",
        ),
        dataframe=MagicMock(),
    )


@pytest.fixture
def persisted_metadata():
    return PersistentDatasetMetadata(
        dataset_id="dataset-123",
        filename="sample.csv",
        original_file_key="raw/dataset-123/original.csv",
        file_size_bytes=100,
        processing_mode="in_memory",
        active_version_id="v1",
    )


@pytest.fixture
def services(ingestion_result, persisted_metadata):
    ingestion_service = MagicMock()
    ingestion_service.ingest.return_value = ingestion_result

    persistence_service = MagicMock()
    persistence_service.persist = AsyncMock(
        return_value=persisted_metadata
    )

    profiling_service = MagicMock()
    profiling_service.profile_dataset = AsyncMock()

    return (
        ingestion_service,
        persistence_service,
        profiling_service,
    )


@pytest.mark.asyncio
async def test_ingest_runs_complete_workflow(
    config,
    ingestion_result,
    persisted_metadata,
    services,
    tmp_path,
):
    (
        ingestion_service,
        persistence_service,
        profiling_service,
    ) = services

    file_path = tmp_path / "sample.csv"

    service = DatasetIngestionService(
        config=config,
        ingestion_service=ingestion_service,
        persistence_service=persistence_service,
        profiling_service=profiling_service,
    )

    result = await service.ingest(file_path)

    # Verify ingestion
    ingestion_service.ingest.assert_called_once_with(file_path)

    # Verify persistence
    persistence_service.persist.assert_awaited_once_with(
        result=ingestion_result,
        config=config,
    )

    # Verify initial profiling
    profiling_service.profile_dataset.assert_awaited_once_with(
        dataset_id=persisted_metadata.dataset_id,
        version_id=persisted_metadata.active_version_id,
    )

    # Verify returned metadata
    assert result == persisted_metadata


@pytest.mark.asyncio
async def test_ingest_propagates_ingestion_error(
    config,
    services,
    tmp_path,
):
    (
        ingestion_service,
        persistence_service,
        profiling_service,
    ) = services

    ingestion_service.ingest.side_effect = RuntimeError(
        "CSV ingestion failed"
    )

    service = DatasetIngestionService(
        config=config,
        ingestion_service=ingestion_service,
        persistence_service=persistence_service,
        profiling_service=profiling_service,
    )

    with pytest.raises(RuntimeError, match="CSV ingestion failed"):
        await service.ingest(tmp_path / "sample.csv")

    persistence_service.persist.assert_not_awaited()
    profiling_service.profile_dataset.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_propagates_persistence_error(
    config,
    services,
    tmp_path,
):
    (
        ingestion_service,
        persistence_service,
        profiling_service,
    ) = services

    persistence_service.persist.side_effect = RuntimeError(
        "Dataset persistence failed"
    )

    service = DatasetIngestionService(
        config=config,
        ingestion_service=ingestion_service,
        persistence_service=persistence_service,
        profiling_service=profiling_service,
    )

    with pytest.raises(
        RuntimeError,
        match="Dataset persistence failed",
    ):
        await service.ingest(tmp_path / "sample.csv")

    ingestion_service.ingest.assert_called_once()
    persistence_service.persist.assert_awaited_once()
    profiling_service.profile_dataset.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_propagates_profiling_error(
    config,
    services,
    tmp_path,
):
    (
        ingestion_service,
        persistence_service,
        profiling_service,
    ) = services

    profiling_service.profile_dataset.side_effect = RuntimeError(
        "Dataset profiling failed"
    )

    service = DatasetIngestionService(
        config=config,
        ingestion_service=ingestion_service,
        persistence_service=persistence_service,
        profiling_service=profiling_service,
    )

    with pytest.raises(
        RuntimeError,
        match="Dataset profiling failed",
    ):
        await service.ingest(tmp_path / "sample.csv")

    ingestion_service.ingest.assert_called_once()
    persistence_service.persist.assert_awaited_once()
    profiling_service.profile_dataset.assert_awaited_once()