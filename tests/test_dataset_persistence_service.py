
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from backend.application.dataset_persistence_service import (
    DatasetPersistenceService,
)
from backend.core.ingestion.models import (
    DatasetMetadata,
    IngestionConfig,
    IngestionResult,
    PersistentDatasetMetadata,
)


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "age": [20, 30, 40],
        }
    )


@pytest.fixture
def ingestion_result(tmp_path, dataframe):
    source_file = tmp_path / "sample.csv"
    dataframe.to_csv(source_file, index=False)

    return IngestionResult(
        dataset_id="dataset-123",
        filename="sample.csv",
        file_path=source_file,
        metadata=DatasetMetadata(
            filename="sample.csv",
            file_size_bytes=source_file.stat().st_size,
            row_count=3,
            column_count=2,
            columns=["name", "age"],
            processing_mode="in_memory",
        ),
        dataframe=dataframe,
    )


@pytest.fixture
def config():
    return IngestionConfig(
        mode="in_memory",
        storage_dir=Path("data/raw"),
    )


@pytest.fixture
def repositories():
    dataset_repository = MagicMock()
    dataset_repository.save_original.return_value = (
        "raw/dataset-123/original.csv"
    )
    dataset_repository.save_version.return_value = (
        "processed/dataset-123/v1/data.parquet"
    )

    metadata_repository = MagicMock()
    metadata_repository.save = AsyncMock(return_value="dataset-123")

    return dataset_repository, metadata_repository


@pytest.mark.asyncio
async def test_persist_saves_original_version_and_metadata(
    ingestion_result,
    config,
    repositories,
):
    dataset_repository, metadata_repository = repositories

    service = DatasetPersistenceService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
    )

    result = await service.persist(ingestion_result, config)

    dataset_repository.save_original.assert_called_once_with(
        source_path=ingestion_result.file_path,
        dataset_id="dataset-123",
    )

    dataset_repository.save_version.assert_called_once_with(
        dataframe=ingestion_result.dataframe,
        dataset_id="dataset-123",
        version_id="v1",
    )

    metadata_repository.save.assert_awaited_once()

    assert isinstance(result, PersistentDatasetMetadata)
    assert result.dataset_id == "dataset-123"
    assert result.active_version_id == "v1"
    assert result.original_file_key == "raw/dataset-123/original.csv"


@pytest.mark.asyncio
async def test_persist_saves_chunked_dataset(
    ingestion_result,
    config,
    repositories,
):
    dataset_repository, metadata_repository = repositories

    chunks = iter(
        [
            ingestion_result.dataframe.iloc[:2],
            ingestion_result.dataframe.iloc[2:],
        ]
    )

    ingestion_result.dataframe = None
    ingestion_result.chunks = chunks
    ingestion_result.metadata.processing_mode = "chunked"

    config.mode = "chunked"

    service = DatasetPersistenceService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
    )

    result = await service.persist(ingestion_result, config)

    dataset_repository.save_version_from_chunks.assert_called_once()

    assert result.processing_mode == "chunked"
    assert result.active_version_id == "v1"


@pytest.mark.asyncio
async def test_persist_rolls_back_files_when_metadata_save_fails(
    ingestion_result,
    config,
    repositories,
):
    dataset_repository, metadata_repository = repositories

    metadata_repository.save.side_effect = RuntimeError(
        "MongoDB insert failed"
    )

    service = DatasetPersistenceService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
    )

    with pytest.raises(RuntimeError, match="MongoDB insert failed"):
        await service.persist(ingestion_result, config)

    dataset_repository.delete_version.assert_called_once_with(
        "dataset-123",
        "v1",
    )

    dataset_repository.delete_original.assert_called_once_with(
        "dataset-123"
    )


@pytest.mark.asyncio
async def test_persist_rolls_back_original_when_version_save_fails(
    ingestion_result,
    config,
    repositories,
):
    dataset_repository, metadata_repository = repositories

    dataset_repository.save_version.side_effect = RuntimeError(
        "Parquet save failed"
    )

    service = DatasetPersistenceService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
    )

    with pytest.raises(RuntimeError, match="Parquet save failed"):
        await service.persist(ingestion_result, config)

    dataset_repository.delete_original.assert_called_once_with(
        "dataset-123"
    )

    dataset_repository.delete_version.assert_not_called()
    metadata_repository.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_rolls_back_original_when_version_is_empty(
    ingestion_result,
    config,
    repositories,
):
    dataset_repository, metadata_repository = repositories

    ingestion_result.dataframe = None
    ingestion_result.chunks = None

    service = DatasetPersistenceService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
    )

    with pytest.raises(ValueError, match="no dataset data"):
        await service.persist(ingestion_result, config)

    dataset_repository.delete_original.assert_called_once_with(
        "dataset-123"
    )

    metadata_repository.save.assert_not_awaited()
