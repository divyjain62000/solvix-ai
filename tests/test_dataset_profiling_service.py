
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from backend.application.dataset_profiling_service import DatasetProfilingService
from backend.core.ingestion.models import PersistentDatasetMetadata
from backend.core.profiling.models import DatasetProfile


@pytest.fixture
def dependencies():
    dataset_repository = MagicMock()
    metadata_repository = MagicMock()
    profile_repository = MagicMock()
    profiling_service = MagicMock()

    service = DatasetProfilingService(
        dataset_repository=dataset_repository,
        metadata_repository=metadata_repository,
        profile_repository=profile_repository,
    )

    service.profiling_service = profiling_service

    metadata_repository.get_by_id = AsyncMock()
    profile_repository.save = AsyncMock()

    return (
        service,
        dataset_repository,
        metadata_repository,
        profile_repository,
        profiling_service,
    )


@pytest.fixture
def dataset_metadata():
    return PersistentDatasetMetadata(
        dataset_id="dataset-123",
        filename="sample.csv",
        original_file_key="raw/dataset-123/original.csv",
        file_size_bytes=1024,
        processing_mode="in_memory",
        ingestion_config={},
        active_version_id="v1",
    )


@pytest.mark.asyncio
async def test_profile_dataset_in_memory(dependencies, dataset_metadata):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    dataframe = pd.DataFrame({"age": [20, 30, 40]})
    expected_profile = DatasetProfile(dataset_id="dataset-123")

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True
    dataset_repo.load_dataframe.return_value = dataframe
    profiler.profile.return_value = expected_profile

    result = await service.profile_dataset("dataset-123")

    assert result is expected_profile

    dataset_repo.load_dataframe.assert_called_once_with(
        "dataset-123", "v1"
    )
    profiler.profile.assert_called_once()
    profile_repo.save.assert_awaited_once_with(
        profile=expected_profile,
        version_id="v1",
    )


@pytest.mark.asyncio
async def test_profile_dataset_chunked(dependencies, dataset_metadata):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    dataset_metadata.processing_mode = "chunked"
    dataset_metadata.ingestion_config = {"chunk_size": 500}

    chunks = iter([
        pd.DataFrame({"age": [20, 30]}),
        pd.DataFrame({"age": [40, 50]}),
    ])

    expected_profile = DatasetProfile(dataset_id="dataset-123")

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True
    dataset_repo.load_chunks.return_value = chunks
    profiler.profile.return_value = expected_profile

    result = await service.profile_dataset("dataset-123")

    assert result is expected_profile

    dataset_repo.load_chunks.assert_called_once_with(
        "dataset-123",
        "v1",
        500,
    )
    profiler.profile.assert_called_once()
    profile_repo.save.assert_awaited_once_with(
        profile=expected_profile,
        version_id="v1",
    )


@pytest.mark.asyncio
async def test_profile_dataset_specific_version(dependencies, dataset_metadata):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    expected_profile = DatasetProfile(dataset_id="dataset-123")

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True
    dataset_repo.load_dataframe.return_value = pd.DataFrame({"age": [25]})
    profiler.profile.return_value = expected_profile

    await service.profile_dataset("dataset-123", version_id="v2")

    dataset_repo.version_exists.assert_called_once_with(
        "dataset-123", "v2"
    )
    dataset_repo.load_dataframe.assert_called_once_with(
        "dataset-123", "v2"
    )
    profile_repo.save.assert_awaited_once_with(
        profile=expected_profile,
        version_id="v2",
    )


@pytest.mark.asyncio
async def test_profile_dataset_not_found(dependencies):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    metadata_repo.get_by_id.return_value = None

    with pytest.raises(ValueError, match="Dataset not found"):
        await service.profile_dataset("unknown-dataset")

    dataset_repo.version_exists.assert_not_called()
    profiler.profile.assert_not_called()
    profile_repo.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_dataset_version_not_found(dependencies, dataset_metadata):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = False

    with pytest.raises(FileNotFoundError, match="Dataset version not found"):
        await service.profile_dataset("dataset-123")

    dataset_repo.load_dataframe.assert_not_called()
    profiler.profile.assert_not_called()
    profile_repo.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_dataset_empty_version_id(dependencies, dataset_metadata):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    metadata_repo.get_by_id.return_value = dataset_metadata

    with pytest.raises(ValueError, match="version ID cannot be empty"):
        await service.profile_dataset("dataset-123", version_id="")

    dataset_repo.version_exists.assert_not_called()
    profiler.profile.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5, "500"])
async def test_profile_dataset_invalid_chunk_size(
    dependencies,
    dataset_metadata,
    chunk_size,
):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    dataset_metadata.processing_mode = "chunked"
    dataset_metadata.ingestion_config = {"chunk_size": chunk_size}

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True

    with pytest.raises(ValueError, match="Invalid chunk_size"):
        await service.profile_dataset("dataset-123")

    dataset_repo.load_chunks.assert_not_called()
    profiler.profile.assert_not_called()


@pytest.mark.asyncio
async def test_profile_dataset_unsupported_processing_mode(
    dependencies,
    dataset_metadata,
):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    dataset_metadata.processing_mode = "unsupported"

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True

    with pytest.raises(ValueError, match="Unsupported processing mode"):
        await service.profile_dataset("dataset-123")

    dataset_repo.load_dataframe.assert_not_called()
    dataset_repo.load_chunks.assert_not_called()
    profiler.profile.assert_not_called()


@pytest.mark.asyncio
async def test_profile_dataset_profile_persistence_failure(
    dependencies,
    dataset_metadata,
):
    service, dataset_repo, metadata_repo, profile_repo, profiler = dependencies

    expected_profile = DatasetProfile(dataset_id="dataset-123")

    metadata_repo.get_by_id.return_value = dataset_metadata
    dataset_repo.version_exists.return_value = True
    dataset_repo.load_dataframe.return_value = pd.DataFrame({"age": [20]})
    profiler.profile.return_value = expected_profile

    profile_repo.save.side_effect = RuntimeError("Database unavailable")

    with pytest.raises(RuntimeError, match="Database unavailable"):
        await service.profile_dataset("dataset-123")

    profiler.profile.assert_called_once()
    profile_repo.save.assert_awaited_once_with(
        profile=expected_profile,
        version_id="v1",
    )
