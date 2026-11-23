
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.core.profiling.models import ColumnProfile, DatasetProfile
from backend.infrastructure.database.repositories.profile_repository import (
    ProfileRepository,
)


@pytest.fixture
def profile():
    return DatasetProfile(
        dataset_id="dataset-123",
        total_rows=100,
        total_columns=2,
        total_missing_values=5,
        duplicate_rows=2,
        memory_usage_bytes=8000,
        columns=[
            ColumnProfile(
                name="age",
                data_type="int64",
                inferred_type="numerical",
                total_values=100,
                missing_values=5,
                missing_percentage=5.0,
                unique_values=40,
                numeric_statistics={"mean": 30.5},
            )
        ],
    )


@pytest.fixture
def repository():
    collection = MagicMock()
    collection.replace_one = AsyncMock()
    collection.find_one = AsyncMock()
    collection.delete_one = AsyncMock()
    collection.delete_many = AsyncMock()

    from backend.infrastructure.database.base_repository import BaseRepository

    repo = object.__new__(ProfileRepository)
    BaseRepository.__init__(repo, collection)

    return repo, collection


def test_to_document(profile):
    document = ProfileRepository._to_document(profile, "v1")

    assert document["_id"] == "dataset-123:v1"
    assert document["dataset_id"] == "dataset-123"
    assert document["version_id"] == "v1"
    assert document["total_rows"] == 100
    assert document["columns"][0]["name"] == "age"


def test_from_document(profile):
    document = ProfileRepository._to_document(profile, "v1")

    result = ProfileRepository._from_document(document)

    assert isinstance(result, DatasetProfile)
    assert result.dataset_id == "dataset-123"
    assert result.total_rows == 100
    assert isinstance(result.columns[0], ColumnProfile)
    assert result.columns[0].name == "age"
    assert result.columns[0].numeric_statistics == {"mean": 30.5}


@pytest.mark.asyncio
async def test_save_uses_upsert(profile, repository):
    repo, collection = repository

    await repo.save(profile, "v1")

    collection.replace_one.assert_awaited_once()

    args, kwargs = collection.replace_one.call_args

    assert args[0] == {"_id": "dataset-123:v1"}
    assert args[1]["version_id"] == "v1"
    assert kwargs["upsert"] is True


@pytest.mark.asyncio
async def test_get_by_version_returns_profile(profile, repository):
    repo, collection = repository

    document = ProfileRepository._to_document(profile, "v1")
    collection.find_one.return_value = document

    result = await repo.get_by_version("dataset-123", "v1")

    collection.find_one.assert_awaited_once_with(
        {"_id": "dataset-123:v1"}
    )

    assert isinstance(result, DatasetProfile)
    assert result.dataset_id == "dataset-123"


@pytest.mark.asyncio
async def test_get_by_version_returns_none_when_missing(repository):
    repo, collection = repository
    collection.find_one.return_value = None

    result = await repo.get_by_version("dataset-123", "v1")

    assert result is None


@pytest.mark.asyncio
async def test_delete_by_version(repository):
    repo, collection = repository
    collection.delete_one.return_value = MagicMock(deleted_count=1)

    result = await repo.delete_by_version("dataset-123", "v1")

    assert result is True
    collection.delete_one.assert_awaited_once_with(
        {"_id": "dataset-123:v1"}
    )


@pytest.mark.asyncio
async def test_delete_by_dataset(repository):
    repo, collection = repository
    collection.delete_many.return_value = MagicMock(deleted_count=3)

    result = await repo.delete_by_dataset("dataset-123")

    assert result == 3
    collection.delete_many.assert_awaited_once_with(
        {"dataset_id": "dataset-123"}
    )


@pytest.mark.asyncio
async def test_save_rejects_empty_version_id(profile, repository):
    repo, collection = repository

    with pytest.raises(ValueError, match="version_id"):
        await repo.save(profile, "")

    collection.replace_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_rejects_empty_dataset_id(repository):
    repo, collection = repository

    with pytest.raises(ValueError, match="dataset_id"):
        await repo.get_by_version("", "v1")

    collection.find_one.assert_not_awaited()
