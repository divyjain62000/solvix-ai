from pathlib import Path

import pytest

from backend.infrastructure.storage.dataset_repository import DatasetRepository
from backend.infrastructure.storage.dataset_storage import DatasetStorage
from backend.infrastructure.storage.local_storage import LocalStorage


@pytest.fixture
def repository(tmp_path):
    backend = LocalStorage(base_dir=tmp_path / "storage")
    storage = DatasetStorage(storage_backend=backend)
    return DatasetRepository(storage=storage)


@pytest.fixture
def source_file(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text(
        "name,age\nAlice,25\nBob,30\n",
        encoding="utf-8",
    )
    return file_path

def test_save_original(repository, source_file):
    dataset_id = "dataset-123"

    key = repository.save_original(source_file, dataset_id)

    assert key == "raw/dataset-123/original.csv"

    backend = repository.storage.backend

    assert backend.exists(key)

    stored_file = backend.base_dir / key

    assert stored_file.read_bytes() == source_file.read_bytes()
    

import pandas as pd
import pytest


def test_save_version(repository):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    })

    key = repository.save_version(
        dataframe=dataframe,
        dataset_id="dataset-123",
        version_id="v1",
    )

    assert key == "processed/dataset-123/v1/data.parquet"

    backend = repository.storage.backend
    assert backend.exists(key)

    stored_path = backend.base_dir / key
    restored = pd.read_parquet(stored_path)

    pd.testing.assert_frame_equal(restored, dataframe)


def test_save_version_rejects_non_dataframe(repository):
    with pytest.raises(TypeError, match="pandas DataFrame"):
        repository.save_version(
            dataframe=[1, 2, 3],
            dataset_id="dataset-123",
            version_id="v1",
        )


def test_save_version_rejects_dataframe_without_columns(repository):
    dataframe = pd.DataFrame()

    with pytest.raises(ValueError, match="without columns"):
        repository.save_version(
            dataframe=dataframe,
            dataset_id="dataset-123",
            version_id="v1",
        )


def test_save_version_allows_zero_rows_with_columns(repository):
    dataframe = pd.DataFrame({
        "name": pd.Series(dtype="object"),
        "age": pd.Series(dtype="int64"),
    })

    key = repository.save_version(
        dataframe=dataframe,
        dataset_id="dataset-123",
        version_id="v1",
    )

    restored = pd.read_parquet(repository.storage.backend.base_dir / key)

    pd.testing.assert_frame_equal(restored, dataframe)



def test_load_dataframe(repository):
    original = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    })

    repository.save_version(
        dataframe=original,
        dataset_id="dataset-123",
        version_id="v1",
    )

    restored = repository.load_dataframe(
        dataset_id="dataset-123",
        version_id="v1",
    )

    pd.testing.assert_frame_equal(restored, original)


def test_load_dataframe_raises_when_version_missing(repository):
    with pytest.raises(FileNotFoundError, match="Dataset version not found"):
        repository.load_dataframe(
            dataset_id="dataset-123",
            version_id="v999",
        )



def test_load_chunks(repository):
    original = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "age": [25, 30, 35, 40, 45],
    })

    repository.save_version(
        dataframe=original,
        dataset_id="dataset-123",
        version_id="v1",
    )

    chunks = list(
        repository.load_chunks(
            dataset_id="dataset-123",
            version_id="v1",
            chunk_size=2,
        )
    )

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]

    restored = pd.concat(chunks, ignore_index=True)

    pd.testing.assert_frame_equal(restored, original)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_load_chunks_rejects_invalid_chunk_size(repository, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        repository.load_chunks(
            dataset_id="dataset-123",
            version_id="v1",
            chunk_size=chunk_size,
        )


def test_load_chunks_raises_when_version_missing(repository):
    with pytest.raises(FileNotFoundError, match="Dataset version not found"):
        repository.load_chunks(
            dataset_id="dataset-123",
            version_id="v999",
            chunk_size=100,
        )



def test_delete_version(repository):
    dataframe = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    })

    repository.save_version(dataframe, "dataset-123", "v1")
    repository.save_version(dataframe, "dataset-123", "v2")

    repository.delete_version("dataset-123", "v1")

    assert not repository.storage.backend.exists(
        "processed/dataset-123/v1/data.parquet"
    )

    assert repository.storage.backend.exists(
        "processed/dataset-123/v2/data.parquet"
    )


def test_delete_version_raises_when_missing(repository):
    with pytest.raises(FileNotFoundError, match="Dataset version not found"):
        repository.delete_version("dataset-123", "v999")

def test_version_exists(repository):
    dataframe = pd.DataFrame({
        "name": ["Alice"],
        "age": [25],
    })

    repository.save_version(dataframe, "dataset-123", "v1")

    assert repository.version_exists("dataset-123", "v1") is True


def test_version_exists_returns_false(repository):
    assert repository.version_exists("dataset-123", "v999") is False