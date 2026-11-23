
from pathlib import Path

import pytest

from backend.infrastructure.storage.dataset_storage import DatasetStorage
from backend.infrastructure.storage.local_storage import LocalStorage


@pytest.fixture
def dataset_storage(tmp_path):
    backend = LocalStorage(base_dir=tmp_path / "storage")
    return DatasetStorage(storage_backend=backend)


@pytest.fixture
def source_file(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("name,age\nAlice,25\n", encoding="utf-8")
    return file_path


def test_store_original(dataset_storage, source_file):
    key = dataset_storage.store_original(source_file, "dataset-123")

    assert key == "raw/dataset-123/original.csv"
    assert dataset_storage.file_exists(key)


def test_store_original_rejects_non_csv(dataset_storage, tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("sample")

    with pytest.raises(ValueError):
        dataset_storage.store_original(source, "dataset-123")


def test_get_version_key(dataset_storage):
    assert dataset_storage.get_version_key("dataset-123", "v1") == (
        "processed/dataset-123/v1/data.parquet"
    )


def test_get_checkpoint_key(dataset_storage):
    assert dataset_storage.get_checkpoint_key("dataset-123", "cp1") == (
        "checkpoints/dataset-123/cp1.parquet"
    )


def test_delete_dataset(dataset_storage, source_file):
    backend = dataset_storage.backend

    keys = [
        "raw/dataset-123/original.csv",
        "processed/dataset-123/v1/data.parquet",
        "checkpoints/dataset-123/cp1.parquet",
        "raw/dataset-456/original.csv",
    ]

    for key in keys:
        backend.upload_file(source_file, key)

    dataset_storage.delete_dataset("dataset-123")

    assert not backend.exists("raw/dataset-123/original.csv")
    assert not backend.exists("processed/dataset-123/v1/data.parquet")
    assert not backend.exists("checkpoints/dataset-123/cp1.parquet")

    # Other datasets must remain untouched.
    assert backend.exists("raw/dataset-456/original.csv")


def test_delete_dataset_with_no_files(dataset_storage):
    dataset_storage.delete_dataset("dataset-123")


@pytest.mark.parametrize("dataset_id", ["", ".", "..", "../outside", "a/b"])
def test_delete_dataset_rejects_invalid_id(dataset_storage, dataset_id):
    with pytest.raises(ValueError):
        dataset_storage.delete_dataset(dataset_id)
