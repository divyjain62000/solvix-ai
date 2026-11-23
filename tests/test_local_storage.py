
from pathlib import Path

import pytest

from backend.infrastructure.storage.local_storage import LocalStorage


@pytest.fixture
def storage(tmp_path):
    """Create isolated local storage for each test."""
    return LocalStorage(base_dir=tmp_path / "storage")


@pytest.fixture
def source_file(tmp_path):
    """Create a sample CSV file."""
    file_path = tmp_path / "sample.csv"
    file_path.write_text(
        "name,age\nAlice,25\nBob,30\n",
        encoding="utf-8",
    )
    return file_path


def test_upload_file(storage, source_file):
    """Verify file upload."""
    key = storage.upload_file(
        source_file,
        "raw/dataset-123/original.csv",
    )

    assert key == "raw/dataset-123/original.csv"
    assert storage.exists(key)

    stored_file = storage.base_dir / key
    assert stored_file.read_text(encoding="utf-8") == (
        "name,age\nAlice,25\nBob,30\n"
    )


def test_upload_creates_nested_directories(storage, source_file):
    """Verify nested directories are created automatically."""
    key = "raw/dataset-123/nested/original.csv"

    storage.upload_file(source_file, key)

    assert (storage.base_dir / key).is_file()


def test_upload_missing_source_file(storage, tmp_path):
    """Verify missing source raises FileNotFoundError."""
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        storage.upload_file(missing_file, "raw/missing.csv")


def test_download_file(storage, source_file, tmp_path):
    """Verify file download."""
    key = storage.upload_file(
        source_file,
        "raw/dataset-123/original.csv",
    )

    destination = tmp_path / "downloads" / "dataset.csv"
    downloaded_file = storage.download_file(key, destination)

    assert downloaded_file == destination
    assert destination.is_file()
    assert destination.read_bytes() == source_file.read_bytes()


def test_download_missing_file(storage, tmp_path):
    """Verify missing stored file raises FileNotFoundError."""
    destination = tmp_path / "downloads" / "missing.csv"

    with pytest.raises(FileNotFoundError):
        storage.download_file("raw/missing.csv", destination)


def test_delete_file(storage, source_file):
    """Verify stored file deletion."""
    key = storage.upload_file(source_file, "raw/dataset-123/original.csv")

    storage.delete_file(key)

    assert not storage.exists(key)


def test_delete_missing_file(storage):
    """Verify deleting a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        storage.delete_file("raw/missing.csv")


def test_exists_returns_true(storage, source_file):
    """Verify exists returns True for stored files."""
    key = storage.upload_file(source_file, "raw/dataset-123/original.csv")

    assert storage.exists(key) is True


def test_exists_returns_false(storage):
    """Verify exists returns False for missing files."""
    assert storage.exists("raw/missing.csv") is False


@pytest.mark.parametrize(
    "key",
    [
        "../outside.csv",
        "raw/../../outside.csv",
        "/absolute/path.csv",
        "",
        "raw\\dataset.csv",
        "C:/outside.csv",
    ],
)
def test_rejects_invalid_storage_keys(storage, key):
    """Verify unsafe storage keys are rejected."""
    with pytest.raises(ValueError):
        storage.exists(key)


def test_storage_base_directory_is_created(tmp_path):
    """Verify base directory is created during initialization."""
    base_dir = tmp_path / "new_storage"

    LocalStorage(base_dir=base_dir)

    assert base_dir.is_dir()
    

def test_list_objects_returns_nested_files(storage, source_file):
    storage.upload_file(source_file, "raw/dataset-123/original.csv")
    storage.upload_file(source_file, "raw/dataset-123/versions/v1/data.csv")
    storage.upload_file(source_file, "raw/dataset-456/original.csv")

    result = storage.list_objects("raw/dataset-123")

    assert result == [
        "raw/dataset-123/original.csv",
        "raw/dataset-123/versions/v1/data.csv",
    ]


def test_list_objects_returns_empty_list_for_missing_prefix(storage):
    assert storage.list_objects("raw/missing-dataset") == []


def test_list_objects_rejects_empty_prefix(storage):
    with pytest.raises(ValueError):
        storage.list_objects("")


def test_list_objects_rejects_invalid_prefix(storage):
    with pytest.raises(ValueError):
        storage.list_objects("../outside")


def test_list_objects_rejects_file_as_prefix(storage, source_file):
    storage.upload_file(source_file, "raw/dataset-123/original.csv")

    with pytest.raises(NotADirectoryError):
        storage.list_objects("raw/dataset-123/original.csv")

