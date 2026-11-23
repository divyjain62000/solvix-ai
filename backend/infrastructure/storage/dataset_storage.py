
"""Dataset-specific storage operations."""

from pathlib import Path

from backend.config.settings import settings
from backend.infrastructure.storage.base import StorageBackend
from backend.infrastructure.storage.local_storage import LocalStorage


class DatasetStorage:
    """Manage dataset storage keys and file operations."""

    def __init__(
        self,
        storage_backend: StorageBackend | None = None,
    ) -> None:
        self.backend = storage_backend or LocalStorage()
        self.base_dir = Path(settings.data_dir)

    @staticmethod
    def _validate_id(value: str, field_name: str) -> None:
        """Validate dataset and version identifiers."""

        if (
            not isinstance(value, str)
            or not value.strip()
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or Path(value).name != value
        ):
            raise ValueError(f"Invalid {field_name}: {value!r}")

    def get_original_key(self, dataset_id: str) -> str:
        """Return the original CSV storage key."""

        self._validate_id(dataset_id, "dataset_id")
        return f"raw/{dataset_id}/original.csv"

    def get_version_key(
        self,
        dataset_id: str,
        version_id: str,
    ) -> str:
        """Return the processed Parquet storage key."""

        self._validate_id(dataset_id, "dataset_id")
        self._validate_id(version_id, "version_id")

        return f"processed/{dataset_id}/{version_id}/data.parquet"

    def get_checkpoint_key(
        self,
        dataset_id: str,
        checkpoint_id: str,
    ) -> str:
        """Return the checkpoint storage key."""

        self._validate_id(dataset_id, "dataset_id")
        self._validate_id(checkpoint_id, "checkpoint_id")

        return (
            f"checkpoints/{dataset_id}/{checkpoint_id}.parquet"
        )

    def store_original(
        self,
        source_path: Path,
        dataset_id: str,
    ) -> str:
        """Store an original CSV file."""

        source_path = Path(source_path)

        if source_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported.")

        key = self.get_original_key(dataset_id)

        return self.backend.upload_file(source_path, key)

    def retrieve_file(
        self,
        key: str,
        destination: Path,
    ) -> Path:
        """Retrieve a stored file."""

        return self.backend.download_file(key, destination)

    def delete_file(self, key: str) -> None:
        """Delete a stored file."""

        self.backend.delete_file(key)

    def file_exists(self, key: str) -> bool:
        """Check whether a stored file exists."""

        return self.backend.exists(key)

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete all stored files belonging to a dataset."""

        self._validate_id(dataset_id, "dataset_id")

        prefixes = (
            f"raw/{dataset_id}/",
            f"processed/{dataset_id}/",
            f"checkpoints/{dataset_id}/",
        )

        for prefix in prefixes:
            keys = self.backend.list_objects(prefix)

            for key in keys:
                self.backend.delete_file(key)