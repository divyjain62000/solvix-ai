
"""Local filesystem storage implementation."""

import shutil
from pathlib import Path, PurePosixPath

from backend.config.settings import settings
from backend.infrastructure.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Store and retrieve files using the local filesystem."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.data_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_key(self, key: str) -> Path:
        """Resolve a storage key safely within the base directory."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError("Storage key cannot be empty.")

        relative_path = PurePosixPath(key)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid storage key: {key!r}")

        if "\\" in key or ":" in key:
            raise ValueError(f"Invalid storage key: {key!r}")

        resolved_path = (self.base_dir / Path(*relative_path.parts)).resolve()

        if not resolved_path.is_relative_to(self.base_dir):
            raise ValueError(f"Invalid storage key: {key!r}")

        return resolved_path

    def upload_file(self, source: Path, key: str) -> str:
        """Copy a local file into managed storage."""

        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        destination = self._resolve_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, destination)

        return PurePosixPath(key).as_posix()

    def download_file(self, key: str, destination: Path) -> Path:
        """Copy a stored file to a local destination."""

        source = self._resolve_key(key)

        if not source.is_file():
            raise FileNotFoundError(f"Stored file not found: {key}")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, destination)

        return destination

    def list_objects(self, prefix: str) -> list[str]:
        """List storage keys that start with the given prefix."""

        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("Storage prefix cannot be empty.")

        if "\\" in prefix or ":" in prefix:
            raise ValueError(f"Invalid storage prefix: {prefix!r}")

        relative_path = PurePosixPath(prefix)

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid storage prefix: {prefix!r}")

        directory = self._resolve_key(prefix)

        if not directory.exists():
            return []

        if not directory.is_dir():
            raise NotADirectoryError(f"Storage prefix is not a directory: {prefix}")

        return sorted(
            path.relative_to(self.base_dir).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )

    def delete_file(self, key: str) -> None:
        """Delete a stored file."""

        file_path = self._resolve_key(key)

        if not file_path.is_file():
            raise FileNotFoundError(f"Stored file not found: {key}")

        file_path.unlink()

    def exists(self, key: str) -> bool:
        """Check whether a stored file exists."""

        return self._resolve_key(key).is_file()
