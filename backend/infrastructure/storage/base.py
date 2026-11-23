"""Base interface for dataset storage providers."""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Common interface for local and cloud storage providers."""

    @abstractmethod
    def upload_file(self, source: Path, key: str) -> str:
        """Upload a local file and return its storage key."""
        raise NotImplementedError

    @abstractmethod
    def download_file(self, key: str, destination: Path) -> Path:
        """Download a stored file to a local destination."""
        raise NotImplementedError
    
    @abstractmethod
    def list_objects(self, prefix: str) -> list[str]:
        """List storage keys that start with the given prefix."""
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, key: str) -> None:
        """Delete a stored file."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a stored file exists."""
        raise NotImplementedError
