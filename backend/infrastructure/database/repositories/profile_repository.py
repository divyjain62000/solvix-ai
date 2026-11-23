
"""Repository for dataset profiling results."""

from dataclasses import asdict
from typing import Any

from backend.core.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)
from backend.infrastructure.database.base_repository import BaseRepository
from backend.infrastructure.database.mongodb import mongodb


class ProfileRepository(BaseRepository):
    """Manage dataset profiles in MongoDB."""

    def __init__(self) -> None:
        database = mongodb.get_database()
        super().__init__(database["dataset_profiles"])

    async def save(
        self,
        profile: DatasetProfile,
        version_id: str,
    ) -> None:
        """Save or replace a profile for a dataset version."""

        if not version_id or not version_id.strip():
            raise ValueError("version_id cannot be empty")

        document = self._to_document(profile, version_id)

        await self.collection.replace_one(
            {"_id": document["_id"]},
            document,
            upsert=True,
        )

    async def get_by_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> DatasetProfile | None:
        """Retrieve a profile for a specific dataset version."""

        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        if not version_id or not version_id.strip():
            raise ValueError("version_id cannot be empty")

        document = await self.find_one(
            {
                "_id": self._get_document_id(dataset_id, version_id),
            }
        )

        if document is None:
            return None

        return self._from_document(document)

    async def delete_by_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> bool:
        """Delete a profile for a specific dataset version."""

        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        if not version_id or not version_id.strip():
            raise ValueError("version_id cannot be empty")

        return await self.delete_one(
            {
                "_id": self._get_document_id(dataset_id, version_id),
            }
        )

    async def delete_by_dataset(
        self,
        dataset_id: str,
    ) -> int:
        """Delete all profiles belonging to a dataset."""

        if not dataset_id or not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        result = await self.collection.delete_many(
            {"dataset_id": dataset_id}
        )

        return result.deleted_count

    @staticmethod
    def _get_document_id(
        dataset_id: str,
        version_id: str,
    ) -> str:
        """Generate a deterministic MongoDB document ID."""

        return f"{dataset_id}:{version_id}"

    @classmethod
    def _to_document(
        cls,
        profile: DatasetProfile,
        version_id: str,
    ) -> dict[str, Any]:
        """Convert a dataset profile into a MongoDB document."""

        if not version_id or not version_id.strip():
            raise ValueError("version_id cannot be empty")

        document = asdict(profile)

        document["_id"] = cls._get_document_id(
            profile.dataset_id,
            version_id,
        )
        document["version_id"] = version_id

        return document

    @staticmethod
    def _from_document(
        document: dict[str, Any],
    ) -> DatasetProfile:
        """Convert a MongoDB document into a DatasetProfile."""

        document = document.copy()

        document.pop("_id", None)
        document.pop("version_id", None)

        document["columns"] = [
            ColumnProfile(**column)
            for column in document.get("columns", [])
        ]

        return DatasetProfile(**document)
