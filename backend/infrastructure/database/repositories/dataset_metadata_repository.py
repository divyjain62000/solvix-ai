from dataclasses import asdict

from backend.core.ingestion.models import PersistentDatasetMetadata
from backend.infrastructure.database.base_repository import BaseRepository
from backend.infrastructure.database.mongodb import mongodb


class DatasetMetadataRepository(BaseRepository):
    """Manage persistent dataset metadata."""

    def __init__(self) -> None:
        database = mongodb.get_database()
        super().__init__(database["datasets"])

    async def save(
        self,
        metadata: PersistentDatasetMetadata,
    ) -> str:
        """Persist dataset metadata in MongoDB."""

        document = self._to_document(metadata)

        return await self.insert_one(document)

    async def get_by_id(
        self,
        dataset_id: str,
    ) -> PersistentDatasetMetadata | None:
        """Retrieve dataset metadata by its ID."""

        document = await self.find_one({"_id": dataset_id})

        if document is None:
            return None

        return self._from_document(document)

    async def update_active_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> PersistentDatasetMetadata | None:
        """Update and return the active dataset version."""

        document = await self.find_one_and_update(
            {"_id": dataset_id},
            {"$set": {"active_version_id": version_id}},
        )

        if document is None:
            return None

        return self._from_document(document)

    async def delete(self, dataset_id: str) -> bool:
        """Delete dataset metadata by its ID."""

        return await self.delete_one({"_id": dataset_id})

    @staticmethod
    def _to_document(
        metadata: PersistentDatasetMetadata,
    ) -> dict:
        """Convert dataset metadata into a MongoDB document."""

        document = asdict(metadata)

        document["_id"] = document.pop("dataset_id")

        return document

    @staticmethod
    def _from_document(
        document: dict,
    ) -> PersistentDatasetMetadata:
        """Convert a MongoDB document into dataset metadata."""

        document = document.copy()
        document["dataset_id"] = document.pop("_id")

        return PersistentDatasetMetadata(**document)
