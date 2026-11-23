
"""Generic MongoDB repository."""

from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection


class BaseRepository:
    """Provide common asynchronous MongoDB operations."""

    def __init__(
        self,
        collection: AsyncCollection,
    ) -> None:
        self.collection = collection

    async def insert_one(self, document: dict[str, Any]) -> str:
        """Insert a document and return its ID."""

        result = await self.collection.insert_one(document)
        return str(result.inserted_id)

    async def find_one(
        self,
        query: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Find a single document."""

        return await self.collection.find_one(query)

    async def find_many(
        self,
        query: dict[str, Any] | None = None,
        limit: int = 0,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """Find multiple documents with pagination."""

        if limit < 0 or skip < 0:
            raise ValueError("limit and skip cannot be negative")

        cursor = self.collection.find(query or {}).skip(skip)

        if limit > 0:
            cursor = cursor.limit(limit)

        return await cursor.to_list()

    async def update_one(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
    ) -> bool:
        """Update a single document.

        Returns True if the document was modified.
        Returns False if no document matched or no values changed.
        """

        result = await self.collection.update_one(query, update)
        return result.modified_count > 0

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update a document and return its updated state."""

        return await self.collection.find_one_and_update(
            query,
            update,
            return_document=ReturnDocument.AFTER,
        )

    async def delete_one(self, query: dict[str, Any]) -> bool:
        """Delete a single document."""

        result = await self.collection.delete_one(query)
        return result.deleted_count > 0

    async def exists(self, query: dict[str, Any]) -> bool:
        """Check whether a matching document exists."""

        return (
            await self.collection.find_one(
                query,
                {"_id": 1},
            )
        ) is not None
