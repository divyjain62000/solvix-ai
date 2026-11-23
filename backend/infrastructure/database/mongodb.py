
"""MongoDB connection management."""

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from backend.config.settings import settings


class MongoDB:
    """Manage the application's MongoDB connection."""

    def __init__(self) -> None:
        self.client: AsyncMongoClient | None = None
        self.database: AsyncDatabase | None = None

    async def connect(self) -> None:
        """Establish and verify MongoDB connection."""

        if self.client is not None:
            return

        client = AsyncMongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )

        try:
            await client.admin.command("ping")

            self.client = client
            self.database = client[settings.mongodb_database]

        except Exception:
            await client.close()
            raise

    async def disconnect(self) -> None:
        """Close MongoDB connection and release resources."""

        if self.client is not None:
            await self.client.close()

        self.client = None
        self.database = None

    def get_database(self) -> AsyncDatabase:
        """Return the active MongoDB database."""

        if self.database is None:
            raise RuntimeError(
                "MongoDB is not connected. Call connect() first."
            )

        return self.database


# Shared MongoDB manager for the application.
mongodb = MongoDB()