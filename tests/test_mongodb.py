
import pytest

from backend.infrastructure.database.mongodb import MongoDB, mongodb
from backend.config.settings import settings


@pytest.fixture
def mongo_manager():
    """Create an independent MongoDB manager for testing."""
    return MongoDB()


@pytest.mark.asyncio
async def test_mongodb_connection(mongo_manager):
    """Verify MongoDB connection succeeds."""
    await mongo_manager.connect()

    try:
        assert mongo_manager.client is not None
        assert mongo_manager.database is not None
    finally:
        await mongo_manager.disconnect()


@pytest.mark.asyncio
async def test_mongodb_database_name(mongo_manager):
    """Verify configured database is selected."""
    await mongo_manager.connect()

    try:
        assert mongo_manager.database.name == settings.mongodb_database
        assert mongo_manager.database.name == "solvixaidb"
    finally:
        await mongo_manager.disconnect()


def test_get_database_before_connection(mongo_manager):
    """Verify database access fails before connecting."""
    with pytest.raises(RuntimeError, match="MongoDB is not connected"):
        mongo_manager.get_database()


@pytest.mark.asyncio
async def test_get_database_after_connection(mongo_manager):
    """Verify database access works after connecting."""
    await mongo_manager.connect()

    try:
        database = mongo_manager.get_database()

        assert database is mongo_manager.database
    finally:
        await mongo_manager.disconnect()


@pytest.mark.asyncio
async def test_mongodb_disconnect(mongo_manager):
    """Verify MongoDB resources are released."""
    await mongo_manager.connect()
    await mongo_manager.disconnect()

    assert mongo_manager.client is None
    assert mongo_manager.database is None


@pytest.mark.asyncio
async def test_mongodb_multiple_connect_calls(mongo_manager):
    """Verify repeated connect calls do not create new clients."""
    await mongo_manager.connect()

    try:
        original_client = mongo_manager.client

        await mongo_manager.connect()

        assert mongo_manager.client is original_client
    finally:
        await mongo_manager.disconnect()


@pytest.mark.asyncio
async def test_shared_mongodb_instance():
    """Verify the shared application instance can connect."""
    await mongodb.connect()

    try:
        assert mongodb.get_database().name == settings.mongodb_database
    finally:
        await mongodb.disconnect()
