"""
Pytest configuration and shared fixtures for NeuroSight AI tests.
Sets up async engine with test database and cleans up after each test.
"""
import os
import asyncio
import pytest
import pytest_asyncio
import sys

# Set a higher recursion limit to avoid Rich RecursionError in deep model objects
sys.setrecursionlimit(5000)

# Force-set environment variables BEFORE any app code is imported.
# Using [] not setdefault so these override anything in .env file.
os.environ["ENVIRONMENT"] = "testing"
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5433"   # user-space test postgres
os.environ["POSTGRES_DB"] = "neurosight_test"
os.environ["POSTGRES_USER"] = "neurosight"
os.environ["POSTGRES_PASSWORD"] = "neurosight_dev_password"
os.environ["SECRET_KEY"] = "test-secret-key-exactly-32-characters-long!"
os.environ["JWT_SECRET_KEY"] = "test-jwt-key-exactly-32-characters-ok!"
os.environ["REDIS_HOST"] = "localhost"
os.environ["QDRANT_HOST"] = "localhost"

# Clear pydantic-settings lru_cache so Settings() re-reads from env
from app.core.config import get_settings
get_settings.cache_clear()



@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="session", autouse=True)
async def fakeredis_client():
    """Create a fake async Redis client for the test session.
    This replaces the real Redis client used by the application with an
    in‑memory mock provided by `fakeredis`. It ensures that integration
    tests that depend on Redis do not require a live Redis server.
    """
    import fakeredis
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    # Monkey‑patch the global redis_client used throughout the codebase
    from app.core import redis as redis_mod
    redis_mod.redis_client = client
    redis_mod.cache = redis_mod.CacheClient(client)
    yield client
    await client.flushall()
    await client.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_test_db():
    """Create all SQLAlchemy tables in the test database before the test session."""
    from app.db.session import engine, Base
    # import all models so they are registered on Base.metadata
    import app.models.models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
