"""
Pytest configuration and shared fixtures for NeuroSight AI tests.
Uses pytest-asyncio in 'auto' mode with a session-scoped event loop.
"""
import os
import sys
import pytest

# Higher recursion limit to avoid Rich RecursionError in deep model objects
sys.setrecursionlimit(5000)

# ---------------------------------------------------------------
# Force test environment BEFORE any app code is imported.
# Using dict-style assignment (not setdefault) so these override .env
# ---------------------------------------------------------------
os.environ["ENVIRONMENT"] = "testing"
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5433"      # user-space test postgres on 5433
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


# ---------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
async def fakeredis_client():
    """
    Inject a FakeAsyncRedis instance for the entire test session.
    Prevents any real Redis connection from being attempted.
    """
    import fakeredis
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    from app.core import redis as redis_mod
    redis_mod.redis_client = client
    redis_mod.cache = redis_mod.CacheClient(client)
    yield client
    await client.flushall()
    await client.close()
