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

# Override database to use test DB before any app imports
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("POSTGRES_DB", os.environ.get("POSTGRES_DB", "neurosight_test"))
os.environ.setdefault("SECRET_KEY", "test-secret-key-exactly-32-characters-long!")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-key-exactly-32-characters-ok!")
os.environ.setdefault("POSTGRES_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "neurosight_dev_password"))
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("QDRANT_HOST", "localhost")


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
