"""
Integration tests for the authentication API endpoints.
Requires a real PostgreSQL database (set via env vars).
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


@pytest_asyncio.fixture
async def registered_user(client):
    """Helper: registers a unique user and returns the response body."""
    import time
    email = f"test_{int(time.time() * 1000)}@neurosight.com"
    resp = await client.post("/api/v1/auth/register", json={
        "name": "Integration Test User",
        "email": email,
        "password": "SecurePass123!",
    })
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.anyio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_register_success(client):
    import time
    resp = await client.post("/api/v1/auth/register", json={
        "name": "New User",
        "email": f"new_{int(time.time()*1000)}@test.com",
        "password": "SecurePass123!",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"].endswith("@test.com")


@pytest.mark.anyio
async def test_register_duplicate_email(client, registered_user):
    resp = await client.post("/api/v1/auth/register", json={
        "name": "Duplicate",
        "email": registered_user["user"]["email"],
        "password": "AnotherPass123!",
    })
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_register_invalid_email(client):
    resp = await client.post("/api/v1/auth/register", json={
        "name": "Bad Email",
        "email": "not-an-email",
        "password": "SecurePass123!",
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_register_short_password(client):
    import time
    resp = await client.post("/api/v1/auth/register", json={
        "name": "User",
        "email": f"short_{int(time.time()*1000)}@test.com",
        "password": "short",
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_login_success(client, registered_user):
    resp = await client.post("/api/v1/auth/login", json={
        "email": registered_user["user"]["email"],
        "password": "SecurePass123!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["id"] == registered_user["user"]["id"]


@pytest.mark.anyio
async def test_login_wrong_password(client, registered_user):
    resp = await client.post("/api/v1/auth/login", json={
        "email": registered_user["user"]["email"],
        "password": "WrongPassword!",
    })
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_login_unknown_email(client):
    resp = await client.post("/api/v1/auth/login", json={
        "email": "nobody@nowhere.com",
        "password": "SomePass123!",
    })
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_protected_without_token(client):
    resp = await client.get("/api/v1/users/me")
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_protected_with_token(client, registered_user):
    token = registered_user["access_token"]
    resp = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == registered_user["user"]["email"]


@pytest.mark.anyio
async def test_protected_with_invalid_token(client):
    resp = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_token_refresh(client, registered_user):
    refresh_token = registered_user["refresh_token"]
    resp = await client.post("/api/v1/auth/refresh", json={
        "refresh_token": refresh_token,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    # New token should differ from original
    assert data["access_token"] != registered_user["access_token"]


@pytest.mark.anyio
async def test_logout(client, registered_user):
    resp = await client.post("/api/v1/auth/logout", json={
        "refresh_token": registered_user["refresh_token"],
    })
    assert resp.status_code == 204
