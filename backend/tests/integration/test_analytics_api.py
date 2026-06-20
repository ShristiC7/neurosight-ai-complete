"""Integration tests for batch analytics API endpoints."""
import uuid
import pytest
import pytest_asyncio
from datetime import datetime, date, timedelta, timezone
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def auth_client():
    """Client with a registered and logged-in user."""
    import time
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        email = f"analytics_test_{int(time.time()*1000)}@test.com"
        reg = await c.post("/api/v1/auth/register", json={
            "name": "Analytics Tester",
            "email": email,
            "password": "SecurePass123!",
        })
        token = reg.json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


@pytest.mark.anyio
async def test_get_batch_analytics_defaults(auth_client):
    resp = await auth_client.get("/api/v1/analytics/batch")
    assert resp.status_code == 200
    data = resp.json()
    assert "user_id" in data
    assert "data_points" in data
    assert "summary" in data
    assert isinstance(data["data_points"], list)
    assert "avg_fatigue_score" in data["summary"]
    assert "total_sessions" in data["summary"]


@pytest.mark.anyio
async def test_get_batch_analytics_date_filters(auth_client):
    today = date.today()
    start = today - timedelta(days=5)
    resp = await auth_client.get(
        f"/api/v1/analytics/batch?start_date={start.isoformat()}&end_date={today.isoformat()}&period=daily"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["start_date"] == start.isoformat()
    assert data["end_date"] == today.isoformat()
    assert data["period"] == "daily"


@pytest.mark.anyio
async def test_get_batch_analytics_invalid_range(auth_client):
    today = date.today()
    start = today + timedelta(days=1)  # start after end
    resp = await auth_client.get(
        f"/api/v1/analytics/batch?start_date={start.isoformat()}&end_date={today.isoformat()}"
    )
    assert resp.status_code == 422
    assert "start_date" in resp.json()["detail"]


@pytest.mark.anyio
async def test_get_batch_analytics_too_long_range(auth_client):
    today = date.today()
    start = today - timedelta(days=95)  # > 90 days
    resp = await auth_client.get(
        f"/api/v1/analytics/batch?start_date={start.isoformat()}&end_date={today.isoformat()}"
    )
    assert resp.status_code == 422
    assert "range" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_get_focus_heatmap(auth_client):
    resp = await auth_client.get("/api/v1/analytics/focus-heatmap")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # The default return length is 7 days * 24 hours = 168 cells
    assert len(data) == 168
    first_cell = data[0]
    assert "day" in first_cell
    assert "hour" in first_cell
    assert "value" in first_cell
    assert 0 <= first_cell["day"] <= 6
    assert 0 <= first_cell["hour"] <= 23


@pytest.mark.anyio
async def test_analytics_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        resp1 = await c.get("/api/v1/analytics/batch")
        assert resp1.status_code == 401

        resp2 = await c.get("/api/v1/analytics/focus-heatmap")
        assert resp2.status_code == 401
