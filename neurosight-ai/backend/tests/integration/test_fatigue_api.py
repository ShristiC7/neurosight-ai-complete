"""Integration tests for fatigue detection API endpoints."""
import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timezone
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
        email = f"fatigue_test_{int(time.time()*1000)}@test.com"
        reg = await c.post("/api/v1/auth/register", json={
            "name": "Fatigue Tester",
            "email": email,
            "password": "SecurePass123!",
        })
        token = reg.json()["access_token"]
        c.headers["Authorization"] = f"Bearer {token}"
        yield c


@pytest.mark.anyio
async def test_store_fatigue_metric(auth_client):
    session_id = str(uuid.uuid4())
    resp = await auth_client.post("/api/v1/fatigue/metrics", json={
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blink_rate": 16.5,
        "eye_aspect_ratio": 0.31,
        "mouth_aspect_ratio": 0.22,
        "fatigue_score": 28.0,
        "drowsiness_level": "mild",
        "confidence": 0.91,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["fatigue_score"] == 28.0
    assert data["drowsiness_level"] == "mild"
    assert "id" in data


@pytest.mark.anyio
async def test_store_fatigue_metric_invalid_score(auth_client):
    resp = await auth_client.post("/api/v1/fatigue/metrics", json={
        "session_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blink_rate": 16.5,
        "eye_aspect_ratio": 0.31,
        "mouth_aspect_ratio": 0.22,
        "fatigue_score": 999.0,  # invalid
        "drowsiness_level": "alert",
        "confidence": 0.9,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_get_fatigue_trend_empty(auth_client):
    resp = await auth_client.get("/api/v1/fatigue/trend?hours=8")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_points" in data
    assert "avg_fatigue_score" in data
    assert isinstance(data["data_points"], list)


@pytest.mark.anyio
async def test_get_fatigue_trend_invalid_hours(auth_client):
    resp = await auth_client.get("/api/v1/fatigue/trend?hours=9999")
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_get_session_fatigue(auth_client):
    session_id = str(uuid.uuid4())
    # Store a metric first
    await auth_client.post("/api/v1/fatigue/metrics", json={
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blink_rate": 18.0,
        "eye_aspect_ratio": 0.29,
        "mouth_aspect_ratio": 0.20,
        "fatigue_score": 45.0,
        "drowsiness_level": "moderate",
        "confidence": 0.88,
    })
    resp = await auth_client.get(f"/api/v1/fatigue/session/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["fatigue_score"] == 45.0


@pytest.mark.anyio
async def test_fatigue_requires_auth():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        resp = await c.get("/api/v1/fatigue/trend")
        assert resp.status_code == 401
