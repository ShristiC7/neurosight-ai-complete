# NeuroSight AI — Next Steps Guide

## Part 3 of 3: Testing Guide

> Step-by-step instructions for testing every layer of the platform —
> from unit tests through to end-to-end browser testing.
> Run in the order shown. Each section builds on the previous.

---

## 3.1 Pre-Flight Check

Before running any tests, verify the environment is healthy:

```bash
# From the project root
cd neurosight-ai

# Check Node version
node --version      # Must be 22+

# Check Python version
python3 --version   # Must be 3.12+

# Check Docker is running
docker info         # Must show server info, not an error

# Check ports are free
lsof -i :3000   # Frontend — must be empty
lsof -i :8000   # Backend  — must be empty
lsof -i :5432   # Postgres — must be empty (or your DB is already running)
lsof -i :6379   # Redis    — must be empty
```

If any port is in use, find and kill the process:

```bash
# Example: kill whatever is on port 8000
lsof -ti :8000 | xargs kill -9
```

---

## 3.2 Start the Infrastructure

All tests below assume the databases are running. The fastest way:

```bash
# Start only the database services (not the app itself)
docker compose -f infrastructure/docker/docker-compose.yml \
  up -d postgres redis qdrant

# Wait ~10 seconds, then verify
docker compose -f infrastructure/docker/docker-compose.yml ps
```

Expected output — all three should show `healthy` or `running`:
```
NAME       STATUS          PORTS
postgres   Up (healthy)    0.0.0.0:5432->5432/tcp
redis      Up (healthy)    0.0.0.0:6379->6379/tcp
qdrant     Up              0.0.0.0:6333->6333/tcp
```

---

## 3.3 Backend Unit Tests

```bash
cd backend

# Create virtual environment if you haven't
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all unit tests
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=70
```

### What the tests cover

The test files to create (put in `backend/tests/unit/`):

**`test_security.py`** — JWT and password utilities:
```python
"""Tests for security utilities."""
import pytest
from app.core.security import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
    create_refresh_token,
)


def test_password_hash_and_verify():
    plain = "MyStr0ngP@ssword!"
    hashed = hash_password(plain)
    assert hashed != plain                        # Must be hashed
    assert verify_password(plain, hashed)         # Correct password works
    assert not verify_password("wrong", hashed)   # Wrong password fails


def test_password_timing_safe():
    """Wrong password must not be noticeably faster than correct one."""
    import time
    hashed = hash_password("correct_password")
    times = []
    for pw in ["correct_password", "wrong_password"]:
        start = time.perf_counter()
        verify_password(pw, hashed)
        times.append(time.perf_counter() - start)
    # Both should take similar time (bcrypt constant-time)
    assert abs(times[0] - times[1]) < 0.1  # Within 100ms


def test_access_token_round_trip():
    user_id = "test-user-123"
    token = create_access_token(user_id)
    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["type"] == "access"


def test_expired_token_raises():
    from datetime import timedelta
    from app.core.config import settings
    # Create token that expired 1 minute ago
    token = create_access_token("user", extra_claims={
        "exp": __import__("time").time() - 60
    })
    with pytest.raises(Exception):  # jose.JWTError
        decode_access_token(token)


def test_refresh_token_returns_unique_values():
    token1, hash1, exp1 = create_refresh_token()
    token2, hash2, exp2 = create_refresh_token()
    assert token1 != token2
    assert hash1 != hash2
```

**`test_fatigue_schemas.py`** — Pydantic schema validation:
```python
"""Tests for fatigue schemas."""
import uuid
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.schemas.fatigue import FatigueMetricCreate
from app.models.models import DrowsinessLevel


def test_valid_fatigue_metric():
    data = FatigueMetricCreate(
        session_id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        blink_rate=15.0,
        eye_aspect_ratio=0.32,
        mouth_aspect_ratio=0.25,
        fatigue_score=25.0,
        drowsiness_level=DrowsinessLevel.ALERT,
        confidence=0.92,
    )
    assert data.fatigue_score == 25.0
    assert data.drowsiness_level == DrowsinessLevel.ALERT


def test_fatigue_score_out_of_range():
    with pytest.raises(ValidationError):
        FatigueMetricCreate(
            session_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            blink_rate=15.0,
            eye_aspect_ratio=0.32,
            mouth_aspect_ratio=0.25,
            fatigue_score=150.0,  # Over 100 — invalid
            drowsiness_level=DrowsinessLevel.ALERT,
        )


def test_ear_bounds():
    with pytest.raises(ValidationError):
        FatigueMetricCreate(
            session_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            blink_rate=15.0,
            eye_aspect_ratio=1.5,  # Over 1.0 — invalid
            mouth_aspect_ratio=0.25,
            fatigue_score=20.0,
            drowsiness_level=DrowsinessLevel.MILD,
        )
```

**`test_ml_models.py`** — Model architecture validation:
```python
"""Tests for ML model architectures (no trained weights needed)."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml-models/eye-fatigue/src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml-models/voice-stress/src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml-models/rl-agent/src"))


class TestFatigueModel:
    def test_model_instantiates(self):
        from model import FatigueClassifier
        model = FatigueClassifier()
        assert model is not None

    def test_forward_with_both_inputs(self):
        from model import FatigueClassifier
        model = FatigueClassifier()
        model.eval()
        eye = torch.randn(2, 1, 48, 48)  # batch of 2
        seq = torch.randn(2, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        assert out["fatigue_score"].shape == (2,)
        assert out["predicted_class"].shape == (2,)
        assert out["probabilities"].shape == (2, 5)

    def test_fatigue_score_range(self):
        from model import FatigueClassifier
        model = FatigueClassifier()
        model.eval()
        eye = torch.randn(4, 1, 48, 48)
        seq = torch.randn(4, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        scores = out["fatigue_score"].numpy()
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_lstm_only_mode(self):
        """Model should work without eye images (LSTM branch only)."""
        from model import FatigueClassifier
        model = FatigueClassifier(use_cnn=False, use_lstm=True)
        seq = torch.randn(2, 30, 4)
        with torch.no_grad():
            out = model(temporal_features=seq)
        assert out["fatigue_score"].shape == (2,)


class TestVoiceStressModel:
    def test_model_instantiates(self):
        from model import VoiceStressModel
        model = VoiceStressModel()
        assert model is not None

    def test_forward_pass(self):
        from model import VoiceStressModel
        model = VoiceStressModel()
        model.eval()
        spec = torch.randn(2, 1, 128, 100)  # (batch, channel, freq, time)
        mfcc = torch.randn(2, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        assert out["stress_score"].shape == (2,)
        assert out["emotion_probs"].shape == (2, 5)

    def test_stress_score_range(self):
        from model import VoiceStressModel
        model = VoiceStressModel()
        spec = torch.randn(4, 1, 128, 100)
        mfcc = torch.randn(4, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        scores = out["stress_score"].numpy()
        assert (scores >= 0).all() and (scores <= 100).all()


class TestRLAgent:
    def test_agent_instantiates(self):
        from agent import ProductivityRLAgent
        agent = ProductivityRLAgent()
        assert agent is not None

    def test_action_in_valid_range(self):
        from agent import ProductivityRLAgent
        agent = ProductivityRLAgent()
        state = np.random.rand(12).astype(np.float32)
        action = agent.select_action(state, user_id="test")
        assert 0 <= action < 10

    def test_greedy_is_deterministic(self):
        """Greedy mode (no exploration) should give same action for same state."""
        from agent import ProductivityRLAgent
        agent = ProductivityRLAgent()
        state = np.array([0.5] * 12, dtype=np.float32)
        actions = [agent.select_action(state, user_id="test", greedy=True) for _ in range(5)]
        assert len(set(actions)) == 1  # All same

    def test_train_step_returns_loss(self):
        from agent import ProductivityRLAgent, Transition
        agent = ProductivityRLAgent()
        # Fill buffer with enough samples
        for _ in range(100):
            s = np.random.rand(12).astype(np.float32)
            agent.store_transition(s, 0, 0.5, s, False)
        loss = agent.train_step()
        assert loss is not None
        assert loss >= 0
```

**Run just the ML tests:**
```bash
cd backend
pytest tests/unit/test_ml_models.py -v
```

---

## 3.4 Backend API Tests (Integration)

These tests hit real API endpoints with a real database.

```bash
cd backend

# Ensure test database is running
# The tests use ENVIRONMENT=testing which uses a separate DB
export ENVIRONMENT=testing
export POSTGRES_DB=neurosight_test
export SECRET_KEY=test-secret-key-exactly-32-characters-long
export JWT_SECRET_KEY=test-jwt-key-exactly-32-characters-long
export POSTGRES_PASSWORD=your_postgres_password
```

Create `backend/tests/integration/test_auth_api.py`:

```python
"""Integration tests for authentication endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_register_new_user(client):
    response = await client.post("/api/v1/auth/register", json={
        "name": "Test User",
        "email": "test@neurosight.ai",
        "password": "SecurePass123!"
    })
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "test@neurosight.ai"


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    payload = {"name": "User", "email": "dup@test.com", "password": "SecurePass123!"}
    await client.post("/api/v1/auth/register", json=payload)
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_success(client):
    # Register first
    await client.post("/api/v1/auth/register", json={
        "name": "Login Test",
        "email": "login@test.com",
        "password": "SecurePass123!"
    })
    # Then login
    response = await client.post("/api/v1/auth/login", json={
        "email": "login@test.com",
        "password": "SecurePass123!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/api/v1/auth/register", json={
        "name": "User",
        "email": "wrong@test.com",
        "password": "CorrectPass123!"
    })
    response = await client.post("/api/v1/auth/login", json={
        "email": "wrong@test.com",
        "password": "WrongPass!"
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_requires_auth(client):
    response = await client.get("/api/v1/fatigue/trend")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_with_token(client):
    # Register and get token
    reg = await client.post("/api/v1/auth/register", json={
        "name": "Auth Test",
        "email": "authtest@test.com",
        "password": "SecurePass123!"
    })
    token = reg.json()["access_token"]

    # Use token on protected endpoint
    response = await client.get(
        "/api/v1/fatigue/trend",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

Run integration tests:
```bash
cd backend
pytest tests/integration/ -v --asyncio-mode=auto
```

---

## 3.5 Manual API Testing with HTTPie

After starting the backend (`uvicorn app.main:app --reload`):

```bash
# Install httpie (better than curl for APIs)
pip install httpie

# 1. Register a user
http POST localhost:8000/api/v1/auth/register \
  name="Your Name" \
  email="you@example.com" \
  password="SecurePass123!"

# 2. Save the access token
TOKEN="paste-your-access-token-here"

# 3. Store a fatigue metric
http POST localhost:8000/api/v1/fatigue/metrics \
  "Authorization: Bearer $TOKEN" \
  session_id="00000000-0000-0000-0000-000000000001" \
  blink_rate:=16.5 \
  eye_aspect_ratio:=0.32 \
  mouth_aspect_ratio:=0.20 \
  fatigue_score:=28.0 \
  drowsiness_level="mild" \
  confidence:=0.91 \
  timestamp="2025-01-01T10:00:00Z"

# 4. Get fatigue trend
http GET "localhost:8000/api/v1/fatigue/trend?hours=8" \
  "Authorization: Bearer $TOKEN"

# 5. Get recommendations
http GET localhost:8000/api/v1/recommendations/ \
  "Authorization: Bearer $TOKEN"
```

---

## 3.6 WebSocket Connection Test

```bash
# Install wscat (WebSocket test client)
npm install -g wscat

# Start the backend first
cd backend && uvicorn app.main:app --reload &

# Get a token (register first via httpie above)
TOKEN="your-access-token"
USER_ID="your-user-id-from-registration"

# Connect to WebSocket
wscat -c "ws://localhost:8000/api/v1/ws?token=${TOKEN}&userId=${USER_ID}"

# In the wscat prompt, send messages:
# Ping
{"event":"ping","payload":null}

# Start session
{"event":"session:start","payload":{"sessionId":"00000000-0000-0000-0000-000000000001"}}

# You should receive:
# {"event":"pong","payload":{"serverTime":1234567890.123}}
# {"event":"session:started","payload":{"sessionId":"..."}}
```

Expected responses:
```json
{"event": "connection:established", "payload": {"userId": "...", "features": {...}}}
{"event": "pong", "payload": {"serverTime": 1735000000.0}}
```

---

## 3.7 Frontend Component Tests

```bash
cd frontend
npm install

# Run tests in watch mode
npm run test

# Run once with coverage
npm run test -- --coverage --reporter=verbose
```

Create `frontend/src/components/dashboard/__tests__/cognitive-score-ring.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { CognitiveScoreRing } from "../cognitive-score-ring";

describe("CognitiveScoreRing", () => {
  it("renders the score value", () => {
    render(
      <CognitiveScoreRing
        score={75}
        fatigueScore={20}
        stressScore={30}
        productivityScore={80}
        isLive={false}
      />
    );
    expect(screen.getByText("75")).toBeTruthy();
  });

  it("shows Focused label at score 75", () => {
    render(
      <CognitiveScoreRing
        score={75}
        fatigueScore={20}
        stressScore={30}
        productivityScore={80}
        isLive={false}
      />
    );
    expect(screen.getByText("FOCUSED")).toBeTruthy();
  });

  it("shows LIVE indicator when monitoring active", () => {
    render(
      <CognitiveScoreRing
        score={75}
        fatigueScore={20}
        stressScore={30}
        productivityScore={80}
        isLive={true}
      />
    );
    expect(screen.getByText("LIVE")).toBeTruthy();
  });

  it("shows Optimal label at score 85", () => {
    render(
      <CognitiveScoreRing
        score={85}
        fatigueScore={10}
        stressScore={15}
        productivityScore={90}
        isLive={false}
      />
    );
    expect(screen.getByText("OPTIMAL")).toBeTruthy();
  });

  it("shows Critical label at score 10", () => {
    render(
      <CognitiveScoreRing
        score={10}
        fatigueScore={90}
        stressScore={85}
        productivityScore={15}
        isLive={false}
      />
    );
    expect(screen.getByText("CRITICAL")).toBeTruthy();
  });
});
```

Create `frontend/src/store/__tests__/dashboard-store.test.ts`:

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { useDashboardStore } from "../dashboard-store";
import type { FatigueMetrics } from "@/types";

describe("Dashboard Store", () => {
  beforeEach(() => {
    useDashboardStore.getState().reset();
  });

  it("initializes with zero fatigue score", () => {
    const { dashboard } = useDashboardStore.getState();
    expect(dashboard.fatigueScore).toBe(0);
  });

  it("updates fatigue score when updateFatigue is called", () => {
    const metrics: FatigueMetrics = {
      id: "test-id",
      userId: "user-1",
      sessionId: "session-1",
      timestamp: new Date().toISOString(),
      blinkRate: 18,
      eyeAspectRatio: 0.3,
      mouthAspectRatio: 0.2,
      headTiltAngle: 2.0,
      gazeDrift: 0.1,
      fatigueScore: 45,
      drowsinessLevel: "moderate",
      confidence: 0.9,
    };

    useDashboardStore.getState().updateFatigue(metrics);
    expect(useDashboardStore.getState().dashboard.fatigueScore).toBe(45);
  });

  it("adds recommendation to the feed", () => {
    const rec = {
      id: "rec-1",
      userId: "user-1",
      sessionId: "session-1",
      timestamp: new Date().toISOString(),
      type: "take_break" as const,
      priority: "medium" as const,
      title: "Take a Break",
      message: "You've been working for a while.",
      accepted: null,
      expiresAt: new Date(Date.now() + 900_000).toISOString(),
      metadata: {},
    };

    useDashboardStore.getState().addRecommendation(rec);
    expect(useDashboardStore.getState().recommendations).toHaveLength(1);
    expect(useDashboardStore.getState().recommendations[0].id).toBe("rec-1");
  });

  it("does not add duplicate recommendations", () => {
    const rec = {
      id: "rec-dup",
      userId: "user-1",
      sessionId: "session-1",
      timestamp: new Date().toISOString(),
      type: "hydrate" as const,
      priority: "low" as const,
      title: "Drink Water",
      message: "Stay hydrated.",
      accepted: null,
      expiresAt: new Date(Date.now() + 900_000).toISOString(),
      metadata: {},
    };

    useDashboardStore.getState().addRecommendation(rec);
    useDashboardStore.getState().addRecommendation(rec); // duplicate
    expect(useDashboardStore.getState().recommendations).toHaveLength(1);
  });

  it("dismisses a recommendation", () => {
    const rec = {
      id: "rec-dismiss",
      userId: "user-1",
      sessionId: "s1",
      timestamp: new Date().toISOString(),
      type: "stretch" as const,
      priority: "low" as const,
      title: "Stretch",
      message: "Move your body.",
      accepted: null,
      expiresAt: new Date(Date.now() + 900_000).toISOString(),
      metadata: {},
    };

    useDashboardStore.getState().addRecommendation(rec);
    useDashboardStore.getState().dismissRecommendation("rec-dismiss");
    expect(useDashboardStore.getState().recommendations).toHaveLength(0);
  });
});
```

---

## 3.8 End-to-End Browser Test

Full flow from login → monitoring → recommendation.

```bash
cd frontend

# Install Playwright
npm install -D @playwright/test
npx playwright install chromium
```

Create `frontend/e2e/dashboard.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";

test.describe("NeuroSight AI — Dashboard Flow", () => {
  test.beforeEach(async ({ page }) => {
    // Register and login via API directly (faster than UI)
    const res = await page.request.post("http://localhost:8000/api/v1/auth/register", {
      data: {
        name: "E2E Test User",
        email: `e2e+${Date.now()}@test.com`,
        password: "SecurePass123!",
      },
    });
    const data = await res.json();

    // Set auth token in localStorage
    await page.goto("http://localhost:3000");
    await page.evaluate((token) => {
      const stored = {
        state: {
          user: token.user,
          accessToken: token.access_token,
          expiresAt: token.expires_at,
        },
      };
      localStorage.setItem("neurosight-auth", JSON.stringify(stored));
    }, data);

    await page.reload();
  });

  test("dashboard loads with all panels", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    await expect(page).toHaveTitle(/NeuroSight/);

    // Check key panels exist
    await expect(page.getByText("Cognitive Score")).toBeVisible();
    await expect(page.getByText("Eye Fatigue")).toBeVisible();
    await expect(page.getByText("Voice Stress")).toBeVisible();
    await expect(page.getByText("Behavioral")).toBeVisible();
    await expect(page.getByText("Productivity")).toBeVisible();
    await expect(page.getByText("AI Coach")).toBeVisible();
    await expect(page.getByText("Weekly Focus Heatmap")).toBeVisible();
  });

  test("session start button is visible", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    const startBtn = page.getByText("▶ Start Session");
    await expect(startBtn).toBeVisible();
  });

  test("sidebar navigation works", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    await page.getByText("Analytics").click();
    await expect(page).toHaveURL(/analytics/);
  });

  test("cognitive timeline chart renders", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    await expect(page.getByText("Cognitive Timeline")).toBeVisible();
  });

  test("burnout risk gauge renders", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    await expect(page.getByText("Burnout Risk")).toBeVisible();
  });

  test("WebSocket connection indicator shows status", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    // Either LIVE or DISCONNECTED/RECONNECTING
    const wsStatus = page.locator("text=/LIVE|DISCONNECTED|RECONNECTING/");
    await expect(wsStatus.first()).toBeVisible({ timeout: 5000 });
  });
});
```

Run E2E tests (requires both frontend and backend running):
```bash
# Terminal 1: Start backend
cd backend && uvicorn app.main:app --reload

# Terminal 2: Start frontend
cd frontend && npm run dev

# Terminal 3: Run Playwright
cd frontend
npx playwright test e2e/ --headed  # --headed shows browser UI
# or headless:
npx playwright test e2e/
```

---

## 3.9 Load Testing (Optional)

Test how the backend handles concurrent users.

```bash
# Install Locust
pip install locust

# Create locustfile.py in project root
```

Create `neurosight-ai/locustfile.py`:

```python
"""
NeuroSight AI — Load Test
Run: locust -f locustfile.py --host http://localhost:8000
Then open: http://localhost:8089
"""

from locust import HttpUser, task, between
import uuid
from datetime import datetime, timezone


class NeuroSightUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    token = None
    session_id = str(uuid.uuid4())

    def on_start(self):
        """Called once per simulated user — register and login."""
        email = f"load+{uuid.uuid4().hex[:8]}@test.com"
        response = self.client.post("/api/v1/auth/register", json={
            "name": "Load Test User",
            "email": email,
            "password": "SecurePass123!",
        })
        if response.status_code == 201:
            self.token = response.json()["access_token"]

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    @task(5)  # Most common — store fatigue metrics every few seconds
    def store_fatigue_metric(self):
        if not self.token:
            return
        self.client.post(
            "/api/v1/fatigue/metrics",
            json={
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "blink_rate": 15.5,
                "eye_aspect_ratio": 0.31,
                "mouth_aspect_ratio": 0.22,
                "fatigue_score": 30.0,
                "drowsiness_level": "mild",
                "confidence": 0.89,
            },
            headers=self.auth_headers(),
            name="/fatigue/metrics",
        )

    @task(3)
    def get_fatigue_trend(self):
        if not self.token:
            return
        self.client.get(
            "/api/v1/fatigue/trend?hours=8",
            headers=self.auth_headers(),
            name="/fatigue/trend",
        )

    @task(2)
    def get_recommendations(self):
        if not self.token:
            return
        self.client.get(
            "/api/v1/recommendations/",
            headers=self.auth_headers(),
            name="/recommendations",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
```

```bash
locust -f locustfile.py --host http://localhost:8000
# Open http://localhost:8089
# Set: Users=50, Spawn rate=5, then Start

# Target thresholds:
# P50 response time < 100ms
# P95 response time < 300ms (our inference target)
# Error rate < 1%
```

---

## 3.10 Full Test Run Checklist

Work through this once before declaring the platform ready:

```
Infrastructure
□ Docker Compose services are healthy (postgres, redis, qdrant)
□ Database migrations ran: cd backend && alembic upgrade head

Backend Unit Tests
□ pytest tests/unit/ passes with 0 failures
□ Coverage report shows ≥70%
□ test_ml_models.py: all architecture tests green

Backend Integration Tests
□ pytest tests/integration/ passes
□ Auth flow (register → login → token refresh) works
□ Protected endpoints return 401 without token, 200 with token

WebSocket
□ wscat connects successfully
□ ping → pong works
□ session:start event acknowledged

Frontend Tests
□ npm run test passes with 0 failures
□ Dashboard store tests all green
□ CognitiveScoreRing renders correct labels

Manual Browser Check
□ Register a new account at http://localhost:3000
□ Dashboard loads without errors (check browser console)
□ WebSocket shows "LIVE" in top bar
□ Click "Start Session" — status changes
□ Sensor toggles appear and respond to clicks
□ Timeline chart appears
□ Focus heatmap renders

E2E Tests (optional but recommended)
□ npx playwright test passes
□ All panels visible on dashboard
□ Navigation between pages works

Load Test (staging only)
□ Locust: 50 users, <300ms P95, <1% errors
```

---

## 3.11 Common Issues & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `POSTGRES_PASSWORD authentication failed` | .env password doesn't match docker-compose | Make them identical |
| `Connection refused :8000` | Backend not started | `cd backend && uvicorn app.main:app --reload` |
| `Cannot import 'model'` in ML tests | `sys.path` not set | Add the path insert at top of test file (see examples above) |
| WebSocket immediately disconnects | Token expired or wrong user_id | Re-login and copy fresh token |
| `ModuleNotFoundError: mediapipe` | Not installed | `pip install mediapipe==0.10.21` |
| Fatigue panel shows no data | Camera permission denied | Allow camera in browser settings → refresh |
| `ONNX model not found` warning | Model not trained yet | Run training scripts first, or ignore (heuristics take over) |
| Frontend: `NEXT_PUBLIC_API_URL` is wrong | Env var not set | Check `frontend/.env.local` exists with correct URL |
| Celery tasks not running | Celery worker not started | `celery -A app.core.celery_app worker --loglevel=info` |
| `Rate limit exceeded` in tests | Too many test requests | Add `time.sleep(1)` between requests in integration tests |

---

*You now have everything needed to configure, train, and test NeuroSight AI end-to-end.*
