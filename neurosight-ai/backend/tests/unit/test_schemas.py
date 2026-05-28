"""Tests for Pydantic schemas — validation and serialisation."""
import uuid
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.schemas.auth import LoginRequest, RegisterRequest, RefreshRequest
from app.schemas.fatigue import FatigueMetricCreate
from app.models.models import DrowsinessLevel


# ── Auth Schemas ──────────────────────────────────────────────────────────────

class TestLoginRequest:
    def test_valid(self):
        req = LoginRequest(email="test@example.com", password="secret123")
        assert req.email == "test@example.com"

    def test_invalid_email(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="not-an-email", password="secret123")

    def test_short_password_rejected(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="a@b.com", password="short")


class TestRegisterRequest:
    def test_valid(self):
        req = RegisterRequest(name="Alice", email="alice@example.com", password="SecurePass1!")
        assert req.name == "Alice"

    def test_name_too_short(self):
        with pytest.raises(ValidationError):
            RegisterRequest(name="A", email="a@b.com", password="SecurePass1!")

    def test_password_too_short(self):
        with pytest.raises(ValidationError):
            RegisterRequest(name="Alice", email="a@b.com", password="ab")


# ── Fatigue Schemas ───────────────────────────────────────────────────────────

class TestFatigueMetricCreate:
    def _valid(self, **overrides):
        base = dict(
            session_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            blink_rate=16.0,
            eye_aspect_ratio=0.32,
            mouth_aspect_ratio=0.22,
            fatigue_score=30.0,
            drowsiness_level=DrowsinessLevel.MILD,
            confidence=0.9,
        )
        base.update(overrides)
        return FatigueMetricCreate(**base)

    def test_valid_metric(self):
        m = self._valid()
        assert m.fatigue_score == 30.0
        assert m.drowsiness_level == DrowsinessLevel.MILD

    def test_fatigue_score_over_100(self):
        with pytest.raises(ValidationError):
            self._valid(fatigue_score=101.0)

    def test_fatigue_score_negative(self):
        with pytest.raises(ValidationError):
            self._valid(fatigue_score=-5.0)

    def test_ear_over_1(self):
        with pytest.raises(ValidationError):
            self._valid(eye_aspect_ratio=1.5)

    def test_ear_negative(self):
        with pytest.raises(ValidationError):
            self._valid(eye_aspect_ratio=-0.1)

    def test_confidence_over_1(self):
        with pytest.raises(ValidationError):
            self._valid(confidence=1.5)

    def test_alert_level(self):
        m = self._valid(fatigue_score=5.0, drowsiness_level=DrowsinessLevel.ALERT)
        assert m.drowsiness_level == DrowsinessLevel.ALERT

    def test_critical_level(self):
        m = self._valid(fatigue_score=95.0, drowsiness_level=DrowsinessLevel.CRITICAL)
        assert m.drowsiness_level == DrowsinessLevel.CRITICAL
