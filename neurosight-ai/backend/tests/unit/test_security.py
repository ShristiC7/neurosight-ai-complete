"""Tests for security utilities — JWT, password hashing."""
import time
import pytest
from app.core.security import (
    hash_password, verify_password,
    create_access_token, decode_access_token,
    create_refresh_token,
)


def test_password_hash_is_not_plain():
    plain = "MyStr0ngP@ssword!"
    hashed = hash_password(plain)
    assert hashed != plain
    assert len(hashed) > 20


def test_password_verify_correct():
    plain = "CorrectHorseBatteryStaple"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_password_verify_wrong():
    hashed = hash_password("correct_password")
    assert verify_password("wrong_password", hashed) is False


def test_password_verify_empty():
    hashed = hash_password("some_password")
    assert verify_password("", hashed) is False


def test_access_token_contains_subject():
    token = create_access_token("user-abc-123")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-abc-123"


def test_access_token_type_is_access():
    token = create_access_token("user-xyz")
    payload = decode_access_token(token)
    assert payload["type"] == "access"


def test_access_token_has_jti():
    token = create_access_token("user-xyz")
    payload = decode_access_token(token)
    assert "jti" in payload
    assert len(payload["jti"]) > 8


def test_access_token_has_exp():
    token = create_access_token("user-xyz")
    payload = decode_access_token(token)
    assert payload["exp"] > time.time()


def test_invalid_token_raises():
    from jose import JWTError
    with pytest.raises(JWTError):
        decode_access_token("not.a.valid.token")


def test_tampered_token_raises():
    from jose import JWTError
    token = create_access_token("user-xyz")
    tampered = token[:-5] + "XXXXX"
    with pytest.raises(JWTError):
        decode_access_token(tampered)


def test_refresh_token_is_unique():
    t1, h1, e1 = create_refresh_token()
    t2, h2, e2 = create_refresh_token()
    assert t1 != t2
    assert h1 != h2


def test_refresh_token_hash_differs_from_raw():
    raw, hashed, _ = create_refresh_token()
    assert raw != hashed


def test_refresh_token_expiry_is_future():
    from datetime import timezone
    _, _, expires_at = create_refresh_token()
    from datetime import datetime
    assert expires_at > datetime.now(timezone.utc)


def test_extra_claims_in_token():
    token = create_access_token("user-xyz", extra_claims={"role": "admin"})
    payload = decode_access_token(token)
    assert payload["role"] == "admin"
