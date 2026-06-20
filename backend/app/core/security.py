"""
NeuroSight AI — Security Utilities
JWT token creation/verification, password hashing, auth dependencies.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Password Hashing
# -----------------------------------------------------------
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # OWASP recommended minimum
)


def hash_password(password: str) -> str:
    """Hash password using bcrypt with cost factor 12."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time password verification."""
    return pwd_context.verify(plain_password, hashed_password)


# -----------------------------------------------------------
# JWT Tokens
# -----------------------------------------------------------
def create_access_token(
    user_id: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Create a signed JWT access token.
    Short-lived (default 60 minutes).
    """
    now = datetime.now(timezone.utc)
    claims = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        "type": "access",
        "jti": secrets.token_hex(16),  # Unique token ID for revocation
    }
    if extra_claims:
        claims.update(extra_claims)

    return jwt.encode(claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT access token.
    Raises JWTError on failure.
    """
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"verify_exp": True},
    )


def create_refresh_token() -> tuple[str, str, datetime]:
    """
    Create a secure refresh token.
    Returns (raw_token, hashed_token, expires_at).
    Refresh tokens are opaque random bytes — not JWTs.
    """
    raw_token = secrets.token_urlsafe(64)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    return raw_token, token_hash, expires_at


def verify_refresh_token(raw_token: str, stored_hash: str) -> bool:
    """Constant-time comparison of refresh token hash."""
    computed_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return secrets.compare_digest(computed_hash, stored_hash)


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


# -----------------------------------------------------------
# FastAPI Security Dependency
# -----------------------------------------------------------
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    FastAPI dependency: extract and validate JWT, return current User.
    Raises 401 if token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials:
        raise credentials_exception

    try:
        payload = decode_access_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        token_type: str | None = payload.get("type")

        if not user_id or token_type != "access":
            raise credentials_exception

    except JWTError as e:
        logger.debug("JWT validation failed", error=str(e))
        raise credentials_exception

    # Load user from DB
    from app.services.user_service import UserService
    user_service = UserService(db)
    user = await user_service.get_by_id(UUID(user_id))

    if not user or not user.is_active:
        raise credentials_exception

    return user
async def get_current_user_optional(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Optional[Any]:
    """Return the current user if a valid JWT is provided, otherwise None.
    Useful for endpoints where authentication is optional.
    """
    if not credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id: str | None = payload.get("sub")
        token_type: str | None = payload.get("type")
        if not user_id or token_type != "access":
            return None
    except JWTError:
        return None
    from app.services.user_service import UserService
    user_service = UserService(db)
    user = await user_service.get_by_id(UUID(user_id))
    if not user or not user.is_active:
        return None
    return user


async def get_current_active_user(
    current_user=Depends(get_current_user),
):
    """Dependency that additionally checks user is active and verified."""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified",
        )
    return current_user
