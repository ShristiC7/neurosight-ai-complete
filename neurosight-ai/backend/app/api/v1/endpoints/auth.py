"""
NeuroSight AI — Authentication Endpoints
JWT-based auth with refresh token rotation.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    hash_password,
    verify_refresh_token,
)
from app.db.session import get_db
from app.models.models import User, RefreshToken
from app.schemas.auth import (
    TokenResponse,
    LoginRequest,
    RegisterRequest,
    RefreshRequest,
)
from app.schemas.users import UserResponse
from app.services.user_service import UserService

logger = structlog.get_logger(__name__)
router = APIRouter()


# -----------------------------------------------------------
# POST /auth/register
# -----------------------------------------------------------
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user account",
)
async def register(
    payload: RegisterRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Create a new user account and return JWT tokens.

    - Validates email uniqueness
    - Hashes password with bcrypt (cost factor 12)
    - Returns access + refresh token pair
    """
    user_service = UserService(db)

    # Check email uniqueness
    existing = await user_service.get_by_email(payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Create user
    user = await user_service.create(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )

    # Issue tokens
    access_token = create_access_token(user_id=str(user.id))
    refresh_token, token_hash, expires_at = create_refresh_token()

    # Persist refresh token
    await user_service.store_refresh_token(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )

    logger.info("User registered", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_at=int(
            (datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
            * 1000
        ),
        user=UserResponse.model_validate(user),
    )


# -----------------------------------------------------------
# POST /auth/login
# -----------------------------------------------------------
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and obtain tokens",
)
async def login(
    payload: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Authenticate with email + password, return JWT token pair.
    Implements constant-time password comparison to prevent timing attacks.
    """
    user_service = UserService(db)
    user = await user_service.get_by_email(payload.email)

    # Constant-time check (always call verify_password even if user not found)
    dummy_hash = "$2b$12$invalidhashtopreventtimingattacks"
    stored_hash = user.password_hash if user else dummy_hash

    if not verify_password(payload.password, stored_hash) or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    access_token = create_access_token(user_id=str(user.id))
    refresh_token, token_hash, expires_at = create_refresh_token()

    await user_service.store_refresh_token(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )

    logger.info("User logged in", user_id=str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_at=int(
            (datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
            * 1000
        ),
        user=UserResponse.model_validate(user),
    )


# -----------------------------------------------------------
# POST /auth/refresh
# -----------------------------------------------------------
@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token using refresh token",
)
async def refresh_token(
    payload: RefreshRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Exchange a valid refresh token for a new access + refresh token pair.
    Implements refresh token rotation — old token is revoked immediately.
    """
    user_service = UserService(db)

    user, stored_token = await user_service.validate_refresh_token(payload.refresh_token)
    if not user or not stored_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Rotate refresh token
    await user_service.revoke_refresh_token(stored_token.id)

    new_access_token = create_access_token(user_id=str(user.id))
    new_refresh_token, token_hash, expires_at = create_refresh_token()

    await user_service.store_refresh_token(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_at=int(
            (datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
            * 1000
        ),
        user=UserResponse.model_validate(user),
    )


# -----------------------------------------------------------
# POST /auth/logout
# -----------------------------------------------------------
@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke refresh token and end session",
)
async def logout(
    payload: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Revoke the provided refresh token. Idempotent."""
    user_service = UserService(db)
    await user_service.revoke_refresh_token_by_value(payload.refresh_token)
    logger.info("User logged out")
