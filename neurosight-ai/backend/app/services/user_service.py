"""
NeuroSight AI — User Service
Database operations for user management and refresh token handling.
"""
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import User, RefreshToken

logger = structlog.get_logger(__name__)


class UserService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, user_id: str) -> Optional[User]:
        try:
            uid = uuid.UUID(user_id)
        except ValueError:
            return None
        result = await self.db.execute(select(User).where(User.id == uid))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    async def create(self, name: str, email: str, password_hash: str) -> User:
        user = User(
            name=name,
            email=email.lower().strip(),
            password_hash=password_hash,
            preferences={
                "workHoursStart": 9,
                "workHoursEnd": 18,
                "breakDuration": 5,
                "timezone": "UTC",
                "theme": "dark",
                "notifications": {
                    "fatigueAlerts": True,
                    "breakReminders": True,
                    "productivityInsights": True,
                    "burnoutWarnings": True,
                },
            },
        )
        self.db.add(user)
        await self.db.flush()
        logger.info("User created", user_id=str(user.id), email=user.email)
        return user

    async def store_refresh_token(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            is_revoked=False,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        self.db.add(token)
        await self.db.flush()
        return token

    async def validate_refresh_token(
        self, raw_token: str
    ) -> tuple[Optional[User], Optional[RefreshToken]]:
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        result = await self.db.execute(
            select(RefreshToken).where(
                and_(
                    RefreshToken.token_hash == token_hash,
                    RefreshToken.is_revoked == False,
                    RefreshToken.expires_at > datetime.now(timezone.utc),
                )
            )
        )
        stored_token = result.scalar_one_or_none()
        if not stored_token:
            return None, None

        user = await self.get_by_id(str(stored_token.user_id))
        return user, stored_token

    async def revoke_refresh_token(self, token_id: uuid.UUID) -> None:
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.id == token_id)
            .values(is_revoked=True)
        )

    async def revoke_refresh_token_by_value(self, raw_token: str) -> None:
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .values(is_revoked=True)
        )

    async def revoke_all_user_tokens(self, user_id: uuid.UUID) -> None:
        await self.db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id)
            .values(is_revoked=True)
        )
