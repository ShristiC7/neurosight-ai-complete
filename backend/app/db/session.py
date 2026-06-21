"""
NeuroSight AI — Async Database Session & Base Model
SQLAlchemy 2.0 async engine with connection pooling.
"""

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


# -----------------------------------------------------------
# Async Engine
# -----------------------------------------------------------
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,  # Detect stale connections
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# -----------------------------------------------------------
# Declarative Base with Mixins
# -----------------------------------------------------------
class Base(DeclarativeBase):
    """Base for all SQLAlchemy models."""
    pass


class TimestampMixin:
    """Adds created_at / updated_at to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDMixin:
    """UUID primary key."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class BaseModel(UUIDMixin, TimestampMixin, Base):
    """Abstract base with UUID PK and timestamps."""

    __abstract__ = True

    def __rich_repr__(self):
        return (self.id,)

    def to_dict(self) -> dict:
        return {
            col.name: getattr(self, col.name)
            for col in self.__table__.columns
        }


# -----------------------------------------------------------
# Dependency — Async DB Session
# -----------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async DB session.
    Automatically commits on success and rolls back on error.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# -----------------------------------------------------------
# Lazy FatigueClassifier Accessor
# -----------------------------------------------------------
_fatigue_classifier = None

def get_fatigue_classifier():
    """
    Lazily load the FatigueClassifier from the ml-models directory.
    Only imported on first call to avoid breaking imports in testing/CI.
    """
    global _fatigue_classifier
    if _fatigue_classifier is None:
        import importlib.util
        import pathlib
        _eye_fatigue_path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "ml-models" / "eye-fatigue" / "src" / "model.py"
        )
        _spec = importlib.util.spec_from_file_location("eye_fatigue_model", _eye_fatigue_path)
        _eye_fatigue_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_eye_fatigue_mod)
        _fatigue_classifier = _eye_fatigue_mod.FatigueClassifier
    return _fatigue_classifier

