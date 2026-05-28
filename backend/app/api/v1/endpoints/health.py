"""NeuroSight AI — Health Check Endpoints"""
import time
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.redis import redis_client

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.get("/", summary="Basic liveness check")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/ready", summary="Readiness — all dependencies")
async def readiness(db: Annotated[AsyncSession, Depends(get_db)]):
    checks: dict[str, str] = {}
    healthy = True
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        healthy = False
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        healthy = False
    try:
        from app.services.ml_registry import ModelRegistry
        loaded = ModelRegistry.loaded_models()
        checks["ml_models"] = f"loaded: {loaded}" if loaded else "heuristics only"
    except Exception as e:
        checks["ml_models"] = f"error: {e}"
    return {"status": "ready" if healthy else "degraded", "checks": checks, "timestamp": time.time()}
