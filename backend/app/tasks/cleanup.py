"""
NeuroSight AI — Data Cleanup Tasks
Purges expired tokens, old metrics, and stale sessions.
"""
from datetime import datetime, timezone, timedelta
import structlog
from app.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="tasks.cleanup.cleanup_expired_data", queue="cleanup")
def cleanup_expired_data():
    """
    Scheduled daily at 2AM UTC. Purges:
    - Expired refresh tokens
    - Raw metric data older than 90 days (aggregates are kept)
    - Expired recommendations
    - Abandoned active sessions older than 24 hours
    """
    import asyncio
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import delete, and_, update
    from app.models.models import (
        RefreshToken, FatigueMetric, VoiceStressMetric,
        BehavioralMetric, Recommendation, WorkSession,
    )

    async def _run():
        async with AsyncSessionLocal() as db:
            now = datetime.now(timezone.utc)
            cutoff_90 = now - timedelta(days=90)
            cutoff_24h = now - timedelta(hours=24)

            # Expired refresh tokens
            r1 = await db.execute(
                delete(RefreshToken).where(
                    RefreshToken.expires_at < now
                ).returning(RefreshToken.id)
            )
            expired_tokens = len(r1.fetchall())

            # Expired recommendations
            r2 = await db.execute(
                delete(Recommendation).where(
                    and_(Recommendation.expires_at < now, Recommendation.accepted.is_(None))
                ).returning(Recommendation.id)
            )
            expired_recs = len(r2.fetchall())

            # Raw metrics older than 90 days
            r3 = await db.execute(
                delete(FatigueMetric).where(FatigueMetric.timestamp < cutoff_90)
                .returning(FatigueMetric.id)
            )
            r4 = await db.execute(
                delete(VoiceStressMetric).where(VoiceStressMetric.timestamp < cutoff_90)
                .returning(VoiceStressMetric.id)
            )
            r5 = await db.execute(
                delete(BehavioralMetric).where(BehavioralMetric.timestamp < cutoff_90)
                .returning(BehavioralMetric.id)
            )
            old_metrics = len(r3.fetchall()) + len(r4.fetchall()) + len(r5.fetchall())

            # Abandon stale active sessions (user closed browser without ending session)
            await db.execute(
                update(WorkSession)
                .where(and_(
                    WorkSession.is_active == True,
                    WorkSession.start_time < cutoff_24h,
                ))
                .values(is_active=False, end_time=cutoff_24h)
            )

            await db.commit()
            logger.info(
                "Cleanup complete",
                expired_tokens=expired_tokens,
                expired_recommendations=expired_recs,
                old_metrics_deleted=old_metrics,
            )

    asyncio.run(_run())


@celery_app.task(name="tasks.cleanup.clear_user_cache", queue="cleanup")
def clear_user_cache(user_id: str):
    """Clears all Redis cache entries for a specific user (called on logout)."""
    import asyncio
    from app.core.redis import redis_client

    async def _run():
        keys = await redis_client.keys(f"neurosight:*:{user_id}")
        if keys:
            await redis_client.delete(*keys)
            logger.info("User cache cleared", user_id=user_id, keys_deleted=len(keys))

    asyncio.run(_run())
