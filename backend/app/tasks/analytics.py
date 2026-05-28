"""
NeuroSight AI — Analytics Aggregation Tasks
Computes session summaries, weekly digests, and focus heatmap data.
"""
from datetime import datetime, timezone, timedelta
import structlog
from app.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="tasks.analytics.aggregate_session_metrics", queue="analytics")
def aggregate_session_metrics():
    """
    Computes rolling averages for all active sessions.
    Updates WorkSession.avg_fatigue_score, avg_stress_score, total_focus_time.
    Scheduled every 5 minutes via Celery Beat.
    """
    import asyncio
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import select, func, and_
    from app.models.models import WorkSession, FatigueMetric, VoiceStressMetric, BehavioralMetric

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(WorkSession).where(WorkSession.is_active == True)
            )
            sessions = result.scalars().all()

            for session in sessions:
                try:
                    # Avg fatigue
                    fat = await db.execute(
                        select(func.avg(FatigueMetric.fatigue_score))
                        .where(FatigueMetric.session_id == session.id)
                    )
                    avg_fat = float(fat.scalar() or 0.0)

                    # Avg stress
                    str_q = await db.execute(
                        select(func.avg(VoiceStressMetric.stress_score))
                        .where(VoiceStressMetric.session_id == session.id)
                    )
                    avg_str = float(str_q.scalar() or 0.0)

                    # Total focus time
                    foc = await db.execute(
                        select(func.sum(BehavioralMetric.focus_session_duration))
                        .where(BehavioralMetric.session_id == session.id)
                    )
                    total_focus = int(foc.scalar() or 0)

                    session.avg_fatigue_score = avg_fat
                    session.avg_stress_score = avg_str
                    session.total_focus_time = total_focus

                except Exception as e:
                    logger.error("Failed to aggregate session", session_id=str(session.id), error=str(e))

            await db.commit()
            logger.debug("Session metrics aggregated", sessions=len(sessions))

    asyncio.run(_run())


@celery_app.task(name="tasks.analytics.compute_focus_heatmap", queue="analytics")
def compute_focus_heatmap(user_id: str):
    """
    Pre-computes the 7-day focus heatmap data for a user and caches in Redis.
    Triggered on session end or daily via Beat.
    """
    import asyncio
    import json
    from app.db.session import AsyncSessionLocal
    from app.core.redis import redis_client
    from sqlalchemy import select, and_
    from app.models.models import BehavioralMetric

    async def _run():
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        async with AsyncSessionLocal() as db:
            import uuid
            result = await db.execute(
                select(BehavioralMetric)
                .where(and_(
                    BehavioralMetric.user_id == uuid.UUID(user_id),
                    BehavioralMetric.timestamp >= cutoff,
                ))
            )
            metrics = result.scalars().all()

        # Aggregate by (day_of_week, hour_of_day)
        grid: dict[str, list[float]] = {}
        for m in metrics:
            ts = m.timestamp.replace(tzinfo=timezone.utc)
            key = f"{ts.weekday()}-{ts.hour}"
            grid.setdefault(key, []).append(m.behavior_score)

        heatmap = []
        for day in range(7):
            for hour in range(24):
                key = f"{day}-{hour}"
                scores = grid.get(key, [])
                avg = sum(scores) / len(scores) if scores else 0.0
                heatmap.append({"day": day, "hour": hour, "value": round(avg, 1)})

        cache_key = f"neurosight:heatmap:{user_id}"
        await redis_client.setex(cache_key, 3600, json.dumps(heatmap))
        logger.info("Focus heatmap cached", user_id=user_id, cells=len(heatmap))

    asyncio.run(_run())
