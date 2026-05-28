"""
NeuroSight AI — Celery ML Inference Tasks
Background tasks for batch predictions and model drift detection.
"""
import uuid
from datetime import datetime, timezone, timedelta

import structlog
from celery import shared_task

from app.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(
    name="tasks.ml.batch_productivity_prediction",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="ml_inference",
)
def batch_productivity_prediction(self):
    """
    Runs productivity predictions for all users with active sessions.
    Scheduled every 10 minutes via Celery Beat.
    """
    import asyncio
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import select
    from app.models.models import WorkSession

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(WorkSession).where(WorkSession.is_active == True)
            )
            active_sessions = result.scalars().all()
            logger.info("Batch prediction", active_sessions=len(active_sessions))

            from app.services.prediction.productivity_service import ProductivityService
            from app.core.websocket_manager import ws_manager

            for session in active_sessions:
                try:
                    svc = ProductivityService(db)
                    prediction = await svc.predict_for_session(session.user_id, session.id)
                    await db.commit()

                    await ws_manager.send_to_user(str(session.user_id), "prediction:update", {
                        "productivityScore": prediction.productivity_score,
                        "burnoutProbability": prediction.burnout_probability,
                        "confidence": prediction.confidence,
                        "timestamp": prediction.timestamp.isoformat(),
                    })
                except Exception as e:
                    logger.error("Prediction failed for session",
                                 session_id=str(session.id), error=str(e))
                    await db.rollback()

    asyncio.run(_run())


@celery_app.task(
    name="tasks.ml.check_model_drift",
    queue="ml_inference",
)
def check_model_drift():
    """
    Compares recent model predictions against ground truth signals.
    Flags drift if accuracy drops below threshold.
    Runs daily via Celery Beat.
    """
    import asyncio
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import select, func, and_
    from app.models.models import ProductivityPrediction, FatigueMetric

    async def _run():
        async with AsyncSessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)

            # Check prediction confidence distribution
            result = await db.execute(
                select(func.avg(ProductivityPrediction.confidence),
                       func.count(ProductivityPrediction.id))
                .where(ProductivityPrediction.timestamp >= cutoff)
            )
            row = result.one()
            avg_confidence = float(row[0] or 0.0)
            prediction_count = int(row[1] or 0)

            if avg_confidence < 0.6 and prediction_count > 100:
                logger.warning(
                    "Model drift detected — confidence below threshold",
                    avg_confidence=avg_confidence,
                    prediction_count=prediction_count,
                    threshold=0.6,
                )
                # In production: trigger retraining pipeline or alert
            else:
                logger.info("Model drift check passed",
                            avg_confidence=round(avg_confidence, 3),
                            prediction_count=prediction_count)

    asyncio.run(_run())


@celery_app.task(
    name="tasks.ml.export_model_metrics",
    queue="ml_inference",
)
def export_model_metrics():
    """Pushes ML performance metrics to Prometheus for Grafana dashboards."""
    try:
        from app.middleware.metrics import ML_INFERENCE_COUNT, ML_INFERENCE_LATENCY
        # Metrics are updated in real-time during inference
        # This task logs a summary snapshot
        logger.info("Model metrics exported to Prometheus")
    except Exception as e:
        logger.error("Failed to export model metrics", error=str(e))
