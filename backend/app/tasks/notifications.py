"""
NeuroSight AI — Notification Tasks
Pushes critical alerts to users via WebSocket and (optionally) email.
"""
import structlog
from app.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="tasks.notifications.send_critical_alert", queue="notifications")
def send_critical_alert(user_id: str, alert_type: str, message: str):
    """
    Sends a critical alert to a connected user via WebSocket.
    Used when fatigue > 85 or burnout probability > 90%.
    """
    import asyncio
    from app.core.websocket_manager import ws_manager

    async def _push():
        delivered = await ws_manager.send_to_user(user_id, "alert:critical", {
            "type": alert_type,
            "message": message,
            "severity": "critical",
        })
        if not delivered:
            logger.warning("Critical alert not delivered — user not connected",
                           user_id=user_id, alert_type=alert_type)

    asyncio.run(_push())


@celery_app.task(name="tasks.notifications.send_break_reminder", queue="notifications")
def send_break_reminder(user_id: str, session_id: str, duration_minutes: int = 5):
    """
    Sends a break reminder recommendation to the user.
    Triggered when focus duration exceeds configured thresholds.
    """
    import asyncio
    from app.core.websocket_manager import ws_manager

    async def _push():
        await ws_manager.push_recommendation(user_id, {
            "type": "take_break",
            "priority": "high",
            "title": "Time for a Break",
            "message": f"You've been focused for a long stretch. A {duration_minutes}-minute break will restore your concentration.",
            "durationMinutes": duration_minutes,
            "sessionId": session_id,
        })

    asyncio.run(_push())


@celery_app.task(name="tasks.notifications.send_burnout_warning", queue="notifications")
def send_burnout_warning(user_id: str, burnout_probability: float):
    """Sends a burnout risk warning when probability exceeds 70%."""
    import asyncio
    from app.core.websocket_manager import ws_manager
    severity = "critical" if burnout_probability > 0.85 else "high"

    async def _push():
        await ws_manager.push_alert(
            user_id,
            severity,
            f"Burnout risk is {round(burnout_probability * 100)}%. "
            "Consider taking a longer break or ending your work session."
        )

    asyncio.run(_push())
