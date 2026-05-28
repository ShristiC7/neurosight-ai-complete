"""
NeuroSight AI — Celery Application
Background task processing for ML inference, recommendations, and notifications.
"""

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

from app.core.config import settings

# -----------------------------------------------------------
# Application
# -----------------------------------------------------------
celery_app = Celery(
    "neurosight",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ml_inference",
        "app.tasks.recommendations",
        "app.tasks.rl_agent",
        "app.tasks.notifications",
        "app.tasks.analytics",
        "app.tasks.cleanup",
    ],
)

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task behavior
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_acks_late=True,  # Acknowledge after completion (safer)
    task_reject_on_worker_lost=True,
    task_track_started=True,

    # Results
    result_expires=3600,  # 1 hour
    result_persistent=True,

    # Worker
    worker_prefetch_multiplier=1,  # Fair distribution for ML tasks
    worker_max_tasks_per_child=100,  # Prevent memory leaks
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Queues and routing
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)

# -----------------------------------------------------------
# Queues
# -----------------------------------------------------------
celery_app.conf.task_queues = (
    Queue(
        "ml_inference",
        Exchange("ml_inference", type="direct"),
        routing_key="ml_inference",
        queue_arguments={"x-max-priority": 10},
    ),
    Queue(
        "recommendations",
        Exchange("recommendations", type="direct"),
        routing_key="recommendations",
    ),
    Queue(
        "notifications",
        Exchange("notifications", type="direct"),
        routing_key="notifications",
    ),
    Queue(
        "analytics",
        Exchange("analytics", type="direct"),
        routing_key="analytics",
    ),
    Queue(
        "cleanup",
        Exchange("cleanup", type="direct"),
        routing_key="cleanup",
    ),
    Queue("default"),
)

# -----------------------------------------------------------
# Task Routing — map task names to queues
# -----------------------------------------------------------
celery_app.conf.task_routes = {
    "app.tasks.ml_inference.*": {"queue": "ml_inference"},
    "app.tasks.recommendations.*": {"queue": "recommendations"},
    "app.tasks.rl_agent.*": {"queue": "recommendations"},
    "app.tasks.notifications.*": {"queue": "notifications"},
    "app.tasks.analytics.*": {"queue": "analytics"},
    "app.tasks.cleanup.*": {"queue": "cleanup"},
}

# -----------------------------------------------------------
# Beat Schedule — Periodic tasks
# -----------------------------------------------------------
celery_app.conf.beat_schedule = {
    # Run productivity predictions every 5 minutes for active sessions
    "predict-productivity-active-sessions": {
        "task": "app.tasks.ml_inference.batch_predict_productivity",
        "schedule": 300,  # Every 5 minutes
        "options": {"queue": "ml_inference", "priority": 5},
    },

    # Train RL agent on accumulated experience
    "train-rl-agent": {
        "task": "app.tasks.rl_agent.train_step",
        "schedule": 600,  # Every 10 minutes
        "options": {"queue": "recommendations"},
    },

    # Clean up expired recommendations
    "cleanup-expired-recommendations": {
        "task": "app.tasks.cleanup.cleanup_expired_recommendations",
        "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
        "options": {"queue": "cleanup"},
    },

    # Aggregate daily analytics
    "aggregate-daily-analytics": {
        "task": "app.tasks.analytics.aggregate_daily",
        "schedule": crontab(minute=0, hour=1),  # Daily at 1am UTC
        "options": {"queue": "analytics"},
    },

    # Update user behavioral embeddings in Qdrant
    "update-behavioral-embeddings": {
        "task": "app.tasks.analytics.update_behavioral_embeddings",
        "schedule": crontab(minute=30, hour="*/2"),  # Every 2 hours
        "options": {"queue": "analytics"},
    },

    # Health check
    "heartbeat": {
        "task": "app.tasks.cleanup.heartbeat",
        "schedule": 60,
    },
}
