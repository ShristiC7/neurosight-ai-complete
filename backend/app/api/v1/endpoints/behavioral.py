"""
NeuroSight AI — Behavioral Analytics Endpoints
Receives typing/mouse/focus metrics from the frontend behavioral tracking hook.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User, BehavioralMetric
from app.core.websocket_manager import ws_manager

logger = structlog.get_logger(__name__)
router = APIRouter()


class BehavioralMetricCreate(BaseModel):
    session_id: uuid.UUID
    timestamp: datetime
    typing_speed: float = Field(default=0.0, ge=0, description="Words per minute")
    typing_rhythm_variance: float = Field(default=0.0, ge=0, description="Std dev of keystroke intervals ms")
    error_rate: float = Field(default=0.0, ge=0, description="Backspace/delete events per minute")
    mouse_movement_entropy: float = Field(default=0.0, ge=0, le=1, description="Shannon entropy of mouse velocity")
    mouse_click_rate: float = Field(default=0.0, ge=0, description="Mouse clicks per minute")
    app_switch_frequency: float = Field(default=0.0, ge=0, description="Visibility change events per hour")
    focus_session_duration: float = Field(default=0.0, ge=0, description="Uninterrupted focus minutes")
    idle_time: float = Field(default=0.0, ge=0, description="Idle seconds in last window")
    behavior_score: float = Field(ge=0, le=100, description="Computed normalcy score 0-100")
    anomaly_score: float = Field(default=0.0, ge=0, le=1, description="Isolation Forest anomaly score 0-1")


class BehavioralMetricResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    timestamp: datetime
    typing_speed: float
    error_rate: float
    mouse_movement_entropy: float
    app_switch_frequency: float
    behavior_score: float
    anomaly_score: float
    created_at: datetime

    class Config:
        from_attributes = True


class AnomalyDetectionResponse(BaseModel):
    is_anomalous: bool
    anomaly_score: float
    behavior_score: float
    anomaly_features: list[str]
    recommendation: str


@router.post("/metrics", response_model=BehavioralMetricResponse, status_code=201,
             summary="Store behavioral analytics metrics")
async def store_behavioral_metrics(
    payload: BehavioralMetricCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BehavioralMetricResponse:
    """
    Persists aggregated behavioral metrics (10-second windows from the frontend hook).
    Triggers anomaly detection and pushes real-time update if anomalous.
    """
    # Run anomaly detection if model available
    anomaly_score = payload.anomaly_score
    try:
        from app.services.analytics.behavioral_service import BehavioralService
        svc = BehavioralService()
        anomaly_score = svc.compute_anomaly_score({
            "typing_speed": payload.typing_speed,
            "typing_rhythm_variance": payload.typing_rhythm_variance,
            "error_rate": payload.error_rate,
            "mouse_movement_entropy": payload.mouse_movement_entropy,
            "mouse_click_rate": payload.mouse_click_rate,
            "app_switch_frequency": payload.app_switch_frequency,
            "focus_session_duration": payload.focus_session_duration,
            "idle_time": payload.idle_time,
            "behavior_score": payload.behavior_score,
            "hour_of_day": payload.timestamp.hour,
        })
    except Exception:
        pass

    metric = BehavioralMetric(
        user_id=current_user.id,
        session_id=payload.session_id,
        timestamp=payload.timestamp,
        typing_speed=payload.typing_speed,
        typing_rhythm_variance=payload.typing_rhythm_variance,
        error_rate=payload.error_rate,
        mouse_movement_entropy=payload.mouse_movement_entropy,
        mouse_click_rate=payload.mouse_click_rate,
        app_switch_frequency=payload.app_switch_frequency,
        focus_session_duration=payload.focus_session_duration,
        idle_time=payload.idle_time,
        behavior_score=payload.behavior_score,
        anomaly_score=anomaly_score,
    )
    db.add(metric)
    await db.flush()

    # Push realtime update
    await ws_manager.send_to_user(str(current_user.id), "behavioral:update", {
        "behaviorScore": payload.behavior_score,
        "anomalyScore": anomaly_score,
        "typingSpeed": payload.typing_speed,
        "focusDuration": payload.focus_session_duration,
        "timestamp": payload.timestamp.isoformat(),
    })

    return BehavioralMetricResponse.model_validate(metric)


@router.get("/trend", summary="Get behavioral trend for current user")
async def get_behavioral_trend(
    hours: int = 8,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Returns time-series behavioral scores for dashboard charts."""
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=422, detail="hours must be 1-168")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await db.execute(
        select(BehavioralMetric)
        .where(and_(BehavioralMetric.user_id == current_user.id,
                    BehavioralMetric.timestamp >= cutoff))
        .order_by(BehavioralMetric.timestamp.asc()).limit(500)
    )
    metrics = result.scalars().all()
    data_points = [{"timestamp": m.timestamp.isoformat(), "value": m.behavior_score,
                    "anomaly": m.anomaly_score} for m in metrics]
    scores = [m.behavior_score for m in metrics]
    return {
        "user_id": str(current_user.id),
        "hours": hours,
        "data_points": data_points,
        "avg_behavior_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "anomalies_detected": sum(1 for m in metrics if m.anomaly_score > 0.5),
    }


@router.get("/session/{session_id}", response_model=list[BehavioralMetricResponse],
            summary="Get behavioral metrics for a session")
async def get_session_behavioral(
    session_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(
        select(BehavioralMetric)
        .where(and_(BehavioralMetric.user_id == current_user.id,
                    BehavioralMetric.session_id == session_id))
        .order_by(BehavioralMetric.timestamp.asc())
    )
    return [BehavioralMetricResponse.model_validate(m) for m in result.scalars().all()]
