"""
NeuroSight AI — Productivity Prediction Endpoints
Triggers ML inference and returns forecasts with explainability.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User, ProductivityPrediction, WorkSession
from app.core.websocket_manager import ws_manager

logger = structlog.get_logger(__name__)
router = APIRouter()


class PredictionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    timestamp: datetime
    productivity_score: float
    burnout_probability: float
    cognitive_load: float
    focus_window_start: datetime | None
    focus_window_end: datetime | None
    recommended_break_at: datetime | None
    predicted_crash_at: datetime | None
    confidence: float
    feature_importance: dict

    class Config:
        from_attributes = True


class FocusWindowsResponse(BaseModel):
    user_id: uuid.UUID
    date: str
    focus_windows: list[dict]
    recommended_work_start: str
    recommended_work_end: str
    total_peak_minutes: int


@router.post("/run", response_model=PredictionResponse, status_code=201,
             summary="Run productivity prediction for current session")
async def run_prediction(
    session_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PredictionResponse:
    """
    Triggers the LSTM+XGBoost ensemble prediction for the given session.
    Aggregates last 30 minutes of sensor data to build the feature vector.
    Also generates an RL recommendation if the agent is loaded.
    """
    # Verify session belongs to user
    sess_result = await db.execute(
        select(WorkSession).where(and_(
            WorkSession.id == session_id,
            WorkSession.user_id == current_user.id,
        ))
    )
    session = sess_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Run productivity prediction
    from app.services.prediction.productivity_service import ProductivityService
    svc = ProductivityService(db)
    prediction = await svc.predict_for_session(current_user.id, session_id)

    # Push to WebSocket
    await ws_manager.send_to_user(str(current_user.id), "prediction:update", {
        "productivityScore": prediction.productivity_score,
        "burnoutProbability": prediction.burnout_probability,
        "cognitiveLoad": prediction.cognitive_load,
        "confidence": prediction.confidence,
        "timestamp": prediction.timestamp.isoformat(),
        "recommendedBreakAt": prediction.recommended_break_at.isoformat() if prediction.recommended_break_at else None,
    })

    # Schedule RL recommendation generation as background task
    background_tasks.add_task(
        _generate_recommendation_background,
        str(current_user.id), str(session_id), prediction.id,
    )

    return PredictionResponse.model_validate(prediction)


@router.get("/latest", response_model=PredictionResponse | None,
            summary="Get the latest prediction for the current user")
async def get_latest_prediction(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(
        select(ProductivityPrediction)
        .where(ProductivityPrediction.user_id == current_user.id)
        .order_by(ProductivityPrediction.timestamp.desc())
        .limit(1)
    )
    pred = result.scalar_one_or_none()
    return PredictionResponse.model_validate(pred) if pred else None


@router.get("/history", response_model=list[PredictionResponse],
            summary="Get prediction history for the last N hours")
async def get_prediction_history(
    hours: int = 24,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=422, detail="hours must be 1-720")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await db.execute(
        select(ProductivityPrediction)
        .where(and_(ProductivityPrediction.user_id == current_user.id,
                    ProductivityPrediction.timestamp >= cutoff))
        .order_by(ProductivityPrediction.timestamp.asc())
        .limit(500)
    )
    return [PredictionResponse.model_validate(p) for p in result.scalars().all()]


@router.get("/focus-windows", response_model=FocusWindowsResponse,
            summary="Get predicted optimal focus windows for today")
async def get_focus_windows(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FocusWindowsResponse:
    """
    Analyzes historical productivity patterns to predict today's peak focus windows.
    Uses 7-day rolling history if available.
    """
    today = datetime.now(timezone.utc).date()
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    result = await db.execute(
        select(ProductivityPrediction)
        .where(and_(ProductivityPrediction.user_id == current_user.id,
                    ProductivityPrediction.timestamp >= week_ago))
        .order_by(ProductivityPrediction.timestamp.asc())
    )
    predictions = result.scalars().all()

    # Aggregate by hour-of-day
    hourly_scores: dict[int, list[float]] = {h: [] for h in range(24)}
    for p in predictions:
        hour = p.timestamp.hour
        hourly_scores[hour].append(p.productivity_score)

    avg_by_hour = {
        h: sum(scores) / len(scores) if scores else 50.0
        for h, scores in hourly_scores.items()
    }

    # Identify peak windows (consecutive hours above 70%)
    focus_windows = []
    in_window = False
    window_start = None
    for hour in range(6, 23):
        if avg_by_hour.get(hour, 0) >= 65:
            if not in_window:
                in_window = True
                window_start = hour
        else:
            if in_window:
                focus_windows.append({
                    "start": f"{window_start:02d}:00",
                    "end": f"{hour:02d}:00",
                    "quality": "peak" if avg_by_hour.get(window_start, 0) >= 80 else "good",
                    "avg_score": round(sum(avg_by_hour.get(h, 0) for h in range(window_start, hour)) / max(hour - window_start, 1), 1),
                })
                in_window = False

    if not focus_windows:
        focus_windows = [{"start": "09:00", "end": "11:00", "quality": "estimated", "avg_score": 70.0}]

    best_hour = max(avg_by_hour, key=avg_by_hour.get)
    peak_minutes = sum(
        60 for h in range(24)
        if avg_by_hour.get(h, 0) >= 70
    )

    return FocusWindowsResponse(
        user_id=current_user.id,
        date=today.isoformat(),
        focus_windows=focus_windows,
        recommended_work_start=f"{best_hour:02d}:00",
        recommended_work_end=f"{min(best_hour + 4, 22):02d}:00",
        total_peak_minutes=peak_minutes,
    )


async def _generate_recommendation_background(user_id: str, session_id: str, prediction_id: uuid.UUID):
    """Background task: generate RL recommendation after prediction completes."""
    try:
        from app.db.session import AsyncSessionLocal
        from app.models.models import ProductivityPrediction, Recommendation
        from app.services.recommendation.recommendation_engine import RecommendationEngine

        async with AsyncSessionLocal() as db:
            pred_result = await db.execute(
                select(ProductivityPrediction).where(ProductivityPrediction.id == prediction_id)
            )
            prediction = pred_result.scalar_one_or_none()
            if not prediction:
                return

            engine = RecommendationEngine(db)
            rec = await engine.generate_recommendation(
                user_id=uuid.UUID(user_id),
                session_id=uuid.UUID(session_id),
                prediction=prediction,
            )
            await db.commit()

            if rec:
                await ws_manager.push_recommendation(user_id, {
                    "id": str(rec.id), "type": rec.type.value, "priority": rec.priority.value,
                    "title": rec.title, "message": rec.message,
                    "durationMinutes": rec.duration_minutes,
                    "expiresAt": rec.expires_at.isoformat(),
                })
    except Exception as e:
        logger.error("Background recommendation generation failed", error=str(e))
