"""
NeuroSight AI — Work Session Endpoints
Manages start/end of monitoring sessions and session summaries.
"""
import uuid
from datetime import datetime, timezone
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User, WorkSession
from app.core.websocket_manager import ws_manager

logger = structlog.get_logger(__name__)
router = APIRouter()


class SessionResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    start_time: datetime
    end_time: datetime | None
    is_active: bool
    avg_fatigue_score: float
    avg_productivity_score: float
    avg_stress_score: float
    total_focus_time: int
    breaks_taken: int
    total_keystrokes: int
    created_at: datetime

    class Config:
        from_attributes = True


class SessionSummary(BaseModel):
    session_id: uuid.UUID
    duration_minutes: float
    avg_fatigue_score: float
    avg_productivity_score: float
    avg_stress_score: float
    total_focus_time: int
    breaks_taken: int
    peak_productivity_hour: int | None
    total_keystrokes: int


@router.post("/start", response_model=SessionResponse, status_code=201,
             summary="Start a new monitoring session")
async def start_session(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionResponse:
    """
    Creates a new WorkSession for the user.
    Closes any previously active session first (handles abandoned sessions).
    """
    # End any stale active sessions
    await db.execute(
        update(WorkSession)
        .where(and_(WorkSession.user_id == current_user.id, WorkSession.is_active == True))
        .values(is_active=False, end_time=datetime.now(timezone.utc))
    )

    session = WorkSession(
        user_id=current_user.id,
        start_time=datetime.now(timezone.utc),
        is_active=True,
    )
    db.add(session)
    await db.flush()

    logger.info("Session started", user_id=str(current_user.id), session_id=str(session.id))

    await ws_manager.send_to_user(str(current_user.id), "session:started", {
        "sessionId": str(session.id),
        "startTime": session.start_time.isoformat(),
    })
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/end", response_model=SessionSummary,
             summary="End a monitoring session and compute summary")
async def end_session(
    session_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SessionSummary:
    """Marks session as ended and returns an aggregated performance summary."""
    result = await db.execute(
        select(WorkSession).where(and_(
            WorkSession.id == session_id,
            WorkSession.user_id == current_user.id,
            WorkSession.is_active == True,
        ))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")

    end_time = datetime.now(timezone.utc)
    duration = (end_time - session.start_time.replace(tzinfo=timezone.utc)).total_seconds() / 60

    # Aggregate fatigue metrics for this session
    from sqlalchemy import func
    from app.models.models import FatigueMetric, VoiceStressMetric, BehavioralMetric

    fat_result = await db.execute(
        select(func.avg(FatigueMetric.fatigue_score))
        .where(FatigueMetric.session_id == session_id)
    )
    avg_fatigue = float(fat_result.scalar() or 0.0)

    stress_result = await db.execute(
        select(func.avg(VoiceStressMetric.stress_score))
        .where(VoiceStressMetric.session_id == session_id)
    )
    avg_stress = float(stress_result.scalar() or 0.0)

    behav_result = await db.execute(
        select(func.avg(BehavioralMetric.typing_speed),
               func.sum(BehavioralMetric.focus_session_duration))
        .where(BehavioralMetric.session_id == session_id)
    )
    behav_row = behav_result.one()
    total_focus = int(behav_row[1] or 0)

    # Update session record
    session.end_time = end_time
    session.is_active = False
    session.avg_fatigue_score = avg_fatigue
    session.avg_stress_score = avg_stress
    session.total_focus_time = total_focus
    await db.flush()

    logger.info("Session ended", session_id=str(session_id), duration_min=round(duration, 1))
    await ws_manager.send_to_user(str(current_user.id), "session:ended", {
        "sessionId": str(session_id), "durationMinutes": round(duration, 1),
    })

    return SessionSummary(
        session_id=session_id,
        duration_minutes=round(duration, 1),
        avg_fatigue_score=round(avg_fatigue, 2),
        avg_productivity_score=round(session.avg_productivity_score, 2),
        avg_stress_score=round(avg_stress, 2),
        total_focus_time=total_focus,
        breaks_taken=session.breaks_taken,
        peak_productivity_hour=None,
        total_keystrokes=session.total_keystrokes,
    )


@router.get("/", response_model=list[SessionResponse], summary="List recent sessions")
async def list_sessions(
    limit: int = 20,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    result = await db.execute(
        select(WorkSession)
        .where(WorkSession.user_id == current_user.id)
        .order_by(WorkSession.start_time.desc())
        .limit(min(limit, 100))
    )
    return [SessionResponse.model_validate(s) for s in result.scalars().all()]


@router.get("/active", response_model=SessionResponse | None,
            summary="Get the currently active session")
async def get_active_session(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(
        select(WorkSession).where(and_(
            WorkSession.user_id == current_user.id,
            WorkSession.is_active == True,
        ))
    )
    session = result.scalar_one_or_none()
    return SessionResponse.model_validate(session) if session else None


@router.get("/{session_id}", response_model=SessionResponse, summary="Get session by ID")
async def get_session(
    session_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(
        select(WorkSession).where(and_(
            WorkSession.id == session_id,
            WorkSession.user_id == current_user.id,
        ))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse.model_validate(session)
