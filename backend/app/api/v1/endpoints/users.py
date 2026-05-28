"""
NeuroSight AI — User Profile & Settings Endpoints
"""
import uuid
from datetime import datetime
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.core.security import hash_password, verify_password
from app.db.session import get_db
from app.models.models import User

logger = structlog.get_logger(__name__)
router = APIRouter()


class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: EmailStr
    avatar_url: str | None
    is_active: bool
    timezone: str
    preferences: dict
    created_at: datetime

    class Config:
        from_attributes = True


class UpdateProfileRequest(BaseModel):
    name: str | None = None
    timezone: str | None = None
    avatar_url: str | None = None


class UpdatePreferencesRequest(BaseModel):
    work_hours_start: int | None = None
    work_hours_end: int | None = None
    break_duration: int | None = None
    theme: str | None = None
    notifications: dict | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UserAnalyticsSummary(BaseModel):
    total_sessions: int
    total_monitoring_hours: float
    avg_fatigue_score: float
    avg_productivity_score: float
    avg_stress_score: float
    recommendations_accepted: int
    recommendations_total: int
    acceptance_rate: float


@router.get("/me", response_model=UserResponse, summary="Get current user profile")
async def get_me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse, summary="Update user profile")
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    if payload.name is not None:
        current_user.name = payload.name
    if payload.timezone is not None:
        current_user.timezone = payload.timezone
    if payload.avatar_url is not None:
        current_user.avatar_url = payload.avatar_url
    await db.flush()
    return UserResponse.model_validate(current_user)


@router.patch("/me/preferences", response_model=UserResponse,
              summary="Update user preferences")
async def update_preferences(
    payload: UpdatePreferencesRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    prefs = dict(current_user.preferences or {})
    updates = payload.model_dump(exclude_none=True)
    prefs.update(updates)
    current_user.preferences = prefs
    await db.flush()
    return UserResponse.model_validate(current_user)


@router.post("/me/change-password", status_code=204, summary="Change user password")
async def change_password(
    payload: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=422, detail="New password must be at least 8 characters")
    current_user.password_hash = hash_password(payload.new_password)
    await db.flush()
    logger.info("Password changed", user_id=str(current_user.id))


@router.get("/me/analytics", response_model=UserAnalyticsSummary,
            summary="Get user-level analytics summary")
async def get_user_analytics(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserAnalyticsSummary:
    """Aggregated lifetime stats for the user profile page."""
    from sqlalchemy import func
    from app.models.models import WorkSession, FatigueMetric, VoiceStressMetric, Recommendation

    # Sessions
    sess_result = await db.execute(
        select(func.count(WorkSession.id), func.sum(WorkSession.total_focus_time))
        .where(WorkSession.user_id == current_user.id)
    )
    sess_row = sess_result.one()
    total_sessions = int(sess_row[0] or 0)
    total_focus_min = float(sess_row[1] or 0.0)

    # Average scores
    fat_result = await db.execute(
        select(func.avg(FatigueMetric.fatigue_score))
        .where(FatigueMetric.user_id == current_user.id)
    )
    avg_fatigue = float(fat_result.scalar() or 0.0)

    stress_result = await db.execute(
        select(func.avg(VoiceStressMetric.stress_score))
        .where(VoiceStressMetric.user_id == current_user.id)
    )
    avg_stress = float(stress_result.scalar() or 0.0)

    # Recommendations
    rec_result = await db.execute(
        select(func.count(Recommendation.id),
               func.sum((Recommendation.accepted == True).cast(func.Integer())))
        .where(Recommendation.user_id == current_user.id)
    )
    rec_row = rec_result.one()
    total_recs = int(rec_row[0] or 0)
    accepted_recs = int(rec_row[1] or 0)

    return UserAnalyticsSummary(
        total_sessions=total_sessions,
        total_monitoring_hours=round(total_focus_min / 60, 1),
        avg_fatigue_score=round(avg_fatigue, 1),
        avg_productivity_score=0.0,
        avg_stress_score=round(avg_stress, 1),
        recommendations_accepted=accepted_recs,
        recommendations_total=total_recs,
        acceptance_rate=round(accepted_recs / max(total_recs, 1), 3),
    )


@router.delete("/me", status_code=204, summary="Delete user account and all data")
async def delete_account(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Permanently deletes the user and all associated data (GDPR compliance)."""
    await db.delete(current_user)
    await db.flush()
    logger.info("Account deleted", user_id=str(current_user.id))
