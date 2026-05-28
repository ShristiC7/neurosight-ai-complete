"""
NeuroSight AI — Recommendation Endpoints
CRUD for AI-generated productivity recommendations + feedback loop for RL training.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User, Recommendation, RecommendationType, RecommendationPriority

logger = structlog.get_logger(__name__)
router = APIRouter()


class RecommendationResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    message: str
    action_label: str | None
    duration_minutes: int | None
    accepted: bool | None
    expires_at: datetime
    timestamp: datetime
    metadata: dict

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    accepted: bool


class RecommendationStats(BaseModel):
    total: int
    accepted: int
    rejected: int
    pending: int
    acceptance_rate: float
    most_accepted_type: str | None


@router.get("/", response_model=list[RecommendationResponse],
            summary="Get active recommendations for current user")
async def get_recommendations(
    limit: int = 10,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    """Returns active (non-expired) recommendations sorted by priority."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Recommendation)
        .where(and_(
            Recommendation.user_id == current_user.id,
            Recommendation.expires_at > now,
            Recommendation.accepted.is_(None),
        ))
        .order_by(Recommendation.timestamp.desc())
        .limit(min(limit, 20))
    )
    return [RecommendationResponse.model_validate(r) for r in result.scalars().all()]


@router.get("/history", response_model=list[RecommendationResponse],
            summary="Get recommendation history (last 7 days)")
async def get_recommendation_history(
    days: int = 7,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=min(days, 30))
    result = await db.execute(
        select(Recommendation)
        .where(and_(Recommendation.user_id == current_user.id,
                    Recommendation.timestamp >= cutoff))
        .order_by(Recommendation.timestamp.desc())
        .limit(200)
    )
    return [RecommendationResponse.model_validate(r) for r in result.scalars().all()]


@router.get("/stats", response_model=RecommendationStats,
            summary="Get recommendation acceptance statistics")
async def get_recommendation_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RecommendationStats:
    """Used by the RL agent to evaluate recommendation effectiveness."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    result = await db.execute(
        select(Recommendation)
        .where(and_(Recommendation.user_id == current_user.id,
                    Recommendation.timestamp >= cutoff))
    )
    recs = result.scalars().all()

    total = len(recs)
    accepted = sum(1 for r in recs if r.accepted is True)
    rejected = sum(1 for r in recs if r.accepted is False)
    pending = total - accepted - rejected

    # Most accepted recommendation type
    type_counts: dict[str, int] = {}
    for r in recs:
        if r.accepted is True:
            type_counts[r.type.value] = type_counts.get(r.type.value, 0) + 1
    most_accepted = max(type_counts, key=type_counts.get) if type_counts else None

    return RecommendationStats(
        total=total,
        accepted=accepted,
        rejected=rejected,
        pending=pending,
        acceptance_rate=round(accepted / max(accepted + rejected, 1), 3),
        most_accepted_type=most_accepted,
    )


@router.patch("/{recommendation_id}/accept", response_model=RecommendationResponse,
              summary="Accept a recommendation (positive RL reward)")
async def accept_recommendation(
    recommendation_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RecommendationResponse:
    """
    Records user acceptance. Triggers a positive reward signal to the RL agent.
    A follow-up check is scheduled in 5 minutes to verify improvement.
    """
    rec = await _get_rec_or_404(recommendation_id, current_user.id, db)
    rec.accepted = True
    rec.reward = 0.5
    await db.flush()

    # Schedule RL reward update via Celery
    try:
        from app.core.celery_app import celery_app
        celery_app.send_task("tasks.rl.record_transition", kwargs={
            "recommendation_id": str(recommendation_id),
            "user_id": str(current_user.id),
            "accepted": True,
        }, countdown=300)
    except Exception:
        pass

    logger.info("Recommendation accepted", rec_id=str(recommendation_id),
                type=rec.type.value, user_id=str(current_user.id))
    return RecommendationResponse.model_validate(rec)


@router.patch("/{recommendation_id}/dismiss", response_model=RecommendationResponse,
              summary="Dismiss a recommendation (negative RL reward)")
async def dismiss_recommendation(
    recommendation_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RecommendationResponse:
    """Records user dismissal. Sends a mild negative reward to the RL agent."""
    rec = await _get_rec_or_404(recommendation_id, current_user.id, db)
    rec.accepted = False
    rec.reward = -0.3
    await db.flush()
    logger.info("Recommendation dismissed", rec_id=str(recommendation_id), type=rec.type.value)
    return RecommendationResponse.model_validate(rec)


@router.delete("/{recommendation_id}", status_code=204, summary="Delete a recommendation")
async def delete_recommendation(
    recommendation_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    rec = await _get_rec_or_404(recommendation_id, current_user.id, db)
    await db.delete(rec)
    await db.flush()


async def _get_rec_or_404(rec_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> Recommendation:
    result = await db.execute(
        select(Recommendation).where(and_(
            Recommendation.id == rec_id,
            Recommendation.user_id == user_id,
        ))
    )
    rec = result.scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return rec
