"""
NeuroSight AI — Batch Analytics Endpoints
"""
import uuid
from datetime import date, timedelta
from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User
from app.schemas.analytics import BatchAnalyticsResponse, HeatmapCell
from app.services.analytics.historical import HistoricalAnalyticsService

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/batch",
    response_model=BatchAnalyticsResponse,
    summary="Get batch historical aggregates",
)
async def get_batch_analytics(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: date | None = None,
    end_date: date | None = None,
    period: Literal["daily", "weekly"] = "daily",
) -> BatchAnalyticsResponse:
    """
    Returns daily or weekly aggregated metrics (fatigue, stress, productivity, focus time)
    for the current user within the specified date range.
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=7)

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_date must be before or equal to end_date",
        )

    # limit date range to 90 days to prevent excessive queries
    if (end_date - start_date).days > 90:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Date range cannot exceed 90 days",
        )

    service = HistoricalAnalyticsService(db)
    result = await service.get_batch_analytics(
        user_id=current_user.id,
        start_date=start_date,
        end_date=end_date,
        period=period,
    )
    return BatchAnalyticsResponse.model_validate(result)


@router.get(
    "/focus-heatmap",
    response_model=list[HeatmapCell],
    summary="Get 7-day focus heatmap data",
)
async def get_focus_heatmap(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[HeatmapCell]:
    """
    Returns focus heatmap data (weekday vs hour of day averages) for the past 7 days.
    """
    service = HistoricalAnalyticsService(db)
    result = await service.get_focus_heatmap(user_id=current_user.id)
    return [HeatmapCell.model_validate(cell) for cell in result]
