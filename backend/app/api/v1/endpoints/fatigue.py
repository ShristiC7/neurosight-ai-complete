"""
NeuroSight AI — Fatigue Detection Endpoints
Real-time and batch fatigue analysis via CV pipeline.
"""

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User
from app.schemas.fatigue import (
    FatigueMetricCreate,
    FatigueMetricResponse,
    FatigueAnalysisRequest,
    FatigueAnalysisResponse,
    FatigueTrendResponse,
)
from app.services.cv.fatigue_service import FatigueService
from app.services.ml_registry import ModelRegistry

logger = structlog.get_logger(__name__)
router = APIRouter()


# -----------------------------------------------------------
# POST /fatigue/metrics — Store fatigue metric from client
# -----------------------------------------------------------
@router.post(
    "/metrics",
    response_model=FatigueMetricResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store fatigue metrics from client-side detection",
)
async def store_fatigue_metrics(
    payload: FatigueMetricCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FatigueMetricResponse:
    """
    Store pre-computed fatigue metrics from the frontend MediaPipe pipeline.
    Metrics are persisted and used for trend analysis and predictions.
    """
    service = FatigueService(db)
    metric = await service.store_metric(
        user_id=current_user.id,
        payload=payload,
    )
    return FatigueMetricResponse.model_validate(metric)


# -----------------------------------------------------------
# POST /fatigue/analyze-frame — Server-side frame analysis
# -----------------------------------------------------------
@router.post(
    "/analyze-frame",
    response_model=FatigueAnalysisResponse,
    summary="Analyze a single video frame for fatigue indicators",
)
async def analyze_frame(
    frame: Annotated[UploadFile, File(description="JPEG/PNG video frame")],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FatigueAnalysisResponse:
    """
    Server-side fatigue analysis using OpenCV + MediaPipe + CNN model.
    Use this when client-side inference is not available (edge cases, mobile).

    - Detects face landmarks
    - Computes EAR, MAR, head pose
    - Classifies drowsiness level via CNN
    - Returns fatigue score + confidence
    """
    if frame.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Frame must be JPEG, PNG, or WebP",
        )

    MAX_SIZE = 2 * 1024 * 1024  # 2MB
    frame_bytes = await frame.read()
    if len(frame_bytes) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Frame must be smaller than 2MB",
        )

    service = FatigueService(db)
    result = await service.analyze_frame(
        user_id=current_user.id,
        frame_bytes=frame_bytes,
    )

    return result


# -----------------------------------------------------------
# GET /fatigue/trend — Historical fatigue trend
# -----------------------------------------------------------
@router.get(
    "/trend",
    response_model=FatigueTrendResponse,
    summary="Get fatigue trend for the current user",
)
async def get_fatigue_trend(
    hours: int = 8,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> FatigueTrendResponse:
    """
    Returns time-series fatigue data for the past N hours.
    Used for dashboard trend visualization.
    """
    if hours < 1 or hours > 168:  # Max 1 week
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="hours must be between 1 and 168",
        )

    service = FatigueService(db)
    return await service.get_trend(user_id=current_user.id, hours=hours)


# -----------------------------------------------------------
# GET /fatigue/session/{session_id} — Session fatigue summary
# -----------------------------------------------------------
@router.get(
    "/session/{session_id}",
    response_model=list[FatigueMetricResponse],
    summary="Get all fatigue metrics for a specific session",
)
async def get_session_fatigue(
    session_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[FatigueMetricResponse]:
    service = FatigueService(db)
    metrics = await service.get_by_session(
        user_id=current_user.id,
        session_id=session_id,
    )
    return [FatigueMetricResponse.model_validate(m) for m in metrics]
