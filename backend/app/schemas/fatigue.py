"""Fatigue detection request/response schemas."""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from app.models.models import DrowsinessLevel


class FatigueMetricCreate(BaseModel):
    session_id: uuid.UUID
    timestamp: datetime
    blink_rate: float = Field(ge=0, le=100)
    eye_aspect_ratio: float = Field(ge=0, le=1)
    mouth_aspect_ratio: float = Field(ge=0, le=1)
    head_tilt_angle: float = Field(default=0.0)
    gaze_drift: float = Field(default=0.0, ge=0, le=1)
    fatigue_score: float = Field(ge=0, le=100)
    drowsiness_level: DrowsinessLevel
    confidence: float = Field(default=0.9, ge=0, le=1)


class FatigueMetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    timestamp: datetime
    blink_rate: float
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    fatigue_score: float
    drowsiness_level: DrowsinessLevel
    confidence: float
    created_at: datetime


class FatigueAnalysisRequest(BaseModel):
    session_id: uuid.UUID


class FatigueAnalysisResponse(BaseModel):
    fatigue_score: float
    drowsiness_level: DrowsinessLevel
    ear: float
    mar: float
    blink_rate: float
    confidence: float
    inference_time_ms: float


class FatigueTrendResponse(BaseModel):
    user_id: uuid.UUID
    hours: int
    data_points: list[dict]
    avg_fatigue_score: float
    max_fatigue_score: float
    trend_direction: str  # "improving" | "stable" | "declining"
