"""
NeuroSight AI — Analytics Schemas
"""
from datetime import date
from pydantic import BaseModel, Field

class BatchAnalyticsDataPoint(BaseModel):
    period: str = Field(description="ISO format date or period label")
    avg_fatigue_score: float
    avg_stress_score: float
    avg_productivity_score: float
    total_focus_time_minutes: int
    sessions_count: int

class BatchAnalyticsSummary(BaseModel):
    avg_fatigue_score: float
    avg_stress_score: float
    avg_productivity_score: float
    total_focus_time_minutes: int
    total_sessions: int

class BatchAnalyticsResponse(BaseModel):
    user_id: str
    start_date: date
    end_date: date
    period: str
    data_points: list[BatchAnalyticsDataPoint]
    summary: BatchAnalyticsSummary

class HeatmapCell(BaseModel):
    day: int = Field(ge=0, le=6, description="0=Monday, 6=Sunday")
    hour: int = Field(ge=0, le=23, description="0-23 hour of day")
    value: float = Field(description="Average behavior score in [0, 100]")
