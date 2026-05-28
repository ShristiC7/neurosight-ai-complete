"""
NeuroSight AI — Database Models
Full schema for all platform entities.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, JSON, Enum as SAEnum,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import BaseModel, Base, TimestampMixin, UUIDMixin
import enum


# -----------------------------------------------------------
# Enums
# -----------------------------------------------------------
class DrowsinessLevel(str, enum.Enum):
    ALERT = "alert"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class EmotionState(str, enum.Enum):
    CALM = "calm"
    STRESSED = "stressed"
    FATIGUED = "fatigued"
    ENERGETIC = "energetic"
    ANXIOUS = "anxious"


class RecommendationType(str, enum.Enum):
    TAKE_BREAK = "take_break"
    STRETCH = "stretch"
    HYDRATE = "hydrate"
    DEEP_WORK = "deep_work"
    LIGHT_TASK = "light_task"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    MEDITATION = "meditation"
    EYE_REST = "eye_rest"
    POSTURE_CHECK = "posture_check"


class RecommendationPriority(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# -----------------------------------------------------------
# User
# -----------------------------------------------------------
class User(BaseModel):
    __tablename__ = "users"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    sessions: Mapped[list["WorkSession"]] = relationship(
        "WorkSession", back_populates="user", cascade="all, delete-orphan"
    )
    fatigue_metrics: Mapped[list["FatigueMetric"]] = relationship(
        "FatigueMetric", back_populates="user", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        "Recommendation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"


# -----------------------------------------------------------
# Refresh Token
# -----------------------------------------------------------
class RefreshToken(BaseModel):
    __tablename__ = "refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)


# -----------------------------------------------------------
# Work Session
# -----------------------------------------------------------
class WorkSession(BaseModel):
    __tablename__ = "work_sessions"
    __table_args__ = (
        Index("ix_work_sessions_user_id_start", "user_id", "start_time"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Aggregated session metrics
    avg_fatigue_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_productivity_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_stress_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_focus_time: Mapped[int] = mapped_column(Integer, default=0)  # minutes
    breaks_taken: Mapped[int] = mapped_column(Integer, default=0)
    total_keystrokes: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    fatigue_metrics: Mapped[list["FatigueMetric"]] = relationship(
        "FatigueMetric", back_populates="session", cascade="all, delete-orphan"
    )
    productivity_predictions: Mapped[list["ProductivityPrediction"]] = relationship(
        "ProductivityPrediction", back_populates="session", cascade="all, delete-orphan"
    )


# -----------------------------------------------------------
# Fatigue Metrics
# -----------------------------------------------------------
class FatigueMetric(BaseModel):
    __tablename__ = "fatigue_metrics"
    __table_args__ = (
        Index("ix_fatigue_metrics_user_time", "user_id", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    blink_rate: Mapped[float] = mapped_column(Float, nullable=False)
    eye_aspect_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    mouth_aspect_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    head_tilt_angle: Mapped[float] = mapped_column(Float, default=0.0)
    gaze_drift: Mapped[float] = mapped_column(Float, default=0.0)
    fatigue_score: Mapped[float] = mapped_column(Float, nullable=False)
    drowsiness_level: Mapped[DrowsinessLevel] = mapped_column(
        SAEnum(DrowsinessLevel), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.9)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="fatigue_metrics")
    session: Mapped["WorkSession"] = relationship("WorkSession", back_populates="fatigue_metrics")


# -----------------------------------------------------------
# Voice Stress Metrics
# -----------------------------------------------------------
class VoiceStressMetric(BaseModel):
    __tablename__ = "voice_stress_metrics"
    __table_args__ = (
        Index("ix_voice_stress_user_time", "user_id", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    pitch_variance: Mapped[float] = mapped_column(Float, default=0.0)
    speech_energy: Mapped[float] = mapped_column(Float, default=0.0)
    pause_duration: Mapped[float] = mapped_column(Float, default=0.0)
    stress_score: Mapped[float] = mapped_column(Float, nullable=False)
    emotion_state: Mapped[EmotionState] = mapped_column(
        SAEnum(EmotionState), nullable=False
    )
    mfcc_features: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.75)


# -----------------------------------------------------------
# Behavioral Metrics
# -----------------------------------------------------------
class BehavioralMetric(BaseModel):
    __tablename__ = "behavioral_metrics"
    __table_args__ = (
        Index("ix_behavioral_user_time", "user_id", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    typing_speed: Mapped[float] = mapped_column(Float, default=0.0)
    typing_rhythm_variance: Mapped[float] = mapped_column(Float, default=0.0)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)
    mouse_movement_entropy: Mapped[float] = mapped_column(Float, default=0.0)
    mouse_click_rate: Mapped[float] = mapped_column(Float, default=0.0)
    app_switch_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    focus_session_duration: Mapped[float] = mapped_column(Float, default=0.0)
    idle_time: Mapped[float] = mapped_column(Float, default=0.0)
    behavior_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)


# -----------------------------------------------------------
# Productivity Prediction
# -----------------------------------------------------------
class ProductivityPrediction(BaseModel):
    __tablename__ = "productivity_predictions"
    __table_args__ = (
        Index("ix_productivity_user_time", "user_id", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    productivity_score: Mapped[float] = mapped_column(Float, nullable=False)
    burnout_probability: Mapped[float] = mapped_column(Float, nullable=False)
    cognitive_load: Mapped[float] = mapped_column(Float, default=0.0)
    focus_window_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    focus_window_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    recommended_break_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    predicted_crash_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    feature_importance: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped["WorkSession"] = relationship(
        "WorkSession", back_populates="productivity_predictions"
    )


# -----------------------------------------------------------
# Recommendations
# -----------------------------------------------------------
class Recommendation(BaseModel):
    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_recommendations_user_time", "user_id", "timestamp"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    type: Mapped[RecommendationType] = mapped_column(SAEnum(RecommendationType), nullable=False)
    priority: Mapped[RecommendationPriority] = mapped_column(
        SAEnum(RecommendationPriority), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    action_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    action_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    # RL tracking
    rl_state_vector: Mapped[list] = mapped_column(JSON, default=list)
    rl_action_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="recommendations")


# -----------------------------------------------------------
# Behavioral Embedding (Vector DB mirror for auditing)
# -----------------------------------------------------------
class BehavioralEmbedding(BaseModel):
    __tablename__ = "behavioral_embeddings"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("work_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=256)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
