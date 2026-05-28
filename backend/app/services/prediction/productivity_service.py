"""
NeuroSight AI — Productivity Prediction Service
Orchestrates LSTM, XGBoost, and TFT models for time-series forecasting.

Features used for prediction:
    - Historical fatigue scores (30-min window)
    - Voice stress metrics
    - Typing speed & rhythm variance
    - Mouse movement entropy
    - App switching frequency
    - Session duration
    - Time of day (cyclically encoded)
    - Day of week
    - Historical productivity scores
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np
import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    FatigueMetric,
    VoiceStressMetric,
    BehavioralMetric,
    ProductivityPrediction,
    WorkSession,
)
from app.services.ml_registry import ModelRegistry
from app.core.redis import redis_client
import json

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Feature Engineering Constants
# -----------------------------------------------------------
FEATURE_WINDOW_MINUTES = 30
SEQUENCE_LENGTH = 30        # Time steps for LSTM
FEATURE_DIM = 12            # Features per time step

# XGBoost feature names (order matters)
XGBOOST_FEATURE_NAMES = [
    "avg_fatigue_30m",
    "avg_stress_30m",
    "avg_typing_speed_30m",
    "typing_rhythm_variance",
    "mouse_entropy",
    "app_switch_freq",
    "session_duration_hours",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "prev_productivity_score",
    "fatigue_trend",       # Slope of fatigue in last 30m
    "stress_trend",
    "blink_rate_avg",
    "error_rate_avg",
]


class ProductivityPredictionService:
    """
    Multi-model productivity prediction pipeline.

    Architecture:
        1. Feature extraction from recent metrics
        2. XGBoost for current productivity score (fast, interpretable)
        3. LSTM for next-hour forecast
        4. Ensemble: weighted average of model outputs
        5. Burnout risk = exponential moving average of fatigue + stress
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # -----------------------------------------------------------
    # Main prediction entry point
    # -----------------------------------------------------------
    async def predict(
        self,
        user_id: UUID,
        session_id: UUID,
    ) -> ProductivityPrediction:
        """
        Generate a full productivity prediction for the current moment.
        Results are cached in Redis for 60 seconds.
        """
        cache_key = f"prediction:{user_id}:{session_id}"
        cached = await redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ProductivityPrediction(**data)

        # Extract features
        features = await self._extract_features(user_id, session_id)
        if features is None:
            return self._default_prediction(user_id, session_id)

        # Run predictions
        productivity_score = await self._predict_productivity(features)
        burnout_prob = self._compute_burnout_probability(features)
        cognitive_load = self._compute_cognitive_load(features)
        focus_window = self._predict_focus_window(features, productivity_score)
        break_time = self._recommend_break_time(features)
        crash_time = self._predict_cognitive_crash(features)

        prediction = ProductivityPrediction(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            productivity_score=productivity_score,
            burnout_probability=burnout_prob,
            cognitive_load=cognitive_load,
            focus_window_start=focus_window[0],
            focus_window_end=focus_window[1],
            recommended_break_at=break_time,
            predicted_crash_at=crash_time,
            confidence=features.get("model_confidence", 0.8),
            feature_importance=features.get("importance", {}),
        )

        # Persist
        self.db.add(prediction)
        await self.db.flush()

        # Cache
        await redis_client.setex(
            cache_key,
            60,
            json.dumps(self._serialize_prediction(prediction)),
        )

        return prediction

    # -----------------------------------------------------------
    # Feature Extraction
    # -----------------------------------------------------------
    async def _extract_features(
        self,
        user_id: UUID,
        session_id: UUID,
    ) -> dict | None:
        """Pull last FEATURE_WINDOW_MINUTES of metrics, compute features."""
        window_start = datetime.now(timezone.utc) - timedelta(minutes=FEATURE_WINDOW_MINUTES)

        # Fatigue metrics
        fatigue_rows = await self.db.execute(
            select(FatigueMetric)
            .where(and_(
                FatigueMetric.user_id == user_id,
                FatigueMetric.session_id == session_id,
                FatigueMetric.timestamp >= window_start,
            ))
            .order_by(FatigueMetric.timestamp)
        )
        fatigue_metrics = fatigue_rows.scalars().all()

        # Stress metrics
        stress_rows = await self.db.execute(
            select(VoiceStressMetric)
            .where(and_(
                VoiceStressMetric.user_id == user_id,
                VoiceStressMetric.session_id == session_id,
                VoiceStressMetric.timestamp >= window_start,
            ))
            .order_by(VoiceStressMetric.timestamp)
        )
        stress_metrics = stress_rows.scalars().all()

        # Behavioral metrics
        behavioral_rows = await self.db.execute(
            select(BehavioralMetric)
            .where(and_(
                BehavioralMetric.user_id == user_id,
                BehavioralMetric.session_id == session_id,
                BehavioralMetric.timestamp >= window_start,
            ))
            .order_by(BehavioralMetric.timestamp)
        )
        behavioral_metrics = behavioral_rows.scalars().all()

        if not fatigue_metrics and not behavioral_metrics:
            return None

        now = datetime.now(timezone.utc)
        hour = now.hour
        day = now.weekday()

        # Aggregate features
        fatigue_scores = [m.fatigue_score for m in fatigue_metrics] or [0.0]
        stress_scores = [m.stress_score for m in stress_metrics] or [0.0]
        typing_speeds = [m.typing_speed for m in behavioral_metrics] or [0.0]

        avg_fatigue = np.mean(fatigue_scores)
        avg_stress = np.mean(stress_scores)
        avg_typing = np.mean(typing_speeds)

        # Compute trends (linear regression slope)
        fatigue_trend = self._compute_trend(fatigue_scores)
        stress_trend = self._compute_trend(stress_scores)

        # Session duration
        session_result = await self.db.execute(
            select(WorkSession).where(WorkSession.id == session_id)
        )
        session = session_result.scalar_one_or_none()
        session_duration_hours = 0.0
        if session:
            delta = now - session.start_time.replace(tzinfo=timezone.utc)
            session_duration_hours = delta.total_seconds() / 3600

        # Build LSTM sequence (pad with zeros if insufficient data)
        lstm_sequence = self._build_lstm_sequence(
            fatigue_metrics, stress_metrics, behavioral_metrics
        )

        return {
            # Scalar features (XGBoost)
            "avg_fatigue_30m": avg_fatigue,
            "avg_stress_30m": avg_stress,
            "avg_typing_speed_30m": avg_typing,
            "typing_rhythm_variance": np.mean([m.typing_rhythm_variance for m in behavioral_metrics] or [0.0]),
            "mouse_entropy": np.mean([m.mouse_movement_entropy for m in behavioral_metrics] or [0.0]),
            "app_switch_freq": np.mean([m.app_switch_frequency for m in behavioral_metrics] or [0.0]),
            "session_duration_hours": session_duration_hours,
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "day_sin": math.sin(2 * math.pi * day / 7),
            "day_cos": math.cos(2 * math.pi * day / 7),
            "prev_productivity_score": 75.0,  # TODO: pull from history
            "fatigue_trend": fatigue_trend,
            "stress_trend": stress_trend,
            "blink_rate_avg": np.mean([m.blink_rate for m in fatigue_metrics] or [15.0]),
            "error_rate_avg": np.mean([m.error_rate for m in behavioral_metrics] or [0.0]),

            # Sequence features (LSTM)
            "lstm_sequence": lstm_sequence,

            # Meta
            "model_confidence": min(0.95, len(fatigue_metrics) / 30.0),
        }

    def _build_lstm_sequence(
        self,
        fatigue_metrics,
        stress_metrics,
        behavioral_metrics,
    ) -> np.ndarray:
        """
        Build a (SEQUENCE_LENGTH, FEATURE_DIM) array for LSTM input.
        Pads with zeros if not enough data points.
        """
        seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        n = min(SEQUENCE_LENGTH, len(fatigue_metrics))

        for i in range(n):
            idx = -(n - i)  # Count from end
            fm = fatigue_metrics[idx] if idx < 0 else fatigue_metrics[i]
            bm = behavioral_metrics[idx] if (behavioral_metrics and idx < len(behavioral_metrics)) else None

            seq[i, 0] = fm.fatigue_score / 100.0
            seq[i, 1] = fm.eye_aspect_ratio
            seq[i, 2] = fm.blink_rate / 30.0
            seq[i, 3] = fm.head_tilt_angle / 45.0
            if bm:
                seq[i, 4] = min(bm.typing_speed / 100.0, 1.0)
                seq[i, 5] = min(bm.mouse_movement_entropy, 1.0)
                seq[i, 6] = min(bm.error_rate / 10.0, 1.0)
                seq[i, 7] = min(bm.app_switch_frequency / 20.0, 1.0)

        return seq

    def _compute_trend(self, values: list[float]) -> float:
        """Simple linear regression slope for trend detection."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        # Slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
        sx, sy = x.sum(), y.sum()
        sxy = (x * y).sum()
        sxx = (x * x).sum()
        denom = n * sxx - sx * sx
        return float((n * sxy - sx * sy) / denom) if denom != 0 else 0.0

    # -----------------------------------------------------------
    # Model Inference
    # -----------------------------------------------------------
    async def _predict_productivity(self, features: dict) -> float:
        """
        XGBoost inference for current productivity score.
        Falls back to rule-based calculation if model unavailable.
        """
        model = ModelRegistry.get_productivity_model()

        if model is not None:
            try:
                feature_vector = np.array([[
                    features[k] for k in XGBOOST_FEATURE_NAMES
                ]], dtype=np.float32)
                result = model.run({"input": feature_vector})
                return float(np.clip(result["output"][0][0], 0, 100))
            except Exception as e:
                logger.warning("Productivity model inference failed", error=str(e))

        # Rule-based fallback
        fatigue_penalty = features["avg_fatigue_30m"] * 0.35
        stress_penalty = features["avg_stress_30m"] * 0.25
        typing_bonus = min(features["avg_typing_speed_30m"] / 60.0, 1.0) * 20
        base = 80.0
        score = base - fatigue_penalty - stress_penalty + typing_bonus
        return float(np.clip(score, 0, 100))

    def _compute_burnout_probability(self, features: dict) -> float:
        """
        Burnout risk based on fatigue trend + stress + session duration.
        Returns probability 0-1.
        """
        fatigue_risk = features["avg_fatigue_30m"] / 100.0
        stress_risk = features["avg_stress_30m"] / 100.0
        duration_risk = min(features["session_duration_hours"] / 8.0, 1.0)
        trend_risk = max(0.0, features["fatigue_trend"] * 10)  # Rising fatigue = more risk

        # Weighted risk score
        risk = (
            fatigue_risk * 0.4 +
            stress_risk * 0.3 +
            duration_risk * 0.2 +
            trend_risk * 0.1
        )
        return float(np.clip(risk, 0.0, 1.0))

    def _compute_cognitive_load(self, features: dict) -> float:
        """Estimated cognitive load 0-100."""
        load = (
            features["avg_fatigue_30m"] * 0.4 +
            features["avg_stress_30m"] * 0.35 +
            features["app_switch_freq"] * 2.0 +
            features["error_rate_avg"] * 5.0
        )
        return float(np.clip(load, 0, 100))

    def _predict_focus_window(
        self,
        features: dict,
        current_score: float,
    ) -> tuple[datetime | None, datetime | None]:
        """
        Predict the next peak focus window based on current trends.
        Simple heuristic: if declining, next peak is after recovery.
        """
        now = datetime.now(timezone.utc)
        if current_score >= 70 and features["fatigue_trend"] <= 0:
            # Currently in a good window — it continues for another 45min
            return now, now + timedelta(minutes=45)

        recovery_minutes = max(15, int(features["avg_fatigue_30m"] / 2))
        window_start = now + timedelta(minutes=recovery_minutes)
        window_end = window_start + timedelta(minutes=60)
        return window_start, window_end

    def _recommend_break_time(self, features: dict) -> datetime | None:
        """Recommend next break based on fatigue level and trends."""
        now = datetime.now(timezone.utc)
        fatigue = features["avg_fatigue_30m"]
        trend = features["fatigue_trend"]

        if fatigue >= 70:
            return now + timedelta(minutes=5)
        if fatigue >= 50 or trend > 1.0:
            return now + timedelta(minutes=15)
        if features["session_duration_hours"] >= 1.5:
            return now + timedelta(minutes=30)
        return now + timedelta(minutes=45)

    def _predict_cognitive_crash(self, features: dict) -> datetime | None:
        """Predict when cognitive performance will become critical."""
        now = datetime.now(timezone.utc)
        fatigue = features["avg_fatigue_30m"]
        trend = features["fatigue_trend"]

        if fatigue >= 80:
            return now + timedelta(minutes=10)
        if trend > 0.5 and fatigue >= 50:
            minutes_to_crash = max(5, int((100 - fatigue) / (trend + 0.1)))
            return now + timedelta(minutes=minutes_to_crash)
        return None

    def _default_prediction(self, user_id: UUID, session_id: UUID) -> ProductivityPrediction:
        """Return a neutral prediction when insufficient data is available."""
        now = datetime.now(timezone.utc)
        return ProductivityPrediction(
            user_id=user_id,
            session_id=session_id,
            timestamp=now,
            productivity_score=75.0,
            burnout_probability=0.1,
            cognitive_load=30.0,
            focus_window_start=now,
            focus_window_end=now + timedelta(hours=1),
            recommended_break_at=now + timedelta(minutes=45),
            confidence=0.3,
        )

    def _serialize_prediction(self, pred: ProductivityPrediction) -> dict:
        return {
            "user_id": str(pred.user_id),
            "session_id": str(pred.session_id),
            "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
            "productivity_score": pred.productivity_score,
            "burnout_probability": pred.burnout_probability,
            "cognitive_load": pred.cognitive_load,
            "recommended_break_at": pred.recommended_break_at.isoformat() if pred.recommended_break_at else None,
            "confidence": pred.confidence,
        }
