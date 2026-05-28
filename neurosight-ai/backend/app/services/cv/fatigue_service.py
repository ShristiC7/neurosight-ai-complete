"""
NeuroSight AI — Fatigue Detection Service
Handles frame analysis, metric persistence, trend queries, and EAR computation.
"""
import io
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import FatigueMetric, DrowsinessLevel
from app.schemas.fatigue import (
    FatigueMetricCreate, FatigueMetricResponse,
    FatigueAnalysisResponse, FatigueTrendResponse,
)
from app.services.ml_registry import ModelRegistry

logger = structlog.get_logger(__name__)

EAR_THRESHOLD = 0.25
BLINK_WINDOW_SECONDS = 60


class FatigueService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def store_metric(self, user_id: uuid.UUID, payload: FatigueMetricCreate) -> FatigueMetric:
        """Persist a fatigue metric received from the frontend MediaPipe pipeline."""
        metric = FatigueMetric(
            user_id=user_id,
            session_id=payload.session_id,
            timestamp=payload.timestamp,
            blink_rate=payload.blink_rate,
            eye_aspect_ratio=payload.eye_aspect_ratio,
            mouth_aspect_ratio=payload.mouth_aspect_ratio,
            head_tilt_angle=payload.head_tilt_angle,
            gaze_drift=payload.gaze_drift,
            fatigue_score=payload.fatigue_score,
            drowsiness_level=payload.drowsiness_level,
            confidence=payload.confidence,
        )
        self.db.add(metric)
        await self.db.flush()
        return metric

    async def analyze_frame(self, user_id: uuid.UUID, frame_bytes: bytes) -> FatigueAnalysisResponse:
        """
        Server-side frame analysis using OpenCV + MediaPipe.
        Falls back to heuristic if models unavailable.
        """
        start = time.perf_counter()
        ear = 0.3
        mar = 0.2
        blink_rate = 15.0
        fatigue_score = 10.0
        confidence = 0.5

        try:
            import cv2
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                import mediapipe as mp
                mp_face = mp.solutions.face_mesh
                with mp_face.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                ) as face_mesh:
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0].landmark
                        h, w = frame.shape[:2]

                        def pt(idx):
                            return np.array([lm[idx].x * w, lm[idx].y * h])

                        # Left eye EAR (indices: 33,160,158,133,153,144)
                        le = [pt(i) for i in [33, 160, 158, 133, 153, 144]]
                        left_ear = _ear(le)
                        re = [pt(i) for i in [362, 385, 387, 263, 373, 380]]
                        right_ear = _ear(re)
                        ear = (left_ear + right_ear) / 2

                        # MAR
                        mouth_pts = [pt(i) for i in [61, 291, 39, 181, 0, 17]]
                        mar = _ear(mouth_pts)

                        fatigue_score = max(0.0, min(100.0, (1 - ear / 0.4) * 70 + (mar / 0.8) * 30))
                        confidence = 0.88

            except ImportError:
                logger.warning("MediaPipe not available, using OpenCV fallback")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fatigue_score = float(255 - gray.mean()) / 255 * 50
                confidence = 0.4

        except Exception as e:
            logger.warning("Frame analysis failed", error=str(e))

        drowsiness = _score_to_level(fatigue_score)
        inference_ms = (time.perf_counter() - start) * 1000

        return FatigueAnalysisResponse(
            fatigue_score=round(fatigue_score, 2),
            drowsiness_level=drowsiness,
            ear=round(ear, 4),
            mar=round(mar, 4),
            blink_rate=round(blink_rate, 1),
            confidence=round(confidence, 3),
            inference_time_ms=round(inference_ms, 2),
        )

    async def get_trend(self, user_id: uuid.UUID, hours: int) -> FatigueTrendResponse:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self.db.execute(
            select(FatigueMetric)
            .where(and_(FatigueMetric.user_id == user_id,
                        FatigueMetric.timestamp >= cutoff))
            .order_by(FatigueMetric.timestamp.asc())
            .limit(500)
        )
        metrics = result.scalars().all()
        data_points = [{"timestamp": m.timestamp.isoformat(), "value": m.fatigue_score,
                        "level": m.drowsiness_level.value} for m in metrics]
        scores = [m.fatigue_score for m in metrics]

        trend = "stable"
        if len(scores) >= 4:
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            if second_half > first_half + 5:
                trend = "declining"
            elif second_half < first_half - 5:
                trend = "improving"

        return FatigueTrendResponse(
            user_id=user_id,
            hours=hours,
            data_points=data_points,
            avg_fatigue_score=round(sum(scores) / len(scores), 2) if scores else 0,
            max_fatigue_score=round(max(scores), 2) if scores else 0,
            trend_direction=trend,
        )

    async def get_by_session(self, user_id: uuid.UUID, session_id: uuid.UUID) -> list[FatigueMetric]:
        result = await self.db.execute(
            select(FatigueMetric)
            .where(and_(FatigueMetric.user_id == user_id,
                        FatigueMetric.session_id == session_id))
            .order_by(FatigueMetric.timestamp.asc())
        )
        return result.scalars().all()


def _ear(pts: list) -> float:
    """Eye Aspect Ratio: (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)"""
    p1,p2,p3,p4,p5,p6 = pts
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    return (v1 + v2) / (2 * h + 1e-6)


def _score_to_level(score: float) -> DrowsinessLevel:
    if score < 20:  return DrowsinessLevel.ALERT
    if score < 40:  return DrowsinessLevel.MILD
    if score < 60:  return DrowsinessLevel.MODERATE
    if score < 80:  return DrowsinessLevel.SEVERE
    return DrowsinessLevel.CRITICAL
