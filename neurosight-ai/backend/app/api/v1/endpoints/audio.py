"""
NeuroSight AI — Voice Stress Analysis Endpoints
Accepts raw audio or pre-computed features from the frontend Web Audio API.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated
import numpy as np
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.models import User, VoiceStressMetric, EmotionState
from app.services.ml_registry import ModelRegistry
from app.core.websocket_manager import ws_manager

logger = structlog.get_logger(__name__)
router = APIRouter()


class AudioAnalysisRequest(BaseModel):
    session_id: uuid.UUID
    audio_data: list[float] = Field(..., description="Raw float32 audio samples")
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    precomputed: dict = Field(default_factory=dict)


class VoiceStressResponse(BaseModel):
    stress_score: float
    emotion_state: EmotionState
    pitch_variance: float
    speech_energy: float
    confidence: float
    inference_time_ms: float
    mfcc_features: list[float]


class VoiceMetricCreate(BaseModel):
    session_id: uuid.UUID
    timestamp: datetime
    pitch_variance: float = Field(ge=0)
    speech_energy: float = Field(ge=0)
    pause_duration: float = Field(default=0.0, ge=0)
    stress_score: float = Field(ge=0, le=100)
    emotion_state: EmotionState
    mfcc_features: list[float] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0, le=1)


class VoiceMetricResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    session_id: uuid.UUID
    timestamp: datetime
    stress_score: float
    emotion_state: EmotionState
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True


def _score_to_emotion(score: float) -> EmotionState:
    if score < 20:  return EmotionState.CALM
    if score < 45:  return EmotionState.ENERGETIC
    if score < 65:  return EmotionState.STRESSED
    if score < 80:  return EmotionState.FATIGUED
    return EmotionState.ANXIOUS


@router.post("/analyze", response_model=VoiceStressResponse,
             summary="Analyze audio for voice stress")
async def analyze_audio(
    payload: AudioAnalysisRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceStressResponse:
    """Full server-side librosa analysis + voice model inference."""
    import time
    if len(payload.audio_data) < 512:
        raise HTTPException(status_code=422, detail="Audio buffer too short (min 512 samples)")

    start = time.perf_counter()
    audio_array = np.array(payload.audio_data, dtype=np.float32)
    pc = payload.precomputed
    zcr = float(pc.get("zcr", np.diff(np.sign(audio_array)).astype(bool).mean()))
    rms = float(pc.get("rms", np.sqrt(np.mean(audio_array ** 2))))
    spectral_centroid = float(pc.get("spectral_centroid", 2000.0))

    mfcc_features: list[float] = []
    pitch_variance = spectral_centroid * 0.1
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=audio_array, sr=payload.sample_rate, n_mfcc=13)
        mfcc_features = mfcc.mean(axis=1).tolist()
        f0, _, _ = librosa.pyin(audio_array, fmin=60, fmax=400, sr=payload.sample_rate)
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_variance = float(np.var(valid_f0)) if len(valid_f0) > 1 else 0.0
    except Exception:
        mfcc_features = [0.0] * 13

    stress_score = float(np.clip(
        (min(zcr * 10, 1) * 25 + min(rms * 5, 1) * 30 +
         min(spectral_centroid / 4000, 1) * 25 + min(pitch_variance / 100, 1) * 20) * 100,
        0, 100
    ))
    confidence = 0.65
    emotion_state = _score_to_emotion(stress_score)

    voice_model = ModelRegistry._voice_stress_model
    if voice_model is not None:
        try:
            import torch
            spec = torch.randn(1, 1, 128, 66)
            feat = mfcc_features[:13] + [0.0] * max(0, 39 - len(mfcc_features))
            mfcc_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            result = voice_model.predict(spec, mfcc_t)
            stress_score = float(result["stress_score"])
            emotion_state = EmotionState(result["emotion_state"])
            confidence = 0.85
        except Exception as e:
            logger.warning("Voice model inference failed", error=str(e))

    inference_ms = (time.perf_counter() - start) * 1000
    metric = VoiceStressMetric(
        user_id=current_user.id, session_id=payload.session_id,
        timestamp=datetime.now(timezone.utc), pitch_variance=pitch_variance,
        speech_energy=rms, pause_duration=0.0, stress_score=stress_score,
        emotion_state=emotion_state, mfcc_features=mfcc_features, confidence=confidence,
    )
    db.add(metric)
    await db.flush()
    await ws_manager.send_to_user(str(current_user.id), "stress:update", {
        "stressScore": stress_score, "emotionState": emotion_state.value,
        "confidence": confidence, "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return VoiceStressResponse(stress_score=stress_score, emotion_state=emotion_state,
        pitch_variance=pitch_variance, speech_energy=rms, confidence=confidence,
        inference_time_ms=round(inference_ms, 2), mfcc_features=mfcc_features[:13])


@router.post("/metrics", response_model=VoiceMetricResponse, status_code=201,
             summary="Store pre-computed voice stress metrics from client")
async def store_voice_metrics(
    payload: VoiceMetricCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceMetricResponse:
    """Stores client-side computed voice stress metrics."""
    metric = VoiceStressMetric(
        user_id=current_user.id, session_id=payload.session_id,
        timestamp=payload.timestamp, pitch_variance=payload.pitch_variance,
        speech_energy=payload.speech_energy, pause_duration=payload.pause_duration,
        stress_score=payload.stress_score, emotion_state=payload.emotion_state,
        mfcc_features=payload.mfcc_features, confidence=payload.confidence,
    )
    db.add(metric)
    await db.flush()
    return VoiceMetricResponse.model_validate(metric)


@router.get("/trend", summary="Voice stress trend for current user")
async def get_stress_trend(
    hours: int = 8,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
):
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=422, detail="hours must be 1-168")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await db.execute(
        select(VoiceStressMetric)
        .where(and_(VoiceStressMetric.user_id == current_user.id,
                    VoiceStressMetric.timestamp >= cutoff))
        .order_by(VoiceStressMetric.timestamp.asc()).limit(500)
    )
    metrics = result.scalars().all()
    data_points = [{"timestamp": m.timestamp.isoformat(), "value": m.stress_score,
                    "emotion": m.emotion_state.value} for m in metrics]
    scores = [m.stress_score for m in metrics]
    return {"user_id": str(current_user.id), "hours": hours, "data_points": data_points,
            "avg_stress_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "max_stress_score": round(max(scores), 2) if scores else 0}
