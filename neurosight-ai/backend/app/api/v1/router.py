"""
NeuroSight AI — API v1 Router
Aggregates all endpoint modules with proper tagging and prefixes.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    fatigue,
    audio,
    behavioral,
    predictions,
    recommendations,
    sessions,
    websocket,
    users,
    health,
)

api_router = APIRouter()

# -----------------------------------------------------------
# Register endpoint modules
# -----------------------------------------------------------
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)

api_router.include_router(
    sessions.router,
    prefix="/sessions",
    tags=["Sessions"],
)

api_router.include_router(
    fatigue.router,
    prefix="/fatigue",
    tags=["Fatigue Detection"],
)

api_router.include_router(
    audio.router,
    prefix="/audio",
    tags=["Voice Stress Analysis"],
)

api_router.include_router(
    behavioral.router,
    prefix="/behavioral",
    tags=["Behavioral Analytics"],
)

api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Productivity Predictions"],
)

api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["AI Recommendations"],
)

api_router.include_router(
    websocket.router,
    prefix="/ws",
    tags=["WebSocket"],
)
