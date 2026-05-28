"""
NeuroSight AI — WebSocket Endpoint
Real-time bidirectional streaming for live metrics and recommendations.
"""

import json
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query, status

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.websocket_manager import ws_manager
from app.db.session import get_db, AsyncSession

logger = structlog.get_logger(__name__)
router = APIRouter()


# -----------------------------------------------------------
# WebSocket Connection Manager
# -----------------------------------------------------------
class ConnectionManager:
    """
    Manages WebSocket connections with user-scoped rooms.
    Supports broadcasting to specific users or all connections.
    """

    def __init__(self):
        # user_id -> list of WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> bool:
        """Accept connection and register. Returns False if limit exceeded."""
        current = self._connections.get(user_id, [])
        if len(current) >= settings.WS_MAX_CONNECTIONS_PER_USER:
            await websocket.close(code=4429, reason="Too many connections")
            return False

        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)
        logger.info("WS connected", user_id=user_id, total=len(current) + 1)
        return True

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        connections = self._connections.get(user_id, [])
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self._connections.pop(user_id, None)
        logger.info("WS disconnected", user_id=user_id)

    async def send_to_user(self, user_id: str, message: dict) -> None:
        """Send message to all connections for a specific user."""
        connections = self._connections.get(user_id, [])
        dead = []
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws, user_id)

    async def broadcast(self, message: dict) -> None:
        """Broadcast to all connected users."""
        for user_id in list(self._connections.keys()):
            await self.send_to_user(user_id, message)

    async def close_all(self) -> None:
        for connections in self._connections.values():
            for ws in connections:
                try:
                    await ws.close(code=1001, reason="Server shutting down")
                except Exception:
                    pass
        self._connections.clear()

    @property
    def connected_users(self) -> int:
        return len(self._connections)

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())


manager = ConnectionManager()


# -----------------------------------------------------------
# WS /ws — Main real-time stream
# -----------------------------------------------------------
@router.websocket("")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
    user_id: str = Query(..., description="User ID"),
):
    """
    Main WebSocket endpoint for real-time cognitive monitoring.

    Message types received from client:
    - ping: heartbeat
    - session:start: begin monitoring session
    - session:end: end monitoring session
    - frame:data: video frame data (if server-side analysis)
    - audio:data: audio chunk data

    Messages pushed to client:
    - fatigue:update: latest fatigue metrics
    - stress:update: voice stress metrics
    - behavioral:update: typing/mouse metrics
    - prediction:update: productivity forecast
    - recommendation:new: AI recommendation
    - alert:critical: critical fatigue/stress alert
    - pong: heartbeat response
    """

    # --- Auth validation ---
    try:
        payload = decode_access_token(token)
        token_user_id = payload.get("sub")
        if token_user_id != user_id:
            await websocket.close(code=4001, reason="Unauthorized")
            return
    except Exception:
        await websocket.close(code=4001, reason="Invalid token")
        return

    # --- Connect ---
    connected = await manager.connect(websocket, user_id)
    if not connected:
        return

    # Register with global manager (for cross-service broadcasting)
    ws_manager.register(user_id, manager)

    try:
        # Send initial connection acknowledgment
        await websocket.send_json({
            "event": "connection:established",
            "payload": {
                "userId": user_id,
                "serverTime": __import__("time").time(),
                "features": {
                    "fatigueDetection": True,
                    "voiceStress": settings.ENABLE_VOICE_ANALYSIS,
                    "behavioralAnalytics": True,
                    "rlRecommendations": settings.ENABLE_RL_RECOMMENDATIONS,
                },
            },
        })

        # --- Message Loop ---
        while True:
            try:
                raw = await websocket.receive_text()
                message = json.loads(raw)
                event = message.get("event", "")
                payload = message.get("payload", {})

            except json.JSONDecodeError:
                await websocket.send_json({
                    "event": "error",
                    "payload": {"message": "Invalid JSON"},
                })
                continue

            # Handle events
            if event == "ping":
                await websocket.send_json({
                    "event": "pong",
                    "payload": {"serverTime": __import__("time").time()},
                })

            elif event == "session:start":
                session_id = payload.get("sessionId")
                logger.info("WS session started", user_id=user_id, session_id=session_id)
                await websocket.send_json({
                    "event": "session:started",
                    "payload": {"sessionId": session_id},
                })

            elif event == "session:end":
                session_id = payload.get("sessionId")
                logger.info("WS session ended", user_id=user_id, session_id=session_id)
                await websocket.send_json({
                    "event": "session:ended",
                    "payload": {"sessionId": session_id},
                })

            elif event == "recommendation:feedback":
                rec_id = payload.get("recommendationId")
                accepted = payload.get("accepted", False)
                # Queue RL reward update
                from app.core.celery_app import celery_app
                celery_app.send_task(
                    "tasks.rl.update_reward",
                    kwargs={
                        "recommendation_id": rec_id,
                        "user_id": user_id,
                        "accepted": accepted,
                    },
                )

            else:
                logger.warning("Unknown WS event", event=event, user_id=user_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        ws_manager.unregister(user_id)
        logger.info("WS client disconnected", user_id=user_id)

    except Exception as e:
        logger.error("WS error", user_id=user_id, error=str(e), exc_info=True)
        manager.disconnect(websocket, user_id)
        ws_manager.unregister(user_id)
