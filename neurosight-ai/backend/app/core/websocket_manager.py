"""
NeuroSight AI — Global WebSocket Manager
Allows backend services (Celery tasks, ML inference) to push messages
to connected clients without direct WebSocket access.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

logger = structlog.get_logger(__name__)


class GlobalWebSocketManager:
    """
    Registry that maps user_ids to their active ConnectionManager instance.
    Used by Celery tasks and background services to push real-time updates.

    Architecture note:
    In a multi-node deployment, use Redis pub/sub here instead of
    direct in-process refs — each node holds its own WS connections,
    and Redis broadcasts across nodes.
    """

    def __init__(self) -> None:
        # user_id → ConnectionManager (from websocket.py)
        self._registry: dict[str, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def register(self, user_id: str, manager: Any) -> None:
        self._registry[user_id] = manager

    def unregister(self, user_id: str) -> None:
        self._registry.pop(user_id, None)

    async def send_to_user(self, user_id: str, event: str, payload: dict) -> bool:
        """
        Send a WebSocket message to a specific user.
        Returns True if user is connected, False otherwise.
        """
        manager = self._registry.get(user_id)
        if not manager:
            logger.debug("User not connected to WS", user_id=user_id)
            return False

        message = {
            "event": event,
            "payload": payload,
            "timestamp": __import__("time").time(),
        }

        try:
            await manager.send_to_user(user_id, message)
            return True
        except Exception as e:
            logger.error("Failed to send WS message", user_id=user_id, error=str(e))
            return False

    async def broadcast_event(self, event: str, payload: dict) -> int:
        """Broadcast to all connected users. Returns count of recipients."""
        count = 0
        for user_id, manager in list(self._registry.items()):
            try:
                await manager.send_to_user(user_id, {"event": event, "payload": payload})
                count += 1
            except Exception:
                pass
        return count

    async def push_fatigue_update(self, user_id: str, metrics: dict) -> None:
        await self.send_to_user(user_id, "fatigue:update", metrics)

    async def push_recommendation(self, user_id: str, recommendation: dict) -> None:
        await self.send_to_user(user_id, "recommendation:new", recommendation)

    async def push_alert(self, user_id: str, level: str, message: str) -> None:
        await self.send_to_user(user_id, "alert:critical", {
            "level": level,
            "message": message,
        })

    async def push_prediction(self, user_id: str, prediction: dict) -> None:
        await self.send_to_user(user_id, "prediction:update", prediction)

    async def close_all(self) -> None:
        for manager in self._registry.values():
            try:
                await manager.close_all()
            except Exception:
                pass
        self._registry.clear()

    @property
    def connected_user_count(self) -> int:
        return len(self._registry)


# Singleton
ws_manager = GlobalWebSocketManager()
