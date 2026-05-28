"""
NeuroSight AI — Redis Client
Async Redis connection with retry logic and helper methods.
"""

from typing import Any

import structlog
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import settings

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Connection Pool
# -----------------------------------------------------------
_pool = ConnectionPool.from_url(
    settings.REDIS_URL,
    max_connections=50,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_error=[ConnectionError, TimeoutError],
    retry=Retry(ExponentialBackoff(base=0.5), 3),
    health_check_interval=30,
)

redis_client = Redis(connection_pool=_pool, decode_responses=True)


# -----------------------------------------------------------
# Cache Helpers
# -----------------------------------------------------------
class CacheClient:
    """High-level cache operations with structured key naming."""

    def __init__(self, client: Redis) -> None:
        self._r = client

    async def get_json(self, key: str) -> Any | None:
        import json
        val = await self._r.get(key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    async def set_json(self, key: str, value: Any, ttl: int = 300) -> None:
        import json
        await self._r.setex(key, ttl, json.dumps(value, default=str))

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def invalidate_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix pattern."""
        keys = await self._r.keys(f"{prefix}*")
        if keys:
            return await self._r.delete(*keys)
        return 0

    # --- User-scoped keys ---
    @staticmethod
    def user_predictions_key(user_id: str) -> str:
        return f"neurosight:predictions:{user_id}"

    @staticmethod
    def user_session_key(user_id: str) -> str:
        return f"neurosight:session:{user_id}"

    @staticmethod
    def user_recommendations_key(user_id: str) -> str:
        return f"neurosight:recs:{user_id}"

    @staticmethod
    def rate_limit_key(identifier: str, window: str) -> str:
        return f"neurosight:ratelimit:{identifier}:{window}"


cache = CacheClient(redis_client)
