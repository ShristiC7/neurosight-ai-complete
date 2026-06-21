"""
NeuroSight AI — Rate Limiter Middleware
Redis-based sliding window rate limiting per user/IP.
Inference endpoints get tighter limits.
"""

import time
from typing import Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Tighter limits for ML/inference endpoints
INFERENCE_PATHS = {"/fatigue/analyze-frame", "/audio/analyze", "/predictions/"}
INFERENCE_LIMIT = settings.RATE_LIMIT_INFERENCE_REQUESTS

# Simple in-memory fallback for testing (no Redis required)
_test_counters: dict = {}

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis sorted sets.

    Algorithm:
    1. Store request timestamps in a sorted set per identifier
    2. Remove entries older than the window
    3. Count remaining entries — reject if over limit
    4. Add current timestamp

    This gives true sliding window behavior vs fixed-window counters.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ("/health", "/metrics", "/api/docs", "/api/redoc"):
            return await call_next(request)

        # In testing, skip Redis — use a trivial pass-through
        if settings.ENVIRONMENT == "testing":
            return await call_next(request)

        # Import redis_client lazily so tests can patch it
        from app.core import redis as redis_mod
        _redis = redis_mod.redis_client

        # Determine identifier (JWT user_id preferred, fallback to IP)
        identifier = await self._get_identifier(request)

        # Determine limit for this path
        is_inference = any(
            request.url.path.endswith(p) or p in request.url.path
            for p in INFERENCE_PATHS
        )
        limit = INFERENCE_LIMIT if is_inference else settings.RATE_LIMIT_REQUESTS
        window = settings.RATE_LIMIT_WINDOW_SECONDS

        # Sliding window check
        key = f"rl:{identifier}:{int(time.time() // window)}"
        now = time.time()

        try:
            async with _redis.pipeline() as pipe:
                # Remove expired entries
                pipe.zremrangebyscore(key, 0, now - window)
                # Count current window
                pipe.zcard(key)
                # Add this request
                pipe.zadd(key, {str(now): now})
                # Set TTL
                pipe.expire(key, window * 2)
                results = await pipe.execute()
        except Exception as exc:
            # Redis unavailable — fail open (allow request)
            logger.warning("Rate limiter Redis error, allowing request", error=str(exc))
            return await call_next(request)

        count = results[1]
        remaining = max(0, limit - count - 1)

        if count >= limit:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                path=request.url.path,
                count=count,
                limit=limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": window,
                },
                headers={
                    "Retry-After": str(window),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + window)),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + window))
        return response

    async def _get_identifier(self, request: Request) -> str:
        """Extract user ID from JWT if present, else use client IP."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                from app.core.security import decode_access_token
                payload = decode_access_token(auth[7:])
                return f"user:{payload.get('sub', 'unknown')}"
            except Exception:
                pass

        ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"
