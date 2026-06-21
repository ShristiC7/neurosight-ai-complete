"""
NeuroSight AI — FastAPI Application Entry Point
Production-grade async API with full middleware stack.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from prometheus_client import make_asgi_app

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import engine, Base
from app.core.redis import redis_client
from app.core.websocket_manager import ws_manager
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.metrics import MetricsMiddleware

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------
# Lifespan — startup / shutdown
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: DB init, Redis, ML model loading."""

    configure_logging()
    logger.info("NeuroSight AI starting up", env=settings.ENVIRONMENT)

    is_testing = settings.ENVIRONMENT == "testing"

    # Initialize database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")

    # Connect to Redis (skip in testing — fakeredis handles it)
    if not is_testing:
        try:
            await redis_client.ping()
            logger.info("Redis connection established")
        except Exception as exc:
            logger.warning("Redis unavailable at startup", error=str(exc))

    # Pre-load ML models into memory (skip in testing to keep tests fast)
    if not is_testing:
        from app.services.ml_registry import ModelRegistry
        await ModelRegistry.initialize()
        logger.info("ML models loaded", models=ModelRegistry.loaded_models())

    logger.info("NeuroSight AI ready", port=8000)

    yield

    # Graceful shutdown
    logger.info("Shutting down NeuroSight AI...")
    await ws_manager.close_all()
    if not is_testing:
        await redis_client.close()
    await engine.dispose()
    logger.info("Shutdown complete")



# -----------------------------------------------------------
# Application Factory
# -----------------------------------------------------------
def create_application() -> FastAPI:
    app = FastAPI(
        title="NeuroSight AI",
        description=(
            "Multimodal AI-powered cognitive fatigue and productivity "
            "intelligence platform. Real-time detection of fatigue, stress, "
            "and burnout using computer vision, audio, and behavioral analytics."
        ),
        version="1.0.0",
        docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # -------------------------------------------------------
    # Middleware (order matters — outermost executes first)
    # -------------------------------------------------------

    # 1. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )

    # 2. GZip compression for responses > 1KB
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # 3. Request ID injection
    app.add_middleware(RequestIDMiddleware)

    # 4. Prometheus metrics
    app.add_middleware(MetricsMiddleware)

    # 5. Rate limiting (per user/IP via Redis)
    app.add_middleware(RateLimitMiddleware)

    # -------------------------------------------------------
    # Routes
    # -------------------------------------------------------
    app.include_router(api_router, prefix="/api/v1")

    # Mount Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # -------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "status": "error",
                "message": f"Endpoint {request.url.path} not found",
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled server error",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error",
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
            },
        )

    # Health check (outside versioned router)
    @app.get("/health", tags=["Health"], include_in_schema=False)
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
        }

    return app


app = create_application()
