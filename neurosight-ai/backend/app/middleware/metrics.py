"""
NeuroSight AI — Prometheus Metrics Middleware
Tracks request counts, latency, and ML inference timing.
"""

import time
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# -----------------------------------------------------------
# Metrics Definitions
# -----------------------------------------------------------
REQUEST_COUNT = Counter(
    "neurosight_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "neurosight_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_REQUESTS = Gauge(
    "neurosight_http_active_requests",
    "Number of active HTTP requests",
    ["method"],
)

WS_CONNECTIONS = Gauge(
    "neurosight_websocket_connections_total",
    "Active WebSocket connections",
)

ML_INFERENCE_LATENCY = Histogram(
    "neurosight_ml_inference_duration_seconds",
    "ML model inference latency",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)

ML_INFERENCE_COUNT = Counter(
    "neurosight_ml_inferences_total",
    "Total ML inferences",
    ["model", "status"],
)

FATIGUE_SCORE_HISTOGRAM = Histogram(
    "neurosight_fatigue_score",
    "Distribution of fatigue scores",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

PRODUCTIVITY_SCORE_HISTOGRAM = Histogram(
    "neurosight_productivity_score",
    "Distribution of productivity scores",
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

RECOMMENDATION_ACCEPTANCE = Counter(
    "neurosight_recommendation_acceptance_total",
    "Recommendation acceptance/rejection counts",
    ["action_type", "accepted"],
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Track HTTP metrics for all requests."""

    SKIP_PATHS = {"/metrics", "/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Normalize endpoint label (avoid high cardinality from path params)
        endpoint = self._normalize_path(request.url.path)
        method = request.method

        ACTIVE_REQUESTS.labels(method=method).inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()
            return response
        except Exception:
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=500).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            ACTIVE_REQUESTS.labels(method=method).dec()

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Replace UUIDs and numbers with placeholders to reduce cardinality."""
        import re
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            ":uuid",
            path,
            flags=re.IGNORECASE,
        )
        path = re.sub(r"/\d+", "/:id", path)
        return path
