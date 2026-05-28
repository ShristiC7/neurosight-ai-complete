"""
NeuroSight AI — Application Configuration
Pydantic Settings v2 with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------
    # Core
    # -----------------------------------------------------------
    APP_NAME: str = "NeuroSight AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    SECRET_KEY: str = Field(min_length=32)
    API_PREFIX: str = "/api/v1"

    # -----------------------------------------------------------
    # Server
    # -----------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False

    # -----------------------------------------------------------
    # CORS
    # -----------------------------------------------------------
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://neurosight.ai",
    ]

    # -----------------------------------------------------------
    # Database
    # -----------------------------------------------------------
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "neurosight"
    POSTGRES_USER: str = "neurosight"
    POSTGRES_PASSWORD: str

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # DB Pool
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    # -----------------------------------------------------------
    # Redis
    # -----------------------------------------------------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Cache TTLs (seconds)
    CACHE_DEFAULT_TTL: int = 300
    CACHE_USER_SESSION_TTL: int = 86400
    CACHE_PREDICTIONS_TTL: int = 60

    # -----------------------------------------------------------
    # Qdrant Vector DB
    # -----------------------------------------------------------
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_BEHAVIORAL: str = "behavioral_embeddings"
    QDRANT_COLLECTION_SESSIONS: str = "session_embeddings"
    QDRANT_EMBEDDING_DIM: int = 256

    # -----------------------------------------------------------
    # JWT
    # -----------------------------------------------------------
    JWT_SECRET_KEY: str = Field(min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # -----------------------------------------------------------
    # Celery
    # -----------------------------------------------------------
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300
    CELERY_TASK_TIME_LIMIT: int = 600

    # -----------------------------------------------------------
    # ML Model Paths
    # -----------------------------------------------------------
    MODEL_DIR: str = "/app/ml-models"
    EYE_FATIGUE_MODEL_PATH: str = "/app/ml-models/eye-fatigue/model.onnx"
    VOICE_STRESS_MODEL_PATH: str = "/app/ml-models/voice-stress/model.onnx"
    BEHAVIORAL_MODEL_PATH: str = "/app/ml-models/behavioral/model.onnx"
    PRODUCTIVITY_MODEL_PATH: str = "/app/ml-models/productivity/model.onnx"
    RL_AGENT_PATH: str = "/app/ml-models/rl-agent/agent.zip"

    # Inference
    INFERENCE_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    MODEL_BATCH_SIZE: int = 1
    INFERENCE_TIMEOUT_MS: int = 300

    # -----------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: str | None = None
    OTEL_EXPORTER_ENDPOINT: str | None = None

    # -----------------------------------------------------------
    # Rate Limiting
    # -----------------------------------------------------------
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_INFERENCE_REQUESTS: int = 30  # Tighter limit for ML endpoints

    # -----------------------------------------------------------
    # WebSocket
    # -----------------------------------------------------------
    WS_MAX_CONNECTIONS_PER_USER: int = 3
    WS_HEARTBEAT_INTERVAL: int = 25
    WS_MESSAGE_SIZE_LIMIT: int = 1_048_576  # 1MB

    # -----------------------------------------------------------
    # Feature Flags
    # -----------------------------------------------------------
    ENABLE_EDGE_INFERENCE: bool = False
    ENABLE_FEDERATED_LEARNING: bool = False
    ENABLE_EXPLAINABLE_AI: bool = True
    ENABLE_VOICE_ANALYSIS: bool = True
    ENABLE_RL_RECOMMENDATIONS: bool = True

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
