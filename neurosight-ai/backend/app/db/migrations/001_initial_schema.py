"""Initial schema — all NeuroSight AI tables

Revision ID: 001_initial
Revises:
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE IF NOT EXISTS drowsiness_level AS ENUM ('alert','mild','moderate','severe','critical')")
    op.execute("CREATE TYPE IF NOT EXISTS emotion_state AS ENUM ('calm','stressed','fatigued','energetic','anxious')")
    op.execute("CREATE TYPE IF NOT EXISTS recommendation_type AS ENUM ('take_break','stretch','hydrate','deep_work','light_task','sleep','exercise','meditation','eye_rest','posture_check')")
    op.execute("CREATE TYPE IF NOT EXISTS recommendation_priority AS ENUM ('critical','high','medium','low')")

    op.create_table("users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("is_verified", sa.Boolean, default=False),
        sa.Column("timezone", sa.String(50), default="UTC"),
        sa.Column("preferences", postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table("refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("token_hash", sa.Text, nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revoked", sa.Boolean, default=False),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table("work_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("avg_fatigue_score", sa.Float, default=0.0),
        sa.Column("avg_productivity_score", sa.Float, default=0.0),
        sa.Column("avg_stress_score", sa.Float, default=0.0),
        sa.Column("total_focus_time", sa.Integer, default=0),
        sa.Column("breaks_taken", sa.Integer, default=0),
        sa.Column("total_keystrokes", sa.Integer, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_work_sessions_user_id", "work_sessions", ["user_id"])

    op.create_table("fatigue_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("blink_rate", sa.Float, nullable=False),
        sa.Column("eye_aspect_ratio", sa.Float, nullable=False),
        sa.Column("mouth_aspect_ratio", sa.Float, nullable=False),
        sa.Column("head_tilt_angle", sa.Float, default=0.0),
        sa.Column("gaze_drift", sa.Float, default=0.0),
        sa.Column("fatigue_score", sa.Float, nullable=False),
        sa.Column("drowsiness_level", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, default=0.9),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_fatigue_metrics_user_time", "fatigue_metrics", ["user_id", "timestamp"])

    op.create_table("voice_stress_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pitch_variance", sa.Float, default=0.0),
        sa.Column("speech_energy", sa.Float, default=0.0),
        sa.Column("pause_duration", sa.Float, default=0.0),
        sa.Column("stress_score", sa.Float, nullable=False),
        sa.Column("emotion_state", sa.Text, nullable=False),
        sa.Column("mfcc_features", postgresql.JSONB, default=[]),
        sa.Column("confidence", sa.Float, default=0.75),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_voice_stress_user_time", "voice_stress_metrics", ["user_id", "timestamp"])

    op.create_table("behavioral_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("typing_speed", sa.Float, default=0.0),
        sa.Column("typing_rhythm_variance", sa.Float, default=0.0),
        sa.Column("error_rate", sa.Float, default=0.0),
        sa.Column("mouse_movement_entropy", sa.Float, default=0.0),
        sa.Column("mouse_click_rate", sa.Float, default=0.0),
        sa.Column("app_switch_frequency", sa.Float, default=0.0),
        sa.Column("focus_session_duration", sa.Float, default=0.0),
        sa.Column("idle_time", sa.Float, default=0.0),
        sa.Column("behavior_score", sa.Float, nullable=False),
        sa.Column("anomaly_score", sa.Float, default=0.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_behavioral_user_time", "behavioral_metrics", ["user_id", "timestamp"])

    op.create_table("productivity_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("productivity_score", sa.Float, nullable=False),
        sa.Column("burnout_probability", sa.Float, nullable=False),
        sa.Column("cognitive_load", sa.Float, default=0.0),
        sa.Column("focus_window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("focus_window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recommended_break_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("predicted_crash_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float, default=0.8),
        sa.Column("feature_importance", postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_productivity_user_time", "productivity_predictions", ["user_id", "timestamp"])

    op.create_table("recommendations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("priority", sa.Text, nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("action_label", sa.String(100), nullable=True),
        sa.Column("action_url", sa.String(500), nullable=True),
        sa.Column("duration_minutes", sa.Integer, nullable=True),
        sa.Column("accepted", sa.Boolean, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB, default={}),
        sa.Column("rl_state_vector", postgresql.JSONB, default=[]),
        sa.Column("rl_action_id", sa.Integer, nullable=True),
        sa.Column("reward", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_recommendations_user_time", "recommendations", ["user_id", "timestamp"])

    op.create_table("behavioral_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("work_sessions.id", ondelete="CASCADE")),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("qdrant_point_id", sa.String(100), nullable=False, unique=True),
        sa.Column("embedding_dim", sa.Integer, default=256),
        sa.Column("metadata", postgresql.JSONB, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )


def downgrade() -> None:
    for table in ["behavioral_embeddings","recommendations","productivity_predictions",
                  "behavioral_metrics","voice_stress_metrics","fatigue_metrics",
                  "work_sessions","refresh_tokens","users"]:
        op.drop_table(table)
    for t in ["recommendation_priority","recommendation_type","emotion_state","drowsiness_level"]:
        op.execute(f"DROP TYPE IF EXISTS {t}")
