"""
NeuroSight AI — Celery RL Training Tasks
Handles reward recording and periodic agent training updates.
"""
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import structlog

from app.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(
    name="tasks.rl.record_transition",
    queue="rl_training",
    max_retries=3,
    default_retry_delay=30,
)
def record_rl_transition(recommendation_id: str, user_id: str, accepted: bool):
    """
    Records a state-action-reward transition in the RL replay buffer.
    Called 5 minutes after a recommendation is accepted/rejected
    so we can measure whether it actually improved the user's state.
    """
    import asyncio
    from app.db.session import AsyncSessionLocal
    from sqlalchemy import select, and_
    from app.models.models import Recommendation, FatigueMetric, VoiceStressMetric

    async def _run():
        async with AsyncSessionLocal() as db:
            rec_result = await db.execute(
                select(Recommendation).where(
                    Recommendation.id == uuid.UUID(recommendation_id)
                )
            )
            rec = rec_result.scalar_one_or_none()
            if not rec or not rec.rl_state_vector:
                return

            # Measure state improvement in the 5 minutes since recommendation
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(minutes=5)

            fat_result = await db.execute(
                select(FatigueMetric)
                .where(and_(
                    FatigueMetric.user_id == uuid.UUID(user_id),
                    FatigueMetric.timestamp >= window_start,
                ))
                .order_by(FatigueMetric.timestamp.desc())
                .limit(5)
            )
            recent_fatigue = fat_result.scalars().all()

            # Compute reward: accepted + measurable improvement = high reward
            base_reward = 0.5 if accepted else -0.3
            improvement_bonus = 0.0

            if accepted and recent_fatigue:
                prev_score = float(rec.rl_state_vector[0]) * 100
                recent_avg = sum(m.fatigue_score for m in recent_fatigue) / len(recent_fatigue)
                if recent_avg < prev_score - 5:
                    improvement_bonus = 0.5   # Fatigue actually decreased
                elif recent_avg > prev_score + 10:
                    improvement_bonus = -0.2  # Fatigue got worse despite acceptance

            final_reward = base_reward + improvement_bonus

            # Update recommendation record
            rec.reward = final_reward
            await db.commit()

            # Store in RL agent replay buffer
            try:
                from app.services.ml_registry import ModelRegistry
                if ModelRegistry._rl_agent is not None:
                    state = np.array(rec.rl_state_vector, dtype=np.float32)
                    next_state = state.copy()
                    if recent_fatigue:
                        next_state[0] = recent_fatigue[0].fatigue_score / 100.0
                    ModelRegistry._rl_agent.store_transition(
                        state=state,
                        action=rec.rl_action_id or 0,
                        reward=final_reward,
                        next_state=next_state,
                        done=False,
                    )
                    logger.info("RL transition recorded",
                                rec_id=recommendation_id,
                                reward=round(final_reward, 3))
            except Exception as e:
                logger.error("Failed to store RL transition", error=str(e))

    asyncio.run(_run())


@celery_app.task(
    name="tasks.rl.training_step",
    queue="rl_training",
)
def rl_training_step():
    """
    Runs a batch of gradient updates on the DQN agent.
    Scheduled every 30 minutes via Celery Beat.
    """
    try:
        from app.services.ml_registry import ModelRegistry
        agent = ModelRegistry._rl_agent
        if agent is None:
            logger.info("RL agent not loaded, skipping training step")
            return

        if len(agent.replay_buffer) < agent.batch_size:
            logger.info("Replay buffer too small for training",
                        size=len(agent.replay_buffer),
                        needed=agent.batch_size)
            return

        # Run 100 gradient steps
        losses = []
        for _ in range(100):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        if losses:
            avg_loss = sum(losses) / len(losses)
            logger.info("RL training step complete",
                        steps=len(losses),
                        avg_loss=round(avg_loss, 6),
                        epsilon=round(agent.epsilon, 4),
                        buffer_size=len(agent.replay_buffer))

            # Periodic checkpoint save every 10 training cycles
            from app.core.config import settings
            import os
            if agent.steps_done % (100 * 10) == 0:
                save_path = os.path.join(settings.MODEL_DIR, "rl-agent", "agent_checkpoint.zip")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                agent.save(save_path)
                logger.info("RL agent checkpoint saved", path=save_path)

    except Exception as e:
        logger.error("RL training step failed", error=str(e), exc_info=True)


@celery_app.task(
    name="tasks.rl.update_reward",
    queue="rl_training",
)
def update_reward(recommendation_id: str, user_id: str, accepted: bool):
    """Alias for record_transition used by WebSocket feedback handler."""
    record_rl_transition(recommendation_id, user_id, accepted)
