"""
NeuroSight AI — Adaptive Recommendation Engine
Integrates the RL agent with business logic for personalized recommendations.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Recommendation,
    RecommendationType,
    RecommendationPriority,
    ProductivityPrediction,
    FatigueMetric,
    VoiceStressMetric,
    BehavioralMetric,
)
from app.services.ml_registry import ModelRegistry
from app.api.v1.endpoints.websocket import manager as ws_manager

logger = structlog.get_logger(__name__)

# -----------------------------------------------------------
# Recommendation Templates
# -----------------------------------------------------------
RECOMMENDATION_TEMPLATES: dict[str, dict] = {
    "take_break": {
        "titles": [
            "Time for a mindful break",
            "Step away from the screen",
            "Rest your mind for a moment",
        ],
        "messages": [
            "Your fatigue levels are rising. A short break now will restore your focus and prevent burnout.",
            "Your cognitive patterns suggest you need to recharge. Even 10 minutes will help significantly.",
            "Taking a break is not a sign of weakness — it's how elite performers sustain output.",
        ],
        "duration_minutes": 10,
        "action_label": "Start Timer",
    },
    "eye_rest": {
        "titles": ["Give your eyes a rest", "20-20-20 Rule", "Eye strain detected"],
        "messages": [
            "Look at something 20 feet away for 20 seconds. Your blink rate has dropped significantly.",
            "Your eye strain indicators are elevated. The 20-20-20 rule: every 20 min, look 20 feet away for 20 sec.",
            "Eye fatigue detected via blink analysis. Rest your eyes to maintain visual clarity.",
        ],
        "duration_minutes": 2,
        "action_label": "Start Eye Rest",
    },
    "hydrate": {
        "titles": ["Stay hydrated", "Water break time", "Hydration check"],
        "messages": [
            "Dehydration affects cognitive performance by up to 20%. When did you last drink water?",
            "Your session has been running for a while. Time to hydrate and refresh.",
            "A glass of water can improve focus and reduce fatigue. Take 2 minutes.",
        ],
        "duration_minutes": 3,
        "action_label": "Got it",
    },
    "deep_work": {
        "titles": ["Peak focus window detected", "Ideal time for deep work", "Cognitive peak ahead"],
        "messages": [
            "Your cognitive indicators are optimal. This is the perfect time for your most demanding tasks.",
            "AI analysis shows you're entering a focus peak. Block distractions and tackle your hardest work now.",
            "Your energy, stress, and fatigue patterns indicate a rare optimal state. Make it count.",
        ],
        "action_label": "Start Deep Work",
    },
    "stretch": {
        "titles": ["Stretch break", "Move your body", "Posture check + stretch"],
        "messages": [
            "Your posture indicators suggest you've been static too long. 5 minutes of stretching improves blood flow and focus.",
            "Standing and stretching for 5 minutes increases alertness by up to 25%.",
            "Movement is medicine for the mind. Quick stretch to reset your cognitive state.",
        ],
        "duration_minutes": 5,
        "action_label": "Start Stretch",
    },
    "meditation": {
        "titles": ["Mindfulness moment", "Stress recovery needed", "Brief meditation"],
        "messages": [
            "Your stress indicators are elevated. A 5-minute mindfulness exercise can reduce cortisol significantly.",
            "Brief meditation shown to reduce stress by 40% in 5 minutes. Your AI coach recommends it now.",
            "Your cognitive load is high. A short meditation will help you reset and refocus.",
        ],
        "duration_minutes": 5,
        "action_label": "Begin Meditation",
    },
    "light_task": {
        "titles": ["Switch to a lighter task", "Cognitive recovery mode", "Easier task recommended"],
        "messages": [
            "Your current fatigue makes complex work less efficient. Switch to lighter tasks to maintain momentum.",
            "Strategic task switching can preserve cognitive resources. Try a routine task for 20 minutes.",
            "Your performance indicators suggest a task change. Lighter work now → sharper focus later.",
        ],
        "action_label": "Noted",
    },
    "sleep": {
        "titles": ["Sleep quality matters", "Sleep debt detected", "Recovery recommendation"],
        "messages": [
            "Your fatigue patterns suggest accumulated sleep debt. Prioritize 7-9 hours tonight for cognitive recovery.",
            "No amount of breaks can replace quality sleep. Your patterns suggest it's time to wind down.",
            "Sleep is the ultimate cognitive reset. Your AI coach recommends ending your session soon.",
        ],
        "action_label": "Wind Down",
    },
    "exercise": {
        "titles": ["Movement for your brain", "Exercise boost recommended", "Physical reset"],
        "messages": [
            "Exercise is the most powerful cognitive enhancer available. Even a 20-minute walk boosts focus for hours.",
            "Your energy patterns suggest physical activity would significantly improve your afternoon performance.",
            "AI insight: users who exercise mid-day report 35% better afternoon productivity.",
        ],
        "duration_minutes": 20,
        "action_label": "Let's Go",
    },
    "posture_check": {
        "titles": ["Check your posture", "Posture alert", "Ergonomic reminder"],
        "messages": [
            "Good posture reduces fatigue and improves breathing. Take a moment to align your spine.",
            "Slouching reduces oxygen intake by 30%. Sit up straight and take a deep breath.",
            "Quick posture check: feet flat, screen at eye level, back supported. Your future self will thank you.",
        ],
        "action_label": "Fixed it",
    },
}


class RecommendationEngine:
    """
    RL-driven adaptive recommendation engine.
    Combines rule-based triggers with RL agent action selection.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def generate_recommendation(
        self,
        user_id: UUID,
        session_id: UUID,
        fatigue_metrics: FatigueMetric | None,
        stress_metrics: VoiceStressMetric | None,
        behavioral_metrics: BehavioralMetric | None,
        productivity_prediction: ProductivityPrediction | None,
    ) -> Recommendation | None:
        """
        Generate a recommendation using the RL agent.
        Returns None if no recommendation is appropriate right now.
        """

        # Build state vector
        state = self._build_state(
            fatigue_metrics,
            stress_metrics,
            behavioral_metrics,
            productivity_prediction,
        )

        # Get RL agent action
        rl_agent = ModelRegistry.get_rl_agent()
        action_idx, rec_type = self._select_action(
            state, rl_agent, user_id, fatigue_metrics, productivity_prediction
        )

        # Determine priority
        priority = self._determine_priority(
            rec_type, fatigue_metrics, stress_metrics, productivity_prediction
        )

        # Generate recommendation text
        recommendation = self._create_recommendation(
            user_id=user_id,
            session_id=session_id,
            rec_type=rec_type,
            priority=priority,
            state_vector=state.tolist(),
            action_id=action_idx,
        )

        # Persist to DB
        self.db.add(recommendation)
        await self.db.flush()

        # Push via WebSocket
        await self._push_to_user(user_id, recommendation)

        logger.info(
            "Recommendation generated",
            user_id=str(user_id),
            type=rec_type.value,
            priority=priority.value,
        )

        return recommendation

    def _build_state(
        self,
        fatigue: FatigueMetric | None,
        stress: VoiceStressMetric | None,
        behavioral: BehavioralMetric | None,
        prediction: ProductivityPrediction | None,
    ) -> "np.ndarray":
        import numpy as np

        now = datetime.now(timezone.utc)
        hour = now.hour
        day = now.weekday()

        return np.array([
            (fatigue.fatigue_score if fatigue else 0.0) / 100.0,
            (stress.stress_score if stress else 0.0) / 100.0,
            (prediction.productivity_score if prediction else 75.0) / 100.0,
            (behavioral.behavior_score if behavioral else 75.0) / 100.0,
            (prediction.burnout_probability if prediction else 0.1),
            0.5,  # session_duration (placeholder)
            0.5,  # time_since_last_break (placeholder)
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day / 7),
            math.cos(2 * math.pi * day / 7),
            0.0,  # last_recommendation_accepted
        ], dtype="float32")

    def _select_action(
        self,
        state,
        rl_agent,
        user_id: UUID,
        fatigue: FatigueMetric | None,
        prediction: ProductivityPrediction | None,
    ) -> tuple[int, RecommendationType]:
        """
        Select action via RL agent or rule-based fallback.
        """
        from ml_models.rl_agent.src.agent import ACTION_NAMES

        # Rule-based override for critical states
        if fatigue and fatigue.fatigue_score >= 85:
            return 0, RecommendationType.TAKE_BREAK
        if fatigue and fatigue.fatigue_score >= 70 and fatigue.drowsiness_level.value in ("severe", "critical"):
            return 8, RecommendationType.EYE_REST
        if prediction and prediction.productivity_score >= 80 and prediction.burnout_probability < 0.2:
            return 3, RecommendationType.DEEP_WORK

        # RL agent selection
        if rl_agent is not None:
            try:
                action_idx = rl_agent.select_action(state, str(user_id), greedy=True)
                rl_agent.record_action(str(user_id), action_idx)
                action_name = ACTION_NAMES[action_idx]
                return action_idx, RecommendationType(action_name)
            except Exception as e:
                logger.warning("RL agent action selection failed", error=str(e))

        # Default fallback
        return 0, RecommendationType.TAKE_BREAK

    def _determine_priority(
        self,
        rec_type: RecommendationType,
        fatigue: FatigueMetric | None,
        stress: VoiceStressMetric | None,
        prediction: ProductivityPrediction | None,
    ) -> RecommendationPriority:
        if fatigue and fatigue.fatigue_score >= 80:
            return RecommendationPriority.CRITICAL
        if prediction and prediction.burnout_probability >= 0.7:
            return RecommendationPriority.CRITICAL
        if (fatigue and fatigue.fatigue_score >= 60) or (stress and stress.stress_score >= 70):
            return RecommendationPriority.HIGH
        if rec_type == RecommendationType.DEEP_WORK:
            return RecommendationPriority.HIGH
        return RecommendationPriority.MEDIUM

    def _create_recommendation(
        self,
        user_id: UUID,
        session_id: UUID,
        rec_type: RecommendationType,
        priority: RecommendationPriority,
        state_vector: list,
        action_id: int,
    ) -> Recommendation:
        import random

        template = RECOMMENDATION_TEMPLATES.get(rec_type.value, {})
        titles = template.get("titles", ["Recommendation"])
        messages = template.get("messages", ["Take care of yourself."])

        idx = random.randint(0, len(titles) - 1)
        msg_idx = random.randint(0, len(messages) - 1)

        return Recommendation(
            id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            type=rec_type,
            priority=priority,
            title=titles[idx],
            message=messages[msg_idx],
            action_label=template.get("action_label"),
            duration_minutes=template.get("duration_minutes"),
            accepted=None,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            metadata={"template_idx": idx},
            rl_state_vector=state_vector,
            rl_action_id=action_id,
        )

    async def _push_to_user(self, user_id: UUID, rec: Recommendation) -> None:
        """Push recommendation via WebSocket to connected user."""
        try:
            await ws_manager.send_to_user(str(user_id), {
                "event": "recommendation:new",
                "payload": {
                    "id": str(rec.id),
                    "type": rec.type.value,
                    "priority": rec.priority.value,
                    "title": rec.title,
                    "message": rec.message,
                    "actionLabel": rec.action_label,
                    "durationMinutes": rec.duration_minutes,
                    "expiresAt": rec.expires_at.isoformat(),
                    "timestamp": rec.timestamp.isoformat(),
                    "accepted": None,
                    "metadata": rec.metadata,
                },
            })
        except Exception as e:
            logger.warning("Failed to push recommendation via WS", error=str(e))
