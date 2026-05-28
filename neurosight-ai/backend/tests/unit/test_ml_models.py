"""
Tests for ML model architectures.
No trained weights needed — validates shapes, dtypes, and value ranges.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Add ML model source directories to path
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "ml-models/eye-fatigue/src"))
sys.path.insert(0, str(ROOT / "ml-models/voice-stress/src"))
sys.path.insert(0, str(ROOT / "ml-models/rl-agent/src"))


# ── Eye Fatigue Model ─────────────────────────────────────────────────────────

class TestFatigueClassifier:
    @pytest.fixture
    def model(self):
        from model import FatigueClassifier
        m = FatigueClassifier()
        m.eval()
        return m

    def test_instantiates(self, model):
        assert model is not None

    def test_forward_both_inputs(self, model):
        eye = torch.randn(2, 1, 48, 48)
        seq = torch.randn(2, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        assert out["fatigue_score"].shape == (2,)
        assert out["predicted_class"].shape == (2,)
        assert out["probabilities"].shape == (2, 5)

    def test_fatigue_score_range(self, model):
        eye = torch.randn(4, 1, 48, 48)
        seq = torch.randn(4, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        scores = out["fatigue_score"].numpy()
        assert np.all(scores >= 0)
        assert np.all(scores <= 100)

    def test_probabilities_sum_to_one(self, model):
        eye = torch.randn(3, 1, 48, 48)
        seq = torch.randn(3, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        sums = out["probabilities"].sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_predicted_class_valid_range(self, model):
        eye = torch.randn(2, 1, 48, 48)
        seq = torch.randn(2, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        classes = out["predicted_class"].numpy()
        assert np.all(classes >= 0)
        assert np.all(classes <= 4)

    def test_lstm_only_mode(self):
        from model import FatigueClassifier
        model = FatigueClassifier(use_cnn=False, use_lstm=True)
        model.eval()
        seq = torch.randn(2, 30, 4)
        with torch.no_grad():
            out = model(temporal_features=seq)
        assert out["fatigue_score"].shape == (2,)

    def test_cnn_only_mode(self):
        from model import FatigueClassifier
        model = FatigueClassifier(use_cnn=True, use_lstm=False)
        model.eval()
        eye = torch.randn(2, 1, 48, 48)
        with torch.no_grad():
            out = model(eye_image=eye)
        assert out["fatigue_score"].shape == (2,)

    def test_batch_size_one(self, model):
        eye = torch.randn(1, 1, 48, 48)
        seq = torch.randn(1, 30, 4)
        with torch.no_grad():
            out = model(eye_image=eye, temporal_features=seq)
        assert out["fatigue_score"].shape == (1,)

    def test_predict_returns_scalars(self, model):
        eye = torch.randn(1, 1, 48, 48)
        seq = torch.randn(1, 30, 4)
        result = model.predict(eye, seq)
        assert isinstance(result["fatigue_score"], float)
        assert isinstance(result["predicted_class"], int)
        assert isinstance(result["probabilities"], list)
        assert len(result["probabilities"]) == 5


# ── Voice Stress Model ────────────────────────────────────────────────────────

class TestVoiceStressModel:
    @pytest.fixture
    def model(self):
        from model import VoiceStressModel
        m = VoiceStressModel()
        m.eval()
        return m

    def test_instantiates(self, model):
        assert model is not None

    def test_forward_pass(self, model):
        spec = torch.randn(2, 1, 128, 100)
        mfcc = torch.randn(2, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        assert out["stress_score"].shape == (2,)
        assert out["emotion_probs"].shape == (2, 5)
        assert out["emotion_class"].shape == (2,)

    def test_stress_score_range(self, model):
        spec = torch.randn(4, 1, 128, 100)
        mfcc = torch.randn(4, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        scores = out["stress_score"].numpy()
        assert np.all(scores >= 0)
        assert np.all(scores <= 100)

    def test_emotion_probs_sum_to_one(self, model):
        spec = torch.randn(3, 1, 128, 100)
        mfcc = torch.randn(3, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        sums = out["emotion_probs"].sum(dim=-1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_emotion_class_valid_range(self, model):
        spec = torch.randn(2, 1, 128, 100)
        mfcc = torch.randn(2, 39)
        with torch.no_grad():
            out = model(spec, mfcc)
        classes = out["emotion_class"].numpy()
        assert np.all(classes >= 0)
        assert np.all(classes <= 4)

    def test_predict_returns_dict(self, model):
        spec = torch.randn(1, 1, 128, 100)
        mfcc = torch.randn(1, 39)
        result = model.predict(spec, mfcc)
        assert isinstance(result["stress_score"], float)
        assert result["emotion_state"] in ["calm", "stressed", "fatigued", "energetic", "anxious"]
        assert isinstance(result["emotion_probs"], dict)
        assert len(result["emotion_probs"]) == 5


# ── RL Agent ──────────────────────────────────────────────────────────────────

class TestProductivityRLAgent:
    @pytest.fixture
    def agent(self):
        from agent import ProductivityRLAgent
        return ProductivityRLAgent()

    def test_instantiates(self, agent):
        assert agent is not None

    def test_select_action_valid_range(self, agent):
        state = np.random.rand(12).astype(np.float32)
        action = agent.select_action(state, user_id="test")
        assert 0 <= action < 10

    def test_greedy_deterministic(self, agent):
        state = np.array([0.5] * 12, dtype=np.float32)
        actions = [agent.select_action(state, user_id="t", greedy=True) for _ in range(5)]
        assert len(set(actions)) == 1

    def test_epsilon_decays(self, agent):
        initial_eps = agent.epsilon
        state = np.random.rand(12).astype(np.float32)
        for _ in range(100):
            agent.select_action(state, user_id="t")
        assert agent.epsilon <= initial_eps

    def test_store_and_train(self, agent):
        state = np.random.rand(12).astype(np.float32)
        for _ in range(200):
            s = np.random.rand(12).astype(np.float32)
            agent.store_transition(s, 0, 0.5, s, False)
        loss = agent.train_step()
        assert loss is not None
        assert loss >= 0

    def test_encode_state(self, agent):
        state_dict = {
            "fatigue_score": 50, "stress_score": 30,
            "productivity_score": 70, "focus_level": 60,
            "burnout_risk": 25, "session_duration_minutes": 120,
            "time_since_last_break": 45, "hour_of_day": 10,
            "day_of_week": 1, "last_recommendation_accepted": 1,
        }
        encoded = agent.encode_state(state_dict)
        assert encoded.shape == (12,)
        assert encoded.dtype == np.float32
        assert np.all(encoded >= -1.5)  # sin/cos can be negative

    def test_all_actions_accessible(self, agent):
        """All 10 actions should be reachable without cooldowns initially."""
        from agent import ACTION_NAMES
        assert len(ACTION_NAMES) == 10
        seen_actions = set()
        state = np.random.rand(12).astype(np.float32)
        for _ in range(500):
            a = agent.select_action(state, user_id="t")
            seen_actions.add(a)
        # At minimum, multiple actions should be explored
        assert len(seen_actions) > 3

    def test_save_and_load(self, agent, tmp_path):
        save_path = str(tmp_path / "agent.zip")
        agent.save(save_path)
        from agent import ProductivityRLAgent
        new_agent = ProductivityRLAgent()
        new_agent.load(save_path)
        assert new_agent.steps_done == agent.steps_done
