"""Tests for the Reinforcement Learning Recommendation Agent."""
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add RL agent src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "ml-models/rl-agent/src"))

from agent import ProductivityRLAgent, Transition, ACTION_DIM, STATE_DIM

def test_agent_instantiation():
    agent = ProductivityRLAgent()
    assert agent is not None
    assert agent.state_dim == STATE_DIM
    assert agent.action_dim == ACTION_DIM

def test_state_encoding():
    agent = ProductivityRLAgent()
    state_dict = {
        "fatigue_score": 50,
        "stress_score": 25,
        "productivity_score": 80,
        "focus_level": 75,
        "burnout_risk": 10,
        "session_duration_minutes": 120,
        "time_since_last_break": 30,
        "hour_of_day": 14,
        "day_of_week": 2,
        "last_recommendation_accepted": 1
    }
    encoded = agent.encode_state(state_dict)
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (STATE_DIM,)
    assert encoded.dtype == np.float32
    assert 0.0 <= encoded[0] <= 1.0  # Fatigue normalized

def test_action_selection_greedy():
    agent = ProductivityRLAgent()
    state = np.random.rand(STATE_DIM).astype(np.float32)
    # Ensure deterministic output when greedy is True
    action1 = agent.select_action(state, user_id="user1", greedy=True)
    action2 = agent.select_action(state, user_id="user1", greedy=True)
    assert action1 == action2
    assert 0 <= action1 < ACTION_DIM

def test_experience_replay_and_training():
    agent = ProductivityRLAgent(batch_size=16)
    
    # Store some fake transitions
    for _ in range(32):
        s = np.random.rand(STATE_DIM).astype(np.float32)
        next_s = np.random.rand(STATE_DIM).astype(np.float32)
        a = np.random.randint(0, ACTION_DIM)
        r = np.random.rand()
        done = False
        agent.store_transition(s, a, r, next_s, done)
    
    # Should have 32 items
    assert len(agent.replay_buffer) == 32
    
    # Perform a train step
    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0

def test_cooldown_logic():
    agent = ProductivityRLAgent()
    state = np.random.rand(STATE_DIM).astype(np.float32)
    
    # Select action and record it
    action = agent.select_action(state, user_id="cooldown_user", greedy=True)
    agent.record_action("cooldown_user", action)
    
    # Get valid actions right after
    valid_actions = agent._get_valid_actions("cooldown_user")
    
    # Since cooldown is long for most actions, the recorded action should not be in valid_actions
    assert action not in valid_actions
