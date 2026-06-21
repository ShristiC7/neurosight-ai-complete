"""
NeuroSight AI — Reinforcement Learning Recommendation Agent
Deep Q-Network (DQN) with experience replay for adaptive productivity recommendations.

State Space:
    - fatigue_score (0-100)
    - stress_score (0-100)
    - productivity_score (0-100)
    - focus_level (0-100)
    - burnout_risk (0-100)
    - session_duration_minutes (0-480)
    - time_since_last_break (0-120)
    - hour_of_day (0-23, cyclically encoded)
    - day_of_week (0-6, cyclically encoded)
    - last_recommendation_accepted (0/1)

Action Space (10 actions):
    0: take_break
    1: stretch
    2: hydrate
    3: deep_work
    4: light_task
    5: sleep
    6: exercise
    7: meditation
    8: eye_rest
    9: posture_check

Reward:
    +1.0 for accepted recommendation that leads to productivity improvement
    +0.5 for accepted recommendation
    -0.5 for rejected recommendation
    -1.0 for ignored critical alert
    +2.0 for preventing a burnout event
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


# -----------------------------------------------------------
# State / Action Constants
# -----------------------------------------------------------
STATE_DIM = 12   # Observation vector dimension
ACTION_DIM = 10  # Number of possible recommendation actions

ACTION_NAMES = [
    "take_break", "stretch", "hydrate", "deep_work",
    "light_task", "sleep", "exercise", "meditation",
    "eye_rest", "posture_check",
]

ACTION_COOLDOWNS = {
    "take_break": 30,   # Minutes before same action can repeat
    "stretch": 20,
    "hydrate": 15,
    "deep_work": 60,
    "light_task": 45,
    "sleep": 240,
    "exercise": 120,
    "meditation": 60,
    "eye_rest": 20,
    "posture_check": 30,
}


# -----------------------------------------------------------
# DQN Network
# -----------------------------------------------------------
class DQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    Separates state value estimation from action advantage estimation
    for more stable training with sparse rewards.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims[:-1]:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.feature_net = nn.Sequential(*layers)

        # Dueling streams
        last_hidden = hidden_dims[-2] if len(hidden_dims) > 1 else hidden_dims[0]
        stream_dim = hidden_dims[-1]

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(last_hidden, stream_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stream_dim, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_hidden, stream_dim),
            nn.ReLU(inplace=True),
            nn.Linear(stream_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Tensor:
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# -----------------------------------------------------------
# Experience Replay Buffer
# -----------------------------------------------------------
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    Samples important transitions more frequently.
    High-reward or high-loss transitions get higher priority.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)

    @property
    def beta(self) -> float:
        return min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames,
        )

    def push(self, transition: Transition, priority: float = 1.0) -> None:
        self.buffer.append(transition)
        self.priorities.append(max(priority, 1e-6) ** self.alpha)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        probs = np.array(self.priorities, dtype=np.float32)
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        transitions = [self.buffer[i] for i in indices]

        # Importance sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.frame += 1
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (abs(priority) + 1e-6) ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)


# -----------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------
class ProductivityRLAgent:
    """
    Deep Q-Network agent for adaptive productivity recommendations.

    Uses:
    - Dueling DQN for stable Q-value estimation
    - Double DQN to reduce overestimation bias
    - Prioritized Experience Replay for efficient learning
    - Epsilon-greedy exploration with decay
    - Target network for training stability
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.steps_done = 0

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Online + target networks (Double DQN)
        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.q_network.eval()
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000
        )

        self.replay_buffer = PrioritizedReplayBuffer(capacity=50_000)

        # Track action cooldowns per user
        self._cooldowns: dict[str, dict[str, float]] = {}

    def encode_state(self, state_dict: dict) -> np.ndarray:
        """
        Encode the state dictionary into a fixed-size vector.
        Uses cyclical encoding for time features.
        """
        hour = state_dict.get("hour_of_day", 12)
        day = state_dict.get("day_of_week", 0)

        return np.array([
            state_dict.get("fatigue_score", 0) / 100.0,
            state_dict.get("stress_score", 0) / 100.0,
            state_dict.get("productivity_score", 50) / 100.0,
            state_dict.get("focus_level", 50) / 100.0,
            state_dict.get("burnout_risk", 0) / 100.0,
            min(state_dict.get("session_duration_minutes", 0) / 480.0, 1.0),
            min(state_dict.get("time_since_last_break", 0) / 120.0, 1.0),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day / 7),
            math.cos(2 * math.pi * day / 7),
            float(state_dict.get("last_recommendation_accepted", 0)),
        ], dtype=np.float32)

    def select_action(
        self,
        state: np.ndarray,
        user_id: str,
        greedy: bool = False,
    ) -> int:
        """
        Epsilon-greedy action selection with cooldown masking.
        Returns action index.
        """
        # Decay epsilon only when not using greedy selection
        if not greedy:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # Get valid actions (not in cooldown)
        valid_actions = self._get_valid_actions(user_id)

        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()

        # Mask invalid actions
        mask = torch.full((self.action_dim,), float("-inf"), device=self.device)
        for a in valid_actions:
            mask[a] = 0.0

        q_values = q_values + mask
        return q_values.argmax().item()

    def _get_valid_actions(self, user_id: str) -> list[int]:
        """Returns action indices not currently in cooldown."""
        import time
        now = time.time()
        cooldowns = self._cooldowns.get(user_id, {})
        valid = []
        for i, name in enumerate(ACTION_NAMES):
            last_used = cooldowns.get(name, 0)
            cooldown_mins = ACTION_COOLDOWNS.get(name, 0)
            if (now - last_used) / 60 >= cooldown_mins:
                valid.append(i)
        return valid or list(range(self.action_dim))

    def record_action(self, user_id: str, action_idx: int) -> None:
        """Record action time for cooldown tracking."""
        import time
        if user_id not in self._cooldowns:
            self._cooldowns[user_id] = {}
        self._cooldowns[user_id][ACTION_NAMES[action_idx]] = time.time()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # New transitions start with max priority
        max_priority = max(self.replay_buffer.priorities, default=1.0)
        self.replay_buffer.push(
            Transition(state, action, reward, next_state, done),
            priority=max_priority,
        )

    def train_step(self) -> Optional[float]:
        """Single gradient update. Returns loss value."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Online network selects action
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Target network evaluates
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Weighted Huber loss (PER)
        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Sync target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save({
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps_done = checkpoint["steps_done"]
        self.epsilon = checkpoint["epsilon"]


import math  # noqa: E402 — needed by encode_state
