"""
Pre-train the RL agent on simulated user interactions.
Run: python simulate.py --steps 50000
"""

import argparse
import numpy as np
from agent import ProductivityRLAgent, ACTION_NAMES


def simulate_user_response(state: np.ndarray, action_name: str) -> float:
    """Simulate whether a simulated user would accept this recommendation."""
    fatigue = state[0]
    stress = state[1]
    productivity = state[2]
    burnout = state[4]

    # Fatigue is high → user is likely to accept break/eye_rest
    if fatigue > 0.7 and action_name in ("take_break", "eye_rest", "stretch"):
        return np.random.choice([0.8, 0.5], p=[0.8, 0.2])
    # Productivity is high → user rejects break suggestions
    if productivity > 0.8 and action_name == "take_break":
        return np.random.choice([-0.3, 0.5], p=[0.7, 0.3])
    # Default acceptance probability
    return np.random.choice([0.5, -0.3], p=[0.55, 0.45])


def run_simulation(steps: int):
    agent = ProductivityRLAgent()
    rng = np.random.default_rng(42)

    print(f"Simulating {steps:,} interactions...")

    for step in range(steps):
        # Random state vector (12-dim)
        state = rng.uniform(0, 1, size=12).astype(np.float32)

        # Agent picks action
        action = agent.select_action(state, user_id="simulation")

        # Simulate next state (small improvement after accepted action)
        next_state = state.copy()
        next_state[0] = max(0, state[0] - rng.uniform(0, 0.1))  # fatigue decreases

        # Simulate reward
        reward = simulate_user_response(state, ACTION_NAMES[action])

        # Store and train
        agent.store_transition(state, action, reward, next_state, done=False)
        loss = agent.train_step()

        if (step + 1) % 5000 == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"  Step {step+1:,} | eps={agent.epsilon:.3f} | Loss={loss_str}")

    agent.save("../agent.zip")
    print(f"\nPre-training complete! Saved to ml-models/rl-agent/agent.zip")
    print(f"Final epsilon: {agent.epsilon:.3f} (lower = more learned behavior)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    run_simulation(parser.parse_args().steps)