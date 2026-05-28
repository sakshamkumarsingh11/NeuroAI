"""RL therapy planner — adapts exercise difficulty based on patient progress.

State  : [current_severity, recent_accuracy, trend, session_count, fatigue_proxy]
Action : 0=easier, 1=same, 2=harder
Reward : based on accuracy gain + successful completion at higher difficulty
"""
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import PPO_ACTOR_PATH, PPO_CRITIC_PATH, RL_CONFIG

STATE_DIM = RL_CONFIG["state_dim"]
ACTION_DIM = RL_CONFIG["action_dim"]
HIDDEN = RL_CONFIG["hidden_size"]

SEVERITY_TO_LEVEL = {"Severe": 0, "Moderate": 1, "Mild": 2, "Normal": 3}
LEVEL_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_LEVEL.items()}

ACTION_LABELS = ["easier", "same", "harder"]


# ---------------------------------------------------------------
# Actor / Critic networks
# ---------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# Therapy Planner class
# ---------------------------------------------------------------
class TherapyPlanner:
    """RL agent that picks difficulty for the next exercise."""

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=RL_CONFIG["learning_rate"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=RL_CONFIG["learning_rate"])
        self.gamma = RL_CONFIG["gamma"]
        self.epsilon = RL_CONFIG["epsilon"]
        self.load_if_exists()

    # -------- persistence --------
    def save(self):
        PPO_ACTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), PPO_ACTOR_PATH)
        torch.save(self.critic.state_dict(), PPO_CRITIC_PATH)

    def load_if_exists(self):
        if PPO_ACTOR_PATH.exists():
            self.actor.load_state_dict(torch.load(PPO_ACTOR_PATH, map_location="cpu"))
        if PPO_CRITIC_PATH.exists():
            self.critic.load_state_dict(torch.load(PPO_CRITIC_PATH, map_location="cpu"))

    # -------- state encoding --------
    @staticmethod
    def encode_state(severity: str,
                     recent_accuracy: float,
                     accuracy_trend: float,
                     session_count: int,
                     fatigue: float) -> torch.Tensor:
        """
        severity:        'Severe'/'Moderate'/'Mild'/'Normal'
        recent_accuracy: 0-100
        accuracy_trend:  -100 to +100 (recent - previous)
        session_count:   integer
        fatigue:         0-1 (estimated from pause ratio or session length)
        """
        state = np.array([
            SEVERITY_TO_LEVEL.get(severity, 1) / 3.0,
            recent_accuracy / 100.0,
            (accuracy_trend + 100) / 200.0,  # normalize to 0-1
            min(session_count, 50) / 50.0,
            min(max(fatigue, 0.0), 1.0),
        ], dtype=np.float32)
        return torch.from_numpy(state)

    # -------- action selection --------
    def select_action(self, state: torch.Tensor, explore: bool = True):
        """Return (action_idx, action_label, action_prob)."""
        with torch.no_grad():
            probs = self.actor(state)
        if explore:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs).item()
        return action_idx, ACTION_LABELS[action_idx], probs[action_idx].item()

    def recommend_difficulty(self, current_severity: str, action: str) -> str:
        """Map current severity + action → next severity/difficulty."""
        level = SEVERITY_TO_LEVEL[current_severity]
        if action == "easier":
            level = max(0, level - 1)
        elif action == "harder":
            level = min(3, level + 1)
        return LEVEL_TO_SEVERITY[level]

    # -------- reward shaping --------
    @staticmethod
    def compute_reward(accuracy_before: float,
                       accuracy_after: float,
                       action: str,
                       completed: bool = True) -> float:
        """
        Reward design:
          + improvement in accuracy
          + bonus for sustaining high accuracy at harder difficulty
          - penalty if accuracy drops sharply (too hard, too soon)
          - small penalty for unfinished exercises (frustration)
        """
        delta = accuracy_after - accuracy_before
        base = delta / 10.0  # normalize

        if action == "harder" and accuracy_after >= 75:
            base += 1.0  # successfully advanced
        elif action == "harder" and accuracy_after < 40:
            base -= 1.5  # too aggressive
        elif action == "easier" and accuracy_after >= 85:
            base -= 0.5  # too conservative
        elif action == "same" and 60 <= accuracy_after <= 80:
            base += 0.3  # good consolidation

        if not completed:
            base -= 0.5

        return float(base)

    # -------- PPO update --------
    def update(self, trajectories: List[dict]):
        """
        trajectories: list of dicts with keys
          state, action_idx, old_prob, reward, next_state
        """
        if not trajectories:
            return

        for t in trajectories:
            state = t["state"]
            action_idx = t["action_idx"]
            old_prob = torch.tensor(t["old_prob"], dtype=torch.float32)
            reward = torch.tensor([t["reward"]], dtype=torch.float32)
            next_state = t.get("next_state")

            # Value estimates
            value = self.critic(state)
            if next_state is not None:
                with torch.no_grad():
                    next_value = self.critic(next_state)
            else:
                next_value = torch.tensor([0.0])

            advantage = reward + self.gamma * next_value - value

            # PPO clipped objective
            new_probs = self.actor(state)
            new_prob = new_probs[action_idx]
            ratio = new_prob / (old_prob + 1e-8)
            clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            actor_loss = -torch.min(ratio * advantage.detach(),
                                    clipped * advantage.detach()).mean()
            critic_loss = (reward - value).pow(2).mean()

            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        self.save()


# ---------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------
_planner: TherapyPlanner | None = None


def get_planner() -> TherapyPlanner:
    global _planner
    if _planner is None:
        _planner = TherapyPlanner()
    return _planner
