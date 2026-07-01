"""
td3_lidar.py — TD3 (Twin Delayed DDPG) for LiDAR-based autonomous racing.

Adapted from Fujimoto et al., "Addressing Function Approximation Error
in Actor-Critic Methods" (2018).

Same architecture as the original TD3.py from the GT3 codebase but
standalone and configured for the LiDAR observation space
(STATE_DIM=23 by default: 3 car-state + 20 LiDAR beams).
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Networks ──────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Deterministic policy: state → [-max_action, max_action]²"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """Twin Q-networks (clipped double-Q)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


# ── TD3 Agent ─────────────────────────────────────────────────────────────────

class TD3:
    """Twin Delayed DDPG agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    # ── inference ─────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action from the deterministic policy (no noise)."""
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state_t).cpu().data.numpy().flatten()

    # ── training ──────────────────────────────────────────────────────────

    def train(self, replay_buffer, batch_size: int = 256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size
        )

        with torch.no_grad():
            # Target policy smoothing
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Clipped double-Q target
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Critic update
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=1.0
            )
            self.actor_optimizer.step()

            # Soft target updates
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    # ── save / load ───────────────────────────────────────────────────────

    def save(self, filename: str):
        """Save full checkpoint (actor + critic + optimizers)."""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(
            self.critic_optimizer.state_dict(), filename + "_critic_optimizer"
        )
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(
            self.actor_optimizer.state_dict(), filename + "_actor_optimizer"
        )

    def load(self, filename: str) -> bool:
        """Load checkpoint.  Returns True if critic was also loaded."""
        self.actor.load_state_dict(
            torch.load(filename + "_actor", weights_only=True)
        )
        self.actor_target = copy.deepcopy(self.actor)

        if os.path.exists(filename + "_actor_optimizer"):
            self.actor_optimizer.load_state_dict(
                torch.load(filename + "_actor_optimizer", weights_only=True)
            )

        if os.path.exists(filename + "_critic"):
            self.critic.load_state_dict(
                torch.load(filename + "_critic", weights_only=True)
            )
            self.critic_target = copy.deepcopy(self.critic)
            if os.path.exists(filename + "_critic_optimizer"):
                self.critic_optimizer.load_state_dict(
                    torch.load(
                        filename + "_critic_optimizer", weights_only=True
                    )
                )
            return True
        return False

    def load_cpu(self, filename: str) -> bool:
        """Load checkpoint onto CPU (for deployment on edge devices)."""
        cpu = torch.device("cpu")
        self.actor.load_state_dict(
            torch.load(filename + "_actor", map_location=cpu, weights_only=True)
        )
        self.actor_target = copy.deepcopy(self.actor)

        if os.path.exists(filename + "_actor_optimizer"):
            self.actor_optimizer.load_state_dict(
                torch.load(
                    filename + "_actor_optimizer",
                    map_location=cpu,
                    weights_only=True,
                )
            )

        if os.path.exists(filename + "_critic"):
            self.critic.load_state_dict(
                torch.load(
                    filename + "_critic", map_location=cpu, weights_only=True
                )
            )
            self.critic_target = copy.deepcopy(self.critic)
            if os.path.exists(filename + "_critic_optimizer"):
                self.critic_optimizer.load_state_dict(
                    torch.load(
                        filename + "_critic_optimizer",
                        map_location=cpu,
                        weights_only=True,
                    )
                )
            return True
        return False


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TD3 LiDAR Self-Test ===")

    STATE_DIM = 23
    ACTION_DIM = 2
    MAX_ACTION = 1.0

    agent = TD3(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
    )
    print(f"  Device      : {device}")
    print(f"  State dim   : {STATE_DIM}")
    print(f"  Action dim  : {ACTION_DIM}")

    # Test inference
    dummy_state = np.random.randn(STATE_DIM).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"  Action shape: {action.shape}")
    print(f"  Action value: {action}")
    assert action.shape == (ACTION_DIM,)
    assert np.all(np.abs(action) <= MAX_ACTION + 1e-6)

    # Test save/load round trip
    os.makedirs("_td3_test_tmp", exist_ok=True)
    agent.save("_td3_test_tmp/test_model")
    agent2 = TD3(STATE_DIM, ACTION_DIM, MAX_ACTION)
    loaded_critic = agent2.load("_td3_test_tmp/test_model")
    action2 = agent2.select_action(dummy_state)
    print(f"  Load critic : {loaded_critic}")
    print(f"  Action diff : {np.abs(action - action2).max():.8f}")
    assert np.allclose(action, action2, atol=1e-6)

    # Clean up
    import shutil
    shutil.rmtree("_td3_test_tmp")

    print("\n=== All TD3 tests passed! ===")
