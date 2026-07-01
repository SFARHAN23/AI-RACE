"""
utils.py — Replay Buffer for off-policy RL (TD3 / SAC).

Stores transitions (s, a, s', r, done) in fixed-size numpy arrays and
returns random mini-batches as PyTorch tensors on the appropriate device.
"""

import numpy as np
import torch


class ReplayBuffer(object):
    """Fixed-size circular replay buffer with numpy backend and PyTorch sampling."""

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """Store a single transition."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions as PyTorch tensors."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


if __name__ == "__main__":
    print("=== ReplayBuffer Self-Test ===")

    STATE_DIM = 22
    ACTION_DIM = 2
    buf = ReplayBuffer(STATE_DIM, ACTION_DIM, max_size=1000)

    # Fill with dummy transitions
    for i in range(50):
        s = np.random.randn(STATE_DIM)
        a = np.random.randn(ACTION_DIM)
        s2 = np.random.randn(STATE_DIM)
        r = np.random.randn()
        done = float(i % 10 == 0)
        buf.add(s, a, s2, r, done)

    assert buf.size == 50, f"Expected size 50, got {buf.size}"
    assert buf.ptr == 50, f"Expected ptr 50, got {buf.ptr}"

    # Sample a batch
    batch = buf.sample(batch_size=16)
    states, actions, next_states, rewards, not_dones = batch

    assert states.shape == (16, STATE_DIM)
    assert actions.shape == (16, ACTION_DIM)
    assert next_states.shape == (16, STATE_DIM)
    assert rewards.shape == (16, 1)
    assert not_dones.shape == (16, 1)

    print(f"  Buffer size: {buf.size}")
    print(f"  Device: {buf.device}")
    print(f"  Sample states shape: {states.shape}")
    print(f"  Sample actions shape: {actions.shape}")
    print(f"  Sample rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")

    # Test wrap-around
    for i in range(1000):
        buf.add(np.zeros(STATE_DIM), np.zeros(ACTION_DIM), np.zeros(STATE_DIM), 0.0, 0.0)
    assert buf.size == 1000, f"Expected size 1000 after wrap, got {buf.size}"
    assert buf.ptr == 50, f"Expected ptr 50 after 1050 inserts into size-1000 buffer, got {buf.ptr}"

    print("  Wrap-around: OK")
    print("=== All ReplayBuffer tests passed! ===")
