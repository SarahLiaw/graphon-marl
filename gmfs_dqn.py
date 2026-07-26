"""
GMFS-DQN: Graphon Mean-Field Subsampling with continuous states via Deep Q-Learning.

This module extends the tabular GMFS approach in `gmfs_core.py` to continuous
state spaces. The tabular approach represents the Q-function as a table
Q[s, a, z] and the neighborhood aggregate z as an exact histogram over a
finite state space, both of which require the state space to be finite. Here:

- States are real vectors (np.ndarray), not hashable discrete symbols.
- The neighborhood aggregate g_hat is a permutation-invariant embedding of the
  kappa sampled neighbor states, produced by a DeepSets encoder
  (phi -> sum-pool -> rho), instead of an exact count histogram.
- The Q-function is a neural network Q_theta(s, g_hat) -> R^{n_actions},
  trained with standard DQN (replay buffer + target network), instead of
  exact synchronous value iteration over every (s, a, z) triple.
- Actions remain discrete (as in the original robotics environment) so that
  vanilla DQN applies directly. Continuous actions would require an
  actor-critic method (e.g. DDPG/TD3) instead.

Contains:
1. ContinuousRoboticsTransition / ContinuousRoboticsReward: continuous-state
   analogue of the discrete robotics env in `gmfs_envs.py`.
2. DeepSetEncoder: permutation-invariant neighbor aggregator.
3. QNetwork: Q_theta(s, g_hat) -> Q-values per discrete action.
4. ReplayBuffer.
5. train_dqn: Algorithm-1 analogue (learns Q via sampled mean-field rollouts).
6. online_execution_dqn: Algorithm-2 analogue (greedy rollout + diagnostics).
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Continuous-state robotics environment
# ---------------------------------------------------------------------------
# State: scalar "task progress" s in [0, 1]. s=0 ~ idle, s=1 ~ actively
# working at a station. Action: discrete target {0: idle, 1: transit, 2:
# work}. The neighbor aggregate is the mean workload of kappa sampled
# neighbors (a continuous congestion proxy), optionally with higher-order
# stats folded in by the DeepSets encoder.

N_ACTIONS = 3
ACTION_TARGETS = {0: 0.0, 1: 0.5, 2: 1.0}


class ContinuousRoboticsTransition:
    def __init__(self, step_rate: float = 0.5, noise_std: float = 0.05, seed: Optional[int] = None):
        self.step_rate = step_rate
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def sample_next_state(self, s: float, a: int, neighbor_states: np.ndarray,
                           rng: Optional[np.random.RandomState] = None) -> float:
        rng = rng if rng is not None else self.rng
        congestion = float(np.mean(neighbor_states)) if len(neighbor_states) > 0 else 0.0
        target = ACTION_TARGETS[a]
        # Reaching "work" (a=2) is harder under congestion; other transitions easier.
        success_prob = max(0.1, 0.9 - 0.8 * congestion) if a == 2 else 0.9
        noise = rng.normal(0.0, self.noise_std)
        if rng.rand() < success_prob:
            s_next = s + self.step_rate * (target - s) + noise
        else:
            s_next = s + 0.1 * noise  # stall near current state
        return float(np.clip(s_next, 0.0, 1.0))


class ContinuousRoboticsReward:
    """Bistable coordination reward:
        r(s, a, congestion) = { +W * s     if congestion < 0.5
                              { -W * s     otherwise
                              - cost[a],   with cost[a=2] = work_cost, else 0.

    Working (s->1) is very good in an uncongested neighborhood and very bad
    in a congested one. Under global-uniform play there is no interior Nash,
    so agents specialize spatially: those in low-density neighborhoods (on
    the graphon's periphery under the radial kernel) work, those in
    high-density (interior) neighborhoods idle. This makes the local
    congestion signal *actually informative*, so higher kappa (better
    estimate of the local mean workload) should yield monotonically higher
    return.
    """

    def __init__(self, W: float = 20.0, work_cost: float = 5.0,
                 threshold: float = 0.5, **_unused):
        self.W = W
        self.work_cost = work_cost
        self.threshold = threshold

    def compute_reward(self, s: float, a: int, neighbor_states: np.ndarray) -> float:
        congestion = float(np.mean(neighbor_states)) if len(neighbor_states) > 0 else 0.0
        sign = 1.0 if congestion < self.threshold else -1.0
        utility = self.W * s * sign
        cost = self.work_cost if a == 2 else 0.0
        return utility - cost


# ---------------------------------------------------------------------------
# DeepSets neighbor encoder: permutation-invariant embedding of {s_j}_{j in N(i)}
# ---------------------------------------------------------------------------
class DeepSetEncoder(nn.Module):
    def __init__(self, state_dim: int = 1, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, neighbor_states: torch.Tensor) -> torch.Tensor:
        """neighbor_states: (batch, kappa, state_dim) -> (batch, embed_dim)."""
        phi = self.psi(neighbor_states)          # (batch, kappa, embed_dim)
        pooled = phi.mean(dim=1)                 # permutation-invariant, kappa-size-agnostic
        return self.rho(pooled)


# ---------------------------------------------------------------------------
# Q-network: Q_theta(s, g_hat) -> Q-values per discrete action
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 1, embed_dim: int = 16, hidden_dim: int = 64,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.encoder = DeepSetEncoder(state_dim=state_dim, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, s: torch.Tensor, neighbor_states: torch.Tensor) -> torch.Tensor:
        """s: (batch, state_dim), neighbor_states: (batch, kappa, state_dim)."""
        g_hat = self.encoder(neighbor_states)
        x = torch.cat([s, g_hat], dim=-1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
Transition = Tuple[float, np.ndarray, int, float, float, np.ndarray, bool]
# (s, neighbor_states, a, r, s_next, neighbor_states_next, done)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, g, a, r, s_next, g_next, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(g, dtype=np.float32),
                np.array(a, dtype=np.int64), np.array(r, dtype=np.float32),
                np.array(s_next, dtype=np.float32), np.array(g_next, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Mean-field simulator helpers (graphon-based neighbor sampling)
# ---------------------------------------------------------------------------
def sample_neighbors(i: int, agent_states: np.ndarray, weights_row: np.ndarray,
                      kappa: int, rng: np.random.RandomState) -> np.ndarray:
    n_agents = len(agent_states)
    idx = rng.choice(n_agents, size=kappa, p=weights_row, replace=True)
    return agent_states[idx]


# ---------------------------------------------------------------------------
# Training loop (Algorithm-1 analogue): sampled DQN over the mean-field system
# ---------------------------------------------------------------------------
def train_dqn(n_agents: int, kappa: int, weights: np.ndarray,
               transition_func: ContinuousRoboticsTransition,
               reward_func: ContinuousRoboticsReward,
               n_episodes: int = 200, horizon: int = 50, gamma: float = 0.95,
               lr: float = 1e-3, batch_size: int = 128, buffer_capacity: int = 50000,
               eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_episodes: int = 150,
               target_sync_every: int = 10, seed: int = 0,
               device: str = "cpu") -> QNetwork:
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    def epsilon(ep):
        frac = min(1.0, ep / max(1, eps_decay_episodes))
        return eps_start + frac * (eps_end - eps_start)

    def select_actions(states: np.ndarray, neighbor_batch: np.ndarray, eps: float) -> np.ndarray:
        n = len(states)
        if rng.rand() < eps:
            return rng.randint(0, N_ACTIONS, size=n)
        with torch.no_grad():
            s_t = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(-1)
            g_t = torch.tensor(neighbor_batch, dtype=torch.float32, device=device).unsqueeze(-1)
            q_vals = q_net(s_t, g_t)
            return q_vals.argmax(dim=-1).cpu().numpy()

    print(f"Training GMFS-DQN (kappa={kappa}) for {n_episodes} episodes...")
    for ep in range(n_episodes):
        agent_states = rng.uniform(0.0, 1.0, size=n_agents)
        eps = epsilon(ep)
        ep_reward = 0.0

        for t in range(horizon):
            neighbor_batch = np.stack([
                sample_neighbors(i, agent_states, weights[i], kappa, rng) for i in range(n_agents)
            ])
            actions = select_actions(agent_states, neighbor_batch, eps)

            next_states = np.zeros(n_agents, dtype=np.float32)
            rewards = np.zeros(n_agents, dtype=np.float32)
            for i in range(n_agents):
                r = reward_func.compute_reward(agent_states[i], int(actions[i]), neighbor_batch[i])
                s_next = transition_func.sample_next_state(agent_states[i], int(actions[i]), neighbor_batch[i], rng=rng)
                rewards[i] = r
                next_states[i] = s_next
            ep_reward += float(np.mean(rewards))

            next_neighbor_batch = np.stack([
                sample_neighbors(i, next_states, weights[i], kappa, rng) for i in range(n_agents)
            ])
            done = (t == horizon - 1)
            for i in range(n_agents):
                buffer.push(agent_states[i], neighbor_batch[i], int(actions[i]), rewards[i],
                            next_states[i], next_neighbor_batch[i], done)

            agent_states = next_states

            if len(buffer) >= batch_size:
                s, g, a, r, s_next, g_next, d = buffer.sample(batch_size)
                s_t = torch.tensor(s, device=device).unsqueeze(-1)
                g_t = torch.tensor(g, device=device).unsqueeze(-1)
                a_t = torch.tensor(a, device=device)
                r_t = torch.tensor(r, device=device)
                s_next_t = torch.tensor(s_next, device=device).unsqueeze(-1)
                g_next_t = torch.tensor(g_next, device=device).unsqueeze(-1)
                d_t = torch.tensor(d, device=device)

                q_values = q_net(s_t, g_t).gather(1, a_t.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    next_q = target_net(s_next_t, g_next_t).max(dim=-1)[0]
                    target = r_t + gamma * (1.0 - d_t) * next_q
                loss = F.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}, avg_reward: {ep_reward/horizon:.4f}, eps: {eps:.3f}", flush=True)

    return q_net


# ---------------------------------------------------------------------------
# Online execution (Algorithm-2 analogue): greedy rollout with the learned Q-net
# ---------------------------------------------------------------------------
def online_execution_dqn(n_agents: int, kappa: int, horizon: int, weights: np.ndarray,
                          q_net: QNetwork, transition_func: ContinuousRoboticsTransition,
                          reward_func: ContinuousRoboticsReward, gamma: float = 0.95,
                          seed: int = 42, device: str = "cpu",
                          return_diagnostics: bool = False) -> Any:
    rng = np.random.RandomState(seed)
    agent_states = rng.uniform(0.0, 1.0, size=n_agents)
    rewards_over_time = []
    mean_state_over_time = []

    q_net.eval()
    with torch.no_grad():
        for t in range(horizon):
            neighbor_batch = np.stack([
                sample_neighbors(i, agent_states, weights[i], kappa, rng) for i in range(n_agents)
            ])
            s_t = torch.tensor(agent_states, dtype=torch.float32, device=device).unsqueeze(-1)
            g_t = torch.tensor(neighbor_batch, dtype=torch.float32, device=device).unsqueeze(-1)
            actions = q_net(s_t, g_t).argmax(dim=-1).cpu().numpy()

            next_states = np.zeros(n_agents, dtype=np.float32)
            step_rewards = np.zeros(n_agents, dtype=np.float32)
            for i in range(n_agents):
                step_rewards[i] = reward_func.compute_reward(agent_states[i], int(actions[i]), neighbor_batch[i])
                next_states[i] = transition_func.sample_next_state(
                    agent_states[i], int(actions[i]), neighbor_batch[i], rng=rng)

            rewards_over_time.append(float(np.mean(step_rewards)))
            mean_state_over_time.append(float(np.mean(agent_states)))
            agent_states = next_states

    ret = sum((gamma ** t) * r for t, r in enumerate(rewards_over_time))
    if return_diagnostics:
        return ret, {"rewards": rewards_over_time, "mean_state": mean_state_over_time}
    return ret
