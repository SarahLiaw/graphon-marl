"""
GMFS Core Implementation.

Contains:
1. Base Classes (State, Action, Agent, Graphon)
2. GMFS Algorithm 1 (Offline Learning) - Optimized for High K
3. GMFS Algorithm 2 (Online Execution)
"""

import numpy as np
from itertools import combinations_with_replacement
from typing import List, Tuple, Callable, Dict, Any, Optional

class State:
    """Base class for environment states."""
    def __hash__(self): raise NotImplementedError
    def __eq__(self, other): raise NotImplementedError

class Action:
    """Base class for environment actions."""
    def __hash__(self): raise NotImplementedError
    def __eq__(self, other): raise NotImplementedError

class TransitionFunction:
    """P(s' | s, a, g)."""
    def sample_next_state(self, local_state: State, local_action: Action, 
                          neighborhood_dist: List[float],
                          rng: Optional[np.random.RandomState] = None) -> State:
        raise NotImplementedError

class RewardFunction:
    """r(s, a, g)."""
    def compute_reward(self, local_state: State, local_action: Action, 
                       neighborhood_dist: List[float]) -> float:
        raise NotImplementedError

class Agent:
    """Agent in the system."""
    def __init__(self, idx: int, position: float, state_space: List[State], 
                 action_space: List[Action]):
        self.idx = idx
        self.position = position  # Latent position alpha_i in [0, 1]
        self.state = None
        
    def reset(self, init_state: State):
        self.state = init_state
        
    def get_state(self) -> State:
        return self.state
    
    def set_state(self, new_state: State):
        self.state = new_state

class Graphon:
    """Graphon function W(x, y)."""
    def eval(self, x: float, y: float) -> float:
        raise NotImplementedError

    def get_normalized_weights(self, n_agents: int, positions: List[Any] = None) -> np.ndarray:
        """Get row-normalized weight matrix W_bar."""
        if positions is None:
            positions = [(i+1)/(n_agents) for i in range(n_agents)]
        W = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    W[i, j] = self.eval(positions[i], positions[j])
        
        # Row normalize
        row_sums = W.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        return W / row_sums

class DistanceDecayGraphon(Graphon):
    """W(x, y) = exp(-beta * |x - y|)."""
    def __init__(self, beta: float = 2.0):
        self.beta = beta
    
    def eval(self, x: float, y: float) -> float:
        dist = min(abs(x - y), 1 - abs(x - y))  # Toroidal distance usually better
        return np.exp(-self.beta * dist)

class RadialGraphon(Graphon):
    """W(x, y) = 1{||x - y||_2 <= r} for 2D positions."""
    def __init__(self, radius: float = 0.3):
        self.radius = radius
    
    def eval(self, x, y) -> float:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        dist = np.linalg.norm(x_arr - y_arr)
        return 1.0 if dist <= self.radius else 0.0

def build_grid_positions(grid_size: Optional[int] = None,
                         n_agents: Optional[int] = None,
                         grid_rows: Optional[int] = None,
                         grid_cols: Optional[int] = None) -> np.ndarray:
    """Create a lattice of positions in [0,1]^2.

    By default this preserves the old square-grid behavior. When ``n_agents`` is
    provided, the lattice can be truncated to exactly that many agents, which lets
    us run non-square population sizes such as n=1000.
    """
    if grid_rows is None and grid_cols is None:
        if grid_size is not None:
            grid_rows = grid_size
            grid_cols = grid_size
        elif n_agents is not None:
            grid_rows = int(np.floor(np.sqrt(n_agents)))
            grid_rows = max(grid_rows, 1)
            grid_cols = int(np.ceil(n_agents / grid_rows))
        else:
            raise ValueError("Must provide grid_size or n_agents to build positions.")
    elif grid_rows is None or grid_cols is None:
        raise ValueError("grid_rows and grid_cols must be provided together.")

    x_coords = np.linspace(0.0, 1.0, grid_cols)
    y_coords = np.linspace(0.0, 1.0, grid_rows)
    positions = [(x, y) for y in y_coords for x in x_coords]
    if n_agents is not None:
        if n_agents > len(positions):
            raise ValueError(
                f"Requested n_agents={n_agents}, but grid only has {len(positions)} positions."
            )
        positions = positions[:n_agents]
    return np.asarray(positions, dtype=float)

def radial_graphon_weights(positions: np.ndarray, radius: float) -> np.ndarray:
    """Compute row-normalized weights from a radial graphon on 2D positions."""
    n_agents = positions.shape[0]
    W = np.zeros((n_agents, n_agents), dtype=float)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            if np.linalg.norm(positions[i] - positions[j]) <= radius:
                W[i, j] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    for i in range(n_agents):
        if row_sums[i, 0] == 0.0:
            W[i, :] = 1.0
            W[i, i] = 0.0
            row_sums[i, 0] = n_agents - 1
    return W / row_sums


def generate_distributions(n_samples: int, n_bins: int) -> List[Tuple[int, ...]]:
    """Generate all count vectors summing to n_samples."""
    distributions = []
    for combo in combinations_with_replacement(range(n_bins), n_samples):
        dist = [0] * n_bins
        for bin_idx in combo:
            dist[bin_idx] += 1
        distributions.append(tuple(dist))
    return sorted(list(set(distributions)))

def sample_neighbor_histogram(states_idx: np.ndarray, weights_row: np.ndarray,
                              kappa: int, n_states: int, rng: np.random.RandomState) -> Tuple[int, ...]:
    """Sample neighbors with replacement and return a histogram of their states."""
    neighbors = rng.choice(len(states_idx), size=kappa, p=weights_row, replace=True)
    counts = [0] * n_states
    for nid in neighbors:
        counts[states_idx[nid]] += 1
    return tuple(counts)

class GMFS_Q_Function:
    """
    Q(s, a, z) table and learning logic.
    Optimized: Uses State-Marginal Histogram (z) instead of S x A historgram for scalability.
    """
    def __init__(self, k: int, state_space: List[State], action_space: List[Action],
                 transition_func: TransitionFunction, reward_func: RewardFunction, 
                 gamma: float = 0.9, mc_samples: int = 10,
                 rng_seed: int = 0):
        self.k = k
        self.state_space = state_space
        self.action_space = action_space
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.gamma = gamma
        self.mc_samples = mc_samples
        self.rng_seed = int(rng_seed)
        self.rng = np.random.RandomState(self.rng_seed)
        
        self.n_states = len(state_space)
        self.n_actions = len(action_space)
        
        self.joint_bins = [(s,) for s in state_space] 
        self.z_distributions = generate_distributions(k, len(self.joint_bins))
        self.n_z = len(self.z_distributions)
        self.marginal_map = {i: z for i, z in enumerate(self.z_distributions)}
        self.z_to_idx = {z: i for i, z in enumerate(self.z_distributions)}
        
        self.Q = np.zeros((self.n_states, self.n_actions, self.n_z), dtype=np.float32)
        self.state_to_idx = {s: i for i, s in enumerate(state_space)}
        
        # Precompute rewards r(s, a, g_z)
        self.reward_matrix = np.zeros((self.n_states, self.n_actions, self.n_z), dtype=np.float32)
        for s_idx, s in enumerate(self.state_space):
            for a_idx, a in enumerate(self.action_space):
                for z_idx, g_tuple in self.marginal_map.items():
                    self.reward_matrix[s_idx, a_idx, z_idx] = \
                        self.reward_func.compute_reward(s, a, list(g_tuple))
        
        self.transition_cache = {}

    def update_step(self) -> float:
        Q_new = np.copy(self.Q)
        max_delta = 0.0

        V_values = np.max(self.Q, axis=1) # Shape: (n_states, n_z)
        
        for s_idx in range(self.n_states):
            s = self.state_space[s_idx]
            for z_idx in range(self.n_z):
                g_current = self.marginal_map[z_idx]
                z_counts = self.z_distributions[z_idx]
                
                for a_idx in range(self.n_actions):
                    a = self.action_space[a_idx]
                    
                    cache_key = (s_idx, a_idx, z_idx)
                    if cache_key not in self.transition_cache:
                        samples = []
                        for _ in range(self.mc_samples):
                            # 1. Evolve focal agent
                            s_next = self.transition_func.sample_next_state(
                                s, a, list(g_current), rng=self.rng
                            )
                            s_next_idx = self.state_to_idx[s_next]
                            
                            # 2. Evolve neighbor population
                            g_next_counts = [0] * self.n_states
                            for bin_i, count in enumerate(z_counts):
                                if count == 0: continue
                                s_n = self.joint_bins[bin_i][0]
                                

                                a_n_idx = np.argmax(self.Q[self.state_to_idx[s_n], :, z_idx])
                                a_n = self.action_space[a_n_idx]
                                
                                for _ in range(count):
                                    sn_next = self.transition_func.sample_next_state(
                                        s_n, a_n, list(g_current), rng=self.rng
                                    )
                                    g_next_counts[self.state_to_idx[sn_next]] += 1
                            

                            g_next_tuple = tuple(g_next_counts)
                            try:
                                z_next_idx = self.z_to_idx[g_next_tuple]
                                samples.append((s_next_idx, z_next_idx))
                            except KeyError:
                                samples.append((s_next_idx, z_idx))
                                
                        self.transition_cache[cache_key] = samples
                    
                    samples = self.transition_cache[cache_key]
                    future_val = np.mean([V_values[s_next, z_next] for s_next, z_next in samples])
                    
                    new_q = self.reward_matrix[s_idx, a_idx, z_idx] + self.gamma * future_val
                    delta = abs(new_q - self.Q[s_idx, a_idx, z_idx])
                    max_delta = max(max_delta, delta)
                    Q_new[s_idx, a_idx, z_idx] = new_q
                    
        self.Q = Q_new
        return max_delta

def offline_learning(q_func: GMFS_Q_Function, steps: int):
    """Run Algo 1."""
    deltas = []
    print(f"Training GMFS (k={q_func.k}) for {steps} steps...")
    for t in range(steps):
        d = q_func.update_step()
        deltas.append(d)
        if (t+1) % 10 == 0:
            print(f"  Step {t+1}/{steps}, Delta: {d:.5f}", flush=True)
    return q_func, deltas

def online_execution(n_agents: int, horizon: int, k: int, graphon: Graphon,
                     q_func: GMFS_Q_Function,
                     state_space: List[State], action_space: List[Action],
                     transition_func: TransitionFunction,
                     seed: int = 42,
                     return_diagnostics: bool = False,
                     focal_idx: int = None,
                     positions: np.ndarray = None,
                     noise_type: Optional[str] = None,
                     noise_sigma: float = 0.0,
                     sampling_mode: str = "graphon",
                     initial_state_mode: str = "uniform",
                     initial_state_params: Optional[Dict[str, Any]] = None):
    """Run Algo 2."""
    rng = np.random.RandomState(seed)
    # Init agents
    agents = [Agent(i, (i+1)/n_agents, state_space, action_space) for i in range(n_agents)]
    if initial_state_params is None:
        initial_state_params = {}

    if initial_state_mode == "uniform" or positions is None:
        for a in agents:
            a.reset(rng.choice(state_space))
    elif initial_state_mode == "x_split_work":
        x_coords = np.asarray(positions, dtype=float)[:, 0]
        split_quantile = float(initial_state_params.get("split_quantile", 0.5))
        split_value = float(np.quantile(x_coords, split_quantile))
        high_work_prob = float(initial_state_params.get("high_work_prob", 0.90))
        low_work_prob = float(initial_state_params.get("low_work_prob", 0.05))
        transit_prob = float(initial_state_params.get("transit_prob", 0.05))
        for i, a in enumerate(agents):
            work_prob = high_work_prob if x_coords[i] <= split_value else low_work_prob
            idle_prob = max(0.0, 1.0 - transit_prob - work_prob)
            probs = np.array([idle_prob, transit_prob, work_prob], dtype=float)
            probs = probs / probs.sum()
            a.reset(state_space[int(rng.choice(len(state_space), p=probs))])
    else:
        raise ValueError(f"Unknown initial_state_mode={initial_state_mode!r}.")

    state_to_idx = q_func.state_to_idx
    if positions is not None:
        if len(positions) != n_agents:
            raise ValueError(
                f"Expected {n_agents} positions, got {len(positions)}."
            )
        W = graphon.get_normalized_weights(n_agents, positions=positions)
    else:
        W = graphon.get_normalized_weights(n_agents)

    if noise_type is not None and noise_sigma > 0.0:
        if noise_type == 'gaussian':
            W = W + rng.normal(0.0, noise_sigma, W.shape)
        elif noise_type == 'subgaussian':
            # Bounded uniform noise; subgaussian with parameter sigma/sqrt(3).
            W = W + rng.uniform(-noise_sigma, noise_sigma, W.shape)
        W = np.clip(W, 0.0, None)
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
    if sampling_mode not in ("graphon", "uniform"):
        raise ValueError(f"Unknown sampling_mode={sampling_mode!r}. Expected 'graphon' or 'uniform'.")
    rewards = []
    avg_hat_g2 = []
    avg_true_g2 = []
    avg_tv = []
    action_fracs = []
    focal_neighbors = None
    focal_neighbors_over_time = []
    focal_hat_g = None
    agent_states_t0 = None
    agent_states_t_end = None
    
    for t in range(horizon):
        state_indices = np.fromiter(
            (state_to_idx[a.get_state()] for a in agents),
            dtype=np.int64,
            count=n_agents
        )
        one_hot_states = np.eye(len(state_space), dtype=float)[state_indices]
        true_g_all = W @ one_hot_states
        if sampling_mode == "graphon":
            sampling_g_all = true_g_all
        else:
            total_state_counts = np.bincount(state_indices, minlength=len(state_space)).astype(float)
            denom = max(n_agents - 1, 1)
            sampling_g_all = (total_state_counts[None, :] - one_hot_states) / float(denom)

        if return_diagnostics and t == 0:
            agent_states_t0 = state_indices.tolist()
        # 1. Action Selection
        actions = []
        action_idx = []
        hat_g2_step = []
        hat_g_probs = []
        if k > 0:
            p0 = np.clip(sampling_g_all[:, 0], 0.0, 1.0)
            g0 = rng.binomial(k, p0)
            remaining = k - g0
            denom = np.clip(1.0 - p0, 1e-12, None)
            p1_cond = np.divide(sampling_g_all[:, 1], denom, out=np.zeros_like(p0), where=denom > 0.0)
            p1_cond = np.clip(p1_cond, 0.0, 1.0)
            g1 = rng.binomial(remaining, p1_cond)
            g2 = remaining - g1
            sampled_counts = np.stack((g0, g1, g2), axis=1).astype(int)
        else:
            sampled_counts = np.zeros((n_agents, len(state_space)), dtype=int)

        for i in range(n_agents):
            g_hat = tuple(sampled_counts[i].tolist())

            if return_diagnostics:
                if i == focal_idx:
                    if k > 0:
                        if sampling_mode == "graphon":
                            neighbor_probs = W[i]
                        else:
                            neighbor_probs = np.full(n_agents, 1.0 / max(n_agents - 1, 1), dtype=float)
                            neighbor_probs[i] = 0.0
                        neighbors = rng.choice(n_agents, size=k, p=neighbor_probs, replace=True)
                        g_hat = tuple(np.bincount(state_indices[neighbors], minlength=len(state_space)).tolist())
                    else:
                        neighbors = np.asarray([], dtype=int)
                    if t == 0:
                        focal_neighbors = neighbors.tolist()
                        focal_hat_g = g_hat
                    focal_neighbors_over_time.append(neighbors.tolist())
                if k > 0:
                    hat_g_probs.append(np.array(g_hat, dtype=float) / float(k))
                else:
                    hat_g_probs.append(np.zeros(len(state_space), dtype=float))
            hat_g2_step.append(g_hat[2] / float(k) if k > 0 else 0.0)
            
            # Greedy Policy
            s_idx = state_indices[i]

            try:
                z_idx = q_func.z_to_idx[g_hat]
                # Greedy action
                a_idx = np.argmax(q_func.Q[s_idx, :, z_idx])
                best_a = action_space[a_idx]
            except KeyError:
                best_a = rng.choice(action_space)
                a_idx = action_space.index(best_a)

            actions.append(best_a)
            action_idx.append(a_idx)
            
        step_rew = 0.0
        next_states = []
        true_g2_step = []
        tv_step = []
        for i in range(n_agents):
            true_g = true_g_all[i].tolist()
            true_g2_step.append(true_g[2])
            if return_diagnostics:
                tv_step.append(0.5 * float(np.sum(np.abs(hat_g_probs[i] - true_g))))
            
            r = q_func.reward_func.compute_reward(agents[i].get_state(), actions[i], true_g)
            step_rew += r
            ns = transition_func.sample_next_state(
                agents[i].get_state(), actions[i], true_g, rng=rng
            )
            next_states.append(ns)
        
        rewards.append(step_rew / n_agents)
        avg_hat_g2.append(float(np.mean(hat_g2_step)))
        avg_true_g2.append(float(np.mean(true_g2_step)))
        if return_diagnostics:
            avg_tv.append(float(np.mean(tv_step)))
        counts = np.bincount(action_idx, minlength=len(action_space))
        action_fracs.append((counts / float(n_agents)).tolist())
        for i, ns in enumerate(next_states):
            agents[i].set_state(ns)
    if return_diagnostics:
        agent_states_t_end = [state_to_idx[a.get_state()] for a in agents]
            
    # Compute discounted return
    ret = 0
    for t, r in enumerate(rewards):
        ret += (q_func.gamma ** t) * r
    if return_diagnostics:
        diagnostics = {
            "avg_hat_g2": avg_hat_g2,
            "avg_true_g2": avg_true_g2,
            "avg_tv": avg_tv,
            "action_fracs": action_fracs,
            "rewards": rewards,
            "focal_idx": focal_idx,
            "focal_neighbors": focal_neighbors,
            "focal_neighbors_over_time": focal_neighbors_over_time,
            "focal_hat_g": focal_hat_g,
            "sampling_mode": sampling_mode,
            "agent_states_t0": agent_states_t0,
            "agent_states_t_end": agent_states_t_end
        }
        return ret, diagnostics
    return ret
