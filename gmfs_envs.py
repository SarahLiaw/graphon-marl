"""
GMFS Environments.

Contains:
1. Formation Control (Reference)
2. Traffic Control
3. Robotics Coordination (Graphon Mean-Field Subsampling)
"""

import numpy as np
from gmfs_core import State, Action, TransitionFunction, RewardFunction

class FormationState(State):
    """State: -1, 0, 1."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, FormationState) and self.value == o.value
    def __repr__(self): return str(self.value)

class FormationAction(Action):
    """Action: -1, 0, 1."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, FormationAction) and self.value == o.value
    def __repr__(self): return str(self.value)

class FormationTransition(TransitionFunction):
    def sample_next_state(self, s_obj, a_obj, g):
        s, a = s_obj.value, a_obj.value
        # Weighted neighborhood average state
        # g is [prob(-1), prob(0), prob(1)]
        current_g_mean = (-1)*g[0] + (0)*g[1] + (1)*g[2]
        
        # Dynamics: s' = clamp(s + a + drift(g_mean))
        # This is a simplification of the reference which makes drift proportional to difference
        drift = 0.5 * (current_g_mean - s)
        noise = np.random.normal(0, 0.1)
        
        val = s + a + drift + noise
        
        # Discretize back to {-1, 0, 1}
        if val < -0.5: return FormationState(-1)
        elif val > 0.5: return FormationState(1)
        else: return FormationState(0)

class FormationReward(RewardFunction):
    def compute_reward(self, s_obj, a_obj, g):
        s, a = s_obj.value, a_obj.value
        # g is probability dist over [-1, 0, 1]
        g_mean = (-1)*g[0] + (0)*g[1] + (1)*g[2]
        
        # Alignment: 1 - |s - g_mean|
        alignment = 1.0 - abs(s - g_mean)
        # Centering: 1 - |s|
        centering = 1.0 - abs(s)
        # Action cost
        cost = 0.1 * abs(a)
        
        return alignment + centering - cost


class TrafficState(State):
    """Congestion: 1 (Free) to 5 (Jam)."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, TrafficState) and self.value == o.value
    def __repr__(self): return f"S{self.value}"

class TrafficAction(Action):
    """0:Wait, 1:Slow, 2:Steady."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, TrafficAction) and self.value == o.value
    def __repr__(self): return f"A{self.value}"

class TrafficTransition(TransitionFunction):
    def sample_next_state(self, s_obj, a_obj, g):
        s, a = s_obj.value, a_obj.value
        # g is probabilities for states [1, 2, 3, 4, 5]
        # Avg congestion
        avg_levels = sum((i+1)*p for i, p in enumerate(g))
        
        # Dynamics
        # If action=Slow (1), reduce congestion chance
        # If neighbors congested (high avg), increase congestion chance
        prob_worsen = 0.1 + (0.1 * avg_levels) - (0.2 if a == 1 else 0.0)
        prob_improve = 0.3 - (0.05 * avg_levels) + (0.1 if a == 0 else 0.0)
        
        rand = np.random.rand()
        if rand < prob_worsen:
            return TrafficState(min(5, s + 1))
        elif rand < prob_worsen + prob_improve:
            return TrafficState(max(1, s - 1))
        else:
            return TrafficState(s)

class TrafficReward(RewardFunction):
    def __init__(self, reward_type='positive'):
        self.reward_type = reward_type
        self.max_cost = 40.0 # Normalization constant
        
    def compute_reward(self, s_obj, a_obj, g):
        s, a = s_obj.value, a_obj.value
        
        # Costs
        wait_cost = 2**(s-1) # Exponential with congestion
        fuel_cost = {0: 2, 1: 5, 2: 3}[a]
        # Density penalty
        avg_levels = sum((i+1)*p for i, p in enumerate(g))
        density_cost = 3 * avg_levels
        
        total_cost = wait_cost + fuel_cost + density_cost
        
        if self.reward_type == 'positive':
            # R = (Max - Cost) / Max
            return max(0.01, (self.max_cost - total_cost)/self.max_cost)
        else:
            return -total_cost

# ROBOTICS COORDINATION 

class RoboticsState(State):
    """State: 0 (Idle), 1 (Transit), 2 (Working)."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, RoboticsState) and self.value == o.value
    def __repr__(self): return f"S{self.value}"

class RoboticsAction(Action):
    """Action: 0, 1, 2 (intended next state)."""
    def __init__(self, value): self.value = value
    def __hash__(self): return hash(self.value)
    def __eq__(self, o): return isinstance(o, RoboticsAction) and self.value == o.value
    def __repr__(self): return f"A{self.value}"

class RoboticsTransition(TransitionFunction):
    def sample_next_state(self, s_obj, a_obj, z, rng=None):
        # Congestion-dependent success; g can be counts or probabilities.
        if rng is None:
            rng = np.random.RandomState()
        g = np.asarray(z, dtype=float)
        total = float(np.sum(g))
        if total > 0:
            g = g / total
        congestion = float(g[2])
        # More congestion reduces success to reach "Work"
        success_prob = max(0.1, 0.9 - 0.8 * congestion)
        if a_obj.value == 2:
            if rng.rand() < success_prob:
                return RoboticsState(2)
            return RoboticsState(1)
        # Other transitions are easier
        if rng.rand() < 0.9:
            return RoboticsState(a_obj.value)
        return RoboticsState(s_obj.value)

class RoboticsReward(RewardFunction):
    def __init__(self, L: float = 2.0):
        self.L = L
        self.base_rewards = {0: 10.0, 1: 5.0, 2: 20.0} 
        self.costs = {0: 0.0, 1: 0.0, 2: 5.0}
    
    def compute_reward(self, s_obj, a_obj, z):
        # z is a histogram (k0, k1, k2)
        kappa = float(sum(z))
        k2_fraction = float(z[2]) / kappa if kappa > 0 else 0
    
        penalty_factor = max(0.4, 1.0 - self.L * k2_fraction)
        utility = self.base_rewards[s_obj.value] * penalty_factor

        # compute costs 
        return utility - self.costs[a_obj.value]


    # def compute_reward(self, s_obj, a_obj, z):
    #     # z is a histogram (k0, k1, k2)
    #     kappa = float(sum(z))
    #     if kappa <= 0:
    #         return float(self.base_rewards[s_obj.value])
    #     k2_fraction = float(z[2]) / kappa
    #     utility = float(self.base_rewards[s_obj.value]) * np.exp(-self.L * k2_fraction)
    #     costs = {0:0.0, 1:0.05, 2:0.10}
    #     action_cost = costs[a_obj.value]
    #     return utility-action_cost
