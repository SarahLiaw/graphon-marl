"""
GMFS Main Runner.

Usage:
  python main.py --config config.json
"""

import argparse
import json
import os
import time
import numpy as np

import matplotlib.pyplot as plt
from math import comb

from gmfs_core import (
    DistanceDecayGraphon, RadialGraphon, GMFS_Q_Function, offline_learning,
    online_execution, build_grid_positions
)
from gmfs_envs import (
    FormationState, FormationAction, FormationTransition, FormationReward,
    TrafficState, TrafficAction, TrafficTransition, TrafficReward,
    RoboticsState, RoboticsAction, RoboticsTransition, RoboticsReward
)

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_env_components(config):
    env_name = config['environment']
    
    if env_name == 'formation':
        states = [FormationState(i) for i in [-1, 0, 1]]
        actions = [FormationAction(i) for i in [-1, 0, 1]]
        trans = FormationTransition()
        rew = FormationReward()
        return states, actions, trans, rew
        
    elif env_name == 'traffic':
        states = [TrafficState(i) for i in range(1, 6)]
        actions = [TrafficAction(i) for i in range(3)]
        trans = TrafficTransition()
        r_type = config.get('traffic_reward_type', 'positive')
        rew = TrafficReward(reward_type=r_type)
        return states, actions, trans, rew
    
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--kappa', type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print(f"Loaded config for: {cfg['environment']}")
    
    base_output_dir = cfg.get('output_dir', 'results')
    os.makedirs(base_output_dir, exist_ok=True)
    
    env_name = cfg['environment']
    k_values = cfg['k_values']
    if args.kappa is not None:
        k_values = [args.kappa]
    n_agents = cfg['n_agents']
    num_runs = cfg.get('num_eval_runs', 100)
    results = {}
    
    if env_name == 'robotics':
        states = [RoboticsState(i) for i in range(3)]
        actions = [RoboticsAction(i) for i in range(3)]
        trans = RoboticsTransition()
        rew = RoboticsReward(L=cfg.get('L', 2.0))
        
        grid_size = cfg.get('grid_size', 5)
        positions = build_grid_positions(grid_size)
        radius = cfg.get('radius', 0.3)
        graphon = RadialGraphon(radius=radius)
        
        # Value Iteration parameters
        training_steps = cfg.get('training_episodes', 100) # Steps of T_hat
        gamma = cfg.get('gamma', 0.95)
        mc_samples = cfg.get('mc_samples', 50)
        debug = bool(cfg.get('debug', False))
        debug_max_runs = int(cfg.get('debug_max_runs', 1))
        focal_idx = cfg.get('debug_focal_idx', n_agents // 2)
        
        for k in k_values:
            output_dir = base_output_dir
            if args.kappa is not None:
                output_dir = os.path.join(base_output_dir, f'kappa_{k}')
                os.makedirs(output_dir, exist_ok=True)
            print(f"\nRunning Value Iteration for k={k}...")
            
            # 1. Initialize Q-Function (Operator T_hat)
            q_func = GMFS_Q_Function(k=k, state_space=states, action_space=actions,
                    transition_func=trans, reward_func=rew, gamma=gamma, mc_samples=mc_samples)
            
            # 2. Offline Value Iteration (T_hat^T Q)
            start_time = time.time()
            q_func, deltas = offline_learning(q_func, steps=training_steps)
            training_time = time.time() - start_time
            
            # 3. Online Execution (Evaluation)
            returns = []
            diagnostics = []
            for run in range(num_runs):
                ret = online_execution(
                    n_agents=n_agents, 
                    horizon=cfg.get('eval_horizon', 100),
                    k=k, 
                    graphon=graphon, 
                    q_func=q_func,
                    state_space=states, 
                    action_space=actions,
                    transition_func=trans, 
                    seed=cfg['seed'] + run
                )
                returns.append(ret)
                if (run+1) % 5 == 0:
                    print(f"  Eval Run {run+1}/{num_runs}: {ret:.4f}", flush=True)
            
            results[k] = {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'stderr': float(np.std(returns)/np.sqrt(num_runs)),
                'returns': [float(r) for r in returns],
                'training_deltas': [float(d) for d in deltas],
                'training_time': float(training_time),
                'q_table_size': int(q_func.n_states * q_func.n_actions * q_func.n_z),
                'n_z': int(q_func.n_z)
            }
            if debug and diagnostics:
                results[k]['diagnostics'] = diagnostics
                # Save quick plots for the first diagnostic run
                diag = diagnostics[0]
                steps = list(range(len(diag["avg_hat_g2"])))
                plt.figure(figsize=(10, 6))
                plt.plot(steps, diag["avg_hat_g2"], label='avg_hat_g2')
                plt.plot(steps, diag["avg_true_g2"], label='avg_true_g2')
                plt.xlabel('t')
                plt.ylabel('g(s=2)')
                plt.title(f'k={k} mean-field g(s=2)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'g2_trace_k{k}.png'))
                
                action_fracs = np.array(diag["action_fracs"])
                plt.figure(figsize=(10, 6))
                plt.plot(steps, action_fracs[:, 0], label='a=0 (Idle)')
                plt.plot(steps, action_fracs[:, 1], label='a=1 (Transit)')
                plt.plot(steps, action_fracs[:, 2], label='a=2 (Work)')
                plt.xlabel('t')
                plt.ylabel('action fraction')
                plt.title(f'k={k} action fractions')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'action_fracs_k{k}.png'))
            print(f" Result k={k}: {results[k]['mean']:.4f} +/- {results[k]['stderr']:.4f}")
        
        # Save graphon data for reuse
        np.savez(os.path.join(base_output_dir, 'graphon_data.npz'), positions=positions, radius=radius, grid_size=grid_size)
    else:
        # Setup for existing environments
        states, actions, trans, rew = get_env_components(cfg)
        graphon = DistanceDecayGraphon(beta=cfg.get('beta', 2.0))
        
        for k in k_values:
            output_dir = base_output_dir
            print(f"\nRunning k={k}...")
            
            # Adaptive T calculation if not specified
            if 'training_steps' in cfg:
                T = cfg['training_steps']
            else:
                # Heuristic: linear scaling with sqrt of bins
                z_bins = comb(k + len(states)*len(actions) - 1, k)
                T = int(100 + 5 * np.sqrt(z_bins))
                T = min(T, 1000) # Cap
                
            print(f" Training steps: {T}")
            
            # 1. Offline Learning
            q_func = GMFS_Q_Function(
                k=k, state_space=states, action_space=actions,
                transition_func=trans, reward_func=rew,
                gamma=cfg.get('gamma', 0.9),
                mc_samples=cfg.get('mc_samples', 20)
            )
            
            q_func, deltas = offline_learning(q_func, steps=T)
            
            # 2. Online Execution
            returns = []
            for i in range(num_runs):
                ret = online_execution(
                    n_agents=n_agents, horizon=cfg.get('horizon', 100),
                    k=k, graphon=graphon, q_func=q_func,
                    state_space=states, action_space=actions,
                    transition_func=trans, seed=i*100 + k
                )
                returns.append(ret)
                if (run+1) % 5 == 0:
                    print(f"  Eval Run {run+1}/{num_runs}: {ret:.4f}", flush=True)
                
            results[k] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'stderr': np.std(returns)/np.sqrt(num_runs)
            }
            print(f" Result k={k}: {results[k]['mean']:.4f} +/- {results[k]['stderr']:.4f}")
    
    # Plotting (only when sweeping multiple k values)
    if args.kappa is None:
        ks = sorted(results.keys())
        means = [results[k]['mean'] for k in ks]
        errs = [results[k]['stderr'] for k in ks]
        
        plt.figure(figsize=(10,6))
        plt.errorbar(ks, means, yerr=errs, fmt='o-', capsize=5)
        plt.xlabel('k (Subsample size)')
        plt.ylabel('Discounted Return')
        plt.title(f"GMFS Performance: {cfg['environment']}")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(base_output_dir, 'plot.png'))
        print(f"\nSaved plot to {base_output_dir}/plot.png")
    
    # Save raw data (compat + extended)
    clean_res = {k: {k2: float(v2) if isinstance(v2, (int, float, np.floating)) else v2
                     for k2, v2 in v.items()}
                 for k, v in results.items()}
    if args.kappa is None:
        out_json_dir = base_output_dir
        clean_payload = clean_res
    else:
        out_json_dir = os.path.join(base_output_dir, f'kappa_{k_values[0]}')
        os.makedirs(out_json_dir, exist_ok=True)
        clean_payload = {k_values[0]: clean_res[k_values[0]]}
    
    with open(os.path.join(out_json_dir, 'data.json'), 'w') as f:
        json.dump(clean_payload, f, indent=2)
    with open(os.path.join(out_json_dir, 'data_full.json'), 'w') as f:
        payload = {
            'environment': env_name,
            'config': cfg,
            'results': clean_payload
        }
        json.dump(payload, f, indent=2)

if __name__ == '__main__':
    main()
