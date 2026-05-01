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


def sanitize_payload(results):
    return {
        k: {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in payload.items()
        }
        for k, payload in results.items()
    }


def write_result_payload(out_dir, env_name, cfg, payload):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'data.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, 'data_full.json'), 'w') as f:
        json.dump({
            'environment': env_name,
            'config': cfg,
            'results': payload
        }, f, indent=2)

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
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print(f"Loaded config for: {cfg['environment']}")
    
    base_output_dir = args.output_dir or cfg.get('output_dir', 'results')
    os.makedirs(base_output_dir, exist_ok=True)

    env_name = cfg['environment']
    k_values = cfg['k_values']
    if args.kappa is not None:
        k_values = [args.kappa]
    n_agents = cfg['n_agents']
    num_runs = cfg.get('num_eval_runs', 100)
    n_training_seeds = int(cfg.get('n_training_seeds', 1))
    base_seed = int(cfg.get('seed', 0))
    training_seed = int(cfg.get('training_seed', base_seed))
    eval_seed = int(cfg.get('eval_seed', base_seed))
    noise_type = cfg.get('noise_type', None)
    noise_sigma = float(cfg.get('noise_sigma', 0.0))
    results = {}
    
    if env_name == 'robotics':
        states = [RoboticsState(i) for i in range(3)]
        actions = [RoboticsAction(i) for i in range(3)]
        trans = RoboticsTransition()
        rew = RoboticsReward(L=cfg.get('L', 2.0))
        
        grid_size = cfg.get('grid_size')
        grid_rows = cfg.get('grid_rows')
        grid_cols = cfg.get('grid_cols')
        positions = build_grid_positions(
            grid_size=grid_size,
            n_agents=n_agents,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
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
            print(f"\nRunning Value Iteration for k={k} ({n_training_seeds} training seed(s))...")

            all_returns = []
            all_deltas = None
            total_training_time = 0.0
            last_q_func = None
            all_diagnostics = []
            per_seed_returns = []

            for seed_run in range(n_training_seeds):
                # Use non-overlapping seed blocks per training seed
                seed_offset = seed_run * 100000
                t_seed = training_seed + seed_offset + k
                e_seed = eval_seed + seed_offset

                # 1. Initialize Q-Function (Operator T_hat)
                q_func = GMFS_Q_Function(
                    k=k,
                    state_space=states,
                    action_space=actions,
                    transition_func=trans,
                    reward_func=rew,
                    gamma=gamma,
                    mc_samples=mc_samples,
                    rng_seed=t_seed,
                )

                # 2. Offline Value Iteration (T_hat^T Q)
                start_time = time.time()
                q_func, deltas = offline_learning(q_func, steps=training_steps)
                training_time = time.time() - start_time
                total_training_time += training_time
                if all_deltas is None:
                    all_deltas = deltas
                last_q_func = q_func

                # 3. Online Execution (Evaluation)
                seed_returns = []
                diagnostics = []
                for run in range(num_runs):
                    capture_diagnostics = debug and run < debug_max_runs and seed_run == 0
                    outcome = online_execution(
                        n_agents=n_agents,
                        horizon=cfg.get('eval_horizon', 100),
                        k=k,
                        graphon=graphon,
                        q_func=q_func,
                        state_space=states,
                        action_space=actions,
                        transition_func=trans,
                        seed=e_seed + 1000 * k + run,
                        return_diagnostics=capture_diagnostics,
                        focal_idx=focal_idx,
                        positions=positions,
                        noise_type=noise_type,
                        noise_sigma=noise_sigma,
                    )
                    if capture_diagnostics:
                        ret, diag = outcome
                        diagnostics.append(diag)
                    else:
                        ret = outcome
                    seed_returns.append(ret)
                    all_returns.append(ret)
                    if (run + 1) % 5 == 0:
                        print(f"  Seed {seed_run} Eval {run+1}/{num_runs}: {ret:.4f}", flush=True)

                per_seed_returns.append([float(r) for r in seed_returns])
                if diagnostics:
                    all_diagnostics.extend(diagnostics)

            total_runs = len(all_returns)
            results[k] = {
                'mean': float(np.mean(all_returns)),
                'std': float(np.std(all_returns)),
                'stderr': float(np.std(all_returns) / np.sqrt(total_runs)),
                'returns': [float(r) for r in all_returns],
                'per_seed_returns': per_seed_returns,
                'n_training_seeds': n_training_seeds,
                'training_deltas': [float(d) for d in all_deltas],
                'training_time': float(total_training_time),
                'q_table_size': int(last_q_func.n_states * last_q_func.n_actions * last_q_func.n_z),
                'n_z': int(last_q_func.n_z),
            }
            if debug and all_diagnostics:
                results[k]['diagnostics'] = all_diagnostics
                diag = all_diagnostics[0]
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
            print(f" Result k={k} ({n_training_seeds} seeds x {num_runs} runs = {total_runs} total): "
                  f"{results[k]['mean']:.4f} +/- {results[k]['stderr']:.4f}")

            clean_single = sanitize_payload({k: results[k]})
            write_result_payload(
                os.path.join(base_output_dir, f'kappa_{k}'),
                env_name,
                cfg,
                clean_single,
            )

        # Save graphon data for reuse
        np.savez(
            os.path.join(base_output_dir, 'graphon_data.npz'),
            positions=positions,
            radius=radius,
            grid_size=-1 if grid_size is None else grid_size,
            grid_rows=-1 if grid_rows is None else grid_rows,
            grid_cols=-1 if grid_cols is None else grid_cols,
            n_agents=n_agents,
        )
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
                mc_samples=cfg.get('mc_samples', 20),
                rng_seed=training_seed + k,
            )
            
            q_func, deltas = offline_learning(q_func, steps=T)
            
            # 2. Online Execution
            returns = []
            for i in range(num_runs):
                ret = online_execution(
                    n_agents=n_agents, horizon=cfg.get('horizon', 100),
                    k=k, graphon=graphon, q_func=q_func,
                    state_space=states, action_space=actions,
                    transition_func=trans, seed=eval_seed + 1000 * k + i
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
    clean_res = sanitize_payload(results)
    if args.kappa is None:
        out_json_dir = base_output_dir
        clean_payload = clean_res
    else:
        out_json_dir = os.path.join(base_output_dir, f'kappa_{k_values[0]}')
        os.makedirs(out_json_dir, exist_ok=True)
        clean_payload = {k_values[0]: clean_res[k_values[0]]}

    write_result_payload(out_json_dir, env_name, cfg, clean_payload)

if __name__ == '__main__':
    main()
