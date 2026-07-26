"""
GMFS-DQN Runner: continuous-state robotics coordination via Deep Q-Learning.

Usage:
  python main_dqn.py --config config_robotics_dqn.json
  python main_dqn.py --config config_robotics_dqn.json --kappa 8
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from gmfs_core import build_grid_positions, radial_graphon_weights
from gmfs_dqn import (
    ContinuousRoboticsTransition, ContinuousRoboticsReward,
    train_dqn, online_execution_dqn,
)


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_robotics_dqn.json')
    parser.add_argument('--kappa', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.get('output_dir', 'results_robotics_dqn')
    os.makedirs(output_dir, exist_ok=True)

    n_agents = cfg['n_agents']
    grid_size = cfg.get('grid_size', 5)
    radius = cfg.get('radius', 0.3)
    positions = build_grid_positions(grid_size)
    weights = radial_graphon_weights(positions, radius)

    k_values = cfg['k_values']
    if args.kappa is not None:
        k_values = [args.kappa]

    trans = ContinuousRoboticsTransition(
        step_rate=cfg.get('step_rate', 0.5), noise_std=cfg.get('noise_std', 0.05))
    rew = ContinuousRoboticsReward(
        W=cfg.get('W', 20.0), work_cost=cfg.get('work_cost', 5.0))

    device = "cuda" if (cfg.get('use_cuda', False) and torch.cuda.is_available()) else "cpu"
    gamma = cfg.get('gamma', 0.95)
    num_eval_runs = cfg.get('num_eval_runs', 10)

    results = {}
    for k in k_values:
        print(f"\n=== GMFS-DQN, kappa={k} ===")
        start = time.time()
        q_net = train_dqn(
            n_agents=n_agents, kappa=k, weights=weights,
            transition_func=trans, reward_func=rew,
            n_episodes=cfg.get('n_episodes', 200),
            horizon=cfg.get('train_horizon', 50),
            gamma=gamma, lr=cfg.get('lr', 1e-3),
            batch_size=cfg.get('batch_size', 128),
            eps_decay_episodes=cfg.get('eps_decay_episodes', 150),
            target_sync_every=cfg.get('target_sync_every', 10),
            seed=cfg.get('seed', 0), device=device,
        )
        train_time = time.time() - start

        returns = []
        for run in range(num_eval_runs):
            ret = online_execution_dqn(
                n_agents=n_agents, kappa=k, horizon=cfg.get('eval_horizon', 100),
                weights=weights, q_net=q_net, transition_func=trans, reward_func=rew,
                gamma=gamma, seed=cfg.get('seed', 0) + 1000 + run, device=device,
            )
            returns.append(ret)

        results[k] = {
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
            'stderr': float(np.std(returns) / np.sqrt(num_eval_runs)),
            'returns': [float(r) for r in returns],
            'train_time': float(train_time),
        }
        print(f" Result k={k}: {results[k]['mean']:.4f} +/- {results[k]['stderr']:.4f}")

        torch.save(q_net.state_dict(), os.path.join(output_dir, f'q_net_k{k}.pt'))

    if len(k_values) > 1:
        ks = sorted(results.keys())
        means = [results[k]['mean'] for k in ks]
        errs = [results[k]['stderr'] for k in ks]
        plt.figure(figsize=(10, 6))
        plt.errorbar(ks, means, yerr=errs, fmt='o-', capsize=5)
        plt.xlabel('kappa (subsample size)')
        plt.ylabel('Discounted Return')
        plt.title('GMFS-DQN Performance (continuous robotics)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'plot.png'))
        print(f"Saved plot to {output_dir}/plot.png")

    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
