"""
Ablate kappa for GMFS-DQN and plot the resulting policy behavior.

For each kappa in the sweep:
  1. Train a GMFS-DQN policy.
  2. Deploy it online (Algorithm-2-style rollout) and record diagnostics:
     mean population workload trajectory and per-step reward trajectory.

Produces:
  - return_vs_kappa.png: discounted return (mean +/- stderr) vs kappa.
  - policy_trace_vs_kappa.png: mean workload s(t) trajectory under the greedy
    policy for each kappa, overlaid, showing how the deployed policy's
    population-level behavior changes as the subsample size k grows.

Returns and workload traces are averaged over `n_train_seeds` independent
DQN training runs (times num_eval_runs eval rollouts each) per kappa, to
distinguish real kappa effects from single-seed DQN training variance.
Error bars (return_vs_kappa.png: +/- stderr; policy_trace_vs_kappa.png:
shaded +/- stderr band) reflect this.

Usage:
  python ablate_kappa_dqn.py --config config_robotics_dqn.json
  python ablate_kappa_dqn.py --config config_robotics_dqn.json --kappas 2 4 8 16 24 --n_train_seeds 3
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--kappas', type=int, nargs='+', default=None,
                         help='Override the k sweep, e.g. --kappas 2 4 8 16 24')
    parser.add_argument('--n_train_seeds', type=int, default=1,
                         help='Number of independent DQN training seeds per kappa, '
                              'averaged together to reduce training variance.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.get('output_dir', 'results_robotics_dqn')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    n_agents = cfg['n_agents']
    grid_size = cfg.get('grid_size', 5)
    radius = cfg.get('radius', 0.3)
    positions = build_grid_positions(grid_size)
    weights = radial_graphon_weights(positions, radius)

    kappas = args.kappas if args.kappas is not None else cfg['k_values']
    num_eval_runs = cfg.get('num_eval_runs', 10)
    gamma = cfg.get('gamma', 0.95)

    trans = ContinuousRoboticsTransition(
        step_rate=cfg.get('step_rate', 0.5), noise_std=cfg.get('noise_std', 0.05))
    rew = ContinuousRoboticsReward(
        W=cfg.get('W', 20.0), work_cost=cfg.get('work_cost', 5.0))

    base_seed = cfg.get('seed', 0)
    n_train_seeds = args.n_train_seeds

    return_means, return_errs = [], []
    policy_trace_means, policy_trace_errs = {}, {}  # kappa -> per-timestep mean/stderr across all (train_seed, eval_run)

    for k in kappas:
        print(f"\n=== Ablation: kappa={k} ({n_train_seeds} training seed(s)) ===")
        returns = []
        traces = []
        for seed_offset in range(n_train_seeds):
            train_seed = base_seed + seed_offset * 100003  # decorrelate training seeds
            q_net = train_dqn(
                n_agents=n_agents, kappa=k, weights=weights,
                transition_func=trans, reward_func=rew,
                n_episodes=cfg.get('n_episodes', 200),
                horizon=cfg.get('train_horizon', 50),
                gamma=gamma, lr=cfg.get('lr', 1e-3),
                batch_size=cfg.get('batch_size', 128),
                eps_decay_episodes=cfg.get('eps_decay_episodes', 150),
                target_sync_every=cfg.get('target_sync_every', 10),
                seed=train_seed,
            )

            for run in range(num_eval_runs):
                ret, diag = online_execution_dqn(
                    n_agents=n_agents, kappa=k, horizon=cfg.get('eval_horizon', 100),
                    weights=weights, q_net=q_net, transition_func=trans, reward_func=rew,
                    gamma=gamma, seed=train_seed + 1000 + run,
                    return_diagnostics=True,
                )
                returns.append(ret)
                traces.append(diag['mean_state'])

        returns = np.array(returns)
        traces = np.array(traces)  # (n_train_seeds * num_eval_runs, horizon)
        n_total = len(returns)

        return_means.append(float(np.mean(returns)))
        return_errs.append(float(np.std(returns) / np.sqrt(n_total)))
        policy_trace_means[k] = traces.mean(axis=0)
        policy_trace_errs[k] = traces.std(axis=0) / np.sqrt(n_total)
        print(f" k={k}: return={return_means[-1]:.3f} +/- {return_errs[-1]:.3f} (n={n_total})")

    # Plot 1: return vs kappa
    plt.figure(figsize=(8, 6))
    plt.errorbar(kappas, return_means, yerr=return_errs, fmt='o-', capsize=5)
    plt.xlabel('kappa (subsample size)')
    plt.ylabel('Discounted Return')
    plt.title('GMFS-DQN: Return vs kappa')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'return_vs_kappa.png'), dpi=150)
    plt.close()

    # Plot 2: deployed policy behavior (mean population workload trajectory) per kappa,
    # with shaded +/- 1 stderr bands across (training seed, eval run).
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(kappas)))
    t = np.arange(cfg.get('eval_horizon', 100))
    for color, k in zip(cmap, kappas):
        mean_trace = policy_trace_means[k]
        err_trace = policy_trace_errs[k]
        plt.plot(t, mean_trace, color=color, label=f'k={k}')
        plt.fill_between(t, mean_trace - err_trace, mean_trace + err_trace, color=color, alpha=0.2)
    plt.xlabel('timestep t')
    plt.ylabel('mean population workload  E[s_t]')
    plt.title('GMFS-DQN Deployed Policy: Mean Workload Trajectory vs kappa (shaded: +/- 1 stderr)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'policy_trace_vs_kappa.png'), dpi=150)
    plt.close()

    print(f"\nSaved plots to {plots_dir}/return_vs_kappa.png and {plots_dir}/policy_trace_vs_kappa.png")

    with open(os.path.join(output_dir, 'ablation_data.json'), 'w') as f:
        json.dump({
            'kappas': kappas,
            'n_train_seeds': n_train_seeds,
            'return_mean': return_means,
            'return_stderr': return_errs,
            'policy_trace_mean': {str(k): v.tolist() for k, v in policy_trace_means.items()},
            'policy_trace_stderr': {str(k): v.tolist() for k, v in policy_trace_errs.items()},
        }, f, indent=2)


if __name__ == '__main__':
    main()
