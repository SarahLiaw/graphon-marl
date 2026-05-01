"""
Plot n=100 clean vs Gaussian vs Subgaussian noise comparison.

Usage:
  python plot_n100_noise_comparison.py [--out_dir plots_n100_noise]
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt


EXPERIMENTS = {
    'clean':       ('results_robotics_n100_clean',       'Clean'),
    'gaussian':    ('results_robotics_n100_gaussian',    r'Gaussian noise ($\sigma=0.05$)'),
    'subgaussian': ('results_robotics_n100_subgaussian', r'Subgaussian noise ($\sigma=0.05$)'),
}
COLORS = {'clean': 'tab:blue', 'gaussian': 'tab:orange', 'subgaussian': 'tab:green'}
MARKERS = {'clean': 'o', 'gaussian': 's', 'subgaussian': '^'}


def load_results(results_dir):
    data = {}
    for path in glob.glob(os.path.join(results_dir, 'kappa_*', 'data_full.json')):
        with open(path) as f:
            payload = json.load(f)
        for k_str, kd in payload.get('results', {}).items():
            data[int(k_str)] = kd
    return data


def save_fig(base_path):
    plt.savefig(base_path + '.png', dpi=150, bbox_inches='tight')
    plt.savefig(base_path + '.pdf', bbox_inches='tight')
    print(f'Saved {base_path}.png / .pdf')


def plot_comparison(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}
    for key, (rdir, _) in EXPERIMENTS.items():
        r = load_results(rdir)
        if r:
            all_results[key] = r
        else:
            print(f'  WARNING: no results found in {rdir}')

    if not all_results:
        print('No results found; run experiments first.')
        return

    plt.figure(figsize=(9, 6))
    for key, results in all_results.items():
        label = EXPERIMENTS[key][1]
        ks = sorted(results.keys())
        means = [results[k]['mean'] for k in ks]
        errs  = [results[k]['stderr'] for k in ks]
        plt.errorbar(ks, means, yerr=errs, fmt=f'{MARKERS[key]}-',
                     color=COLORS[key], capsize=5, label=label, linewidth=1.8)

    plt.xlabel(r'$\kappa$ (subsample size)', fontsize=13)
    plt.ylabel('Discounted Return', fontsize=13)
    plt.title(r'GMFS Robustness to Graphon Noise ($n=100$)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    save_fig(os.path.join(out_dir, 'n100_noise_comparison'))

    # Also plot absolute gap (clean - noisy)
    if 'clean' in all_results:
        plt.figure(figsize=(9, 5))
        clean = all_results['clean']
        ks_clean = sorted(clean.keys())
        for key in ('gaussian', 'subgaussian'):
            if key not in all_results:
                continue
            noisy = all_results[key]
            label = EXPERIMENTS[key][1]
            ks = [k for k in ks_clean if k in noisy]
            gaps = [clean[k]['mean'] - noisy[k]['mean'] for k in ks]
            plt.plot(ks, gaps, f'{MARKERS[key]}-', color=COLORS[key],
                     label=f'Gap: clean - {label}', linewidth=1.8)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel(r'$\kappa$ (subsample size)', fontsize=13)
        plt.ylabel('Return gap (clean - noisy)', fontsize=13)
        plt.title(r'Performance degradation from graphon noise ($n=100$)', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        save_fig(os.path.join(out_dir, 'n100_noise_gap'))

    # CSV summary
    rows = []
    for key, results in all_results.items():
        for k in sorted(results.keys()):
            rows.append({
                'condition': key,
                'kappa': k,
                'mean': results[k]['mean'],
                'std': results[k].get('std', float('nan')),
                'stderr': results[k]['stderr'],
                'n_training_seeds': results[k].get('n_training_seeds', 1),
                'n_total_returns': len(results[k].get('returns', [])),
            })
    if rows:
        out_csv = os.path.join(out_dir, 'n100_noise_summary.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {out_csv}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='plots_n100_noise')
    args = parser.parse_args()
    plot_comparison(args.out_dir)


if __name__ == '__main__':
    main()
