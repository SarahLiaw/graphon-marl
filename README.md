# GMFS: Graphon Mean-Field Subsampling

This repository contains the GMFS implementation for the robotics warehouse coordination task, with offline value iteration (Algorithm 1) and online execution (Algorithm 2).

## Structure
- `gmfs_core.py`: GMFS core logic (graphon weights, sampled Bellman operator, online execution diagnostics).
- `gmfs_envs.py`: Robotics environment definitions (state/action, reward, congestion-sensitive transition).
- `main.py`: Runner for training/evaluation from JSON configs.
- `plot_robotics_results.py`: Plots one scaling sweep.
- `plot_n100_noise_comparison.py`: Plots the n=100 clean/noisy comparison.
- `config_robotics.json`: Legacy n=25 smoke/default config.
- `config_robotics_n25.json`: Legacy n=25 sweep config.
- `config_robotics_n1000.json`: n=1000 scaling experiment.
- `config_robotics_n100_clean.json`, `config_robotics_n100_gaussian.json`, `config_robotics_n100_subgaussian.json`: n=100 robustness experiments.
- `slurm/`: Generic Slurm templates. They intentionally avoid account names, personal paths, and site-specific partitions.

## Quick Start
Run the n=1000 scaling sweep:
```bash
python main.py --config config_robotics_n1000.json
python plot_robotics_results.py --results_dir results_robotics_n1000
```

Run the legacy n=25 default smoke config:
```bash
python main.py --config config_robotics.json
```

Run or plot the legacy n=25 sweep:
```bash
python main.py --config config_robotics_n25.json
python plot_robotics_results.py --results_dir results_robotics_n25
```

Run one n=1000 kappa:
```bash
python main.py --config config_robotics_n1000.json --kappa 30
```

Run the n=100 clean/noisy sweeps:
```bash
python main.py --config config_robotics_n100_clean.json
python main.py --config config_robotics_n100_gaussian.json
python main.py --config config_robotics_n100_subgaussian.json
python plot_n100_noise_comparison.py
```

Slurm entrypoints:
```bash
sbatch slurm/run_robotics_n1000_array.slurm
sbatch slurm/run_robotics_n25_array.slurm
sbatch slurm/run_robotics_n100_noise_array.slurm
sbatch slurm/plot_robotics.slurm
```

The Slurm templates default to `python` from the active environment. Override with:
```bash
PYTHON=/path/to/python sbatch slurm/run_robotics_n1000_array.slurm
```

## Outputs
Each kappa run writes:
- `results_robotics_n1000/kappa_<k>/data.json` and `data_full.json`
- Debug plots (when `debug=true`) like `g2_trace_k<k>.png`, `action_fracs_k<k>.png`
- Perception/coordination plots in `<results_dir>/plots/`
- Summary tables in `<results_dir>/plots/kappa_summary.csv` and `.json`

## Notes
- The perception heatmaps visualize **one focal agent** (center agent by default).
- \(\kappa\) controls **sampling resolution**, not the physical interaction radius.
- Generated result directories, plots, logs, and caches are intentionally gitignored.

## Graphon Notes
This project uses a radial graphon for sampling neighbor interactions. The key pieces are:
- **Graphon function**: \(W(x,y)=\mathbf{1}\{\|x-y\|_2 \le r\}\) with \(r=0.3\).
- **Sampling**: each agent draws \(\kappa\) neighbors with replacement from the normalized graphon weights.
- **Effect**: larger \(\kappa\) improves the accuracy of the empirical neighborhood estimate \(\hat g\).

Example usage inside the runner:
```python
positions = build_grid_positions(n_agents=1000, grid_rows=25, grid_cols=40)
graphon = RadialGraphon(radius=radius)
W = graphon.get_normalized_weights(n_agents, positions=positions)
```
