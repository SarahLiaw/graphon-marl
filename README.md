# GMFS: Graphon Mean-Field Subsampling

This repository contains the GMFS implementation for the robotics warehouse coordination task, with offline value iteration (Algorithm 1) and online execution (Algorithm 2).

## Structure
- `gmfs_core.py`: GMFS core logic (graphon weights, sampled Bellman operator, online execution diagnostics).
- `gmfs_envs.py`: Robotics environment definitions (state/action, reward, congestion-sensitive transition).
- `main.py`: Runner for training/evaluation using `config_robotics.json`.
- `plot_robotics_results.py`
- `config_robotics.json`: Current experiment configuration.
- `results_robotics/`

## Quick Start
Run a single kappa:
```bash
python main.py --config config_robotics.json --kappa 8
```

Run a sweep (k values inside config):
```bash
python main.py --config config_robotics.json
```

Generate all plots (PNG + PDF):
```bash
python plot_robotics_results.py --results_dir results_robotics
```

## Outputs
Each kappa run writes:
- `results_robotics/kappa_<k>/data.json` and `data_full.json`
- Debug plots (when `debug=true`) like `g2_trace_k<k>.png`, `action_fracs_k<k>.png`
- Perception/coordination plots in `results_robotics/plots/`

Logs (kept): `linear_run_k*_more.log`

## Notes
- The perception heatmaps visualize **one focal agent** (center agent by default).
- \(\kappa\) controls **sampling resolution**, not the physical interaction radius.

## Graphon Notes
This project uses a radial graphon for sampling neighbor interactions. The key pieces are:
- **Graphon function**: \(W(x,y)=\mathbf{1}\{\|x-y\|_2 \le r\}\) with \(r=0.3\).
- **Sampling**: each agent draws \(\kappa\) neighbors with replacement from the normalized graphon weights.
- **Effect**: larger \(\kappa\) improves the accuracy of the empirical neighborhood estimate \(\hat g\).

Example usage inside the runner:
```python
positions = build_grid_positions(grid_size)
graphon = RadialGraphon(radius=radius)
W = graphon.get_normalized_weights(n_agents, positions=positions)
```
