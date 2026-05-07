# GMFS: Graphon Mean-Field Subsampling

This repository contains the GMFS implementation used to run the cooperative
robotics coordination experiments, including offline value iteration and
online execution. It is set up so the full sweep set used in the paper can be
re-run from scratch.

## Layout
- `gmfs_core.py` — GMFS core (graphon weights, sampled Bellman operator,
  online execution diagnostics, optional graphon-noise and uniform-sampling
  modes).
- `gmfs_envs.py` — Robotics environment definitions, including the localized
  congestion-sensitive transition and reward.
- `main.py` — Runner for training/evaluation from JSON configs.
- `plot_robotics_results.py` — Plot one scaling sweep.
- `plot_n100_noise_comparison.py` — Legacy n=100 clean / Gaussian / subgaussian
  comparison.
- `plot_graphon_learning_experiment.py` — n=100 graphon-learning noise figures
  (reference, Gaussian, subgaussian; weight maps; reference-minus-perturbed
  return gap).
- `plot_localized_experiment.py` — n=1000 localized experiment figures
  (return-vs-κ for graphon vs. uniform sampling, perception heatmaps,
  power-law layout, noise robustness).
- `plot_robotics_uniform_comparison.py` — Side-by-side comparison of graphon
  and uniform sampling sweeps.
- `slurm/` — Portable Slurm templates. They use a relative `REPO_DIR` and a
  `PYTHON` env override, with no hard-coded site paths or accounts.

## Configs
All run configs live in `configs/`:
- `configs/config_robotics.json`, `configs/config_robotics_n25.json` —
  legacy n=25 smoke / sweep configs.
- `configs/config_robotics_n1000.json` — n=1000 baseline scaling experiment.
- `configs/config_robotics_n1000_uniform.json` — n=1000 with uniform
  (non-graphon) neighbor sampling, used as the baseline for the
  topology-aware comparison.
- `configs/config_robotics_n100_clean.json`,
  `configs/config_robotics_n100_gaussian.json`,
  `configs/config_robotics_n100_subgaussian.json` — n=100 graphon-noise
  robustness.
- `configs/config_robotics_n1000_localized_graphon.json`,
  `configs/config_robotics_n1000_localized_uniform.json` — n=1000 localized
  power-law-x setup, with graphon-weighted vs. uniform sampling.
- `configs/config_robotics_n1000_localized_gaussian.json`,
  `configs/config_robotics_n1000_localized_subgaussian.json` — n=1000
  localized setup under Gaussian and bounded sub-gaussian graphon-weight
  perturbations.

## Quick Start
n=1000 baseline scaling sweep:
```bash
python main.py --config configs/config_robotics_n1000.json
python plot_robotics_results.py --results_dir results_robotics_n1000
```

n=1000 graphon vs. uniform sampling (localized, power-law layout):
```bash
python main.py --config configs/config_robotics_n1000_localized_graphon.json
python main.py --config configs/config_robotics_n1000_localized_uniform.json
python plot_localized_experiment.py
```

n=1000 localized noise robustness (Gaussian / bounded sub-gaussian):
```bash
python main.py --config configs/config_robotics_n1000_localized_gaussian.json
python main.py --config configs/config_robotics_n1000_localized_subgaussian.json
python plot_localized_experiment.py
```

n=100 graphon-noise robustness:
```bash
python main.py --config configs/config_robotics_n100_clean.json
python main.py --config configs/config_robotics_n100_gaussian.json
python main.py --config configs/config_robotics_n100_subgaussian.json
python plot_graphon_learning_experiment.py
```

Run a single κ inside a sweep:
```bash
python main.py --config configs/config_robotics_n1000.json --kappa 30
```

## Slurm
Generic array templates (no site-specific account / partition / paths):
```bash
sbatch slurm/run_robotics_n1000_array.slurm
sbatch slurm/run_robotics_n1000_uniform_array.slurm
sbatch slurm/run_robotics_n1000_localized_sampling_array.slurm
sbatch slurm/run_robotics_n1000_localized_noise_array.slurm
sbatch slurm/run_robotics_n100_noise_array.slurm
sbatch slurm/run_robotics_n25_array.slurm
sbatch slurm/plot_robotics.slurm
```

Override the Python interpreter or repo location:
```bash
PYTHON=/path/to/python REPO_DIR=/path/to/repo \
  sbatch slurm/run_robotics_n1000_array.slurm
```

Add `--partition`, `--account`, or other site-specific `#SBATCH` directives
on the `sbatch` command line as appropriate to your cluster.

## Outputs
Each κ run writes:
- `<results_dir>/kappa_<k>/data.json` and `data_full.json`
- Optional debug plots (when `debug=true`): `g2_trace_k<k>.png`,
  `action_fracs_k<k>.png`
- Perception/coordination plots in `<results_dir>/plots/`
- Summary tables in `<results_dir>/plots/kappa_summary.csv` and `.json`

Generated result directories, plots, logs, and caches are gitignored.

## Graphon
The default neighbor graphon is the radius graphon
\(W(\alpha,\beta)=\mathbf{1}\{\|\alpha-\beta\|_2\le r\}\) (`r=0.3` for the
n=100 grid, `r=0.15` for the n=1000 power-law-x layout). Each agent draws κ
neighbors with replacement from the row-normalized graphon weights; if a row
has zero graphon mass after self-removal, the sampling distribution falls
back to uniform over the other agents.
