#!/bin/bash
#SBATCH --job-name=gmfs-plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
mkdir -p logs
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${PYTHON}" plot_robotics_results.py --results_dir results_robotics_n1000
"${PYTHON}" plot_n100_noise_comparison.py
