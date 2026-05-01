#!/bin/bash
#SBATCH --job-name=gmfs-n25
#SBATCH --output=logs/n25_%A_%a.out
#SBATCH --error=logs/n25_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-8

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${REPO_DIR}/config_robotics_n25.json"
KAPPA_LIST=(1 5 10 15 20 25 30 60 100)
KAPPA="${KAPPA_LIST[$SLURM_ARRAY_TASK_ID]}"

cd "${REPO_DIR}"
mkdir -p logs results_robotics_n25/logs
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Running n=25 GMFS for kappa=${KAPPA}"
"${PYTHON}" main.py --config "${CONFIG}" --kappa "${KAPPA}"
echo "Finished n=25 kappa=${KAPPA}"
