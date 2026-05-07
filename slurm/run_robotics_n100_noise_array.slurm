#!/bin/bash
#SBATCH --job-name=gmfs-n100-noise
#SBATCH --output=logs/n100_noise_%A_%a.out
#SBATCH --error=logs/n100_noise_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-17

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIGS=(
  "configs/config_robotics_n100_clean.json"
  "configs/config_robotics_n100_gaussian.json"
  "configs/config_robotics_n100_subgaussian.json"
)
KAPPA_LIST=(1 5 10 15 20 25)

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / ${#KAPPA_LIST[@]}))
KAPPA_IDX=$((SLURM_ARRAY_TASK_ID % ${#KAPPA_LIST[@]}))
CONFIG="${REPO_DIR}/${CONFIGS[$CONFIG_IDX]}"
KAPPA="${KAPPA_LIST[$KAPPA_IDX]}"

cd "${REPO_DIR}"
mkdir -p logs
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Running n=100 GMFS for config=${CONFIGS[$CONFIG_IDX]} kappa=${KAPPA}"
"${PYTHON}" main.py --config "${CONFIG}" --kappa "${KAPPA}"
echo "Finished n=100 config=${CONFIGS[$CONFIG_IDX]} kappa=${KAPPA}"
