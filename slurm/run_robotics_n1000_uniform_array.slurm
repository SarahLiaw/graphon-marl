#!/bin/bash
#SBATCH --job-name=gmfs-n1000-uniform
#SBATCH --output=logs/n1000_uniform_%A_%a.out
#SBATCH --error=logs/n1000_uniform_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${REPO_DIR}/configs/config_robotics_n1000_uniform.json"
RESULTS_DIR="${REPO_DIR}/results_robotics_n1000_uniform"
KAPPA_LIST=(1 3 5 10 15 20 25 30 60 100)
KAPPA="${KAPPA_LIST[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${RESULTS_DIR}/logs"
cd "${REPO_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    cp "${BASH_SOURCE[0]}" "${RESULTS_DIR}/run_robotics_n1000_uniform_array.slurm"
    cp "${CONFIG}" "${RESULTS_DIR}/config_robotics_n1000_uniform.json"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Running n=1000 uniform-sampling GMFS for kappa=${KAPPA} (5 training seeds x 20 eval runs)"
"${PYTHON}" main.py --config "${CONFIG}" --kappa "${KAPPA}"
echo "Finished n=1000 uniform-sampling kappa=${KAPPA}"
