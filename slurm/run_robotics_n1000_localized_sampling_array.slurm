#!/bin/bash
#SBATCH --job-name=gmfs-localized
#SBATCH --output=logs/n1000_localized_%A_%a.out
#SBATCH --error=logs/n1000_localized_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
KAPPA_LIST=(1 3 5 10 15 20 25 30 60 100)
MODE_LIST=(graphon uniform)

MODE_IDX=$((SLURM_ARRAY_TASK_ID / ${#KAPPA_LIST[@]}))
KAPPA_IDX=$((SLURM_ARRAY_TASK_ID % ${#KAPPA_LIST[@]}))
MODE="${MODE_LIST[$MODE_IDX]}"
KAPPA="${KAPPA_LIST[$KAPPA_IDX]}"
CONFIG="${REPO_DIR}/configs/config_robotics_n1000_localized_${MODE}.json"
RESULTS_DIR="${REPO_DIR}/results_robotics_n1000_localized_${MODE}"

mkdir -p "${REPO_DIR}/logs" "${RESULTS_DIR}/logs"
cd "${REPO_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
fi

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    cp "${BASH_SOURCE[0]}" "${RESULTS_DIR}/run_robotics_n1000_localized_sampling_array.slurm"
    cp "${CONFIG}" "${RESULTS_DIR}/config_robotics_n1000_localized_${MODE}.json"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

echo "Running localized n=1000 ${MODE} GMFS for kappa=${KAPPA}"
"${PYTHON}" main.py --config "${CONFIG}" --kappa "${KAPPA}"
echo "Finished localized n=1000 ${MODE} kappa=${KAPPA}"
