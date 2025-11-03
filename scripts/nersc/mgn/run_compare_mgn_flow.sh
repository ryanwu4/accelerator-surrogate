#!/bin/bash
#SBATCH -A m669
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/run_compare_mgn_flow_%j.out
#SBATCH --error=logs/run_compare_mgn_flow_%j.err

# Minimal SLURM harness to compare MeshGraphNet and conditional flow models.

set -euo pipefail

export SLURM_CPU_BIND="cores"

module load conda
module load cudatoolkit
module load pytorch/2.3.1

source activate ignn

REPO_ROOT=/global/homes/t/tiffan/repo/slac_bunch_modeling
SURROGATE_DIR=${REPO_ROOT}/accelerator-surrogate
EVAL_SCRIPT=${SURROGATE_DIR}/src/evaluation/compare_mgn_vs_flow.py

PARTICLE_DIR=/global/cfs/cdirs/m669/tiffan/data/particle_data_eval
FLOW_CHECKPOINT=${REPO_ROOT}/accelerator_flow_model/conditional_flow_model.pt
FLOW_SCALERS=${REPO_ROOT}/accelerator_flow_model/conditional_flow_scalers.pkl
MGN_CHECKPOINT=${SURROGATE_DIR}/results/mgn/latest/checkpoint.pth
METADATA_PATH=${REPO_ROOT}/metadata.json
SETTINGS_PATH=${REPO_ROOT}/settings.pt
OUTPUT_DIR=${SURROGATE_DIR}/results/mgn_vs_flow_eval
LIMIT=0  # Set to >0 to cap the number of particle files

for required_path in "${EVAL_SCRIPT}" "${PARTICLE_DIR}" "${FLOW_CHECKPOINT}" "${FLOW_SCALERS}" "${MGN_CHECKPOINT}" "${METADATA_PATH}" "${SETTINGS_PATH}"; do
    if [ ! -e "${required_path}" ]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p logs "${OUTPUT_DIR}"

# Ensure both project roots are importable so accelerator_flow_model and src.* resolve.
export PYTHONPATH=${REPO_ROOT}:${SURROGATE_DIR}:${PYTHONPATH:-}

cd "${REPO_ROOT}"

if [ "${LIMIT}" -gt 0 ]; then
    LIMIT_ARGS=("--limit" "${LIMIT}")
else
    LIMIT_ARGS=()
fi

python "${EVAL_SCRIPT}" \
    --particle-dir "${PARTICLE_DIR}" \
    --flow-checkpoint "${FLOW_CHECKPOINT}" \
    --flow-scalers "${FLOW_SCALERS}" \
    --mgn-checkpoint "${MGN_CHECKPOINT}" \
    --metadata "${METADATA_PATH}" \
    --settings "${SETTINGS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --representative-count 5 \
    "${LIMIT_ARGS[@]}"
