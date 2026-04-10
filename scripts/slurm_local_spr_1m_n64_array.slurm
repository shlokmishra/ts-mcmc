#!/bin/bash
#SBATCH --job-name=ts-local-spr-1m-n64
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 3 independent guarded 1M local_spr chains at n=64, one seed per array task.
# 200k post-fix run projected about 2h per 1M chain; request 8h for safe headroom.
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${PROJECT_ROOT:-/home/shlokmishra08/code/ts-mcmc}"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${HOME}/activate_gkdvenv"

export MPLBACKEND=Agg
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/user/shlokmishra08/uv-cache}"

readonly PY="/scratch/user/shlokmishra08/virtual_envs/gkdvenv/bin/python"
readonly SEEDS=(11 22 33)
readonly SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
readonly OUTPUT_ROOT="benchmark_outputs/local_spr_1m_n64_seed_${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "=== $(date -Is) host=$(hostname) cwd=$(pwd) ==="
echo "python=${PY} ($("${PY}" --version 2>&1))"
echo "seed=${SEED}"
echo "output_root=${OUTPUT_ROOT}"

progress_monitor() {
  local run_dir="${OUTPUT_ROOT}/proposal_local_spr__seed_${SEED}__n_64__L_1400__steps_1000000"
  while true; do
    if [[ -f "${run_dir}/run_status.json" ]]; then
      echo "[$(date -Is)] run_status=$(cat "${run_dir}/run_status.json")"
    elif [[ -f "${run_dir}/checkpoints/latest.json" ]]; then
      echo "[$(date -Is)] checkpoint=$(cat "${run_dir}/checkpoints/latest.json")"
    else
      echo "[$(date -Is)] waiting_for_first_checkpoint output_root=${OUTPUT_ROOT}"
    fi
    sleep 300
  done
}
progress_monitor &
MON_PID=$!
trap 'kill "${MON_PID}" 2>/dev/null || true' EXIT

"${PY}" scripts/run_mcmc_experiments.py \
  --output-root "${OUTPUT_ROOT}" \
  --proposal-type local_spr \
  --seed "${SEED}" \
  --sample-size 64 \
  --seq-length 1400 \
  --steps 1000000 \
  --burn-in 200000 \
  --time-move local \
  --time-step-size 1.0 \
  --mutation-step-size 0.1 \
  --spr-moves-per-step 2 \
  --acceptance-window 1000 \
  --checkpoint-every 10000 \
  --guard-magnitude-limit 1e100 \
  --make-plots

echo "=== $(date -Is) finished OK ==="
