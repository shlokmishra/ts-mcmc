#!/bin/bash
#SBATCH --job-name=ts-local-spr-1m-n64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 1M-step guarded local_spr pilot at n=64.
# Recent medium sweep runtime was about 4.2h projected; request 8h for headroom.
#SBATCH --time=08:00:00
#SBATCH --partition=long
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${PROJECT_ROOT:-/home/shlokmishra08/code/ts-mcmc}"
cd "${PROJECT_ROOT}"

# shellcheck source=/dev/null
source "${HOME}/activate_gkdvenv"

export MPLBACKEND=Agg
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/user/shlokmishra08/uv-cache}"

readonly PY="/scratch/user/shlokmishra08/virtual_envs/gkdvenv/bin/python"
readonly MANIFEST="manifests/guarded_local_spr_1m_n64.json"
readonly OUTPUT_ROOT="benchmark_outputs/local_spr_1m_n64_${SLURM_JOB_ID}"

echo "=== $(date -Is) host=$(hostname) cwd=$(pwd) ==="
echo "python=${PY} ($("${PY}" --version 2>&1))"
echo "manifest=${MANIFEST}"
echo "output_root=${OUTPUT_ROOT}"

progress_monitor() {
  local run_dir="${OUTPUT_ROOT}/proposal_local_spr__seed_11__n_64__L_1400__steps_1000000"
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
  --manifest "${MANIFEST}" \
  --make-plots

echo "=== $(date -Is) finished OK ==="
