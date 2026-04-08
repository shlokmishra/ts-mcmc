#!/bin/bash
#SBATCH --job-name=ts-mcmc-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Medium comparison manifest is many long chains; increase if jobs time out.
# Grace: short max 2h, medium max 1d, long max 7d (see hprc.tamu.edu Grace batch guide).
#SBATCH --time=48:00:00
#
# CPU job — use a CPU queue (not gpu). Match your working jobs: no bogus --account lines.
# Add #SBATCH --account=... only if `sbatch` or your PI requires a project account.
#SBATCH --partition=long
#
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Optional notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.edu
#
# Slurm 20.02+: run batch stage from repo root so logs and paths resolve.
# #SBATCH --chdir=/home/shlokmishra08/code/ts-mcmc

set -euo pipefail

# Project root (repo with benchmark_outputs/, scripts/)
readonly PROJECT_ROOT="${PROJECT_ROOT:-/home/shlokmishra08/code/ts-mcmc}"
cd "${PROJECT_ROOT}"

# Load same toolchain + venv as interactive runs (PyTorch module provides libs python links against).
# shellcheck source=/dev/null
source "${HOME}/activate_gkdvenv"

# Batch-safe matplotlib; keep uv cache on scratch
export MPLBACKEND=Agg
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/user/shlokmishra08/uv-cache}"

# Same interpreter as ~/activate_gkdvenv (adjust if you move the venv).
readonly PY="/scratch/user/shlokmishra08/virtual_envs/gkdvenv/bin/python"

echo "=== $(date -Is) host=$(hostname) cwd=$(pwd) ==="
echo "python=${PY} ($("${PY}" --version 2>&1))"

# Unique output directory per job to avoid overwrites across restarts.
readonly OUTPUT_ROOT="benchmark_outputs/mcmc_medium_comparison_${SLURM_JOB_ID}"

# Lightweight progress heartbeat (prints to slurm-%x-%j.out).
progress_monitor() {
  local expected=40
  while true; do
    local done
    done=$(ls -d "${OUTPUT_ROOT}"/proposal_* 2>/dev/null | wc -l || true)
    echo "[$(date -Is)] progress: ${done}/${expected} (output_root=${OUTPUT_ROOT})"
    sleep 300
  done
}
progress_monitor &
MON_PID=$!
trap 'kill "${MON_PID}" 2>/dev/null || true' EXIT

# 1) Run all manifest configs
"${PY}" scripts/run_mcmc_experiments.py \
  --output-root "${OUTPUT_ROOT}" \
  --manifest benchmark_outputs/manifests/medium_local_comparison.json \
  --make-plots

# 2) Aggregate proposal_* run directories
"${PY}" scripts/aggregate_mcmc_runs.py \
  "${OUTPUT_ROOT}"/proposal_* \
  --output-dir "${OUTPUT_ROOT}"/aggregate

# 3) Paper-style medium comparison table
"${PY}" scripts/make_medium_comparison_table.py \
  --input-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}"/aggregate/medium_comparison_table.csv

echo "=== $(date -Is) finished OK ==="
