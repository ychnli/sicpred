#!/bin/bash
#
# Each array task runs one worker of the Python downloader.
#
# Usage examples:
#  sbatch --array=1-10 run_download_cesm_array.slurm      # submit 10 workers
#  sbatch --array=1-10%3 run_download_cesm_array.slurm    # limit concurrent tasks to 3
#

#SBATCH --job-name=cesm_download
#SBATCH --partition=serc
#SBATCH --array=1-16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cesm_download_logs/%x_%A_%a.out
#SBATCH --error=logs/cesm_download_logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuchenli713@gmail.com

set -euo pipefail

source /home/groups/earlew/yuchen/miniconda3/bin/activate
conda activate sicpred_with_regridding

cd /home/users/yucli/sicpred
mkdir -p logs/cesm_download_logs/

# get worker id and total workers (script expects 0-indexed worker-id)
NUM_WORKERS=${SLURM_ARRAY_TASK_COUNT:-1}          # default to 1 worker
WORKER_ID=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))    # convert SLURM 1..N to 0..N-1

echo "NUM_WORKERS=${NUM_WORKERS}, WORKER_ID=${WORKER_ID}"
date

python -m src.download.download_cesm_data --num-workers ${NUM_WORKERS} --worker-id ${WORKER_ID}

date