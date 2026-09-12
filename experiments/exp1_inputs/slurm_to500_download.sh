#!/usr/bin/bash
#SBATCH --job-name=sicto500dl
#SBATCH --output=logs/exp1_to500_download.%A_%a.out
#SBATCH --error=logs/exp1_to500_download.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-39

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

python -m src.download.download_cesm_data \
    --variables to500 \
    --num-workers "${SLURM_ARRAY_TASK_COUNT}" \
    --worker-id "${SLURM_ARRAY_TASK_ID}"
