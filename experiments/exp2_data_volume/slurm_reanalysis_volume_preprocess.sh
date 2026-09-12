#!/usr/bin/bash
#SBATCH --job-name=sicrvprep
#SBATCH --output=logs/exp2_reanalysis_volume_preprocess.%A_%a.out
#SBATCH --error=logs/exp2_reanalysis_volume_preprocess.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-3

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

variants=(
    reanalysis_volume_r2i1251p1f1
    reanalysis_volume_r2i1281p1f1
    reanalysis_volume_r2i1301p1f1
    reanalysis_volume_r3i1041p1f1
)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"

python -m src.preprocessing.preprocess_cesm_data \
    --config "exp2_data_volume:${variant}"
