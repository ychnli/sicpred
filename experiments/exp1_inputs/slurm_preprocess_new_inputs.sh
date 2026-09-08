#!/usr/bin/bash
#SBATCH --job-name=sicpred_preprocess_new_inputs
#SBATCH --output=logs/sicpred_preprocess_new_inputs.%A_%a.out
#SBATCH --error=logs/sicpred_preprocess_new_inputs.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-5

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(input2 input3a input3b input3c input3d input4)
config="${configs[$SLURM_ARRAY_TASK_ID]}"

python -m src.preprocessing.preprocess_cesm_data \
    --config "exp1_inputs:${config}"
