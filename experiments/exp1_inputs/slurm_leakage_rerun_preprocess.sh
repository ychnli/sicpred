#!/usr/bin/bash
#SBATCH --job-name=sicexp1prep
#SBATCH --output=logs/exp1_leakage_preprocess.%A_%a.out
#SBATCH --error=logs/exp1_leakage_preprocess.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --array=0-12

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(
    input2 input3a input3b input3c input3d input3e input3f
    input3g input3h_to500 input4a input4b input5 input5_noSIC
)
config="${configs[$SLURM_ARRAY_TASK_ID]}"

python -m src.preprocessing.preprocess_cesm_data \
    --config "exp1_inputs:${config}" \
    --overwrite
