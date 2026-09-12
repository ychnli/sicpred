#!/usr/bin/bash
#SBATCH --job-name=sicicethickprep
#SBATCH --output=logs/exp1_icethick_preprocess.%A_%a.out
#SBATCH --error=logs/exp1_icethick_preprocess.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

python -m src.preprocessing.preprocess_cesm_data --config "exp1_inputs:input3g" --overwrite