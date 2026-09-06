#!/usr/bin/bash
#SBATCH --job-name=sicpred_input2_notrain
#SBATCH --output=logs/sicpred_input2_notrain.%j.out
#SBATCH --error=logs/sicpred_input2_notrain.%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

bash experiments/exp1_inputs/run_all_notrain.sh
