#!/usr/bin/bash
#SBATCH --job-name=sicpred_input2_notrain
#SBATCH --output=logs/sicpred_input2_notrain.%j.out
#SBATCH --error=logs/sicpred_input2_notrain.%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuchenli713@gmail.com

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

python3 -m src.models.evaluate --config exp1_inputs:input3c
python3 -m src.models.evaluate --config exp1_inputs:input3d
python3 -m src.models.evaluate --config exp1_inputs:input4

bash experiments/exp1_inputs/diagnostics.sh
bash experiments/exp1_inputs/permute_and_predict.sh