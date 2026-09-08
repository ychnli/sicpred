#!/usr/bin/bash
#SBATCH --job-name=sicnwin
#SBATCH --output=logs/sicpred_new_inputs.%A_%a.out
#SBATCH --error=logs/sicpred_new_inputs.%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5
#SBATCH --array=0-6

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(input3e input3f input3g input4a input4b input5 input5_noSIC)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
selector="exp1_inputs:${config}"
experiment_name="exp1_${config}"

python -m src.models.train --config "$selector" --members 5
python -m src.models.evaluate --config "$selector" --device cuda
python -m src.models.diagnostics --config "$selector"
python -m src.utils.bootstrap \
    --metric acc \
    --config_a exp1_input2 \
    --config_b "$experiment_name" \
    --transform fisher_z
python -m src.utils.bootstrap \
    --metric rmse \
    --config_a exp1_input2 \
    --config_b "$experiment_name"
