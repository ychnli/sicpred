#!/usr/bin/bash
#SBATCH --job-name=sicexp1boot
#SBATCH --output=logs/exp1_leakage_bootstrap.%A_%a.out
#SBATCH --error=logs/exp1_leakage_bootstrap.%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --array=0-11

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(
    input3a input3b input3c input3d input3e input3f input3g
    input3h_to500 input4a input4b input5 input5_noSIC
)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
experiment_name="exp1_${config}"

python -m src.utils.bootstrap \
    --metric acc \
    --config_a exp1_input2 \
    --config_b "${experiment_name}" \
    --transform none \
    --random_seed 42 \
    --overwrite
python -m src.utils.bootstrap \
    --metric rmse \
    --config_a exp1_input2 \
    --config_b "${experiment_name}" \
    --random_seed 42 \
    --overwrite
