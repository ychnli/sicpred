#!/bin/bash
#SBATCH --job-name=exp1_inputs_ensembling
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --mem=16G
#SBATCH --time=15:00:00
#SBATCH --output=logs/training_logs/%x_%A_%a.out
#SBATCH --error=logs/training_logs/%x_%A_%a.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=yucli@stanford.edu

set -euo pipefail

set +u
source /home/groups/earlew/yuchen/miniconda3/bin/activate
conda activate sicpred_env_conda
set -u

cd /home/users/yucli/sicpred

bash experiments/exp2_data_volume/train4.sh