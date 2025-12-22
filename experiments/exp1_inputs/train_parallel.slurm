#!/bin/bash
#SBATCH --job-name=exp1_inputs_ensembling
#SBATCH --partition=serc
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
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
mkdir -p logs/train_eval

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
IDX=$((TASK_ID - 1))

# Map array index -> config stem
CONFIG_STEMS=(input3a input3b input3c input3d input4)

if (( IDX < 0 || IDX >= ${#CONFIG_STEMS[@]} )); then
  echo "Error: TASK_ID=${TASK_ID} out of range (1..${#CONFIG_STEMS[@]})"
  exit 1
fi

STEM="${CONFIG_STEMS[$IDX]}"
CONFIG="src/experiment_configs/exp1_inputs/${STEM}.py"

echo "SLURM job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Using config: ${CONFIG}"
date

# Each task = train then evaluate (multiline)
CMD="$(cat <<EOF
python3 -m src.models.train --config "${CONFIG}" --members 4 --start_ens_id 1

python3 -m src.models.evaluate --config "${CONFIG}"
EOF
)"

echo "----------------------------------------"
echo "${CMD}"
echo "----------------------------------------"

eval "${CMD}"

date
