#!/usr/bin/bash
#SBATCH --job-name=sicexp1train
#SBATCH --output=logs/exp1_leakage_train_eval.%A_%a.out
#SBATCH --error=logs/exp1_leakage_train_eval.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5
#SBATCH --array=0-6

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(
    input2 input3a input3b input3c input3d input3e input3f
    input3g input3h_to500 input4a input4b input5 input5_noSIC
)

for offset in 0 1; do
    config_index=$((SLURM_ARRAY_TASK_ID * 2 + offset))
    if ((config_index >= ${#configs[@]})); then
        continue
    fi

    config="${configs[$config_index]}"
    selector="exp1_inputs:${config}"
    experiment_name="exp1_${config}"
    model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"

    python -m src.models.train \
        --config "${selector}" \
        --members 5 \
        --start_ens_id 0 \
        --data-source dynamic \
        --num-workers 2 \
        --prefetch-factor 2 \
        --overwrite

    for seed in 0 1 2 3 4; do
        checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        [[ -s "${checkpoint}" ]] || {
            echo "Expected checkpoint is missing or empty: ${checkpoint}" >&2
            exit 1
        }
    done

    python -m src.models.evaluate \
        --config "${selector}" \
        --device cuda \
        --data-source dynamic \
        --num-workers 4 \
        --prefetch-factor 2 \
        --overwrite
done
