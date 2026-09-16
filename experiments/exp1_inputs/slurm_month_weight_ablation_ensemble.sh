#!/usr/bin/bash
#SBATCH --job-name=sicmonthens
#SBATCH --output=logs/month_weight_ablation_ensemble.%A_%a.out
#SBATCH --error=logs/month_weight_ablation_ensemble.%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5
#SBATCH --array=0-1

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(input2_weights_train input2_no_month_weights)
experiments=(exp1_input2_month_weights_train exp1_input2_no_month_weights)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
experiment_name="${experiments[$SLURM_ARRAY_TASK_ID]}"
selector="exp1_inputs:${config}"
model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"

seed_zero_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_0_best.pth"
[[ -s "${seed_zero_checkpoint}" ]] || {
    echo "Missing completed seed-0 checkpoint: ${seed_zero_checkpoint}" >&2
    exit 1
}

for seed in 1 2 3 4; do
    if compgen -G "${model_dir}/UNetRes3_${experiment_name}_member_${seed}_*.pth" > /dev/null; then
        echo "Seed ${seed} already has artifacts; refusing to mix runs." >&2
        exit 1
    fi
done

python -m src.models.train \
    --config "${selector}" \
    --members 4 \
    --start_ens_id 1 \
    --data-source dynamic \
    --num-workers 2 \
    --prefetch-factor 2

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

python -m src.models.diagnostics --config "${selector}"
