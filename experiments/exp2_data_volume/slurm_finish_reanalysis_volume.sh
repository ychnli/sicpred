#!/usr/bin/bash
# Finish the audited incomplete reanalysis-volume ensemble and evaluate all four.
#SBATCH --job-name=sicrvfinish
#SBATCH --output=logs/exp2_reanalysis_volume_finish.%j.out
#SBATCH --error=logs/exp2_reanalysis_volume_finish.%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u

cd /home/users/yucli/sicpred

model_root="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models"
prediction_root="/scratch/users/yucli/sicpred_model_predictions"
variants=(
    reanalysis_volume_r2i1251p1f1
    reanalysis_volume_r2i1281p1f1
    reanalysis_volume_r2i1301p1f1
    reanalysis_volume_r3i1041p1f1
)

require_complete_member() {
    local experiment_name="$1"
    local seed="$2"
    local prefix="${model_root}/${experiment_name}/UNetRes3_${experiment_name}_member_${seed}"
    [[ -s "${prefix}_best.pth" && -s "${prefix}_final.pth" ]] || {
        echo "Incomplete member: ${experiment_name} seed ${seed}" >&2
        exit 1
    }
}

require_no_member_artifacts() {
    local experiment_name="$1"
    local seed="$2"
    local prefix="${model_root}/${experiment_name}/UNetRes3_${experiment_name}_member_${seed}"
    if compgen -G "${prefix}_*.pth" > /dev/null; then
        echo "Unexpected artifacts for ${experiment_name} seed ${seed}; re-audit before submitting." >&2
        exit 1
    fi
}

# The first three variants and member 0 of the fourth are complete.
for variant in "${variants[@]:0:3}"; do
    experiment_name="exp2_${variant}"
    for seed in 0 1 2 3 4; do
        require_complete_member "${experiment_name}" "${seed}"
    done
done
experiment_name="exp2_reanalysis_volume_r3i1041p1f1"
require_complete_member "${experiment_name}" 0

# Member 1 stopped before the first interval checkpoint (epoch 10), leaving
# only a best checkpoint, so it cannot be resumed with optimizer state.
member1_prefix="${model_root}/${experiment_name}/UNetRes3_${experiment_name}_member_1"
[[ -s "${member1_prefix}_best.pth" ]] || {
    echo "Expected best-only artifact is missing for ${experiment_name} seed 1." >&2
    exit 1
}
if compgen -G "${member1_prefix}_epoch_*.pth" > /dev/null || [[ -e "${member1_prefix}_final.pth" ]]; then
    echo "Seed 1 state changed since the audit; re-audit before submitting." >&2
    exit 1
fi
for seed in 2 3 4; do
    require_no_member_artifacts "${experiment_name}" "${seed}"
done

# Train all four unfinished members together so the eager input store is loaded
# once. Overwrite is scoped to members 1-4; the completed member 0 is untouched.
python -m src.models.train \
    --config exp2_data_volume:reanalysis_volume_r3i1041p1f1 \
    --members 4 \
    --start_ens_id 1 \
    --overwrite \
    --data-source dynamic \
    --num-workers 2 \
    --prefetch-factor 2

for seed in 0 1 2 3 4; do
    require_complete_member "${experiment_name}" "${seed}"
done

for variant in "${variants[@]}"; do
    experiment_name="exp2_${variant}"
    prediction="${prediction_root}/${experiment_name}/UNetRes3_best_predictions.nc"
    if [[ -s "${prediction}" ]]; then
        echo "Prediction artifact already exists for ${experiment_name}; skipping evaluation."
        continue
    fi
    python -m src.models.evaluate \
        --config "exp2_data_volume:${variant}" \
        --device cuda \
        --data-source dynamic \
        --num-workers 4 \
        --prefetch-factor 2 \
        --overwrite
    [[ -s "${prediction}" ]] || {
        echo "Evaluation did not create ${prediction}." >&2
        exit 1
    }
done
