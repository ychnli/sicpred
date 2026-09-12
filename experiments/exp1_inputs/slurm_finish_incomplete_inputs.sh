#!/usr/bin/bash
# Finish the audited incomplete exp1 ensembles, then create missing predictions.
# This script intentionally reuses completed members and only overwrites a member
# when its partial artifacts contain no resumable epoch checkpoint.
#SBATCH --job-name=sicexp1finish
#SBATCH --output=logs/exp1_finish_incomplete.%j.out
#SBATCH --error=logs/exp1_finish_incomplete.%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
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

member_prefix() {
    local experiment_name="$1"
    local seed="$2"
    printf '%s/%s/UNetRes3_%s_member_%s' \
        "${model_root}" "${experiment_name}" "${experiment_name}" "${seed}"
}

require_complete_member() {
    local experiment_name="$1"
    local seed="$2"
    local prefix
    prefix="$(member_prefix "${experiment_name}" "${seed}")"
    [[ -s "${prefix}_best.pth" ]] || {
        echo "Missing best checkpoint: ${prefix}_best.pth" >&2
        exit 1
    }
    [[ -s "${prefix}_final.pth" ]] || {
        echo "Missing final checkpoint: ${prefix}_final.pth" >&2
        exit 1
    }
}

require_no_member_artifacts() {
    local experiment_name="$1"
    local seed="$2"
    local prefix
    prefix="$(member_prefix "${experiment_name}" "${seed}")"
    if compgen -G "${prefix}_*.pth" > /dev/null; then
        echo "Unexpected artifacts for ${experiment_name} member ${seed}; refusing to mix runs." >&2
        exit 1
    fi
}

train_fresh_group() {
    local config="$1"
    local first_seed="$2"
    local members="$3"
    local experiment_name="exp1_${config}"
    local last_seed=$((first_seed + members - 1))
    local seed

    for ((seed = first_seed; seed <= last_seed; seed++)); do
        require_no_member_artifacts "${experiment_name}" "${seed}"
    done

    python -m src.models.train \
        --config "exp1_inputs:${config}" \
        --members "${members}" \
        --start_ens_id "${first_seed}" \
        --data-source dynamic \
        --num-workers 2 \
        --prefetch-factor 2

    for ((seed = first_seed; seed <= last_seed; seed++)); do
        require_complete_member "${experiment_name}" "${seed}"
    done
}

resume_member() {
    local config="$1"
    local seed="$2"
    local expected_epoch="$3"
    local additional_epochs="$4"
    local experiment_name="exp1_${config}"
    local prefix
    prefix="$(member_prefix "${experiment_name}" "${seed}")"

    [[ -s "${prefix}_best.pth" ]] || {
        echo "Missing best checkpoint required to resume ${experiment_name} member ${seed}." >&2
        exit 1
    }
    [[ -s "${prefix}_epoch_${expected_epoch}.pth" ]] || {
        echo "Missing expected resume checkpoint: ${prefix}_epoch_${expected_epoch}.pth" >&2
        exit 1
    }
    [[ ! -e "${prefix}_epoch_$((expected_epoch + 1)).pth" ]] || {
        echo "A newer checkpoint exists for ${experiment_name} member ${seed}; re-audit before submitting." >&2
        exit 1
    }
    [[ ! -e "${prefix}_final.pth" ]] || {
        echo "Final checkpoint already exists for ${experiment_name} member ${seed}; re-audit before submitting." >&2
        exit 1
    }

    python -m src.models.train \
        --config "exp1_inputs:${config}" \
        --members 1 \
        --start_ens_id "${seed}" \
        --resume "${additional_epochs}" \
        --data-source dynamic \
        --num-workers 2 \
        --prefetch-factor 2
    require_complete_member "${experiment_name}" "${seed}"
}

evaluate_if_missing() {
    local config="$1"
    local experiment_name="exp1_${config}"
    local prediction="${prediction_root}/${experiment_name}/UNetRes3_best_predictions.nc"
    local seed

    for seed in 0 1 2 3 4; do
        require_complete_member "${experiment_name}" "${seed}"
    done
    if [[ -s "${prediction}" ]]; then
        echo "Complete prediction artifact already exists for ${experiment_name}; skipping evaluation."
        return
    fi

    python -m src.models.evaluate \
        --config "exp1_inputs:${config}" \
        --device cuda \
        --data-source dynamic \
        --num-workers 4 \
        --prefetch-factor 2 \
        --overwrite
    [[ -s "${prediction}" ]] || {
        echo "Evaluation did not create ${prediction}." >&2
        exit 1
    }
}

# Audit snapshot (2026-09-11): input3g members 0-3 and input4b members 0-4
# are complete. The two partial members below have per-epoch optimizer state.
for seed in 0 1 2 3; do
    require_complete_member exp1_input3g "${seed}"
done
for seed in 0 1 2 3 4; do
    require_complete_member exp1_input4b "${seed}"
done

resume_member input3g 4 2 8
resume_member input3h_to500 0 8 2
train_fresh_group input3h_to500 1 4
train_fresh_group input5 0 5
train_fresh_group input5_noSIC 0 5

for config in input3g input4b input3h_to500 input5 input5_noSIC; do
    evaluate_if_missing "${config}"
done
