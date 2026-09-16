#!/usr/bin/bash
# Submit leakage-free, area-only-loss exp1 reruns with fail-closed dependencies.
set -euo pipefail

check_only=false
if [[ "${1:-}" == "--check-only" ]]; then
    check_only=true
elif (($# != 0)); then
    echo "Usage: $0 [--check-only]" >&2
    exit 2
fi

cd /home/users/yucli/sicpred
mkdir -p logs

original_backup_root="/scratch/users/yucli/sicpred_exp1_leakage_backup_20260912_170731_PDT"
supplemental_backup_root="/scratch/users/yucli/sicpred_exp1_no_month_weights_supplemental_backup_20260912_211011_PDT"
original_configs=(input2 input3a input3b input3c input3d input3e input3f input4a)
supplemental_configs=(input3g input3h_to500 input4b input5 input5_noSIC)

[[ -f "${original_backup_root}/BACKUP_COMPLETE" ]] || {
    echo "Backup is not marked complete: ${original_backup_root}" >&2
    exit 1
}

for config in "${original_configs[@]}"; do
    experiment_name="exp1_${config}"
    model_backup="${original_backup_root}/models/${experiment_name}"
    prediction_backup="${original_backup_root}/predictions/${experiment_name}/UNetRes3_best_predictions.nc"
    diagnostics_backup="${original_backup_root}/predictions/${experiment_name}/diagnostics"

    for seed in 0 1 2 3 4; do
        checkpoint="${model_backup}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        [[ -s "${checkpoint}" ]] || {
            echo "Backup checkpoint is missing or empty: ${checkpoint}" >&2
            exit 1
        }
    done
    [[ -s "${prediction_backup}" ]] || {
        echo "Backup prediction is missing or empty: ${prediction_backup}" >&2
        exit 1
    }
    for metric in acc acc_agg rmse rmse_agg iiee iiee_agg pred_abs truth_abs; do
        diagnostic="${diagnostics_backup}/${metric}.nc"
        [[ -s "${diagnostic}" ]] || {
            echo "Backup diagnostic is missing or empty: ${diagnostic}" >&2
            exit 1
        }
    done
done

[[ -f "${supplemental_backup_root}/BACKUP_COMPLETE" ]] || {
    echo "Backup is not marked complete: ${supplemental_backup_root}" >&2
    exit 1
}

for config in "${supplemental_configs[@]}"; do
    experiment_name="exp1_${config}"
    model_backup="${supplemental_backup_root}/models/${experiment_name}"
    prediction_backup="${supplemental_backup_root}/predictions/${experiment_name}/UNetRes3_best_predictions.nc"
    diagnostics_backup="${supplemental_backup_root}/predictions/${experiment_name}/diagnostics"

    for seed in 0 1 2 3 4; do
        checkpoint="${model_backup}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        [[ -s "${checkpoint}" ]] || {
            echo "Backup checkpoint is missing or empty: ${checkpoint}" >&2
            exit 1
        }
    done
    [[ -s "${prediction_backup}" ]] || {
        echo "Backup prediction is missing or empty: ${prediction_backup}" >&2
        exit 1
    }
    for metric in acc acc_agg rmse rmse_agg iiee iiee_agg; do
        diagnostic="${diagnostics_backup}/${metric}.nc"
        [[ -s "${diagnostic}" ]] || {
            echo "Backup diagnostic is missing or empty: ${diagnostic}" >&2
            exit 1
        }
    done
done

for config in input3a input3b input3c input3d input3e input3f input4a; do
    for metric in acc rmse; do
        interval="${original_backup_root}/bootstrap_intervals/exp1_input2_exp1_${config}_${metric}.nc"
        [[ -s "${interval}" ]] || {
            echo "Backup bootstrap interval is missing or empty: ${interval}" >&2
            exit 1
        }
    done
done

for config in "${supplemental_configs[@]}"; do
    for metric in acc rmse; do
        interval="${supplemental_backup_root}/bootstrap_intervals/exp1_input2_exp1_${config}_${metric}.nc"
        [[ -s "${interval}" ]] || {
            echo "Backup bootstrap interval is missing or empty: ${interval}" >&2
            exit 1
        }
    done
done

if [[ "${check_only}" == true ]]; then
    printf "%s\n" \
        "Backup verification passed for all 13 configurations." \
        "Preprocess: 13-task CPU array (0-12)." \
        "Train/evaluate: 7-task GPU array (0-6), up to two configurations per GPU." \
        "Diagnostics: 13-task CPU array (0-12)." \
        "Bootstrap: 12-task CPU array (0-11), each versus input2."
    exit 0
fi

preprocess_submission="$(sbatch --parsable experiments/exp1_inputs/slurm_leakage_rerun_preprocess.sh)"
preprocess_job_id="${preprocess_submission%%;*}"

train_submission="$(sbatch --parsable \
    --dependency="afterok:${preprocess_job_id}" \
    experiments/exp1_inputs/slurm_leakage_rerun_train_eval.sh)"
train_job_id="${train_submission%%;*}"

postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${train_job_id}" \
    experiments/exp1_inputs/slurm_leakage_rerun_postprocess.sh)"
postprocess_job_id="${postprocess_submission%%;*}"

bootstrap_submission="$(sbatch --parsable \
    --dependency="afterok:${postprocess_job_id}" \
    experiments/exp1_inputs/slurm_leakage_rerun_bootstrap.sh)"
bootstrap_job_id="${bootstrap_submission%%;*}"

printf '%s\n' \
    "preprocess_job_id=${preprocess_job_id}" \
    "train_eval_job_id=${train_job_id}" \
    "diagnostics_job_id=${postprocess_job_id}" \
    "bootstrap_job_id=${bootstrap_job_id}"
