#!/usr/bin/bash
# Submit the input3g, input5, and input5_noSIC icethick-fix reruns.
set -euo pipefail

training_time="20:00:00"
training_constraint="GPU_SKU:A100_SXM4"
preprocess_job_id="42833757"

cd /home/users/yucli/sicpred
mkdir -p logs

input3g_training_submission="$(sbatch --parsable \
    --time="${training_time}" \
    --constraint="${training_constraint}" \
    experiments/exp1_inputs/slurm_icethick_rerun_train.sh input3g)"
input3g_training_job_id="${input3g_training_submission%%;*}"

input5_training_submission="$(sbatch --parsable \
    --dependency="afterok:${preprocess_job_id}" \
    --time="${training_time}" \
    --constraint="${training_constraint}" \
    experiments/exp1_inputs/slurm_icethick_rerun_train.sh input5)"
input5_training_job_id="${input5_training_submission%%;*}"

input5_noSIC_training_submission="$(sbatch --parsable \
    --dependency="afterok:${preprocess_job_id}" \
    --time="${training_time}" \
    --constraint="${training_constraint}" \
    experiments/exp1_inputs/slurm_icethick_rerun_train.sh input5_noSIC)"
input5_noSIC_training_job_id="${input5_noSIC_training_submission%%;*}"

input3g_postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${input3g_training_job_id}" \
    experiments/exp1_inputs/slurm_icethick_rerun_postprocess.sh input3g)"
input3g_postprocess_job_id="${input3g_postprocess_submission%%;*}"

input5_postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${input5_training_job_id}" \
    experiments/exp1_inputs/slurm_icethick_rerun_postprocess.sh input5)"
input5_postprocess_job_id="${input5_postprocess_submission%%;*}"

input5_noSIC_postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${input5_noSIC_training_job_id}" \
    experiments/exp1_inputs/slurm_icethick_rerun_postprocess.sh input5_noSIC)"
input5_noSIC_postprocess_job_id="${input5_noSIC_postprocess_submission%%;*}"

printf "%s\n" \
    "preprocess_job_id=${preprocess_job_id}" \
    "input3g_training_job_id=${input3g_training_job_id}" \
    "input3g_postprocess_job_id=${input3g_postprocess_job_id}" \
    "input5_training_job_id=${input5_training_job_id}" \
    "input5_postprocess_job_id=${input5_postprocess_job_id}" \
    "input5_noSIC_training_job_id=${input5_noSIC_training_job_id}" \
    "input5_noSIC_postprocess_job_id=${input5_noSIC_postprocess_job_id}"
