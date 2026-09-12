#!/usr/bin/bash
# Submit exp1 to500 runs with fail-closed Slurm dependencies.
set -euo pipefail

input3h_time="20:00:00"
input3h_constraint="GPU_SKU:A100_SXM4"
input4b_time="20:00:00"
input4b_constraint="GPU_SKU:A100_SXM4"

cd /home/users/yucli/sicpred
mkdir -p logs

for experiment_name in exp1_input3h_to500 exp1_input4b; do
    model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"
    if [[ -d "${model_dir}" ]] && compgen -G "${model_dir}/*" > /dev/null; then
        echo "Artifacts already exist in ${model_dir}; refusing to submit." >&2
        exit 1
    fi
done

# download_submission="$(sbatch --parsable experiments/exp1_inputs/slurm_to500_download.sh)"
# download_job_id="${download_submission%%;*}"
# preprocess_submission="$(sbatch --parsable --dependency="afterok:${download_job_id}" experiments/exp1_inputs/slurm_to500_preprocess.sh)"
preprocess_job_id="42823282"

input3h_training_submission="$(sbatch --parsable \
    --dependency="afterok:${preprocess_job_id}" \
    --time="${input3h_time}" \
    --constraint="${input3h_constraint}" \
    experiments/exp1_inputs/slurm_to500_train.sh input3h_to500)"
input3h_training_job_id="${input3h_training_submission%%;*}"

input4b_training_submission="$(sbatch --parsable \
    --time="${input4b_time}" \
    --constraint="${input4b_constraint}" \
    experiments/exp1_inputs/slurm_to500_train.sh input4b)"
input4b_training_job_id="${input4b_training_submission%%;*}"

input3h_postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${input3h_training_job_id}" \
    experiments/exp1_inputs/slurm_to500_postprocess.sh input3h_to500)"
input3h_postprocess_job_id="${input3h_postprocess_submission%%;*}"

input4b_postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${input4b_training_job_id}" \
    experiments/exp1_inputs/slurm_to500_postprocess.sh input4b)"
input4b_postprocess_job_id="${input4b_postprocess_submission%%;*}"

# printf "%s\n" \
#     "download_job_id=${download_job_id}" \
#     "preprocess_job_id=${preprocess_job_id}" \
#     "input3h_training_job_id=${input3h_training_job_id}" \
#     "input3h_postprocess_job_id=${input3h_postprocess_job_id}" \
#     "input4b_training_job_id=${input4b_training_job_id}" \
#     "input4b_postprocess_job_id=${input4b_postprocess_job_id}"
