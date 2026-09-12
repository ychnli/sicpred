#!/usr/bin/bash
# Submit the isolated reanalysis-volume CESM workflow with fail-closed dependencies.

set -euo pipefail

repo_root="/home/users/yucli/sicpred"
cd "${repo_root}"
mkdir -p logs

preprocess_submission="$(sbatch --parsable experiments/exp2_data_volume/slurm_reanalysis_volume_preprocess.sh)"
preprocess_job_id="${preprocess_submission%%;*}"

training_submission="$(sbatch --parsable \
    --dependency="afterok:${preprocess_job_id}" \
    experiments/exp2_data_volume/slurm_reanalysis_volume_train.sh)"
training_job_id="${training_submission%%;*}"

postprocess_submission="$(sbatch --parsable \
    --dependency="afterok:${training_job_id}" \
    experiments/exp2_data_volume/slurm_reanalysis_volume_postprocess.sh)"
postprocess_job_id="${postprocess_submission%%;*}"

echo "preprocess_job_id=${preprocess_job_id}"
echo "training_job_id=${training_job_id}"
echo "postprocess_job_id=${postprocess_job_id}"
