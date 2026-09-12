#!/usr/bin/bash
# Submit the audited exp1 and exp2 completion workflows.
# Review this file and the referenced batch scripts before running it.

set -euo pipefail

cd /home/users/yucli/sicpred
mkdir -p logs

# The relevant preprocessing jobs (42823282 and 42833757) were audited as
# COMPLETED on 2026-09-11. The dynamic pipeline consumes their normalized data,
# so there is no preprocessing submission in this completion graph.
exp1_submission="$(sbatch --parsable \
    experiments/exp1_inputs/slurm_finish_incomplete_inputs.sh)"
exp1_job_id="${exp1_submission%%;*}"

exp2_submission="$(sbatch --parsable \
    experiments/exp2_data_volume/slurm_finish_reanalysis_volume.sh)"
exp2_job_id="${exp2_submission%%;*}"

exp1_post_submission="$(sbatch --parsable \
    --dependency="afterok:${exp1_job_id}" \
    experiments/exp1_inputs/slurm_finish_incomplete_inputs_postprocess.sh)"
exp1_post_job_id="${exp1_post_submission%%;*}"

exp2_post_submission="$(sbatch --parsable \
    --dependency="afterok:${exp2_job_id}" \
    experiments/exp2_data_volume/slurm_finish_reanalysis_volume_postprocess.sh)"
exp2_post_job_id="${exp2_post_submission%%;*}"

printf '%s\n' \
    "exp1_gpu_job_id=${exp1_job_id}" \
    "exp1_postprocess_job_id=${exp1_post_job_id}" \
    "exp2_gpu_job_id=${exp2_job_id}" \
    "exp2_postprocess_job_id=${exp2_post_job_id}"
