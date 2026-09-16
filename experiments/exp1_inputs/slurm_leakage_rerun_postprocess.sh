#!/usr/bin/bash
#SBATCH --job-name=sicexp1post
#SBATCH --output=logs/exp1_leakage_postprocess.%A_%a.out
#SBATCH --error=logs/exp1_leakage_postprocess.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --array=0-12

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

configs=(
    input2 input3a input3b input3c input3d input3e input3f
    input3g input3h_to500 input4a input4b input5 input5_noSIC
)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
selector="exp1_inputs:${config}"
experiment_name="exp1_${config}"
prediction="/scratch/users/yucli/sicpred_model_predictions/${experiment_name}/UNetRes3_best_predictions.nc"
diagnostics_dir="/scratch/users/yucli/sicpred_model_predictions/${experiment_name}/diagnostics"

[[ -s "${prediction}" ]] || {
    echo "Expected prediction artifact is missing or empty: ${prediction}" >&2
    exit 1
}

python -m src.models.diagnostics \
    --config "${selector}" \
    --data-source dynamic \
    --overwrite

# The refactored diagnostics compute reconstruction lazily and no longer write
# these historical caches. Remove the stale leakage-contaminated copies only
# after all replacement metrics have been written successfully.
rm -f "${diagnostics_dir}/pred_abs.nc" "${diagnostics_dir}/truth_abs.nc"
