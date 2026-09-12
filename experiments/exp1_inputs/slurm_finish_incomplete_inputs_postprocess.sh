#!/usr/bin/bash
# Recompute diagnostics and input2-relative bootstrap intervals after exp1 finish.
#SBATCH --job-name=sicexp1post
#SBATCH --output=logs/exp1_finish_postprocess.%j.out
#SBATCH --error=logs/exp1_finish_postprocess.%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=48GB

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u

cd /home/users/yucli/sicpred

model_root="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models"
prediction_root="/scratch/users/yucli/sicpred_model_predictions"
configs=(input3g input4b input3h_to500 input5 input5_noSIC)

for config in "${configs[@]}"; do
    experiment_name="exp1_${config}"
    model_dir="${model_root}/${experiment_name}"
    prediction="${prediction_root}/${experiment_name}/UNetRes3_best_predictions.nc"

    for seed in 0 1 2 3 4; do
        checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        [[ -s "${checkpoint}" ]] || {
            echo "Missing checkpoint: ${checkpoint}" >&2
            exit 1
        }
    done
    [[ -s "${prediction}" ]] || {
        echo "Missing prediction artifact: ${prediction}" >&2
        exit 1
    }

    python -m src.models.diagnostics \
        --config "exp1_inputs:${config}" \
        --data-source dynamic \
        --overwrite
    python -m src.utils.bootstrap \
        --metric acc \
        --config_a exp1_input2 \
        --config_b "${experiment_name}" \
        --transform fisher_z \
        --overwrite
    python -m src.utils.bootstrap \
        --metric rmse \
        --config_a exp1_input2 \
        --config_b "${experiment_name}" \
        --overwrite
done
