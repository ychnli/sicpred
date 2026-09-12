#!/usr/bin/bash
# Run all reanalysis-volume diagnostics serially in one CPU allocation.
#SBATCH --job-name=sicrvfinishpost
#SBATCH --output=logs/exp2_reanalysis_volume_finish_post.%j.out
#SBATCH --error=logs/exp2_reanalysis_volume_finish_post.%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB

set -eo pipefail

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

for variant in "${variants[@]}"; do
    experiment_name="exp2_${variant}"
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
        --config "exp2_data_volume:${variant}" \
        --data-source dynamic \
        --ensemble-mean \
        --baselines \
        --overwrite
done
