#!/usr/bin/bash
#SBATCH --job-name=sicrvpost
#SBATCH --output=logs/exp2_reanalysis_volume_postprocess.%A_%a.out
#SBATCH --error=logs/exp2_reanalysis_volume_postprocess.%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-3

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

variants=(
    reanalysis_volume_r2i1251p1f1
    reanalysis_volume_r2i1281p1f1
    reanalysis_volume_r2i1301p1f1
    reanalysis_volume_r3i1041p1f1
)
variant="${variants[$SLURM_ARRAY_TASK_ID]}"
selector="exp2_data_volume:${variant}"
experiment_name="exp2_${variant}"
model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"

for seed in 0 1 2 3 4; do
    checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Expected checkpoint is missing: ${checkpoint}" >&2
        exit 1
    fi
done

predictions="/scratch/users/yucli/sicpred_model_predictions/${experiment_name}/UNetRes3_best_predictions.nc"
if [[ ! -f "${predictions}" ]]; then
    echo "Expected predictions are missing: ${predictions}" >&2
    exit 1
fi

python -m src.models.diagnostics --config "${selector}" --ensemble-mean --baselines --overwrite
