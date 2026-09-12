#!/usr/bin/bash
#SBATCH --job-name=sicto500train
#SBATCH --output=logs/exp1_to500_train.%j.out
#SBATCH --error=logs/exp1_to500_train.%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

config="${1:?Pass input3h_to500 or input4b as the first argument}"
case "${config}" in
    input3h_to500|input4b) ;;
    *) echo "Unsupported configuration: ${config}" >&2; exit 2 ;;
esac
selector="exp1_inputs:${config}"
experiment_name="exp1_${config}"
model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"

if [[ -d "${model_dir}" ]] && compgen -G "${model_dir}/*" > /dev/null; then
    echo "Artifacts already exist in ${model_dir}; refusing to overwrite." >&2
    exit 1
fi

python -m src.models.train --config "${selector}" --members 5 --start_ens_id 0

for seed in 0 1 2 3 4; do
    checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
    [[ -s "${checkpoint}" ]] || { echo "Expected checkpoint is missing or empty: ${checkpoint}" >&2; exit 1; }
done

python -m src.models.evaluate --config "${selector}" --device cuda --overwrite
