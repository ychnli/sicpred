#!/usr/bin/bash
#SBATCH --job-name=sicto500post
#SBATCH --output=logs/exp1_to500_postprocess.%j.out
#SBATCH --error=logs/exp1_to500_postprocess.%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

set -eo pipefail

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

for seed in 0 1 2 3 4; do
    checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
    [[ -s "${checkpoint}" ]] || { echo "Expected checkpoint is missing or empty: ${checkpoint}" >&2; exit 1; }
done

python -m src.models.diagnostics --config "${selector}"
python -m src.utils.bootstrap --metric acc --config_a exp1_input2 --config_b "${experiment_name}" --transform none
python -m src.utils.bootstrap --metric rmse --config_a exp1_input2 --config_b "${experiment_name}"
