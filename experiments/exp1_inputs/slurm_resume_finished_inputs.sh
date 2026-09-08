#!/usr/bin/bash
#SBATCH --job-name=sicresume
#SBATCH --output=logs/sicpred_resume_finished.%A_%a.out
#SBATCH --error=logs/sicpred_resume_finished.%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5
#SBATCH --array=0-2

# Resume only the completed jobs from array 42348878. input3g is deliberately
# excluded because its icethick inputs are under investigation. input4b was
# still running when this recovery array was submitted. input5 and
# input5_noSIC were subsequently stopped because they contain the same
# corrupted icethick channel found during that investigation.

set -eo pipefail

ml load cuda/12.4.0
source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        config="input3e"
        partial_member=3
        checkpoint_epoch=8
        resume_epochs=2
        fresh_start=4
        fresh_members=1
        ;;
    1)
        config="input3f"
        partial_member=2
        checkpoint_epoch=6
        resume_epochs=4
        fresh_start=3
        fresh_members=2
        ;;
    2)
        config="input4a"
        partial_member=3
        checkpoint_epoch=5
        resume_epochs=5
        fresh_start=4
        fresh_members=1
        ;;
    *)
        echo "Unexpected array task ID: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

selector="exp1_inputs:${config}"
experiment_name="exp1_${config}"
model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"
expected_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${partial_member}_epoch_${checkpoint_epoch}.pth"

if [[ ! -f "${expected_checkpoint}" ]]; then
    echo "Expected restart checkpoint is missing: ${expected_checkpoint}" >&2
    exit 1
fi

if compgen -G "${model_dir}/*member_${partial_member}_epoch_$((checkpoint_epoch + 1)).pth" > /dev/null; then
    echo "A newer checkpoint now exists for ${config} member ${partial_member}; refusing a stale fixed-length resume." >&2
    exit 1
fi

for ((member_id = fresh_start; member_id < fresh_start + fresh_members; member_id++)); do
    if compgen -G "${model_dir}/*member_${member_id}_*" > /dev/null; then
        echo "Artifacts already exist for intended fresh member ${member_id}; refusing to overwrite them." >&2
        exit 1
    fi
done

python -m src.models.train \
    --config "${selector}" \
    --members 1 \
    --start_ens_id "${partial_member}" \
    --resume "${resume_epochs}"

python -m src.models.train \
    --config "${selector}" \
    --members "${fresh_members}" \
    --start_ens_id "${fresh_start}"

python -m src.models.evaluate --config "${selector}" --device cuda
python -m src.models.diagnostics --config "${selector}"
python -m src.utils.bootstrap \
    --metric acc \
    --config_a exp1_input2 \
    --config_b "${experiment_name}" \
    --transform fisher_z
python -m src.utils.bootstrap \
    --metric rmse \
    --config_a exp1_input2 \
    --config_b "${experiment_name}"
