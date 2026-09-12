#!/usr/bin/bash
#SBATCH --job-name=sicrvgpu
#SBATCH --output=logs/exp2_reanalysis_volume_gpu.%j.out
#SBATCH --error=logs/exp2_reanalysis_volume_gpu.%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5

set -eo pipefail

ml load cuda/12.4.0
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
# Audit every target namespace before starting so completed seeds can be reused,
# while partial or ambiguous runs fail closed instead of mixing checkpoints.
for variant in "${variants[@]}"; do
    experiment_name="exp2_${variant}"
    model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"
    for seed in 0 1 2 3 4; do
        best_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        final_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_final.pth"
        member_artifacts=("${model_dir}"/UNetRes3_"${experiment_name}"_member_"${seed}"_*.pth)

        if [[ -f "${best_checkpoint}" && -f "${final_checkpoint}" ]]; then
            continue
        fi
        if [[ -f "${best_checkpoint}" || -f "${final_checkpoint}" || -e "${member_artifacts[0]}" ]]; then
            echo "Partial or ambiguous artifacts found for ${experiment_name} seed ${seed}." >&2
            echo "Archive that member's artifacts before restarting; runs will not be mixed." >&2
            exit 1
        fi
    done
done

# Keep all 20 short trainings in one allocation to avoid flooding the scheduler.
# A seed with both best and final markers is an unambiguous completed boundary.
for variant in "${variants[@]}"; do
    experiment_name="exp2_${variant}"
    model_dir="/oak/stanford/groups/earlew/yuchen/sicpred/sicpred_models/${experiment_name}"
    for seed in 0 1 2 3 4; do
        best_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_best.pth"
        final_checkpoint="${model_dir}/UNetRes3_${experiment_name}_member_${seed}_final.pth"
        if [[ -f "${best_checkpoint}" && -f "${final_checkpoint}" ]]; then
            echo "Completed ${experiment_name} seed ${seed} already exists; skipping training."
            continue
        fi
        python -m src.models.train \
            --config "exp2_data_volume:${variant}" \
            --members 1 \
            --start_ens_id "${seed}"
    done
done

# Evaluate each completed five-member ensemble before releasing the GPU.
# Overwrite prevents a stale canonical prediction file from surviving a fresh run.
for variant in "${variants[@]}"; do
    python -m src.models.evaluate \
        --config "exp2_data_volume:${variant}" \
        --device cuda \
        --overwrite
done
