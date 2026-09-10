#!/usr/bin/bash
#SBATCH --job-name=sic_detrend_triage
#SBATCH --output=logs/sic_detrend_triage.%A_%a.out
#SBATCH --error=logs/sic_detrend_triage.%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --partition=serc
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --array=0-4

set -eo pipefail

source /home/groups/earlew/yuchen/miniconda3/etc/profile.d/conda.sh
conda activate sicpred_env_conda
set -u
cd /home/users/yucli/sicpred

variables=(icefrac sst psl z500 t2m)
variable="${variables[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python experiments/exp1_inputs/detrending_triage/generate_and_compare.py \
    --variable "${variable}"
