set -euo pipefail

vars=(sst psl geopotential t2m)
lags=(lag1 lag2 lag3 lag4 lag5 lag6)

for lag in "${lags[@]}"; do
    for var in "${vars[@]}"; do
        # python -m src.models.permute_and_predict --config src/experiment_configs/exp1_inputs/input4.py --var_name "${var}_${lag}" --overwrite
        python3 -m src.models.diagnostics --config src/experiment_configs/exp1_inputs/input4.py --permute-var "${var}_${lag}"
    done
done